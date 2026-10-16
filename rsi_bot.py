#!/usr/bin/env python3
"""RSI-бот для спотовой торговли на Binance.

Стратегия (только long, спот):
  * ПОКУПКА — RSI закрытой свечи пересёк уровень RSI_BUY снизу вверх
    (выход из зоны перепроданности) и позиции нет.
  * ПРОДАЖА — RSI пересёк уровень RSI_SELL сверху вниз (выход из зоны
    перекупленности), либо сработал стоп-лосс / тейк-профит.

Режимы (переменная MODE):
  * paper   — по умолчанию: реальные котировки, сделки только симулируются.
  * signals — бот только присылает сигналы; сделку вы подтверждаете и
              исполняете через Binance MCP в Claude Code (/rsi-signals).
  * testnet — ордера на тестовой бирже Binance (testnet.binance.vision).
  * live    — реальные деньги; дополнительно требует CONFIRM_LIVE=yes.

Команды:
  python3 rsi_bot.py            — запустить бота
  python3 rsi_bot.py backtest   — прогнать стратегию на истории
  python3 rsi_bot.py signals    — ожидающие сигналы (JSON)
  python3 rsi_bot.py confirm ID --price P --amount A [--quote Q]
  python3 rsi_bot.py reject ID
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger("rsi_bot")

FEE_RATE = 0.001  # комиссия Binance spot 0.1% (для paper и бэктеста)


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass
class Config:
    symbol: str = os.getenv("SYMBOL", "BTC/USDT")
    timeframe: str = os.getenv("TIMEFRAME", "15m")
    rsi_period: int = int(os.getenv("RSI_PERIOD", 14))
    rsi_buy: float = _env_float("RSI_BUY", 30)
    rsi_sell: float = _env_float("RSI_SELL", 70)
    order_usdt: float = _env_float("ORDER_USDT", 20)
    stop_loss_pct: float = _env_float("STOP_LOSS_PCT", 3)
    take_profit_pct: float = _env_float("TAKE_PROFIT_PCT", 6)
    mode: str = os.getenv("MODE", "paper").lower()
    poll_seconds: int = int(os.getenv("POLL_SECONDS", 30))
    signal_ttl_minutes: float = _env_float("SIGNAL_TTL_MINUTES", 30)
    state_file: Path = Path(os.getenv("STATE_FILE", "rsi_bot_state.json"))
    api_key: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("BINANCE_API_SECRET", ""))
    telegram_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))

    def validate(self) -> None:
        if self.mode not in ("paper", "signals", "testnet", "live"):
            raise ValueError(f"MODE должен быть paper, signals, testnet или live, а не {self.mode!r}")
        if not 0 < self.rsi_buy < self.rsi_sell < 100:
            raise ValueError("Нужно 0 < RSI_BUY < RSI_SELL < 100")
        if self.mode in ("testnet", "live") and not (self.api_key and self.api_secret):
            raise ValueError("Для testnet/live задайте BINANCE_API_KEY и BINANCE_API_SECRET")
        if self.mode == "live" and os.getenv("CONFIRM_LIVE", "").lower() != "yes":
            raise ValueError("Режим live торгует реальными деньгами: задайте CONFIRM_LIVE=yes")


# --------------------------------------------------------------------------- #
# Индикатор и сигналы
# --------------------------------------------------------------------------- #

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI по Уайлдеру (как в TradingView и на Binance)."""
    delta = close.diff().to_numpy()
    rsi = pd.Series(float("nan"), index=close.index)
    if len(close) <= period:
        return rsi
    gains, losses = delta.clip(min=0), -delta.clip(max=0)
    # Затравка — простое среднее первых `period` изменений, далее сглаживание Уайлдера.
    avg_gain, avg_loss = gains[1:period + 1].mean(), losses[1:period + 1].mean()
    for i in range(period, len(close)):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rsi.iloc[i] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    return rsi


def rsi_signal(prev_rsi: float, curr_rsi: float, buy_level: float, sell_level: float) -> str | None:
    """'buy' / 'sell' при пересечении уровней, иначе None."""
    if pd.isna(prev_rsi) or pd.isna(curr_rsi):
        return None
    if prev_rsi < buy_level <= curr_rsi:
        return "buy"
    if prev_rsi > sell_level >= curr_rsi:
        return "sell"
    return None


def exit_reason(entry_price: float, price: float, cfg: Config) -> str | None:
    if cfg.stop_loss_pct > 0 and price <= entry_price * (1 - cfg.stop_loss_pct / 100):
        return "стоп-лосс"
    if cfg.take_profit_pct > 0 and price >= entry_price * (1 + cfg.take_profit_pct / 100):
        return "тейк-профит"
    return None


# --------------------------------------------------------------------------- #
# Биржа
# --------------------------------------------------------------------------- #

def make_exchange(cfg: Config):
    import ccxt

    params = {"enableRateLimit": True}
    if cfg.mode in ("testnet", "live"):
        params.update(apiKey=cfg.api_key, secret=cfg.api_secret)
    exchange = ccxt.binance(params)
    if cfg.mode == "testnet":
        exchange.set_sandbox_mode(True)
    exchange.load_markets()
    return exchange


def fetch_candles(exchange, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def notify(cfg: Config, text: str) -> None:
    logger.info(text.replace("\n", " | "))
    if not (cfg.telegram_token and cfg.telegram_chat_id):
        return
    url = f"https://api.telegram.org/bot{cfg.telegram_token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": cfg.telegram_chat_id, "text": text}).encode()
    try:
        urllib.request.urlopen(url, data=data, timeout=10).read()
    except Exception as exc:  # уведомление не должно ронять бота
        logger.warning("Не удалось отправить сообщение в Telegram: %s", exc)


# --------------------------------------------------------------------------- #
# Состояние и торговля
# --------------------------------------------------------------------------- #

def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"position": None, "last_candle": None, "realized_pnl": 0.0, "trades": 0,
            "signals": []}


def save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    tmp.replace(path)


class Trader:
    def __init__(self, cfg: Config, exchange):
        self.cfg = cfg
        self.ex = exchange
        self.base = cfg.symbol.split("/")[0]

    def buy(self, price: float) -> dict:
        """Покупка на ORDER_USDT. Возвращает {'amount', 'price'}."""
        if self.cfg.mode == "paper":
            amount = self.cfg.order_usdt * (1 - FEE_RATE) / price
            return {"amount": amount, "price": price}
        order = self.ex.create_market_buy_order_with_cost(self.cfg.symbol, self.cfg.order_usdt)
        filled = float(order.get("filled") or 0)
        avg = float(order.get("average") or price)
        # Если комиссия списана в базовой монете — продать можно только остаток.
        fee = order.get("fee") or {}
        if fee.get("currency") == self.base:
            filled -= float(fee.get("cost") or 0)
        return {"amount": filled, "price": avg}

    def sell(self, amount: float, price: float) -> dict:
        """Продажа позиции. Возвращает {'amount', 'price', 'proceeds'}."""
        if self.cfg.mode == "paper":
            return {"amount": amount, "price": price, "proceeds": amount * price * (1 - FEE_RATE)}
        free = float(self.ex.fetch_balance().get(self.base, {}).get("free") or 0)
        qty = float(self.ex.amount_to_precision(self.cfg.symbol, min(amount, free)))
        order = self.ex.create_market_sell_order(self.cfg.symbol, qty)
        avg = float(order.get("average") or price)
        proceeds = float(order.get("cost") or qty * avg)
        fee = order.get("fee") or {}
        if fee.get("currency") == self.cfg.symbol.split("/")[1]:
            proceeds -= float(fee.get("cost") or 0)
        return {"amount": qty, "price": avg, "proceeds": proceeds}


def evaluate(df: pd.DataFrame, state: dict, cfg: Config) -> dict | None:
    """Решение стратегии на текущем шаге: {'side', 'price', 'reason'} или None.

    df — свечи, последняя может быть ещё не закрыта. Отмечает обработанную
    свечу в state['last_candle'].
    """
    closed = df.iloc[:-1]  # сигналы RSI только по закрытым свечам
    if len(closed) < cfg.rsi_period + 2:
        return None
    last_price = float(df["close"].iloc[-1])
    pos = state["position"]

    # Стоп-лосс / тейк-профит проверяем по текущей цене на каждом опросе.
    if pos:
        reason = exit_reason(pos["entry_price"], last_price, cfg)
        if reason:
            return {"side": "sell", "price": last_price, "reason": reason}

    candle_ts = str(closed["ts"].iloc[-1])
    if candle_ts == state.get("last_candle"):
        return None  # эту свечу уже обработали
    state["last_candle"] = candle_ts

    rsi = compute_rsi(closed["close"], cfg.rsi_period)
    signal = rsi_signal(rsi.iloc[-2], rsi.iloc[-1], cfg.rsi_buy, cfg.rsi_sell)
    logger.info("%s %s close=%.4f RSI=%.2f signal=%s", cfg.symbol, candle_ts,
                closed["close"].iloc[-1], rsi.iloc[-1], signal)
    if (signal == "buy" and not pos) or (signal == "sell" and pos):
        return {"side": signal, "price": last_price, "reason": f"RSI={rsi.iloc[-1]:.1f}"}
    return None


def process_candles(df: pd.DataFrame, state: dict, cfg: Config, trader: Trader) -> dict:
    """Один шаг автоторговли (paper / testnet / live)."""
    decision = evaluate(df, state, cfg)
    if not decision:
        return state
    if decision["side"] == "sell":
        return _close(state, cfg, trader, decision["price"], decision["reason"])
    fill = trader.buy(decision["price"])
    state["position"] = {"amount": fill["amount"], "entry_price": fill["price"],
                         "cost": cfg.order_usdt, "opened": state["last_candle"]}
    notify(cfg, f"🟢 [{cfg.mode}] КУПЛЕНО {fill['amount']:.6f} {cfg.symbol} "
                f"по {fill['price']:.4f}\n{decision['reason']}")
    return state


# --------------------------------------------------------------------------- #
# Режим сигналов: бот не торгует, сделку подтверждает человек через Binance MCP
# --------------------------------------------------------------------------- #

def pending_signal(state: dict) -> dict | None:
    return next((s for s in state.get("signals", []) if s["status"] == "pending"), None)


def expire_signals(state: dict, cfg: Config, now: float | None = None) -> None:
    now = time.time() if now is None else now
    for sig in state.get("signals", []):
        if sig["status"] == "pending" and now - sig["created"] > cfg.signal_ttl_minutes * 60:
            sig["status"] = "expired"


def process_signals(df: pd.DataFrame, state: dict, cfg: Config, now: float | None = None) -> dict:
    """Один шаг режима signals: вместо ордера создаёт сигнал на подтверждение."""
    state.setdefault("signals", [])
    expire_signals(state, cfg, now)
    decision = evaluate(df, state, cfg)
    if not decision:
        return state
    now = time.time() if now is None else now
    current = pending_signal(state)
    if current and current["side"] == decision["side"]:
        return state  # такой сигнал уже ждёт подтверждения (например, повторный стоп-лосс)
    last_same = next((s for s in reversed(state["signals"]) if s["side"] == decision["side"]), None)
    if (last_same and last_same["status"] == "rejected"
            and now - last_same["created"] < cfg.signal_ttl_minutes * 60):
        return state  # вы только что отклонили такой сигнал — не повторяем его каждый опрос
    if current:
        current["status"] = "superseded"
    pos = state["position"]
    sig = {
        "id": uuid.uuid4().hex[:8],
        "side": decision["side"],
        "symbol": cfg.symbol,
        "mcp_symbol": cfg.symbol.replace("/", ""),
        "price": decision["price"],
        "reason": decision["reason"],
        "created": now,
        "status": "pending",
    }
    if sig["side"] == "buy":
        sig["quote_usdt"] = cfg.order_usdt
        action = f"КУПИТЬ {cfg.symbol} на {cfg.order_usdt} USDT по рынку"
    else:
        sig["amount"] = pos["amount"]
        action = f"ПРОДАТЬ {pos['amount']:.6f} {cfg.symbol} по рынку"
    state["signals"].append(sig)
    state["signals"] = state["signals"][-50:]  # не раздуваем файл состояния
    notify(cfg, f"🔔 Сигнал {sig['id']}: {action}\nЦена ~{sig['price']:.4f}, {sig['reason']}\n"
                f"Действует {cfg.signal_ttl_minutes} мин. Подтвердите в Claude Code: /rsi-signals")
    return state


def confirm_signal(state: dict, signal_id: str, price: float, amount: float,
                   quote: float | None = None) -> dict:
    """Записать исполненный через MCP сигнал: фактическая цена и количество базовой монеты.

    quote — потраченные (buy) или полученные (sell) USDT с учётом комиссии;
    по умолчанию price * amount.
    """
    sig = next((s for s in state.get("signals", []) if s["id"] == signal_id), None)
    if not sig:
        raise ValueError(f"Сигнал {signal_id} не найден")
    if sig["status"] != "pending":
        raise ValueError(f"Сигнал {signal_id} уже в статусе {sig['status']}")
    quote = price * amount if quote is None else quote
    if sig["side"] == "buy":
        state["position"] = {"amount": amount, "entry_price": price, "cost": quote,
                             "opened": state.get("last_candle")}
    else:
        pos = state["position"]
        if not pos:
            raise ValueError("Нет открытой позиции для продажи")
        state["realized_pnl"] += quote - pos["cost"]
        state["trades"] += 1
        state["position"] = None
    sig.update(status="executed", fill_price=price, fill_amount=amount, fill_quote=quote)
    return state


def reject_signal(state: dict, signal_id: str) -> dict:
    sig = next((s for s in state.get("signals", []) if s["id"] == signal_id), None)
    if not sig or sig["status"] != "pending":
        raise ValueError(f"Нет ожидающего сигнала {signal_id}")
    sig["status"] = "rejected"
    return state


def _close(state: dict, cfg: Config, trader: Trader, price: float, reason: str) -> dict:
    pos = state["position"]
    fill = trader.sell(pos["amount"], price)
    pnl = fill["proceeds"] - pos["cost"]
    state["realized_pnl"] += pnl
    state["trades"] += 1
    state["position"] = None
    notify(cfg, f"🔴 [{cfg.mode}] ПРОДАНО {fill['amount']:.6f} {cfg.symbol} по {fill['price']:.4f} "
                f"({reason})\nP&L сделки: {pnl:+.2f} USDT, всего: {state['realized_pnl']:+.2f} USDT")
    return state


def run(cfg: Config) -> None:
    cfg.validate()
    exchange = make_exchange(cfg)
    trader = Trader(cfg, exchange)
    notify(cfg, f"🤖 RSI-бот запущен: {cfg.symbol} {cfg.timeframe}, режим {cfg.mode}, "
                f"RSI({cfg.rsi_period}) {cfg.rsi_buy}/{cfg.rsi_sell}, ордер {cfg.order_usdt} USDT")
    while True:
        try:
            df = fetch_candles(exchange, cfg.symbol, cfg.timeframe, limit=cfg.rsi_period * 20)
            # Перечитываем файл: в режиме signals его меняют команды confirm/reject.
            state = load_state(cfg.state_file)
            if cfg.mode == "signals":
                state = process_signals(df, state, cfg)
            else:
                state = process_candles(df, state, cfg, trader)
            save_state(cfg.state_file, state)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # сеть, лимиты API и т.п. — повторяем позже
            logger.exception("Ошибка шага: %s", exc)
        time.sleep(cfg.poll_seconds)


# --------------------------------------------------------------------------- #
# Бэктест
# --------------------------------------------------------------------------- #

def backtest(df: pd.DataFrame, cfg: Config) -> dict:
    """Симуляция на закрытых свечах: вход/выход по цене закрытия, SL/TP по high/low."""
    rsi = compute_rsi(df["close"], cfg.rsi_period)
    equity, pos, trades = cfg.order_usdt, None, []
    for i in range(1, len(df)):
        row = df.iloc[i]
        if pos:
            sl = pos["entry"] * (1 - cfg.stop_loss_pct / 100) if cfg.stop_loss_pct > 0 else None
            tp = pos["entry"] * (1 + cfg.take_profit_pct / 100) if cfg.take_profit_pct > 0 else None
            exit_price = None
            if sl and row["low"] <= sl:
                exit_price = sl  # консервативно: стоп раньше тейка
            elif tp and row["high"] >= tp:
                exit_price = tp
            elif rsi_signal(rsi.iloc[i - 1], rsi.iloc[i], cfg.rsi_buy, cfg.rsi_sell) == "sell":
                exit_price = row["close"]
            if exit_price is not None:
                equity = pos["amount"] * exit_price * (1 - FEE_RATE)
                trades.append(equity / pos["cost"] - 1)
                pos = None
                continue
        if not pos and rsi_signal(rsi.iloc[i - 1], rsi.iloc[i], cfg.rsi_buy, cfg.rsi_sell) == "buy":
            pos = {"entry": row["close"], "cost": equity,
                   "amount": equity * (1 - FEE_RATE) / row["close"]}
    if pos:  # закрываем открытую позицию по последней цене
        equity = pos["amount"] * df["close"].iloc[-1] * (1 - FEE_RATE)
        trades.append(equity / pos["cost"] - 1)
    wins = sum(1 for t in trades if t > 0)
    return {
        "trades": len(trades),
        "win_rate": wins / len(trades) * 100 if trades else 0.0,
        "return_pct": (equity / cfg.order_usdt - 1) * 100,
        "buy_and_hold_pct": (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100,
    }


def run_backtest(cfg: Config, candles: int = 1000) -> None:
    exchange = make_exchange(Config(symbol=cfg.symbol, mode="paper"))
    df = fetch_candles(exchange, cfg.symbol, cfg.timeframe, limit=candles)
    res = backtest(df, cfg)
    print(f"Бэктест {cfg.symbol} {cfg.timeframe}, {len(df)} свечей "
          f"({df['ts'].iloc[0]:%Y-%m-%d} — {df['ts'].iloc[-1]:%Y-%m-%d})")
    print(f"RSI({cfg.rsi_period}) {cfg.rsi_buy}/{cfg.rsi_sell}, SL {cfg.stop_loss_pct}%, "
          f"TP {cfg.take_profit_pct}%")
    print(f"Сделок: {res['trades']}, прибыльных: {res['win_rate']:.1f}%")
    print(f"Доходность стратегии: {res['return_pct']:+.2f}% "
          f"(купить и держать: {res['buy_and_hold_pct']:+.2f}%)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RSI-бот для Binance")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run", help="запустить бота (по умолчанию)")
    sub.add_parser("backtest", help="прогнать стратегию на истории")
    sub.add_parser("signals", help="показать ожидающие сигналы и позицию (JSON)")
    p_confirm = sub.add_parser("confirm", help="записать сигнал, исполненный через MCP")
    p_confirm.add_argument("id")
    p_confirm.add_argument("--price", type=float, required=True, help="средняя цена исполнения")
    p_confirm.add_argument("--amount", type=float, required=True,
                           help="количество базовой монеты (после комиссии)")
    p_confirm.add_argument("--quote", type=float, help="USDT потрачено/получено с учётом комиссии")
    p_reject = sub.add_parser("reject", help="отклонить сигнал")
    p_reject.add_argument("id")
    args = parser.parse_args(argv)
    config = Config()

    if args.command == "backtest":
        run_backtest(config)
    elif args.command in ("signals", "confirm", "reject"):
        state = load_state(config.state_file)
        if args.command == "signals":
            expire_signals(state, config)
            print(json.dumps({"symbol": config.symbol, "position": state["position"],
                              "realized_pnl": state["realized_pnl"],
                              "pending": [s for s in state.get("signals", [])
                                          if s["status"] == "pending"]},
                             indent=2, ensure_ascii=False))
            return
        if args.command == "confirm":
            state = confirm_signal(state, args.id, args.price, args.amount, args.quote)
        else:
            state = reject_signal(state, args.id)
        save_state(config.state_file, state)
        status = next(s["status"] for s in state["signals"] if s["id"] == args.id)
        print(f"Сигнал {args.id}: {status}")
    else:
        run(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        main()
    except ValueError as exc:
        sys.exit(f"Ошибка: {exc}")
