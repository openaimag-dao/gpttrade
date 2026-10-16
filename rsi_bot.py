#!/usr/bin/env python3
"""Торговый бот для спота Binance (только long) с выбором стратегии.

Стратегии (переменная STRATEGY):
  * rsi       — по умолчанию: покупка, когда RSI выходит из перепроданности
                (пересекает RSI_BUY снизу вверх), продажа — RSI_SELL сверху вниз.
  * ema_cross — пересечение быстрой и медленной EMA (следование за трендом).
  * donchian  — пробой максимума за N свечей, выход по минимуму за M свечей.
  * bollinger — возврат цены внутрь нижней полосы Боллинджера, выход у средней.
  * macd      — пересечение MACD(12,26) и сигнальной линии (9).
Поверх любой стратегии: стоп-лосс, тейк-профит, трейлинг-стоп, фильтр тренда
EMA и пауза после серии убытков.

Режимы (переменная MODE):
  * paper   — по умолчанию: реальные котировки, сделки только симулируются.
  * signals — бот только присылает сигналы; сделку вы подтверждаете и
              исполняете через Binance MCP в Claude Code (/rsi-signals).
  * testnet — ордера на тестовой бирже Binance (testnet.binance.vision).
  * live    — реальные деньги; дополнительно требует CONFIRM_LIVE=yes.

Команды:
  python3 rsi_bot.py            — запустить бота
  python3 rsi_bot.py backtest   — прогнать стратегию на истории
  python3 rsi_bot.py optimize --strategy all — сравнить стратегии (walk-forward)
  python3 rsi_bot.py signals    — ожидающие сигналы (JSON)
  python3 rsi_bot.py confirm ID --price P --amount A [--quote Q]
  python3 rsi_bot.py reject ID
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
import math
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("rsi_bot")

FEE_RATE = 0.001  # комиссия Binance spot 0.1% (для paper и бэктеста)


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass
class Config:
    symbol: str = os.getenv("SYMBOL", "BTC/USDT")
    # Портфель: несколько пар через запятую, например "BTC/USDT,ETH/USDT,SOL/USDT".
    # У каждой пары своя позиция, свой файл состояния и свой ордер ORDER_USDT.
    symbols: str = os.getenv("SYMBOLS", "")
    timeframe: str = os.getenv("TIMEFRAME", "1h")
    strategy: str = os.getenv("STRATEGY", "rsi").lower()
    rsi_period: int = int(os.getenv("RSI_PERIOD", 14))
    rsi_buy: float = _env_float("RSI_BUY", 30)
    rsi_sell: float = _env_float("RSI_SELL", 70)
    fast_ema: int = int(os.getenv("FAST_EMA", 20))           # ema_cross
    slow_ema: int = int(os.getenv("SLOW_EMA", 100))          # ema_cross
    channel_entry: int = int(os.getenv("CHANNEL_ENTRY", 20))  # donchian
    channel_exit: int = int(os.getenv("CHANNEL_EXIT", 10))    # donchian
    bb_period: int = int(os.getenv("BB_PERIOD", 20))          # bollinger
    bb_std: float = _env_float("BB_STD", 2)                   # bollinger
    order_usdt: float = _env_float("ORDER_USDT", 20)
    stop_loss_pct: float = _env_float("STOP_LOSS_PCT", 3)
    take_profit_pct: float = _env_float("TAKE_PROFIT_PCT", 6)
    # Фильтр тренда: покупать только если цена выше EMA(TREND_EMA). 0 — выключен.
    trend_ema: int = int(os.getenv("TREND_EMA", 200))
    # Трейлинг-стоп: продать, если цена упала на TRAILING_PCT% от максимума после входа. 0 — выкл.
    trailing_pct: float = _env_float("TRAILING_PCT", 0)
    # После MAX_LOSS_STREAK убыточных сделок подряд — пауза в покупках на PAUSE_HOURS часов.
    max_loss_streak: int = int(os.getenv("MAX_LOSS_STREAK", 3))
    pause_hours: float = _env_float("PAUSE_HOURS", 24)
    mode: str = os.getenv("MODE", "paper").lower()
    poll_seconds: int = int(os.getenv("POLL_SECONDS", 30))
    signal_ttl_minutes: float = _env_float("SIGNAL_TTL_MINUTES", 30)
    # Публичное зеркало рыночных данных Binance для paper/signals/backtest, например
    # https://data-api.binance.vision/api/v3 — помогает, если api.binance.com недоступен.
    data_url: str = os.getenv("BINANCE_DATA_URL", "")
    state_file: Path = Path(os.getenv("STATE_FILE", "rsi_bot_state.json"))
    api_key: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("BINANCE_API_SECRET", ""))
    telegram_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))

    def symbol_list(self) -> list[str]:
        if not self.symbols.strip():
            return [self.symbol]
        return [x.strip().upper() for x in self.symbols.split(",") if x.strip()]

    def for_symbol(self, symbol: str) -> "Config":
        """Настройки для одной пары портфеля: свой SYMBOL и свой файл состояния."""
        if len(self.symbol_list()) == 1:
            return replace(self, symbol=symbol, symbols="")
        path = self.state_file
        state_file = path.with_name(f"{path.stem}_{symbol.replace('/', '-')}{path.suffix}")
        return replace(self, symbol=symbol, symbols="", state_file=state_file)

    def validate(self) -> None:
        if self.mode not in ("paper", "signals", "testnet", "live"):
            raise ValueError(f"MODE должен быть paper, signals, testnet или live, а не {self.mode!r}")
        if self.strategy not in STRATEGIES:
            raise ValueError(f"STRATEGY должна быть одной из {', '.join(STRATEGIES)}")
        if not 0 < self.rsi_buy < self.rsi_sell < 100:
            raise ValueError("Нужно 0 < RSI_BUY < RSI_SELL < 100")
        if self.strategy == "ema_cross" and self.fast_ema >= self.slow_ema:
            raise ValueError("Нужно FAST_EMA < SLOW_EMA")
        if self.mode in ("testnet", "live") and not (self.api_key and self.api_secret):
            raise ValueError("Для testnet/live задайте BINANCE_API_KEY и BINANCE_API_SECRET")
        if self.mode == "live" and os.getenv("CONFIRM_LIVE", "").lower() != "yes":
            raise ValueError("Режим live торгует реальными деньгами: задайте CONFIRM_LIVE=yes")


# --------------------------------------------------------------------------- #
# Индикатор и сигналы
# --------------------------------------------------------------------------- #

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI по Уайлдеру (как в TradingView и на Binance)."""
    delta = close.diff().to_numpy(float)
    out = np.full(len(close), np.nan)
    if len(close) > period:
        gains, losses = delta.clip(min=0), -delta.clip(max=0)
        # Затравка — простое среднее первых `period` изменений, далее сглаживание Уайлдера.
        avg_gain, avg_loss = gains[1:period + 1].mean(), losses[1:period + 1].mean()
        for i in range(period, len(close)):
            if i > period:
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            out[i] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    return pd.Series(out, index=close.index)


def compute_ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False, min_periods=period).mean()


def _prev(a: np.ndarray, fill) -> np.ndarray:
    out = np.roll(a, 1)
    out[:1] = fill
    return out


def _crosses(fast: np.ndarray, slow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(пересечение снизу вверх, сверху вниз); NaN считается отсутствием сигнала."""
    valid = ~np.isnan(fast) & ~np.isnan(slow)
    above = valid & (fast > slow)
    below = valid & (fast < slow)
    return above & _prev(below, False), below & _prev(above, False)


def strategy_signals(df: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, np.ndarray, str]:
    """Сигналы стратегии по закрытым свечам: (входы, выходы, описание последней свечи).

    Индикаторы используют только прошлые и текущую свечу — заглядывания в
    будущее нет, поэтому один расчёт по всей истории годится и для бэктеста.
    """
    close = df["close"]
    c = close.to_numpy(float)
    with np.errstate(invalid="ignore"):
        if cfg.strategy == "rsi":
            rsi = compute_rsi(close, cfg.rsi_period).to_numpy()
            prev = _prev(rsi, np.nan)
            entries = (prev < cfg.rsi_buy) & (rsi >= cfg.rsi_buy)
            exits = (prev > cfg.rsi_sell) & (rsi <= cfg.rsi_sell)
            label = f"RSI={rsi[-1]:.1f}"
        elif cfg.strategy == "ema_cross":
            fast = compute_ema(close, cfg.fast_ema).to_numpy()
            slow = compute_ema(close, cfg.slow_ema).to_numpy()
            entries, exits = _crosses(fast, slow)
            label = f"EMA{cfg.fast_ema}={fast[-1]:.4f} EMA{cfg.slow_ema}={slow[-1]:.4f}"
        elif cfg.strategy == "donchian":
            upper = df["high"].rolling(cfg.channel_entry).max().shift(1).to_numpy()
            lower = df["low"].rolling(cfg.channel_exit).min().shift(1).to_numpy()
            entries, exits = c > upper, c < lower
            label = f"канал {lower[-1]:.4f}–{upper[-1]:.4f}"
        elif cfg.strategy == "bollinger":
            mid = close.rolling(cfg.bb_period).mean()
            low_band = (mid - cfg.bb_std * close.rolling(cfg.bb_period).std(ddof=0)).to_numpy()
            mid = mid.to_numpy()
            entries = (_prev(c, np.nan) < _prev(low_band, np.nan)) & (c >= low_band)
            exits = c >= mid
            label = f"BB нижняя={low_band[-1]:.4f} средняя={mid[-1]:.4f}"
        elif cfg.strategy == "macd":
            macd = (compute_ema(close, 12) - compute_ema(close, 26))
            signal = macd.ewm(span=9, adjust=False).mean().to_numpy()
            macd = macd.to_numpy()
            entries, exits = _crosses(macd, signal)
            label = f"MACD={macd[-1]:.4f} сигнал={signal[-1]:.4f}"
        else:
            raise ValueError(f"Неизвестная стратегия {cfg.strategy!r}")
    return entries, exits, label


STRATEGIES = ("rsi", "ema_cross", "donchian", "bollinger", "macd")


def trend_mask(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """True там, где фильтр тренда разрешает покупку (цена выше EMA)."""
    if cfg.trend_ema <= 0:
        return np.ones(len(df), dtype=bool)
    ema = compute_ema(df["close"], cfg.trend_ema).to_numpy()
    with np.errstate(invalid="ignore"):
        return df["close"].to_numpy(float) > ema


def stop_price(entry: float, peak: float, cfg: Config) -> tuple[float | None, str]:
    """Уровень стопа с учётом трейлинга: (цена, 'стоп-лосс' | 'трейлинг-стоп')."""
    fixed = entry * (1 - cfg.stop_loss_pct / 100) if cfg.stop_loss_pct > 0 else None
    trail = peak * (1 - cfg.trailing_pct / 100) if cfg.trailing_pct > 0 else None
    if trail is not None and (fixed is None or trail > fixed):
        return trail, "трейлинг-стоп"
    return fixed, "стоп-лосс"


def exit_reason(pos: dict, price: float, cfg: Config) -> str | None:
    entry = pos["entry_price"]
    stop, label = stop_price(entry, pos.get("peak", entry), cfg)
    if stop is not None and price <= stop:
        return label
    if cfg.take_profit_pct > 0 and price >= entry * (1 + cfg.take_profit_pct / 100):
        return "тейк-профит"
    return None


def record_result(state: dict, pnl: float, cfg: Config) -> None:
    """Учёт закрытой сделки и пауза после серии убытков."""
    state["realized_pnl"] += pnl
    state["trades"] += 1
    state["loss_streak"] = state.get("loss_streak", 0) + 1 if pnl < 0 else 0
    if cfg.max_loss_streak > 0 and state["loss_streak"] >= cfg.max_loss_streak:
        last = pd.Timestamp(state["last_candle"]).timestamp() if state.get("last_candle") else time.time()
        state["paused_until"] = last + cfg.pause_hours * 3600
        state["loss_streak"] = 0
        notify(cfg, f"⏸ {cfg.max_loss_streak} убыточные сделки подряд — покупки на паузе "
                    f"{cfg.pause_hours:g} ч")


# --------------------------------------------------------------------------- #
# Биржа
# --------------------------------------------------------------------------- #

def make_exchange(cfg: Config):
    import ccxt

    params = {
        "enableRateLimit": True,
        "requests_trust_env": True,  # учитывать HTTPS_PROXY / REQUESTS_CA_BUNDLE
        # Бот торгует только спот: не запрашиваем рынки фьючерсов.
        "options": {"fetchMarkets": {"types": ["spot"]}, "fetchCurrencies": False},
    }
    trading = cfg.mode in ("testnet", "live")
    if trading:
        params.update(apiKey=cfg.api_key, secret=cfg.api_secret)
    exchange = ccxt.binance(params)
    if cfg.mode == "testnet":
        exchange.set_sandbox_mode(True)
    if cfg.data_url and not trading:
        exchange.urls["api"]["public"] = cfg.data_url.rstrip("/")
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
    closed = df.iloc[:-1]  # сигналы только по закрытым свечам
    if len(closed) < 3:
        return None
    last_price = float(df["close"].iloc[-1])
    pos = state["position"]

    # Стоп-лосс / трейлинг / тейк-профит проверяем по текущей цене на каждом опросе.
    if pos:
        pos["peak"] = max(pos.get("peak", pos["entry_price"]), last_price)
        reason = exit_reason(pos, last_price, cfg)
        if reason:
            return {"side": "sell", "price": last_price, "reason": reason}

    candle_ts = str(closed["ts"].iloc[-1])
    if candle_ts == state.get("last_candle"):
        return None  # эту свечу уже обработали
    state["last_candle"] = candle_ts

    entries, exits, label = strategy_signals(closed, cfg)
    signal = "buy" if entries[-1] else "sell" if exits[-1] else None
    logger.info("%s %s close=%.4f %s %s signal=%s", cfg.symbol, candle_ts,
                closed["close"].iloc[-1], cfg.strategy, label, signal)
    reason = f"{cfg.strategy}: {label}"
    if exits[-1] and pos:
        return {"side": "sell", "price": last_price, "reason": reason}
    if not entries[-1] or pos:
        return None
    if closed["ts"].iloc[-1].timestamp() < state.get("paused_until", 0):
        logger.info("Сигнал на покупку пропущен: пауза после серии убытков")
        return None
    if not trend_mask(closed, cfg)[-1]:
        logger.info("Сигнал на покупку пропущен: цена ниже EMA(%d) — нисходящий тренд",
                    cfg.trend_ema)
        return None
    return {"side": "buy", "price": last_price, "reason": reason}


def process_candles(df: pd.DataFrame, state: dict, cfg: Config, trader: Trader) -> dict:
    """Один шаг автоторговли (paper / testnet / live)."""
    decision = evaluate(df, state, cfg)
    if not decision:
        return state
    if decision["side"] == "sell":
        return _close(state, cfg, trader, decision["price"], decision["reason"])
    fill = trader.buy(decision["price"])
    state["position"] = {"amount": fill["amount"], "entry_price": fill["price"],
                         "peak": fill["price"], "cost": cfg.order_usdt,
                         "opened": state["last_candle"]}
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
        "timeframe": cfg.timeframe,
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
                   quote: float | None = None, cfg: Config | None = None) -> dict:
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
        state["position"] = {"amount": amount, "entry_price": price, "peak": price,
                             "cost": quote, "opened": state.get("last_candle")}
    else:
        pos = state["position"]
        if not pos:
            raise ValueError("Нет открытой позиции для продажи")
        record_result(state, quote - pos["cost"], cfg or Config())
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
    record_result(state, pnl, cfg)
    state["position"] = None
    notify(cfg, f"🔴 [{cfg.mode}] ПРОДАНО {fill['amount']:.6f} {cfg.symbol} по {fill['price']:.4f} "
                f"({reason})\nP&L сделки: {pnl:+.2f} USDT, всего: {state['realized_pnl']:+.2f} USDT")
    return state


def run(cfg: Config) -> None:
    cfg.validate()
    exchange = make_exchange(cfg)
    configs = [cfg.for_symbol(sym) for sym in cfg.symbol_list()]
    for c in configs:
        if c.symbol not in exchange.markets:
            raise ValueError(f"Пара {c.symbol} не найдена на Binance")
    traders = {c.symbol: Trader(c, exchange) for c in configs}
    pairs = ", ".join(c.symbol for c in configs)
    notify(cfg, f"🤖 Бот запущен: {pairs} {cfg.timeframe}, режим {cfg.mode}, "
                f"{describe(cfg)}, ордер {cfg.order_usdt} USDT на каждую пару")
    limit = min(1000, max(300, cfg.trend_ema * 3, cfg.slow_ema * 3))
    while True:
        for c in configs:
            try:
                df = fetch_candles(exchange, c.symbol, c.timeframe, limit=limit)
                # Перечитываем файл: в режиме signals его меняют команды confirm/reject.
                state = load_state(c.state_file)
                if c.mode == "signals":
                    state = process_signals(df, state, c)
                else:
                    state = process_candles(df, state, c, traders[c.symbol])
                save_state(c.state_file, state)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # сеть, лимиты API и т.п. — повторяем позже
                logger.exception("Ошибка шага %s: %s", c.symbol, exc)
        time.sleep(cfg.poll_seconds)


# --------------------------------------------------------------------------- #
# Бэктест
# --------------------------------------------------------------------------- #

def backtest(df: pd.DataFrame, cfg: Config,
             signals: tuple[np.ndarray, np.ndarray] | None = None,
             trend: np.ndarray | None = None, start: int = 1, end: int | None = None) -> dict:
    """Симуляция той же логики, что и в живом режиме, на свечах df[start:end].

    Вход и выход по сигналу стратегии — по цене закрытия; стоп/трейлинг/тейк —
    по high/low (при гэпе — по цене открытия). Если стоп и тейк задеты одной
    свечой, считаем стоп (консервативно). Комиссия FEE_RATE на каждую сторону.
    signals/trend можно передать заранее посчитанными по всей истории.
    """
    end = len(df) if end is None else end
    start = max(start, 1)
    close, high = df["close"].to_numpy(float), df["high"].to_numpy(float)
    low, opn = df["low"].to_numpy(float), df["open"].to_numpy(float)
    entries, exits = signals if signals is not None else strategy_signals(df, cfg)[:2]
    trend = trend_mask(df, cfg) if trend is None else trend
    bar_seconds = (df["ts"].iloc[1] - df["ts"].iloc[0]).total_seconds() if len(df) > 1 else 60
    pause_bars = math.ceil(cfg.pause_hours * 3600 / bar_seconds)

    equity = peak_equity = cfg.order_usdt
    max_dd, pos, trades, streak, paused_until = 0.0, None, [], 0, -1
    curve = np.empty(end - start)
    for i in range(start, end):
        if pos:
            stop, _ = stop_price(pos["entry"], pos["peak"], cfg)
            tp = pos["entry"] * (1 + cfg.take_profit_pct / 100) if cfg.take_profit_pct > 0 else None
            exit_price = None
            if stop is not None and low[i] <= stop:
                exit_price = min(stop, opn[i])
            elif tp is not None and high[i] >= tp:
                exit_price = max(tp, opn[i])
            elif exits[i]:
                exit_price = close[i]
            if exit_price is not None:
                equity = pos["amount"] * exit_price * (1 - FEE_RATE)
                ret = equity / pos["cost"] - 1
                trades.append(ret)
                streak = streak + 1 if ret < 0 else 0
                if cfg.max_loss_streak > 0 and streak >= cfg.max_loss_streak:
                    paused_until, streak = i + pause_bars, 0
                pos = None
            else:
                pos["peak"] = max(pos["peak"], high[i])
        elif entries[i] and trend[i] and i > paused_until:
            pos = {"entry": close[i], "peak": close[i], "cost": equity,
                   "amount": equity * (1 - FEE_RATE) / close[i]}
        mark = pos["amount"] * close[i] if pos else equity
        curve[i - start] = mark
        peak_equity = max(peak_equity, mark)
        max_dd = max(max_dd, 1 - mark / peak_equity)
    if pos:  # закрываем открытую позицию по последней цене отрезка
        equity = pos["amount"] * close[end - 1] * (1 - FEE_RATE)
        trades.append(equity / pos["cost"] - 1)
        if len(curve):
            curve[-1] = equity  # кривая заканчивается тем же капиталом, что и итог
    gains = sum(t for t in trades if t > 0)
    losses = -sum(t for t in trades if t < 0)
    return {
        "trades": len(trades),
        "win_rate": sum(t > 0 for t in trades) / len(trades) * 100 if trades else 0.0,
        "return_pct": (equity / cfg.order_usdt - 1) * 100,
        "max_drawdown_pct": max_dd * 100,
        "profit_factor": gains / losses if losses else (math.inf if gains else 0.0),
        "buy_and_hold_pct": (close[end - 1] / close[start - 1] - 1) * 100,
        "equity_curve": curve,  # стоимость позиции/кэша на каждой свече отрезка
    }


def fetch_history(exchange, symbol: str, timeframe: str, candles: int) -> pd.DataFrame:
    """Загрузить больше 1000 свечей постранично (лимит Binance — 1000 за запрос)."""
    step_ms = exchange.parse_timeframe(timeframe) * 1000
    since = exchange.milliseconds() - candles * step_ms
    rows: list = []
    while len(rows) < candles:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + step_ms
        if len(batch) < 1000:
            break
    df = pd.DataFrame(rows[-candles:], columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


STRATEGY_PARAMS = {
    "rsi": ("rsi_buy", "rsi_sell"),
    "ema_cross": ("fast_ema", "slow_ema"),
    "donchian": ("channel_entry", "channel_exit"),
    "bollinger": ("bb_period", "bb_std"),
    "macd": (),
}


def describe(cfg: Config) -> str:
    strat = {
        "rsi": f"RSI({cfg.rsi_period}) {cfg.rsi_buy:g}/{cfg.rsi_sell:g}",
        "ema_cross": f"EMA {cfg.fast_ema}/{cfg.slow_ema}",
        "donchian": f"Donchian {cfg.channel_entry}/{cfg.channel_exit}",
        "bollinger": f"Bollinger {cfg.bb_period}, {cfg.bb_std:g}σ",
        "macd": "MACD 12/26/9",
    }[cfg.strategy]
    parts = [strat,
             f"SL {cfg.stop_loss_pct:g}%" if cfg.stop_loss_pct else "SL выкл",
             f"TP {cfg.take_profit_pct:g}%" if cfg.take_profit_pct else "TP выкл",
             f"трейлинг {cfg.trailing_pct:g}%" if cfg.trailing_pct else "трейлинг выкл",
             f"EMA{cfg.trend_ema}" if cfg.trend_ema else "без фильтра тренда"]
    return ", ".join(parts)


def env_line(cfg: Config) -> str:
    keys = ("strategy",) + STRATEGY_PARAMS[cfg.strategy] + (
        "stop_loss_pct", "take_profit_pct", "trailing_pct", "trend_ema", "timeframe")
    return " ".join(f"{k.upper()}={getattr(cfg, k)}" for k in keys)


def format_result(res: dict) -> str:
    pf = "∞" if math.isinf(res["profit_factor"]) else f"{res['profit_factor']:.2f}"
    return (f"доходность {res['return_pct']:+.2f}%, сделок {res['trades']}, "
            f"прибыльных {res['win_rate']:.0f}%, profit factor {pf}, "
            f"макс. просадка {res['max_drawdown_pct']:.1f}% "
            f"(купить и держать {res['buy_and_hold_pct']:+.2f}%)")


def run_backtest(cfg: Config, candles: int) -> None:
    exchange = make_exchange(Config(symbol=cfg.symbol, mode="paper"))
    df = fetch_history(exchange, cfg.symbol, cfg.timeframe, candles)
    print(f"Бэктест {cfg.symbol} {cfg.timeframe}, {len(df)} свечей "
          f"({df['ts'].iloc[0]:%Y-%m-%d} — {df['ts'].iloc[-1]:%Y-%m-%d})")
    print(describe(cfg))
    print(format_result(backtest(df, cfg)))


def max_drawdown(curve: np.ndarray) -> float:
    return float((1 - curve / np.maximum.accumulate(curve)).max() * 100) if len(curve) else 0.0


def portfolio_backtest(frames: dict[str, pd.DataFrame], cfg: Config) -> dict:
    """Бэктест портфеля: каждой паре — своя равная доля (ORDER_USDT), торгуют независимо.

    Сравнение идёт на общем периоде, когда торговались все пары; индикаторы
    считаются по всей истории каждой пары, чтобы к началу периода они были «прогреты».
    """
    start_ts = max(df["ts"].iloc[0] for df in frames.values())
    end_ts = min(df["ts"].iloc[-1] for df in frames.values())
    per_symbol, curves, holds = {}, [], []
    for sym, df in frames.items():
        c = cfg.for_symbol(sym)
        start = max(1, int(df["ts"].searchsorted(start_ts)))
        end = int(df["ts"].searchsorted(end_ts, side="right"))
        res = backtest(df, c, start=start, end=end)
        per_symbol[sym] = res
        curves.append(res["equity_curve"])
        closes = df["close"].to_numpy(float)[start - 1:end]
        holds.append(cfg.order_usdt * closes[1:] / closes[0])
    n = min(len(x) for x in curves)
    total = sum(x[-n:] for x in curves)
    hold = sum(x[-n:] for x in holds)
    capital = cfg.order_usdt * len(frames)
    return {
        "period": (start_ts, end_ts),
        "per_symbol": per_symbol,
        "return_pct": (total[-1] / capital - 1) * 100,
        "max_drawdown_pct": max_drawdown(np.concatenate([[capital], total])),
        "trades": sum(r["trades"] for r in per_symbol.values()),
        "buy_and_hold_pct": (hold[-1] / capital - 1) * 100,
        "buy_and_hold_drawdown_pct": max_drawdown(np.concatenate([[capital], hold])),
    }


def run_portfolio(cfg: Config, candles: int) -> None:
    symbols = cfg.symbol_list()
    if len(symbols) < 2:
        raise ValueError("Для портфеля задайте SYMBOLS, например BTC/USDT,ETH/USDT,SOL/USDT")
    exchange = make_exchange(Config(mode="paper"))
    frames = {sym: fetch_history(exchange, sym, cfg.timeframe, candles) for sym in symbols}
    r = portfolio_backtest(frames, cfg)
    print(f"Портфель {len(symbols)} пар, {cfg.timeframe}, {describe(cfg)}")
    print(f"Период {r['period'][0]:%Y-%m-%d} — {r['period'][1]:%Y-%m-%d}, "
          f"по {cfg.order_usdt:g} USDT на пару\n")
    for sym, res in r["per_symbol"].items():
        print(f"  {sym:10} {format_result(res)}")
    print(f"\nПОРТФЕЛЬ: доходность {r['return_pct']:+.2f}%, макс. просадка "
          f"{r['max_drawdown_pct']:.1f}%, сделок {r['trades']}")
    print(f"Купить и держать те же пары: {r['buy_and_hold_pct']:+.2f}%, "
          f"просадка {r['buy_and_hold_drawdown_pct']:.1f}%")


STRATEGY_GRIDS = {
    "rsi": {"rsi_buy": [25, 30, 35], "rsi_sell": [60, 70, 75]},
    "ema_cross": {"fast_ema": [10, 20, 50], "slow_ema": [50, 100, 200]},
    "donchian": {"channel_entry": [20, 55], "channel_exit": [10, 20]},
    "bollinger": {"bb_period": [20, 40], "bb_std": [2.0, 2.5]},
    "macd": {},
}
RISK_GRID = {
    "stop_loss_pct": [0, 3, 6],
    "take_profit_pct": [0, 6, 12],
    "trailing_pct": [0, 3, 6],
    "trend_ema": [0, 200],
}


def _grid(grid: dict) -> list[dict]:
    return [dict(zip(grid, v)) for v in itertools.product(*grid.values())]


def walk_forward(df: pd.DataFrame, cfg: Config, strategy: str, folds: int = 3,
                 min_trades: int = 5) -> dict:
    """Walk-forward проверка стратегии.

    История делится на folds+1 равных частей. На шаге k параметры подбираются
    по всем данным до части k, а результат считается на части k — данных,
    которые при подборе не использовались. Итог по тестовым частям — честная
    оценка «как бы бот торговал вживую». Высокий результат только на обучении —
    это подгонка под историю.
    """
    base = Config(**{**cfg.__dict__, "strategy": strategy})
    strat_combos = [c for c in _grid(STRATEGY_GRIDS[strategy])
                    if not (strategy == "ema_cross" and c["fast_ema"] >= c["slow_ema"])
                    and not (strategy == "rsi" and c["rsi_buy"] >= c["rsi_sell"])]
    signals = {}
    for sc in strat_combos:
        signals[tuple(sc.items())] = strategy_signals(df, Config(**{**base.__dict__, **sc}))[:2]
    trends = {t: trend_mask(df, Config(**{**base.__dict__, "trend_ema": t}))
              for t in RISK_GRID["trend_ema"]}
    risk_combos = _grid(RISK_GRID)

    chunk = len(df) // (folds + 1)
    fold_results, growth = [], 1.0
    for k in range(1, folds + 1):
        train_end, test_end = k * chunk, (k + 1) * chunk if k < folds else len(df)
        best = None
        for sc in strat_combos:
            sig = signals[tuple(sc.items())]
            for rc in risk_combos:
                c = Config(**{**base.__dict__, **sc, **rc})
                res = backtest(df, c, sig, trends[c.trend_ema], start=1, end=train_end)
                if res["trades"] < min_trades:
                    continue
                score = res["return_pct"] - 0.5 * res["max_drawdown_pct"]
                if best is None or score > best[0]:
                    best = (score, c, res)
        if best is None:  # нечего торговать — остаёмся в деньгах
            fold_results.append({"cfg": None, "train": None, "test": None,
                                 "period": (df["ts"].iloc[train_end], df["ts"].iloc[test_end - 1])})
            continue
        _, c, train_res = best
        sig = signals[tuple((key, getattr(c, key)) for key in STRATEGY_GRIDS[strategy])]
        test_res = backtest(df, c, sig, trends[c.trend_ema], start=train_end, end=test_end)
        growth *= 1 + test_res["return_pct"] / 100
        fold_results.append({"cfg": c, "train": train_res, "test": test_res,
                             "period": (df["ts"].iloc[train_end], df["ts"].iloc[test_end - 1])})
    tested = [f for f in fold_results if f["test"]]
    last_cfg = next((f["cfg"] for f in reversed(fold_results) if f["cfg"]), None)
    return {
        "strategy": strategy,
        "folds": fold_results,
        "oos_return_pct": (growth - 1) * 100,
        "oos_max_drawdown_pct": max((f["test"]["max_drawdown_pct"] for f in tested), default=0.0),
        "oos_trades": sum(f["test"]["trades"] for f in tested),
        "positive_folds": sum(f["test"]["return_pct"] > 0 for f in tested),
        "buy_and_hold_pct": (df["close"].iloc[-1] / df["close"].iloc[chunk - 1] - 1) * 100,
        "last_cfg": last_cfg,
        "combos": len(strat_combos) * len(risk_combos),
    }


def run_optimize(cfg: Config, candles: int, strategy: str, folds: int) -> None:
    exchange = make_exchange(Config(symbol=cfg.symbol, mode="paper"))
    df = fetch_history(exchange, cfg.symbol, cfg.timeframe, candles)
    names = STRATEGIES if strategy == "all" else (strategy,)
    print(f"Walk-forward {cfg.symbol} {cfg.timeframe}: {len(df)} свечей "
          f"({df['ts'].iloc[0]:%Y-%m-%d} — {df['ts'].iloc[-1]:%Y-%m-%d}), {folds} тестовых отрезка\n")
    results = []
    for name in names:
        r = walk_forward(df, cfg, name, folds)
        results.append(r)
        print(f"== {name} ({r['combos']} комбинаций на отрезок)")
        for i, f in enumerate(r["folds"], 1):
            per = f"{f['period'][0]:%Y-%m-%d} — {f['period'][1]:%Y-%m-%d}"
            if not f["test"]:
                print(f"  тест {i} ({per}): мало сделок на обучении — не торгуем")
                continue
            print(f"  тест {i} ({per}): {describe(f['cfg'])}")
            print(f"     {format_result(f['test'])}")
        print(f"  ИТОГО на новых данных: {r['oos_return_pct']:+.2f}%, сделок {r['oos_trades']}, "
              f"плюсовых отрезков {r['positive_folds']}/{folds}, "
              f"худшая просадка {r['oos_max_drawdown_pct']:.1f}% "
              f"(купить и держать {r['buy_and_hold_pct']:+.2f}%)\n")
    ok = [r for r in results if r["oos_return_pct"] > 0 and r["positive_folds"] >= folds - 1
          and r["last_cfg"]]
    if not ok:
        print("⚠️ Ни одна стратегия не была стабильно прибыльной на новых данных для "
              f"{cfg.symbol} {cfg.timeframe} — не торгуйте её на реальные деньги.")
        return
    best = max(ok, key=lambda r: r["oos_return_pct"])
    print(f"Лучшая устойчивая стратегия: {best['strategy']} "
          f"({best['oos_return_pct']:+.2f}% на новых данных). Параметры последнего отрезка:")
    print(f"  {env_line(best['last_cfg'])}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RSI-бот для Binance")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run", help="запустить бота (по умолчанию)")
    p_bt = sub.add_parser("backtest", help="прогнать стратегию на истории")
    p_bt.add_argument("--candles", type=int, default=1000)
    p_opt = sub.add_parser("optimize", help="подобрать параметры с проверкой на новых данных")
    p_opt.add_argument("--candles", type=int, default=6000)
    p_opt.add_argument("--strategy", default=None, help="rsi, ema_cross, donchian, bollinger, "
                       "macd или all (по умолчанию — STRATEGY)")
    p_opt.add_argument("--folds", type=int, default=3)
    p_pf = sub.add_parser("portfolio", help="бэктест портфеля из пар SYMBOLS")
    p_pf.add_argument("--candles", type=int, default=3000)
    sub.add_parser("signals", help="показать ожидающие сигналы и позиции (JSON)")
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
        run_backtest(config, args.candles)
    elif args.command == "optimize":
        run_optimize(config, args.candles, args.strategy or config.strategy, args.folds)
    elif args.command == "portfolio":
        run_portfolio(config, args.candles)
    elif args.command == "signals":
        report = {"pairs": [], "pending": []}
        for c in (config.for_symbol(sym) for sym in config.symbol_list()):
            state = load_state(c.state_file)
            expire_signals(state, c)
            report["pairs"].append({"symbol": c.symbol, "position": state["position"],
                                    "realized_pnl": state["realized_pnl"]})
            report["pending"] += [s for s in state.get("signals", []) if s["status"] == "pending"]
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.command in ("confirm", "reject"):
        for c in (config.for_symbol(sym) for sym in config.symbol_list()):
            state = load_state(c.state_file)
            if any(s["id"] == args.id for s in state.get("signals", [])):
                break
        else:
            raise ValueError(f"Сигнал {args.id} не найден")
        if args.command == "confirm":
            state = confirm_signal(state, args.id, args.price, args.amount, args.quote, c)
        else:
            state = reject_signal(state, args.id)
        save_state(c.state_file, state)
        status = next(s["status"] for s in state["signals"] if s["id"] == args.id)
        print(f"Сигнал {args.id} ({c.symbol}): {status}")
    else:
        run(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        main()
    except ValueError as exc:
        sys.exit(f"Ошибка: {exc}")
