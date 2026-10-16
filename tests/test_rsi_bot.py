import math

import numpy as np
import pandas as pd
import pytest

import rsi_bot
from rsi_bot import Config, Trader, backtest, compute_rsi, process_candles, rsi_signal


def make_df(closes):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "ts": pd.date_range("2026-01-01", periods=len(closes), freq="15min", tz="UTC"),
        "open": closes, "high": closes * 1.001, "low": closes * 0.999,
        "close": closes, "volume": 1.0,
    })


def test_rsi_matches_wilder_reference():
    # Классический пример из книги Уайлдера / StockCharts: RSI(14) на 15-й точке ≈ 70.53
    closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
              45.89, 46.03, 45.61, 46.28, 46.28]
    rsi = compute_rsi(pd.Series(closes), 14)
    assert rsi.iloc[:14].isna().all()
    assert rsi.iloc[14] == pytest.approx(70.46, abs=0.5)


def test_rsi_bounds_and_only_gains():
    assert compute_rsi(pd.Series(range(1, 40)), 14).dropna().eq(100).all()
    rsi = compute_rsi(pd.Series(np.random.default_rng(0).normal(0, 1, 500).cumsum() + 100), 14)
    assert rsi.dropna().between(0, 100).all()


def test_rsi_signal_crossings():
    assert rsi_signal(25, 31, 30, 70) == "buy"
    assert rsi_signal(75, 69, 30, 70) == "sell"
    assert rsi_signal(25, 28, 30, 70) is None
    assert rsi_signal(math.nan, 31, 30, 70) is None


def test_config_validation(monkeypatch):
    with pytest.raises(ValueError):
        Config(mode="live", api_key="k", api_secret="s").validate()
    monkeypatch.setenv("CONFIRM_LIVE", "yes")
    Config(mode="live", api_key="k", api_secret="s").validate()
    with pytest.raises(ValueError):
        Config(mode="testnet").validate()
    with pytest.raises(ValueError):
        Config(rsi_buy=70, rsi_sell=30).validate()


def oversold_then_bounce():
    down = list(np.linspace(100, 80, 30))
    up = list(np.linspace(80.5, 84, 6))
    return down + up


def test_paper_buy_then_sell(tmp_path):
    cfg = Config(mode="paper", stop_loss_pct=0, take_profit_pct=0, state_file=tmp_path / "s.json",
                 telegram_token="")
    trader = Trader(cfg, exchange=None)
    state = rsi_bot.load_state(cfg.state_file)
    closes = oversold_then_bounce()
    # Прокручиваем свечи по одной, как в живом цикле.
    for n in range(20, len(closes) + 1):
        state = process_candles(make_df(closes[:n]), state, cfg, trader)
        if state["position"]:
            break
    assert state["position"] is not None
    # Рост до перекупленности, затем разворот вниз.
    closes += list(np.linspace(85, 110, 25)) + [105, 100]
    for n in range(len(closes) - 26, len(closes) + 1):
        state = process_candles(make_df(closes[:n]), state, cfg, trader)
    assert state["position"] is None
    assert state["trades"] == 1
    assert state["realized_pnl"] > 0


def test_stop_loss_triggers(tmp_path):
    cfg = Config(mode="paper", stop_loss_pct=3, take_profit_pct=0, state_file=tmp_path / "s.json")
    trader = Trader(cfg, exchange=None)
    state = {"position": {"amount": 1.0, "entry_price": 100.0, "cost": 100.0, "opened": "x"},
             "last_candle": None, "realized_pnl": 0.0, "trades": 0}
    state = process_candles(make_df([100] * 20 + [96]), state, cfg, trader)
    assert state["position"] is None and state["realized_pnl"] < 0


class FakeExchange:
    def __init__(self):
        self.orders = []

    def create_market_buy_order_with_cost(self, symbol, cost):
        self.orders.append(("buy", symbol, cost))
        return {"filled": 0.001, "average": 20000.0, "fee": {"currency": "BTC", "cost": 0.000001}}

    def fetch_balance(self):
        return {"BTC": {"free": 0.000999}}

    def amount_to_precision(self, symbol, amount):
        return str(math.floor(amount * 1e5) / 1e5)  # ccxt по умолчанию отбрасывает лишние знаки

    def create_market_sell_order(self, symbol, qty):
        self.orders.append(("sell", symbol, qty))
        return {"average": 21000.0, "cost": qty * 21000.0, "fee": {"currency": "USDT", "cost": 0.02}}


def test_real_orders_account_for_fees():
    ex = FakeExchange()
    trader = Trader(Config(mode="testnet", order_usdt=20), ex)
    fill = trader.buy(20000)
    assert fill["amount"] == pytest.approx(0.000999)
    sold = trader.sell(fill["amount"], 21000)
    assert ex.orders[1] == ("sell", "BTC/USDT", 0.00099)  # округлено вниз до шага лота
    assert sold["proceeds"] == pytest.approx(0.00099 * 21000 - 0.02)


def test_backtest_runs():
    rng = np.random.default_rng(1)
    closes = 100 * np.exp(rng.normal(0, 0.01, 2000).cumsum())
    res = backtest(make_df(closes), Config())
    assert res["trades"] > 0
    assert 0 <= res["win_rate"] <= 100
