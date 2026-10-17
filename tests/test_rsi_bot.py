import math

import numpy as np
import pandas as pd
import pytest

import rsi_bot
from rsi_bot import Config, Trader, backtest, compute_rsi, process_candles, strategy_signals


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


def test_rsi_crossings(monkeypatch):
    fake = pd.Series([math.nan, 25, 31, 28, 75, 69, 72])
    monkeypatch.setattr(rsi_bot, "compute_rsi", lambda close, period: fake)
    entries, exits, _ = strategy_signals(make_df(range(1, 8)), Config(rsi_buy=30, rsi_sell=70))
    assert entries.tolist() == [False, False, True, False, True, False, False]  # 28→75 тоже пересечение
    assert exits.tolist() == [False, False, False, False, False, True, False]


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
    cfg = Config(mode="paper", stop_loss_pct=0, take_profit_pct=0, trend_ema=0,
                 state_file=tmp_path / "s.json", telegram_token="")
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


# ----------------------------- режим signals ------------------------------ #

from rsi_bot import confirm_signal, pending_signal, process_signals, reject_signal


def run_until_signal(closes, state, cfg, now=1000.0):
    for n in range(20, len(closes) + 1):
        state = process_signals(make_df(closes[:n]), state, cfg, now=now)
        if pending_signal(state):
            return state
    return state


def test_signals_mode_does_not_trade_until_confirmed(tmp_path):
    cfg = Config(mode="signals", stop_loss_pct=0, take_profit_pct=0, trend_ema=0, telegram_token="")
    cfg.validate()  # ключи Binance не нужны
    state = run_until_signal(oversold_then_bounce(), rsi_bot.load_state(tmp_path / "s.json"), cfg)
    sig = pending_signal(state)
    assert sig["side"] == "buy" and sig["quote_usdt"] == cfg.order_usdt
    assert sig["mcp_symbol"] == "BTCUSDT"
    assert state["position"] is None  # без подтверждения позиции нет

    state = confirm_signal(state, sig["id"], price=81.0, amount=0.2466, quote=20.0)
    assert state["position"] == {"amount": 0.2466, "entry_price": 81.0, "peak": 81.0,
                                 "cost": 20.0, "opened": state["last_candle"]}
    assert pending_signal(state) is None
    with pytest.raises(ValueError):
        confirm_signal(state, sig["id"], price=81.0, amount=0.2466)  # повторно нельзя


def test_signal_sell_records_pnl_and_dedupes_stop_loss(tmp_path):
    cfg = Config(mode="signals", stop_loss_pct=3, take_profit_pct=0)
    state = {"position": {"amount": 1.0, "entry_price": 100.0, "cost": 100.0, "opened": "x"},
             "last_candle": None, "realized_pnl": 0.0, "trades": 0, "signals": []}
    df = make_df([100] * 20 + [96])
    state = process_signals(df, state, cfg, now=0)
    state = process_signals(df, state, cfg, now=30)  # стоп-лосс снова, но сигнал один
    assert [s["side"] for s in state["signals"]] == ["sell"]
    sig = pending_signal(state)
    assert sig["amount"] == 1.0 and sig["reason"] == "стоп-лосс"
    state = confirm_signal(state, sig["id"], price=96.0, amount=1.0, quote=95.9)
    assert state["position"] is None
    assert state["realized_pnl"] == pytest.approx(-4.1)
    assert state["trades"] == 1


def test_signal_expiry_and_reject_cooldown():
    cfg = Config(mode="signals", stop_loss_pct=3, take_profit_pct=0, signal_ttl_minutes=10)
    state = {"position": {"amount": 1.0, "entry_price": 100.0, "cost": 100.0, "opened": "x"},
             "last_candle": None, "realized_pnl": 0.0, "trades": 0, "signals": []}
    df = make_df([100] * 20 + [96])
    state = process_signals(df, state, cfg, now=0)
    state = reject_signal(state, state["signals"][0]["id"])
    state = process_signals(df, state, cfg, now=60)  # сразу после отказа — не повторяем
    assert pending_signal(state) is None
    state = process_signals(df, state, cfg, now=11 * 60)  # после TTL — новый сигнал
    assert pending_signal(state) is not None
    state = process_signals(df, state, cfg, now=30 * 60)  # не подтвердили — истёк
    assert state["signals"][1]["status"] == "expired"


def test_cli_signals_confirm(tmp_path, monkeypatch, capsys):
    path = tmp_path / "s.json"
    rsi_bot.save_state(path, {"position": None, "last_candle": "t", "realized_pnl": 0.0,
                              "trades": 0, "signals": [
                                  {"id": "abc12345", "side": "buy", "symbol": "BTC/USDT",
                                   "mcp_symbol": "BTCUSDT", "price": 100.0, "reason": "RSI=31",
                                   "created": __import__("time").time(), "status": "pending",
                                   "quote_usdt": 20.0}]})
    monkeypatch.setattr(rsi_bot, "Config", lambda: Config(state_file=path))
    rsi_bot.main(["signals"])
    assert '"abc12345"' in capsys.readouterr().out
    rsi_bot.main(["confirm", "abc12345", "--price", "100", "--amount", "0.1998", "--quote", "20"])
    assert "executed" in capsys.readouterr().out
    assert rsi_bot.load_state(path)["position"]["amount"] == 0.1998


# ------------------------- защита от убытков ------------------------------ #

def test_trend_filter_blocks_buy_in_downtrend(tmp_path):
    closes = oversold_then_bounce()
    with_filter = Config(mode="signals", trend_ema=20, stop_loss_pct=0, take_profit_pct=0)
    state = run_until_signal(closes, rsi_bot.load_state(tmp_path / "a.json"), with_filter)
    assert pending_signal(state) is None  # цена ниже EMA — нож не ловим
    without = Config(mode="signals", trend_ema=0, stop_loss_pct=0, take_profit_pct=0)
    state = run_until_signal(closes, rsi_bot.load_state(tmp_path / "b.json"), without)
    assert pending_signal(state)["side"] == "buy"


def test_trailing_stop_locks_profit():
    cfg = Config(mode="paper", stop_loss_pct=3, take_profit_pct=0, trailing_pct=2, trend_ema=0)
    trader = Trader(cfg, exchange=None)
    state = {"position": {"amount": 1.0, "entry_price": 100.0, "peak": 100.0, "cost": 100.0,
                          "opened": "x"},
             "last_candle": None, "realized_pnl": 0.0, "trades": 0}
    state = process_candles(make_df([100] * 20 + [110]), state, cfg, trader)
    assert state["position"]["peak"] == 110  # максимум запомнили
    state = process_candles(make_df([100] * 20 + [110, 107.5]), state, cfg, trader)
    assert state["position"] is None  # 107.5 < 110 * 0.98 — сработал трейлинг
    assert state["realized_pnl"] > 0


def test_loss_streak_pauses_buying():
    cfg = Config(max_loss_streak=2, pause_hours=10)
    state = {"position": None, "last_candle": "2026-01-01 00:00:00+00:00", "realized_pnl": 0.0,
             "trades": 0}
    rsi_bot.record_result(state, -1.0, cfg)
    assert "paused_until" not in state
    rsi_bot.record_result(state, -1.0, cfg)
    assert state["paused_until"] == pd.Timestamp("2026-01-01", tz="UTC").timestamp() + 10 * 3600
    assert state["loss_streak"] == 0


def test_backtest_reports_risk_metrics():
    rng = np.random.default_rng(2)
    df = make_df(100 * np.exp(rng.normal(0, 0.01, 3000).cumsum()))
    res = backtest(df, Config(trend_ema=0, trailing_pct=2))
    assert 0 <= res["max_drawdown_pct"] <= 100
    assert res["profit_factor"] >= 0


def test_walk_forward_uses_only_unseen_data_for_tests():
    rng = np.random.default_rng(3)
    df = make_df(100 * np.exp(rng.normal(0, 0.01, 3000).cumsum()))
    r = rsi_bot.walk_forward(df, Config(), "ema_cross", folds=3, min_trades=1)
    assert len(r["folds"]) == 3
    chunk = len(df) // 4
    for k, f in enumerate(r["folds"], 1):
        assert f["period"][0] == df["ts"].iloc[k * chunk]  # тест начинается после обучения
    assert r["oos_trades"] == sum(f["test"]["trades"] for f in r["folds"] if f["test"])


@pytest.mark.parametrize("strategy", rsi_bot.STRATEGIES)
def test_every_strategy_backtests_and_signals(strategy):
    rng = np.random.default_rng(4)
    df = make_df(100 * np.exp(rng.normal(0, 0.01, 1500).cumsum()))
    cfg = Config(strategy=strategy, trend_ema=0, stop_loss_pct=5, take_profit_pct=0)
    cfg.validate()
    entries, exits, label = strategy_signals(df, cfg)
    assert entries.dtype == bool and len(entries) == len(df) and entries.any() and exits.any()
    assert label
    assert backtest(df, cfg)["trades"] > 0


def test_signals_have_no_lookahead():
    """Сигнал на свече i не должен зависеть от будущих свечей."""
    rng = np.random.default_rng(5)
    df = make_df(100 * np.exp(rng.normal(0, 0.01, 800).cumsum()))
    for strategy in rsi_bot.STRATEGIES:
        cfg = Config(strategy=strategy)
        full_e, full_x, _ = strategy_signals(df, cfg)
        part_e, part_x, _ = strategy_signals(df.iloc[:600], cfg)
        assert (full_e[:600] == part_e).all() and (full_x[:600] == part_x).all(), strategy


class PagedExchange:
    def __init__(self, n):
        self.rows = [[i * 60_000, 1, 1, 1, 1, 1] for i in range(n)]

    def parse_timeframe(self, tf):
        return 60

    def milliseconds(self):
        return len(self.rows) * 60_000

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        return [r for r in self.rows if r[0] >= since][:limit]


def test_fetch_history_pages_past_1000():
    df = rsi_bot.fetch_history(PagedExchange(2500), "BTC/USDT", "1m", 2500)
    assert len(df) == 2500 and df["ts"].is_monotonic_increasing


# ------------------------------- портфель --------------------------------- #

def test_portfolio_configs_get_separate_state_files(tmp_path):
    cfg = Config(symbols="BTC/USDT, eth/usdt", state_file=tmp_path / "state.json")
    assert cfg.symbol_list() == ["BTC/USDT", "ETH/USDT"]
    eth = cfg.for_symbol("ETH/USDT")
    assert eth.symbol == "ETH/USDT" and eth.state_file == tmp_path / "state_ETH-USDT.json"
    single = Config(symbol="SOL/USDT", state_file=tmp_path / "state.json")
    assert single.symbol_list() == ["SOL/USDT"]
    assert single.for_symbol("SOL/USDT").state_file == tmp_path / "state.json"  # как раньше


def test_portfolio_backtest_sums_equal_shares():
    rng = np.random.default_rng(6)
    frames = {"AAA/USDT": make_df(100 * np.exp(rng.normal(0, 0.01, 1200).cumsum())),
              "BBB/USDT": make_df(50 * np.exp(rng.normal(0, 0.01, 1200).cumsum()))}
    cfg = Config(strategy="ema_cross", fast_ema=10, slow_ema=50, trend_ema=0, order_usdt=10)
    r = rsi_bot.portfolio_backtest(frames, cfg)
    per = r["per_symbol"]
    expected = (sum(10 * (1 + x["return_pct"] / 100) for x in per.values()) / 20 - 1) * 100
    assert r["return_pct"] == pytest.approx(expected)
    assert r["trades"] == sum(x["trades"] for x in per.values())
    assert 0 <= r["max_drawdown_pct"] <= 100


def test_cli_portfolio_signals_and_confirm(tmp_path, monkeypatch, capsys):
    base = tmp_path / "s.json"
    cfg = Config(symbols="BTC/USDT,ETH/USDT", state_file=base)
    for sym, sid in (("BTC/USDT", "aaaa1111"), ("ETH/USDT", "bbbb2222")):
        rsi_bot.save_state(cfg.for_symbol(sym).state_file, {
            "position": None, "last_candle": "t", "realized_pnl": 0.0, "trades": 0,
            "signals": [{"id": sid, "side": "buy", "symbol": sym, "price": 10.0, "reason": "x",
                         "created": __import__("time").time(), "status": "pending",
                         "quote_usdt": 20.0}]})
    monkeypatch.setattr(rsi_bot, "Config", lambda: cfg)
    rsi_bot.main(["signals"])
    out = __import__("json").loads(capsys.readouterr().out)
    assert {s["id"] for s in out["pending"]} == {"aaaa1111", "bbbb2222"}
    assert [p["symbol"] for p in out["pairs"]] == ["BTC/USDT", "ETH/USDT"]
    rsi_bot.main(["confirm", "bbbb2222", "--price", "10", "--amount", "2", "--quote", "20"])
    assert "ETH/USDT" in capsys.readouterr().out
    assert rsi_bot.load_state(cfg.for_symbol("ETH/USDT").state_file)["position"]["amount"] == 2
    assert rsi_bot.load_state(cfg.for_symbol("BTC/USDT").state_file)["position"] is None


# --------------------- «умные» фильтры и размер позиции -------------------- #

def test_adx_high_in_trend_low_in_chop():
    trend = make_df(np.linspace(100, 200, 300))
    chop = make_df(100 + np.tile([0, 1, 0, -1], 75))
    _, adx_trend = rsi_bot.compute_atr_adx(trend, 14)
    _, adx_chop = rsi_bot.compute_atr_adx(chop, 14)
    assert adx_trend[-1] > 50 > adx_chop[-1]


def test_position_fraction_shrinks_for_volatile_coin():
    calm = make_df(100 + np.tile([0.0, 0.5], 100))      # ATR ≈ 0.6%
    wild = calm.copy()
    wild["high"], wild["low"] = wild["close"] * 1.03, wild["close"] * 0.97  # ATR ≈ 6%
    cfg = Config(vol_target_pct=3)
    assert rsi_bot.position_fraction(calm, cfg)[-1] == 1.0
    assert rsi_bot.position_fraction(wild, cfg)[-1] == pytest.approx(0.5, rel=0.1)
    assert (rsi_bot.position_fraction(wild, Config(vol_target_pct=0)) == 1).all()


def test_market_mask_follows_btc_trend():
    alt = make_df(np.full(120, 10.0))
    btc_up = make_df(np.linspace(100, 200, 120))
    btc_down = make_df(np.linspace(200, 100, 120))
    cfg = Config(market_ema=20)
    assert rsi_bot.market_mask(alt, cfg, btc_up)[-1]
    assert not rsi_bot.market_mask(alt, cfg, btc_down)[-1]
    with pytest.raises(ValueError):
        rsi_bot.market_mask(alt, cfg, None)


def test_backtest_keeps_uninvested_part_in_cash():
    df = make_df(np.r_[np.linspace(100, 50, 50), np.full(10, 50.0)])
    entries = np.zeros(len(df), bool); entries[1] = True
    exits = np.zeros(len(df), bool); exits[-1] = True
    cfg = Config(trend_ema=0, stop_loss_pct=0, take_profit_pct=0, order_usdt=100)
    full = backtest(df, cfg, (entries, exits), fractions=np.ones(len(df)))
    half = backtest(df, cfg, (entries, exits), fractions=np.full(len(df), 0.5))
    assert half["return_pct"] == pytest.approx(full["return_pct"] / 2, rel=0.01)
    assert half["max_drawdown_pct"] < full["max_drawdown_pct"]


def test_signal_quote_scaled_by_volatility_and_btc_filter(tmp_path):
    closes = oversold_then_bounce()
    btc_down = make_df(np.linspace(200, 100, len(closes)))
    cfg = Config(mode="signals", trend_ema=0, stop_loss_pct=0, take_profit_pct=0,
                 market_ema=10, order_usdt=20)
    state = rsi_bot.load_state(tmp_path / "a.json")
    for n in range(20, len(closes) + 1):
        state = process_signals(make_df(closes[:n]), state, cfg, market=btc_down.iloc[:n])
    assert pending_signal(state) is None  # BTC падает — альткоин не покупаем

    cfg = Config(mode="signals", trend_ema=0, stop_loss_pct=0, take_profit_pct=0,
                 vol_target_pct=0.1, order_usdt=20)
    state = run_until_signal(closes, rsi_bot.load_state(tmp_path / "b.json"), cfg)
    sig = pending_signal(state)
    assert 0 < sig["quote_usdt"] < 20 and "размер" in sig["reason"]
