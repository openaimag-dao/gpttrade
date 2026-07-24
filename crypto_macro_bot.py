#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto-Macro Correlation Bot
============================

Локальный скрипт для макро- и корреляционного анализа BTC относительно
традиционных рынков (DXY, QQQ, SPY, NVDA, MSFT) за период 2018-2026 гг.

Запуск:
    python3 crypto_macro_bot.py [--leverage 5]
"""

import argparse
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0. КОНФИГУРАЦИЯ
# ---------------------------------------------------------------------------

# Тикеры Yahoo Finance -> короткие алиасы, которые используются в отчёте
TICKERS = {
    "BTC": "BTC-USD",     # Bitcoin
    "DXY": "DX-Y.NYB",    # Индекс доллара (ICE US Dollar Index)
    "QQQ": "QQQ",         # ETF на Nasdaq-100 (тех. сектор)
    "SPY": "SPY",         # ETF на S&P 500
    "NVDA": "NVDA",       # Nvidia
    "MSFT": "MSFT",       # Microsoft
}

START_DATE = "2018-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Окна скользящей корреляции (в днях)
CORR_WINDOWS = (30, 90, 180)

# Пары активов, для которых считается корреляция с BTC
CORR_PAIRS = [
    ("BTC", "DXY"),
    ("BTC", "QQQ"),
    ("BTC", "SPY"),
    ("BTC", "NVDA"),
    ("BTC", "MSFT"),
]

DEFAULT_LEVERAGE = 5  # плечо пользователя по умолчанию (диапазон 3x-5x)


# ---------------------------------------------------------------------------
# 1. СБОР И СИНХРОНИЗАЦИЯ ДАННЫХ
# ---------------------------------------------------------------------------
def download_prices(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """
    Скачивает дневные цены закрытия (Close) по всем активам из TICKERS
    и синхронизирует их в единый календарный ряд.
    """
    print(f"[i] Скачивание дневных данных с {start} по {end} ...")

    raw = yf.download(
        list(TICKERS.values()),
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        threads=True,
    )

    if raw is None or raw.empty:
        raise RuntimeError(
            "yfinance вернул пустой набор данных. "
            "Проверьте интернет-соединение и корректность тикеров."
        )

    # При загрузке нескольких тикеров yfinance отдаёт колонки в виде
    # MultiIndex вида (Поле, Тикер) -> берём срез по полю "Close".
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        # На случай, если по какой-то причине вернулся только один тикер
        close = raw[["Close"]].copy()
        close.columns = list(TICKERS.values())

    # Переименовываем тикеры Yahoo в короткие алиасы (BTC, DXY, QQQ, ...)
    reverse_map = {yahoo_symbol: alias for alias, yahoo_symbol in TICKERS.items()}
    close = close.rename(columns=reverse_map)

    missing = [alias for alias in TICKERS if alias not in close.columns]
    if missing:
        raise RuntimeError(f"Не удалось загрузить данные по тикерам: {missing}")

    close = close[list(TICKERS.keys())]

    # --- Синхронизация временных рядов ---
    # BTC торгуется 24/7 (365 дней в году), тогда как фондовый рынок
    # работает только по будням (~252 дня в году). Чтобы ряды были
    # сопоставимы день-в-день, строим полный календарный диапазон дат
    # и переносим последнюю известную цену вперёд (forward fill) на
    # выходные и праздники фондового рынка.
    full_range = pd.date_range(start=close.index.min(), end=close.index.max(), freq="D")
    close = close.reindex(full_range).ffill()

    # Отбрасываем самые первые дни, пока история есть не по всем активам
    close = close.dropna(how="any")
    close.index.name = "Date"

    print(
        f"[i] Данные синхронизированы: {len(close)} дневных наблюдений "
        f"({close.index[0].date()} - {close.index[-1].date()})"
    )
    return close


# ---------------------------------------------------------------------------
# 2. МАТЕМАТИЧЕСКИЙ БЛОК (Correlation Engine)
# ---------------------------------------------------------------------------
def build_analysis_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Строит единый DataFrame с ценами, дневной доходностью и скользящими
    корреляциями. Именно этот DataFrame передаётся в движок правил
    (apply_custom_expert_rules), в определение фазы рынка и в отчёт.
    """
    returns = prices.pct_change()
    returns.columns = [f"ret_{col}" for col in returns.columns]

    df = prices.join(returns)

    # Скользящая корреляция доходностей BTC с каждым активом на трёх окнах
    for asset_a, asset_b in CORR_PAIRS:
        ret_a, ret_b = f"ret_{asset_a}", f"ret_{asset_b}"
        for window in CORR_WINDOWS:
            col_name = f"corr_{asset_a}_{asset_b}_{window}"
            df[col_name] = df[ret_a].rolling(window).corr(df[ret_b])

    return df


# ---------------------------------------------------------------------------
# 3. ДВИЖОК ПОЛЬЗОВАТЕЛЬСКОЙ ЭКСПЕРТИЗЫ (Rules Engine)
# ---------------------------------------------------------------------------
def rule_catchup_effect(df: pd.DataFrame):
    """
    Правило 1: Раскорреляция / Catch-up effect.
    Если QQQ растёт 3+ дня подряд, DXY за это время снижается,
    а BTC при этом стоит в узком диапазоне (консолидация) -
    вероятен догоняющий импульсный рост BTC.
    """
    if len(df) < 10:
        return None

    qqq_growing_3d = (df["ret_QQQ"].iloc[-3:] > 0).all()
    dxy_falling = df["ret_DXY"].iloc[-3:].sum() < 0

    btc_window = df["BTC"].iloc[-10:]
    btc_range_pct = (btc_window.max() - btc_window.min()) / btc_window.min() * 100
    btc_consolidating = btc_range_pct < 4.0  # диапазон < 4% за последние 10 дней

    if qqq_growing_3d and dxy_falling and btc_consolidating:
        return {
            "id": "CATCHUP",
            "level": "INFO",
            "title": "Раскорреляция (Catch-up effect)",
            "message": (
                "QQQ растёт 3+ дня подряд, DXY снижается, а BTC консолидируется "
                f"в диапазоне {btc_range_pct:.1f}% за 10 дней -> "
                "высокая вероятность догоняющего импульса BTC."
            ),
        }
    return None


def rule_dxy_pressure(df: pd.DataFrame):
    """
    Правило 2: Давление DXY.
    Резкий рост DXY (>1.5% за неделю) в сочетании с сильной обратной
    30-дневной корреляцией BTC/DXY (< -0.6) -> высокий риск локального
    дампа BTC, стоит снизить плечи.
    """
    if len(df) < 8:
        return None

    dxy_week_change = (df["DXY"].iloc[-1] / df["DXY"].iloc[-8] - 1) * 100
    corr_30 = df["corr_BTC_DXY_30"].iloc[-1]

    if pd.notna(corr_30) and dxy_week_change > 1.5 and corr_30 < -0.6:
        return {
            "id": "DXY_PRESSURE",
            "level": "WARNING",
            "title": "Давление DXY",
            "message": (
                f"DXY вырос на {dxy_week_change:.2f}% за неделю при сильной обратной "
                f"корреляции BTC/DXY (30д = {corr_30:.2f}) -> "
                "высокий риск локального дампа BTC, рекомендуется снизить плечо."
            ),
        }
    return None


def rule_leverage_risk(df: pd.DataFrame, leverage: int = DEFAULT_LEVERAGE):
    """
    Правило 3: Управление рисками при торговле с плечом.
    Если пользователь торгует с плечом 3x-5x, предупреждаем при аномальном
    росте краткосрочной волатильности DXY или S&P относительно её
    30-дневной нормы (расширение волатильности > x1.5).
    """
    if leverage < 3 or len(df) < 35:
        return None

    triggered = []
    for asset in ("DXY", "SPY"):
        vol_short = df[f"ret_{asset}"].rolling(10).std().iloc[-1]
        vol_long = df[f"ret_{asset}"].rolling(30).std().iloc[-1]
        if vol_long and vol_long > 0 and (vol_short / vol_long) > 1.5:
            triggered.append((asset, vol_short / vol_long))

    if triggered:
        details = ", ".join(f"{asset} (x{ratio:.1f} к норме)" for asset, ratio in triggered)
        return {
            "id": "LEVERAGE_RISK",
            "level": "CRITICAL",
            "title": "Риск при торговле с плечом",
            "message": (
                f"Аномальный рост краткосрочной волатильности: {details}. "
                f"При текущем плече {leverage}x рекомендуется уменьшить размер "
                "позиции или временно снизить плечо."
            ),
        }
    return None


# Реестр правил без параметров, которые применяются автоматически.
# Чтобы добавить новое правило: напишите функцию вида rule_xxx(df) -> dict | None
# и добавьте её в этот список.
REGISTERED_RULES = [
    rule_catchup_effect,
    rule_dxy_pressure,
]


def apply_custom_expert_rules(df: pd.DataFrame, leverage: int = DEFAULT_LEVERAGE):
    """
    Точка расширения экспертизы. Прогоняет DataFrame через все
    зарегистрированные правила и возвращает список сработавших флагов.
    """
    flags = []

    for rule_func in REGISTERED_RULES:
        result = rule_func(df)
        if result:
            flags.append(result)

    # Правило по управлению риском зависит от плеча пользователя,
    # поэтому вызывается отдельно с параметром leverage.
    leverage_flag = rule_leverage_risk(df, leverage=leverage)
    if leverage_flag:
        flags.append(leverage_flag)

    return flags


# ---------------------------------------------------------------------------
# 4. ФАЗА РЫНКА И СЦЕНАРИЙ
# ---------------------------------------------------------------------------
def determine_market_phase(df: pd.DataFrame) -> str:
    """Определяет текущую макро-фазу рынка: Risk-On / Risk-Off / Divergence."""
    if len(df) < 10:
        return "НЕДОСТАТОЧНО ДАННЫХ"

    spy_trend_up = df["SPY"].iloc[-1] > df["SPY"].iloc[-6]
    dxy_trend_up = df["DXY"].iloc[-1] > df["DXY"].iloc[-6]

    corr_qqq_30 = df["corr_BTC_QQQ_30"].iloc[-1]
    strong_positive_corr = pd.notna(corr_qqq_30) and corr_qqq_30 > 0.3

    if spy_trend_up and not dxy_trend_up:
        return "RISK-ON (аппетит к риску)" if strong_positive_corr else "RISK-ON (слабая связь с крипто)"
    if not spy_trend_up and dxy_trend_up:
        return "RISK-OFF (бегство в защитные активы)" if strong_positive_corr else "RISK-OFF (слабая связь с крипто)"
    return "DIVERGENCE (рассинхронизация активов)"


def generate_scenario(df: pd.DataFrame, flags: list, phase: str) -> str:
    """
    Формирует вероятностный сценарий движения BTC на ближайшие 24-72 часа
    на основе фазы рынка, сработавших правил экспертизы и текущего моментума.
    Это эвристическая оценка, а не финансовая рекомендация.
    """
    btc_mom_3d = (df["BTC"].iloc[-1] / df["BTC"].iloc[-4] - 1) * 100
    active_ids = {flag["id"] for flag in flags}

    if "CATCHUP" in active_ids and "DXY_PRESSURE" not in active_ids:
        return (
            "БЫЧИЙ (умеренно) - консолидация BTC на фоне роста QQQ и слабеющего доллара "
            "повышает шансы догоняющего движения вверх. Возможен пробой диапазона "
            "консолидации в пределах 24-72ч."
        )

    if "DXY_PRESSURE" in active_ids and "CATCHUP" not in active_ids:
        return (
            "МЕДВЕЖИЙ - сильный доллар при устойчивой обратной корреляции повышает риск "
            "локальной коррекции/дампа BTC в ближайшие 24-72ч. Открытие новых лонгов "
            "стоит отложить."
        )

    if "CATCHUP" in active_ids and "DXY_PRESSURE" in active_ids:
        return (
            "СМЕШАННЫЙ / ПОВЫШЕННАЯ ВОЛАТИЛЬНОСТЬ - одновременно активны сигналы роста "
            "риск-аппетита и давления доллара. Возможны резкие движения в обе стороны, "
            "чёткого направления нет."
        )

    if phase.startswith("RISK-ON"):
        direction = "умеренно вверх" if btc_mom_3d >= 0 else "вверх после короткой паузы"
    elif phase.startswith("RISK-OFF"):
        direction = "умеренно вниз" if btc_mom_3d <= 0 else "вниз после короткого отскока"
    else:
        direction = "боковое движение (диапазон) без выраженной направленности"

    return (
        f"БАЗОВЫЙ СЦЕНАРИЙ ({phase.split(' ')[0]}) - моментум BTC за 3 дня: "
        f"{btc_mom_3d:+.2f}%. Ожидается {direction} на горизонте 24-72ч без "
        "дополнительных экспертных триггеров."
    )


# ---------------------------------------------------------------------------
# 5. ГЕНЕРАТОР СВОДКИ (Output Report)
# ---------------------------------------------------------------------------
def _fmt_corr(value: float) -> str:
    return "  н/д " if pd.isna(value) else f"{value:+.2f}"


def print_report(df: pd.DataFrame, flags: list, phase: str, scenario: str, leverage: int) -> None:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"

    width = 84
    last = df.iloc[-1]
    date_str = df.index[-1].strftime("%Y-%m-%d")

    print("\n" + "=" * width)
    print(f"{BOLD}{'МАКРО-КОРРЕЛЯЦИОННЫЙ ОТЧЁТ: BTC vs TRADFI'.center(width)}{RESET}")
    print(f"{('Дата: ' + date_str + '  |  Плечо пользователя: ' + str(leverage) + 'x').center(width)}")
    print("=" * width)

    print(f"\n{BOLD}Текущие цены закрытия:{RESET}")
    print(
        f"  BTC: ${last['BTC']:,.0f}    DXY: {last['DXY']:.2f}    "
        f"QQQ: ${last['QQQ']:.2f}    SPY: ${last['SPY']:.2f}    "
        f"NVDA: ${last['NVDA']:.2f}    MSFT: ${last['MSFT']:.2f}"
    )

    print(f"\n{BOLD}Скользящая корреляция доходностей BTC:{RESET}")
    header = f"  {'Пара':<14}{'30д':>10}{'90д':>10}{'180д':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for asset_a, asset_b in CORR_PAIRS:
        row = f"  {asset_a + '/' + asset_b:<14}"
        for window in CORR_WINDOWS:
            row += f"{_fmt_corr(last[f'corr_{asset_a}_{asset_b}_{window}']):>10}"
        print(row)

    print(f"\n{BOLD}Фаза рынка:{RESET} {CYAN}{phase}{RESET}")

    print(f"\n{BOLD}Сработавшие правила экспертизы:{RESET}")
    if not flags:
        print("  Нет активных сигналов.")
    else:
        level_color = {"INFO": GREEN, "WARNING": YELLOW, "CRITICAL": RED}
        for flag in flags:
            color = level_color.get(flag["level"], RESET)
            print(f"  [{color}{flag['level']}{RESET}] {BOLD}{flag['title']}{RESET}")
            print(f"      -> {flag['message']}")

    print(f"\n{BOLD}Вероятный сценарий BTC (24-72ч):{RESET}")
    print(f"  {scenario}")

    print(f"\n{'-' * width}")
    print(
        "  ВНИМАНИЕ: отчёт основан на статистических корреляциях и эвристических\n"
        "  правилах. Это не является финансовой рекомендацией. Всегда проверяйте\n"
        "  риски самостоятельно перед принятием торговых решений."
    )
    print("=" * width + "\n")


# ---------------------------------------------------------------------------
# 6. ТОЧКА ВХОДА
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Crypto-Macro Correlation Bot")
    parser.add_argument(
        "--leverage",
        type=float,
        default=DEFAULT_LEVERAGE,
        help="Плечо пользователя для правила управления рисками (по умолчанию 5)",
    )
    parser.add_argument("--start", type=str, default=START_DATE, help="Дата начала выборки (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=END_DATE, help="Дата конца выборки (YYYY-MM-DD)")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        prices = download_prices(start=args.start, end=args.end)
    except Exception as exc:
        print(f"[ошибка] Не удалось получить данные: {exc}", file=sys.stderr)
        sys.exit(1)

    df = build_analysis_frame(prices)

    flags = apply_custom_expert_rules(df, leverage=args.leverage)
    phase = determine_market_phase(df)
    scenario = generate_scenario(df, flags, phase)

    print_report(df, flags, phase, scenario, leverage=args.leverage)


if __name__ == "__main__":
    main()
