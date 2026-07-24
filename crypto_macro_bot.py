#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto-Macro Correlation Bot
============================

Профессиональный локальный комбайн для макро- и корреляционного анализа BTC
относительно традиционных рынков (DXY, QQQ, SPY, NVDA, MSFT) за период
2018-2026 гг., дополненный ML-моделью прогноза роста BTC и отправкой
итогового отчёта в Telegram.

Запуск:
    python3 crypto_macro_bot.py [--leverage 5] [--telegram]
"""

import argparse
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests
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

# Горизонты доходностей для ML-признаков (в днях)
ML_RETURN_HORIZONS = (1, 3, 7, 14)
# Активы, для которых считаются лаговые доходности как признаки ML-модели
ML_RETURN_ASSETS = ("BTC", "DXY", "QQQ", "SPY", "NVDA")

# Целевая переменная: рост BTC более чем на ML_TARGET_THRESHOLD
# на горизонте ML_TARGET_HORIZON дней вперёд (бинарная классификация)
ML_TARGET_HORIZON = 3
ML_TARGET_THRESHOLD = 0.02

# Граница обучение/тест по умолчанию: обучаемся на 2018-2025, тестируем на 2025-2026
ML_SPLIT_DATE = "2025-01-01"


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
    (apply_custom_expert_rules), в ML-модуль, в определение фазы рынка и в отчёт.
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
# 3. МОДУЛЬ МАШИННОГО ОБУЧЕНИЯ
# ---------------------------------------------------------------------------
def build_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Формирует таблицу признаков для ML-модели:
      - Доходности за 1/3/7/14 дней для BTC, DXY, QQQ, SPY, NVDA.
      - Скользящие корреляции BTC с DXY/QQQ/SPY/NVDA/MSFT (30/90/180д),
        уже посчитанные в build_analysis_frame.
      - Волатильность (rolling std дневной доходности) DXY и QQQ.
      - Отношение цены BTC к его 50- и 200-дневной SMA.
    """
    features = pd.DataFrame(index=df.index)

    for asset in ML_RETURN_ASSETS:
        for horizon in ML_RETURN_HORIZONS:
            features[f"ret_{asset}_{horizon}d"] = df[asset].pct_change(horizon)

    for asset_a, asset_b in CORR_PAIRS:
        for window in CORR_WINDOWS:
            col_name = f"corr_{asset_a}_{asset_b}_{window}"
            features[col_name] = df[col_name]

    features["vol_DXY_10"] = df["ret_DXY"].rolling(10).std()
    features["vol_DXY_30"] = df["ret_DXY"].rolling(30).std()
    features["vol_QQQ_10"] = df["ret_QQQ"].rolling(10).std()
    features["vol_QQQ_30"] = df["ret_QQQ"].rolling(30).std()

    sma_50 = df["BTC"].rolling(50).mean()
    sma_200 = df["BTC"].rolling(200).mean()
    features["btc_to_sma50"] = df["BTC"] / sma_50
    features["btc_to_sma200"] = df["BTC"] / sma_200

    return features


def build_ml_target(df: pd.DataFrame, horizon: int = ML_TARGET_HORIZON,
                     threshold: float = ML_TARGET_THRESHOLD) -> pd.Series:
    """
    Целевая переменная: 1, если BTC вырастет более чем на threshold
    на горизонте horizon дней вперёд, иначе 0.
    """
    future_return = df["BTC"].shift(-horizon) / df["BTC"] - 1
    return (future_return > threshold).astype(int)


def train_ml_model(df: pd.DataFrame, split_date: str = ML_SPLIT_DATE) -> dict:
    """
    Обучает классификатор (XGBoost, либо RandomForest как запасной вариант,
    если xgboost не установлен) предсказывать рост BTC > ML_TARGET_THRESHOLD
    на горизонте ML_TARGET_HORIZON дней. Обучение - на данных до split_date,
    тест - на данных после split_date. Возвращает вероятность роста BTC
    на текущий момент вместе с метриками качества модели.
    """
    try:
        features = build_ml_features(df)
        target = build_ml_target(df)

        # Последняя строка с полностью посчитанными признаками (для инференса);
        # её таргет ещё не наступил, поэтому берём отдельно от обучающей выборки.
        features_complete = features.dropna()
        if features_complete.empty:
            return {"available": False, "reason": "Недостаточно истории для расчёта всех признаков."}
        latest_features = features_complete.iloc[[-1]]

        dataset = features.copy()
        dataset["target"] = target
        dataset = dataset.dropna()

        if dataset.empty:
            return {"available": False, "reason": "Недостаточно данных с известным исходом для обучения."}

        feature_cols = features.columns
        train_data = dataset[dataset.index < split_date]
        test_data = dataset[dataset.index >= split_date]

        if len(train_data) < 100:
            return {"available": False, "reason": "Недостаточно обучающих данных (нужно минимум 100 наблюдений)."}

        X_train, y_train = train_data[feature_cols], train_data["target"]
        X_test, y_test = test_data[feature_cols], test_data["target"]

        model_name = "XGBoost"
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
            )
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            model_name = "RandomForest"
            model = RandomForestClassifier(n_estimators=400, max_depth=6, random_state=42, n_jobs=-1)

        model.fit(X_train, y_train)

        test_accuracy = None
        if len(X_test) > 0 and y_test.nunique() > 1:
            test_accuracy = float(model.score(X_test, y_test))

        probability_up = float(model.predict_proba(latest_features)[0][1]) * 100

        return {
            "available": True,
            "model_name": model_name,
            "probability_up": probability_up,
            "test_accuracy": test_accuracy,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "as_of": latest_features.index[-1],
        }
    except Exception as exc:  # ML-модуль не должен обрушивать весь отчёт
        return {"available": False, "reason": f"Ошибка обучения ML-модели: {exc}"}


# ---------------------------------------------------------------------------
# 4. ДВИЖОК ПОЛЬЗОВАТЕЛЬСКОЙ ЭКСПЕРТИЗЫ (Rules Engine)
# ---------------------------------------------------------------------------
def rule_catchup_effect(df: pd.DataFrame):
    """
    Правило 1: Раскорреляция и догоняющий рост (Catch-up).
    Если QQQ растёт 3+ дня подряд, DXY за это время падает, а волатильность
    BTC за последние 5 дней заметно ниже её 30-дневной нормы (накопление
    в узком диапазоне) - высокий шанс догоняющего пробоя BTC вверх.
    """
    if len(df) < 35:
        return None

    qqq_growing_3d = (df["ret_QQQ"].iloc[-3:] > 0).all()
    dxy_falling = df["ret_DXY"].iloc[-3:].sum() < 0

    btc_vol_5d = df["ret_BTC"].rolling(5).std().iloc[-1]
    btc_vol_30d = df["ret_BTC"].rolling(30).std().iloc[-1]
    btc_low_vol = (
        pd.notna(btc_vol_5d) and pd.notna(btc_vol_30d)
        and btc_vol_30d > 0 and btc_vol_5d < btc_vol_30d * 0.7
    )

    if qqq_growing_3d and dxy_falling and btc_low_vol:
        return {
            "id": "CATCHUP",
            "level": "INFO",
            "title": "Раскорреляция и догоняющий рост (Catch-up)",
            "message": (
                f"QQQ растёт 3+ дня подряд, DXY снижается, а волатильность BTC "
                f"за 5 дней ({btc_vol_5d * 100:.2f}%) заметно ниже 30-дневной нормы "
                f"({btc_vol_30d * 100:.2f}%) -> накопление в рэндже на фоне сильного "
                "TradFi. Высокий шанс пробоя вверх."
            ),
        }
    return None


def rule_dxy_pressure(df: pd.DataFrame, leverage: float = DEFAULT_LEVERAGE):
    """
    Правило 2: Давление DXY и управление риском для плеча 3x-5x.
    Недельный рост DXY более чем на 1% в сочетании с сильной обратной
    30-дневной корреляцией BTC/DXY (< -0.5) -> опасность локального сброса,
    рекомендация снизить плечо до минимума или подтянуть стопы.
    """
    if len(df) < 8:
        return None

    dxy_week_change = (df["DXY"].iloc[-1] / df["DXY"].iloc[-8] - 1) * 100
    corr_30 = df["corr_BTC_DXY_30"].iloc[-1]

    if pd.notna(corr_30) and dxy_week_change > 1.0 and corr_30 < -0.5:
        return {
            "id": "DXY_PRESSURE",
            "level": "WARNING",
            "title": "Давление DXY & риск для плеча 3x-5x",
            "message": (
                f"DXY вырос на {dxy_week_change:.2f}% за неделю при сильной обратной "
                f"корреляции BTC/DXY (30д = {corr_30:.2f}). Опасность локального "
                f"сброса. При текущем плече {leverage:.0f}x рекомендация: снизить "
                "плечо до минимума или подтянуть стопы."
            ),
        }
    return None


def rule_vol_compression(df: pd.DataFrame):
    """
    Правило 3: Сжатие волатильности перед импульсом.
    Если 30-дневная корреляция BTC с QQQ и с SPY одновременно упала ниже
    0.15 по модулю (полная раскорреляция с TradFi), а DXY при этом торгуется
    в узком диапазоне ("упёрся в уровень") - BTC движется на собственных
    идиосинкразических драйверах, стоит следить за внутренними объёмами.
    """
    if len(df) < 30:
        return None

    corr_qqq_30 = df["corr_BTC_QQQ_30"].iloc[-1]
    corr_spy_30 = df["corr_BTC_SPY_30"].iloc[-1]

    if pd.isna(corr_qqq_30) or pd.isna(corr_spy_30):
        return None

    fully_decorrelated = abs(corr_qqq_30) < 0.15 and abs(corr_spy_30) < 0.15

    dxy_window = df["DXY"].iloc[-10:]
    dxy_range_pct = (dxy_window.max() - dxy_window.min()) / dxy_window.min() * 100
    dxy_at_level = dxy_range_pct < 1.0  # DXY зажат в диапазоне < 1% за 10 дней

    if fully_decorrelated and dxy_at_level:
        return {
            "id": "VOL_COMPRESSION",
            "level": "INFO",
            "title": "Сжатие волатильности перед импульсом",
            "message": (
                f"30д корреляция BTC/QQQ ({corr_qqq_30:.2f}) и BTC/SPY ({corr_spy_30:.2f}) "
                f"близки к нулю, а DXY зажат в диапазоне {dxy_range_pct:.2f}% за 10 дней -> "
                "BTC торгуется на собственных идиосинкразических драйверах. "
                "Внимание к внутренним объёмам."
            ),
        }
    return None


# Правила без параметров, которые применяются автоматически.
# Чтобы добавить новое правило: напишите функцию вида rule_xxx(df) -> dict | None
# и добавьте её в этот список.
REGISTERED_RULES = [
    rule_catchup_effect,
    rule_vol_compression,
]


def apply_custom_expert_rules(df: pd.DataFrame, leverage: float = DEFAULT_LEVERAGE):
    """
    Точка расширения экспертизы. Прогоняет DataFrame через все
    зарегистрированные правила и возвращает список сработавших флагов.
    """
    flags = []

    for rule_func in REGISTERED_RULES:
        result = rule_func(df)
        if result:
            flags.append(result)

    # Правило давления DXY зависит от плеча пользователя (используется
    # в тексте рекомендации), поэтому вызывается отдельно.
    dxy_flag = rule_dxy_pressure(df, leverage=leverage)
    if dxy_flag:
        flags.append(dxy_flag)

    return flags


# ---------------------------------------------------------------------------
# 5. ФАЗА РЫНКА И СЦЕНАРИЙ
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


def generate_scenario(df: pd.DataFrame, flags: list, phase: str, ml_result: dict = None) -> str:
    """
    Формирует вероятностный сценарий движения BTC на ближайшие 24-72 часа
    на основе фазы рынка, сработавших правил экспертизы, текущего моментума
    и (если доступна) вероятности ML-модели. Это эвристическая оценка,
    а не финансовая рекомендация.
    """
    btc_mom_3d = (df["BTC"].iloc[-1] / df["BTC"].iloc[-4] - 1) * 100
    active_ids = {flag["id"] for flag in flags}

    if "VOL_COMPRESSION" in active_ids:
        base = (
            "ИДИОСИНКРАЗИЧЕСКИЙ - корреляция BTC с QQQ/SPY развалилась, а DXY застрял "
            "в узком диапазоне. Макро-факторы временно не определяют движение BTC, "
            "решающими на горизонте 24-72ч будут внутренние объёмы и локальные потоки."
        )
    elif "CATCHUP" in active_ids and "DXY_PRESSURE" not in active_ids:
        base = (
            "БЫЧИЙ (умеренно) - накопление BTC в узком диапазоне на фоне роста QQQ и "
            "слабеющего доллара повышает шансы догоняющего пробоя вверх в пределах 24-72ч."
        )
    elif "DXY_PRESSURE" in active_ids and "CATCHUP" not in active_ids:
        base = (
            "МЕДВЕЖИЙ - сильный доллар при устойчивой обратной корреляции повышает риск "
            "локального сброса BTC в ближайшие 24-72ч. Новые лонги стоит отложить."
        )
    elif "CATCHUP" in active_ids and "DXY_PRESSURE" in active_ids:
        base = (
            "СМЕШАННЫЙ / ПОВЫШЕННАЯ ВОЛАТИЛЬНОСТЬ - одновременно активны сигналы роста "
            "риск-аппетита и давления доллара. Возможны резкие движения в обе стороны."
        )
    else:
        if phase.startswith("RISK-ON"):
            direction = "умеренно вверх" if btc_mom_3d >= 0 else "вверх после короткой паузы"
        elif phase.startswith("RISK-OFF"):
            direction = "умеренно вниз" if btc_mom_3d <= 0 else "вниз после короткого отскока"
        else:
            direction = "боковое движение (диапазон) без выраженной направленности"
        base = (
            f"БАЗОВЫЙ СЦЕНАРИЙ ({phase.split(' ')[0]}) - моментум BTC за 3 дня: "
            f"{btc_mom_3d:+.2f}%. Ожидается {direction} на горизонте 24-72ч без "
            "дополнительных экспертных триггеров."
        )

    if ml_result and ml_result.get("available"):
        base += (
            f" ML-модель ({ml_result['model_name']}) оценивает вероятность роста BTC "
            f"более чем на {ML_TARGET_THRESHOLD * 100:.0f}% за {ML_TARGET_HORIZON} дня "
            f"в {ml_result['probability_up']:.0f}%."
        )

    return base


# ---------------------------------------------------------------------------
# 6. ГЕНЕРАТОР СВОДКИ (Output Report)
# ---------------------------------------------------------------------------
def _fmt_corr(value: float) -> str:
    return "  н/д " if pd.isna(value) else f"{value:+.2f}"


def print_report(df: pd.DataFrame, flags: list, phase: str, scenario: str,
                  leverage: float, ml_result: dict) -> None:
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

    print(f"\n{BOLD}ML-модель:{RESET}")
    if ml_result.get("available"):
        print(
            f"  Модель: {ml_result['model_name']}  |  Обучена на {ml_result['train_size']} "
            f"наблюдениях, протестирована на {ml_result['test_size']}"
        )
        accuracy_str = (
            f"{ml_result['test_accuracy'] * 100:.1f}%" if ml_result["test_accuracy"] is not None else "н/д"
        )
        print(f"  Точность на тестовой выборке (2025-2026): {accuracy_str}")
        print(
            f"  Вероятность роста BTC > {ML_TARGET_THRESHOLD * 100:.0f}% за "
            f"{ML_TARGET_HORIZON} дня: {BOLD}{ml_result['probability_up']:.0f}%{RESET}"
        )
    else:
        print(f"  Недоступно: {ml_result.get('reason', 'неизвестная причина')}")

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
        "  ВНИМАНИЕ: отчёт основан на статистических корреляциях, эвристических\n"
        "  правилах и ML-модели, обученной на исторических данных. Это не является\n"
        "  финансовой рекомендацией. Всегда проверяйте риски самостоятельно."
    )
    print("=" * width + "\n")


def format_telegram_message(df: pd.DataFrame, flags: list, phase: str, scenario: str,
                             leverage: float, ml_result: dict) -> str:
    """Форматирует отчёт с эмодзи для отправки в Telegram (HTML parse_mode)."""
    last = df.iloc[-1]
    date_str = df.index[-1].strftime("%Y-%m-%d")

    if phase.startswith("RISK-ON"):
        phase_emoji = "🟢"
    elif phase.startswith("RISK-OFF"):
        phase_emoji = "🔴"
    else:
        phase_emoji = "🟡"

    level_emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨"}

    lines = [
        "📊 <b>МАКРО-ОТЧЁТ BTC vs TRADFI</b>",
        f"🗓 {date_str}  |  Плечо: {leverage:.0f}x",
        "",
        f"₿ BTC: ${last['BTC']:,.0f}    💵 DXY: {last['DXY']:.2f}",
        f"📈 QQQ: ${last['QQQ']:.2f}    📈 SPY: ${last['SPY']:.2f}",
        "",
        f"{phase_emoji} <b>Фаза рынка:</b> {phase}",
        "",
    ]

    if ml_result.get("available"):
        lines.append(
            f"🤖 <b>ML-модель ({ml_result['model_name']}):</b> вероятность роста BTC "
            f"&gt;{ML_TARGET_THRESHOLD * 100:.0f}% за {ML_TARGET_HORIZON}д — "
            f"<b>{ml_result['probability_up']:.0f}%</b>"
        )
        if ml_result.get("test_accuracy") is not None:
            lines.append(f"   Точность на тесте 2025-2026: {ml_result['test_accuracy'] * 100:.1f}%")
        lines.append("")

    lines.append("🔎 <b>Сработавшие правила:</b>")
    if not flags:
        lines.append("Нет активных сигналов.")
    else:
        for flag in flags:
            emoji = level_emoji.get(flag["level"], "•")
            lines.append(f"{emoji} <b>{flag['title']}</b>\n{flag['message']}")
    lines.append("")
    lines.append(f"🔮 <b>Сценарий (24-72ч):</b>\n{scenario}")
    lines.append("")
    lines.append("⚠️ <i>Не является финансовой рекомендацией.</i>")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. TELEGRAM-ИНТЕГРАЦИЯ
# ---------------------------------------------------------------------------
def send_telegram_report(message: str, bot_token: str = None, chat_id: str = None) -> bool:
    """
    Отправляет текстовый отчёт в Telegram через Bot API.
    Токен и chat_id берутся из аргументов, а если не переданы -
    из переменных окружения TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID.
    """
    bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        print(
            "[i] Отправка в Telegram пропущена: не заданы TELEGRAM_BOT_TOKEN / "
            "TELEGRAM_CHAT_ID (переменные окружения или --telegram-token/--telegram-chat-id)."
        )
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        print("[i] Отчёт успешно отправлен в Telegram.")
        return True
    except requests.RequestException as exc:
        print(f"[ошибка] Не удалось отправить отчёт в Telegram: {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# 8. ТОЧКА ВХОДА
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
    parser.add_argument(
        "--ml-split-date",
        type=str,
        default=ML_SPLIT_DATE,
        help="Граница между обучающей и тестовой выборкой ML-модели (по умолчанию 2025-01-01)",
    )
    parser.add_argument(
        "--telegram",
        action="store_true",
        help="Отправить отчёт в Telegram (требует TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID)",
    )
    parser.add_argument("--telegram-token", type=str, default=None, help="Токен Telegram-бота (переопределяет env)")
    parser.add_argument("--telegram-chat-id", type=str, default=None, help="Chat ID Telegram (переопределяет env)")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        prices = download_prices(start=args.start, end=args.end)
    except Exception as exc:
        print(f"[ошибка] Не удалось получить данные: {exc}", file=sys.stderr)
        sys.exit(1)

    df = build_analysis_frame(prices)

    print("[i] Обучение ML-модели прогноза роста BTC ...")
    ml_result = train_ml_model(df, split_date=args.ml_split_date)

    flags = apply_custom_expert_rules(df, leverage=args.leverage)
    phase = determine_market_phase(df)
    scenario = generate_scenario(df, flags, phase, ml_result=ml_result)

    print_report(df, flags, phase, scenario, leverage=args.leverage, ml_result=ml_result)

    if args.telegram:
        message = format_telegram_message(
            df, flags, phase, scenario, leverage=args.leverage, ml_result=ml_result
        )
        send_telegram_report(message, bot_token=args.telegram_token, chat_id=args.telegram_chat_id)


if __name__ == "__main__":
    main()
