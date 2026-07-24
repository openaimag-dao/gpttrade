#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto-Macro Correlation Bot (Telegram)
========================================

Интерактивный Telegram-бот для макро- и корреляционного анализа BTC
относительно традиционных рынков (DXY, QQQ, SPY, NVDA, MSFT), с ML-моделью
прогноза роста BTC, экспертными правилами и ежедневной автоматической
рассылкой полного отчёта.

Запуск:
    export TELEGRAM_BOT_TOKEN="..."
    export TELEGRAM_CHAT_ID="..."   # чат для ежедневной рассылки в 09:00 UTC
    python3 crypto_macro_bot.py
"""

import asyncio
import logging
import os
import sys
import time
import warnings
from datetime import datetime, time as dt_time, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("crypto_macro_bot")

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
ML_RETURN_ASSETS = ("BTC", "DXY", "QQQ", "SPY", "NVDA")

# Целевая переменная: рост BTC более чем на ML_TARGET_THRESHOLD
# на горизонте ML_TARGET_HORIZON дней вперёд (бинарная классификация)
ML_TARGET_HORIZON = 3
ML_TARGET_THRESHOLD = 0.02

# Граница обучение/тест по умолчанию: обучаемся на 2018-2025, тестируем на 2025-2026
ML_SPLIT_DATE = "2025-01-01"

# Сколько секунд держать скачанные данные в кэше, прежде чем обновлять их
# заново (защищает от повторных запросов к Yahoo Finance при частых нажатиях
# кнопок и от rate-limit'ов yfinance).
CACHE_TTL_SECONDS = 15 * 60

# Время ежедневной автоматической рассылки полного отчёта (UTC)
DAILY_REPORT_TIME_UTC = dt_time(hour=9, minute=0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# 1. СБОР И СИНХРОНИЗАЦИЯ ДАННЫХ
# ---------------------------------------------------------------------------
def download_prices(start: str = START_DATE, end: str = None) -> pd.DataFrame:
    """
    Скачивает дневные цены закрытия (Close) по всем активам из TICKERS
    и синхронизирует их в единый календарный ряд. Синхронная (блокирующая)
    функция - в боте всегда вызывается через asyncio.to_thread.
    """
    end = end or datetime.now().strftime("%Y-%m-%d")
    logger.info("Скачивание дневных данных с %s по %s ...", start, end)

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
        close = raw[["Close"]].copy()
        close.columns = list(TICKERS.values())

    reverse_map = {yahoo_symbol: alias for alias, yahoo_symbol in TICKERS.items()}
    close = close.rename(columns=reverse_map)

    missing = [alias for alias in TICKERS if alias not in close.columns]
    if missing:
        raise RuntimeError(f"Не удалось загрузить данные по тикерам: {missing}")

    close = close[list(TICKERS.keys())]

    # --- Синхронизация временных рядов ---
    # BTC торгуется 24/7, фондовый рынок - 5 дней в неделю. Строим полный
    # календарный диапазон дат и переносим последнюю известную цену вперёд
    # (forward fill) на выходные и праздники фондового рынка.
    full_range = pd.date_range(start=close.index.min(), end=close.index.max(), freq="D")
    close = close.reindex(full_range).ffill()
    close = close.dropna(how="any")
    close.index.name = "Date"

    logger.info(
        "Данные синхронизированы: %d наблюдений (%s - %s)",
        len(close), close.index[0].date(), close.index[-1].date(),
    )
    return close


def build_analysis_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Строит единый DataFrame с ценами, дневной доходностью и скользящими
    корреляциями. Именно этот DataFrame используется ML-модулем, движком
    правил и генератором отчётов.
    """
    returns = prices.pct_change()
    returns.columns = [f"ret_{col}" for col in returns.columns]

    df = prices.join(returns)

    for asset_a, asset_b in CORR_PAIRS:
        ret_a, ret_b = f"ret_{asset_a}", f"ret_{asset_b}"
        for window in CORR_WINDOWS:
            col_name = f"corr_{asset_a}_{asset_b}_{window}"
            df[col_name] = df[ret_a].rolling(window).corr(df[ret_b])

    return df


# --- Кэш скачанных и обработанных данных (общий для всех пользователей бота) ---
_cache = {"df": None, "timestamp": 0.0}
_cache_lock = asyncio.Lock()


async def get_analysis_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Возвращает актуальный DataFrame с ценами/доходностями/корреляциями,
    используя кэш с TTL, чтобы не дёргать yfinance на каждое нажатие кнопки.
    Скачивание и построение фрейма выполняются в отдельном потоке, чтобы
    не блокировать asyncio event loop.
    """
    async with _cache_lock:
        now = time.monotonic()
        is_fresh = _cache["df"] is not None and (now - _cache["timestamp"]) < CACHE_TTL_SECONDS
        if not force_refresh and is_fresh:
            return _cache["df"]

        prices = await asyncio.to_thread(download_prices)
        df = await asyncio.to_thread(build_analysis_frame, prices)
        _cache["df"] = df
        _cache["timestamp"] = now
        return df


# ---------------------------------------------------------------------------
# 2. МОДУЛЬ МАШИННОГО ОБУЧЕНИЯ
# ---------------------------------------------------------------------------
def build_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Формирует таблицу признаков для ML-модели:
      - Доходности за 1/3/7/14 дней для BTC, DXY, QQQ, SPY, NVDA.
      - Скользящие корреляции BTC с DXY/QQQ/SPY/NVDA/MSFT (30/90/180д).
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
    тест - на данных после split_date. Синхронная (CPU-bound) функция -
    в боте вызывается через asyncio.to_thread.
    """
    try:
        features = build_ml_features(df)
        target = build_ml_target(df)

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
    except Exception as exc:  # ML-модуль не должен обрушивать бота
        return {"available": False, "reason": f"Ошибка обучения ML-модели: {exc}"}


# ---------------------------------------------------------------------------
# 3. ДВИЖОК ПОЛЬЗОВАТЕЛЬСКОЙ ЭКСПЕРТИЗЫ (Rules Engine)
# ---------------------------------------------------------------------------
def rule_catchup_effect(df: pd.DataFrame):
    """
    Правило 1: Раскорреляция и догоняющий рост (Catch-up).
    QQQ растёт 3+ дня подряд, DXY падает, а волатильность BTC за 5 дней
    заметно ниже 30-дневной нормы (накопление в узком диапазоне) -
    высокий шанс догоняющего пробоя BTC вверх.
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
    30-дневная корреляция BTC с QQQ и с SPY одновременно упала ниже 0.15
    по модулю (полная раскорреляция с TradFi), а DXY торгуется в узком
    диапазоне ("упёрся в уровень") - BTC движется на собственных
    идиосинкразических драйверах.
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
    dxy_at_level = dxy_range_pct < 1.0

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

    dxy_flag = rule_dxy_pressure(df, leverage=leverage)
    if dxy_flag:
        flags.append(dxy_flag)

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
# 5. ФОРМАТИРОВАНИЕ ОТВЕТОВ (HTML для Telegram)
# ---------------------------------------------------------------------------
def _fmt_corr(value: float) -> str:
    return "  н/д " if pd.isna(value) else f"{value:+.2f}"


def format_correlation_section(df: pd.DataFrame) -> str:
    """Раздел '📊 Макро-корреляции': только табличка корреляций 30/90/180д."""
    last = df.iloc[-1]
    date_str = df.index[-1].strftime("%Y-%m-%d")

    lines = [f"📊 <b>Макро-корреляции BTC</b>", f"🗓 {date_str}", ""]
    lines.append("<pre>")
    lines.append(f"{'Пара':<12}{'30д':>8}{'90д':>8}{'180д':>8}")
    lines.append("-" * 36)
    for asset_a, asset_b in CORR_PAIRS:
        row = f"{asset_a + '/' + asset_b:<12}"
        for window in CORR_WINDOWS:
            value = last[f"corr_{asset_a}_{asset_b}_{window}"]
            row += f"{_fmt_corr(value):>8}"
        lines.append(row)
    lines.append("</pre>")
    return "\n".join(lines)


def format_ml_section(ml_result: dict) -> str:
    """Раздел '🤖 ML-Прогноз': вероятность роста BTC по модели."""
    if not ml_result.get("available"):
        return (
            "🤖 <b>ML-Прогноз</b>\n\n"
            f"⚠️ Недоступно: {ml_result.get('reason', 'неизвестная причина')}"
        )

    lines = [
        "🤖 <b>ML-Прогноз</b>",
        "",
        f"Модель: <b>{ml_result['model_name']}</b>",
        f"Обучена на {ml_result['train_size']} набл., тест — {ml_result['test_size']} набл.",
    ]
    if ml_result["test_accuracy"] is not None:
        lines.append(f"Точность на тестовой выборке (2025-2026): {ml_result['test_accuracy'] * 100:.1f}%")
    lines.append("")
    lines.append(
        f"📈 Вероятность роста BTC &gt;{ML_TARGET_THRESHOLD * 100:.0f}% за "
        f"{ML_TARGET_HORIZON} дня: <b>{ml_result['probability_up']:.0f}%</b>"
    )
    return "\n".join(lines)


def format_rules_section(flags: list, leverage: float) -> str:
    """Раздел '🧠 Экспертные сигналы': сработавшие правила для плеча 3x-5x."""
    level_emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨"}
    lines = [f"🧠 <b>Экспертные сигналы</b> (плечо {leverage:.0f}x)", ""]

    if not flags:
        lines.append("Нет активных сигналов.")
    else:
        for flag in flags:
            emoji = level_emoji.get(flag["level"], "•")
            lines.append(f"{emoji} <b>{flag['title']}</b>\n{flag['message']}\n")

    return "\n".join(lines)


def format_full_report(df: pd.DataFrame, flags: list, phase: str, scenario: str,
                        leverage: float, ml_result: dict) -> str:
    """Раздел '📝 Полный отчёт': сводка всех блоков."""
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
        "📝 <b>МАКРО-ОТЧЁТ BTC vs TRADFI</b>",
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


async def build_full_report_text(df: pd.DataFrame, leverage: float = DEFAULT_LEVERAGE) -> str:
    """Собирает ML-прогноз, правила, фазу и сценарий в единый текст отчёта."""
    ml_result = await asyncio.to_thread(train_ml_model, df)
    flags = apply_custom_expert_rules(df, leverage=leverage)
    phase = determine_market_phase(df)
    scenario = generate_scenario(df, flags, phase, ml_result=ml_result)
    return format_full_report(df, flags, phase, scenario, leverage, ml_result)


# ---------------------------------------------------------------------------
# 6. TELEGRAM: МЕНЮ И ОБРАБОТЧИКИ
# ---------------------------------------------------------------------------
MENU_BUTTONS = [
    [InlineKeyboardButton("📊 Макро-корреляции", callback_data="corr")],
    [InlineKeyboardButton("🤖 ML-Прогноз", callback_data="ml")],
    [InlineKeyboardButton("🧠 Экспертные сигналы", callback_data="rules")],
    [InlineKeyboardButton("📝 Полный отчёт", callback_data="full")],
]


def main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(MENU_BUTTONS)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/start и /menu - приветствие и главное меню с inline-кнопками."""
    await update.effective_message.reply_text(
        "👋 <b>Crypto-Macro Correlation Bot</b>\n\n"
        "Я слежу за корреляцией BTC с DXY, QQQ, SPY, NVDA и MSFT, прогоняю "
        "данные через ML-модель и набор экспертных правил для торговли "
        "с плечом 3x-5x.\n\n"
        "Выберите раздел:",
        parse_mode=ParseMode.HTML,
        reply_markup=main_menu_keyboard(),
    )


async def on_menu_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает нажатия на inline-кнопки главного меню."""
    query = update.callback_query
    await query.answer()  # подтверждаем нажатие, чтобы убрать "часики" в клиенте

    await query.edit_message_text("⏳ Считаю, минутку...", reply_markup=None)

    try:
        df = await get_analysis_data()
    except Exception as exc:
        logger.exception("Не удалось получить рыночные данные")
        await query.edit_message_text(
            f"⚠️ Не удалось получить данные с Yahoo Finance: {exc}\n\n"
            "Попробуйте ещё раз чуть позже.",
            reply_markup=main_menu_keyboard(),
        )
        return

    action = query.data
    leverage = DEFAULT_LEVERAGE

    try:
        if action == "corr":
            text = format_correlation_section(df)
        elif action == "ml":
            ml_result = await asyncio.to_thread(train_ml_model, df)
            text = format_ml_section(ml_result)
        elif action == "rules":
            flags = apply_custom_expert_rules(df, leverage=leverage)
            text = format_rules_section(flags, leverage)
        elif action == "full":
            text = await build_full_report_text(df, leverage)
        else:
            text = "Неизвестная команда. Нажмите /start, чтобы открыть меню заново."
    except Exception as exc:
        logger.exception("Ошибка при формировании раздела '%s'", action)
        text = f"⚠️ Произошла ошибка при расчёте: {exc}"

    await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=main_menu_keyboard())


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Глобальный обработчик ошибок - логирует и не даёт боту упасть."""
    logger.error("Необработанное исключение при обработке апдейта", exc_info=context.error)
    if isinstance(update, Update) and update.effective_chat:
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"⚠️ Внутренняя ошибка бота: {context.error}",
            )
        except Exception:
            pass  # если даже сообщение об ошибке не отправляется - молча логируем


# ---------------------------------------------------------------------------
# 7. ЕЖЕДНЕВНАЯ АВТОМАТИЧЕСКАЯ РАССЫЛКА (JobQueue / APScheduler)
# ---------------------------------------------------------------------------
async def send_daily_report(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Каждый день в DAILY_REPORT_TIME_UTC отправляет полный отчёт в TELEGRAM_CHAT_ID."""
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not chat_id:
        logger.warning("TELEGRAM_CHAT_ID не задан - ежедневная рассылка пропущена.")
        return

    try:
        df = await get_analysis_data(force_refresh=True)
        text = await build_full_report_text(df, DEFAULT_LEVERAGE)
    except Exception as exc:
        logger.exception("Не удалось сформировать ежедневный отчёт")
        text = f"⚠️ Не удалось сформировать ежедневный отчёт: {exc}"

    await context.bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.HTML)


# ---------------------------------------------------------------------------
# 8. ТОЧКА ВХОДА
# ---------------------------------------------------------------------------
def main():
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        print(
            "[ошибка] Переменная окружения TELEGRAM_BOT_TOKEN не задана.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.environ.get("TELEGRAM_CHAT_ID"):
        logger.warning(
            "TELEGRAM_CHAT_ID не задан - ежедневная автоматическая рассылка работать не будет "
            "(интерактивное меню по кнопкам продолжит работать для любого пользователя)."
        )

    application = ApplicationBuilder().token(bot_token).build()

    application.add_handler(CommandHandler(["start", "menu"], cmd_start))
    application.add_handler(CallbackQueryHandler(on_menu_button))
    application.add_error_handler(error_handler)

    application.job_queue.run_daily(
        send_daily_report,
        time=DAILY_REPORT_TIME_UTC,
        name="daily_full_report",
    )

    logger.info("Бот запущен, ожидаю сообщений... (ежедневный отчёт в %s UTC)", DAILY_REPORT_TIME_UTC)
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
