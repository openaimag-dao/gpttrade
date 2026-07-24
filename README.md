# gpttrade

## Crypto-Macro Correlation Bot

Локальный комбайн для макро- и корреляционного анализа BTC относительно
традиционных рынков (DXY, QQQ, SPY, NVDA, MSFT) за период 2018–2026 гг.,
дополненный ML-моделью прогноза роста BTC и отправкой отчёта в Telegram.

### Установка

```bash
pip install -r requirements.txt
```

Новые зависимости для ML-модуля и Telegram-интеграции:

```bash
pip install xgboost scikit-learn requests
```

(`xgboost` используется, если установлен; иначе автоматически используется
`RandomForestClassifier` из `scikit-learn`.)

### Запуск

```bash
python3 crypto_macro_bot.py
```

Опциональные параметры:

```bash
python3 crypto_macro_bot.py --leverage 5 --start 2018-01-01 --end 2026-07-24 \
    --ml-split-date 2025-01-01 --telegram
```

Для отправки отчёта в Telegram задайте переменные окружения (или флаги
`--telegram-token` / `--telegram-chat-id`):

```bash
export TELEGRAM_BOT_TOKEN="..."
export TELEGRAM_CHAT_ID="..."
python3 crypto_macro_bot.py --telegram
```

### Что делает скрипт

1. Скачивает дневные цены закрытия по BTC-USD, DX-Y.NYB (DXY), QQQ, SPY, NVDA, MSFT через `yfinance`
   и синхронизирует ряды (крипта торгуется 24/7, фондовый рынок — 5 дней в неделю, используется `.ffill()`).
2. Считает дневную доходность и скользящую корреляцию BTC с каждым активом на окнах 30/90/180 дней.
3. Обучает ML-модель (`train_ml_model`) — XGBoost (или RandomForest как запасной вариант) на признаках
   (лаговые доходности, скользящие корреляции, волатильность DXY/QQQ, отношение цены BTC к SMA50/SMA200) —
   и возвращает вероятность роста BTC более чем на 2% за 3 дня. Обучение на 2018–2025, тест на 2025–2026.
4. Прогоняет данные через движок пользовательской экспертизы (`apply_custom_expert_rules`) с тремя
   правилами: раскорреляция/catch-up, давление DXY с учётом плеча 3x-5x, сжатие волатильности перед импульсом.
5. Выводит в консоль отчёт: цены, таблицу корреляций, фазу рынка (Risk-On/Risk-Off/Divergence),
   ML-прогноз, сработавшие правила и вероятный сценарий движения BTC на 24–72 часа.
6. По флагу `--telegram` отправляет отформатированный отчёт с эмодзи в Telegram через Bot API.

Новые торговые правила добавляются функцией вида `rule_xxx(df) -> dict | None`,
зарегистрированной в списке `REGISTERED_RULES`.
