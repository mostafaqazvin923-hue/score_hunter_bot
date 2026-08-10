import os
import json
import time
import requests

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

CMC_URL = "https://pro-api.coinmarketcap.com/public-api/v1/cryptocurrency/quotes/latest"

ETH_ID = 1027

TIMEFRAME_HOURS = 4

TP_PERCENT = 1.0
SL_PERCENT = 0.50
REQUIRED_SCORE = 5

STATE_FILE = "state.json"


def load_state():
    if not os.path.exists(STATE_FILE):
        return {}

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    response = requests.post(
        url,
        data={
            "chat_id": CHAT_ID,
            "text": message
        },
        timeout=20
    )

    print("Telegram:", response.status_code)
    print(response.text)

    response.raise_for_status()


def get_eth_price():
    response = requests.get(
        CMC_URL,
        params={
            "id": ETH_ID,
            "convert": "USD"
        },
        headers={
            "X-CMC_PRO_API_KEY": os.environ["CMC_API_KEY"]
        },
        timeout=20
    )

    print("CoinMarketCap:", response.status_code)

    response.raise_for_status()

    data = response.json()["data"]

    return float(
        data[str(ETH_ID)]["quote"]["USD"]["price"]
    )


def get_4h_data():
    """
    این تابع باید داده‌های کندل 4H شامل:
    open, high, low, close, volume
    را دریافت کند.

    برای اینکه منطق Pine Script دقیقاً روی کندل‌های 4H
    اجرا شود، منبع داده باید OHLCV واقعی ارائه دهد.
    """

    url = "https://api.binance.com/api/v3/klines"

    response = requests.get(
        url,
        params={
            "symbol": "ETHUSDT",
            "interval": "4h",
            "limit": 250
        },
        timeout=20
    )

    print("Market data:", response.status_code)

    response.raise_for_status()

    return response.json()


def ema(values, period):
    if len(values) < period:
        return None

    multiplier = 2 / (period + 1)

    result = sum(values[:period]) / period

    for price in values[period:]:
        result = (
            (price - result) * multiplier
            + result
        )

    return result


def sma(values, period):
    if len(values) < period:
        return None

    return sum(values[-period:]) / period


def rsi(values, period=14):
    if len(values) <= period:
        return None

    gains = []
    losses = []

    for i in range(1, len(values)):
        change = values[i] - values[i - 1]

        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (
            (avg_gain * (period - 1) + gains[i])
            / period
        )

        avg_loss = (
            (avg_loss * (period - 1) + losses[i])
            / period
        )

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def atr(highs, lows, closes, period=14):
    if len(closes) <= period:
        return None

    true_ranges = []

    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )

        true_ranges.append(tr)

    return sum(true_ranges[-period:]) / period


def calculate_signal(candles):
    if len(candles) < 210:
        print("Not enough 4H candles.")
        return None

    opens = [float(x[1]) for x in candles]
    highs = [float(x[2]) for x in candles]
    lows = [float(x[3]) for x in candles]
    closes = [float(x[4]) for x in candles]
    volumes = [float(x[5]) for x in candles]

    close = closes[-1]
    open_price = opens[-1]
    high = highs[-1]
    low = lows[-1]

    ema20 = ema(closes, 20)
    ema50 = ema(closes, 50)
    ema200 = ema(closes, 200)

    current_rsi = rsi(closes, 14)

    previous_rsi = rsi(closes[:-1], 14)

    volume_ma = sma(volumes, 20)

    current_atr = atr(
        highs,
        lows,
        closes,
        14
    )

    if any(
        x is None
        for x in [
            ema20,
            ema50,
            ema200,
            current_rsi,
            previous_rsi,
            volume_ma,
            current_atr
        ]
    ):
        return None

    long_trend = (
        close > ema200
        and ema20 > ema50
    )

    short_trend = (
        close < ema200
        and ema20 < ema50
    )

    long_rsi = (
        current_rsi > 50
        and current_rsi < 72
        and current_rsi > previous_rsi
    )

    short_rsi = (
        current_rsi < 50
        and current_rsi > 28
        and current_rsi < previous_rsi
    )

    volume_ok = volume >= volume_ma

    recent_high = max(highs[-7:-1])
    recent_low = min(lows[-7:-1])

    bull_break = close > recent_high
    bear_break = close < recent_low

    long_pullback = (
        low <= ema20
        and close > ema20
    )

    short_pullback = (
        high >= ema20
        and close < ema20
    )

    candle_range = high - low

    bull_candle = (
        close > open_price
        and candle_range > 0
        and (
            (close - open_price)
            / candle_range
        ) >= 0.40
    )

    bear_candle = (
        close < open_price
        and candle_range > 0
        and (
            (open_price - close)
            / candle_range
        ) >= 0.40
    )

    volatility_ok = (
        current_atr / close
    ) >= 0.002

    long_score = (
        int(long_trend)
        + int(long_rsi)
        + int(volume_ok)
        + int(bull_break)
        + int(long_pullback)
        + int(bull_candle)
        + int(volatility_ok)
    )

    short_score = (
        int(short_trend)
        + int(short_rsi)
        + int(volume_ok)
        + int(bear_break)
        + int(short_pullback)
        + int(bear_candle)
        + int(volatility_ok)
    )

    print(f"ETHUSDT 4H close: {close}")
    print(f"RSI: {current_rsi:.2f}")
    print(f"EMA20: {ema20}")
    print(f"EMA50: {ema50}")
    print(f"EMA200: {ema200}")
    print(f"ATR: {current_atr}")
    print(f"LONG SCORE: {long_score}/7")
    print(f"SHORT SCORE: {short_score}/7")

    if long_score >= REQUIRED_SCORE:
        return {
            "direction": "LONG",
            "score": long_score,
            "price": close
        }

    if short_score >= REQUIRED_SCORE:
        return {
            "direction": "SHORT",
            "score": short_score,
            "price": close
        }

    return None


def main():
    print("🟢 SCORE HUNTER 4H SCANNING")

    state = load_state()

    candles = get_4h_data()

    signal = calculate_signal(candles)

    now = int(time.time())

    today = time.strftime(
        "%Y-%m-%d",
        time.gmtime(now)
    )

    last_signal_day = state.get(
        "last_signal_day"
    )

    if signal is None:
        print("No valid signal.")
        save_state(state)
        print("✅ Scan completed.")
        return

    if last_signal_day == today:
        print("Signal already sent today.")
        save_state(state)
        print("✅ Scan completed.")
        return

    direction = signal["direction"]
    score = signal["score"]
    price = signal["price"]

    if direction == "LONG":
        tp = price * (1 + TP_PERCENT / 100)
        sl = price * (1 - SL_PERCENT / 100)

        message = (
            "🚨 SCORE HUNTER 4H 🚨\n\n"
            "💰 ETHUSDT\n"
            "📊 🟢 LONG\n"
            f"⭐ Score: {score}/7\n"
            f"💵 Entry: {price:.2f}\n"
            f"🎯 TP: {tp:.2f} (+{TP_PERCENT}%)\n"
            f"🛑 SL: {sl:.2f} (-{SL_PERCENT}%)\n\n"
            "⏱ Timeframe: 4H"
        )

    else:
        tp = price * (1 - TP_PERCENT / 100)
        sl = price * (1 + SL_PERCENT / 100)

        message = (
            "🚨 SCORE HUNTER 4H 🚨\n\n"
            "💰 ETHUSDT\n"
            "📊 🔴 SHORT\n"
            f"⭐ Score: {score}/7\n"
            f"💵 Entry: {price:.2f}\n"
            f"🎯 TP: {tp:.2f} (-{TP_PERCENT}%)\n"
            f"🛑 SL: {sl:.2f} (+{SL_PERCENT}%)\n\n"
            "⏱ Timeframe: 4H"
        )

    send_telegram(message)

    state["last_signal_day"] = today
    state["last_signal"] = signal
    state["signal_time"] = now

    save_state(state)

    print("🚨 SIGNAL SENT")
    print("✅ Scan completed.")


if __name__ == "__main__":
    main()
