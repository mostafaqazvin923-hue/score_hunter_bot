import os
import json
import requests
from datetime import datetime, timezone

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

KRAKEN_URL = "https://api.kraken.com/0/public/OHLC"

STATE_FILE = "state.json"

SYMBOL = "ETHUSDT"
INTERVAL = 240  # 4 hours

TP_PERCENT = 1.0
SL_PERCENT = 0.50
REQUIRED_SCORE = 5


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


def get_4h_data():
    print("Getting ETHUSDT 4H candles from Kraken...")

    response = requests.get(
        KRAKEN_URL,
        params={
            "pair": "ETHUSDT",
            "interval": INTERVAL
        },
        timeout=20
    )

    print("Kraken:", response.status_code)

    response.raise_for_status()

    payload = response.json()

    if payload.get("error"):
        raise RuntimeError(
            f"Kraken API error: {payload['error']}"
        )

    result = payload.get("result", {})

    pair_key = next(
        (key for key in result if key != "last"),
        None
    )

    if pair_key is None:
        raise RuntimeError(
            f"Kraken returned no candle data: {payload}"
        )

    raw = result[pair_key]

    candles = []

    for row in raw:
        candles.append({
            "time": int(row[0]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[6])
        })

    # آخرین کندل ممکن است هنوز در حال تشکیل باشد.
    # فقط کندل بسته‌شده را استفاده می‌کنیم.
    if len(candles) > 1:
        candles = candles[:-1]

    if len(candles) < 210:
        raise RuntimeError(
            f"Not enough 4H candles: {len(candles)}"
        )

    print(
        f"Kraken 4H closed candles: {len(candles)}"
    )

    return candles


def ema(values, period):
    if len(values) < period:
        return None

    value = sum(values[:period]) / period
    multiplier = 2.0 / (period + 1)

    for price in values[period:]:
        value = (
            (price - value) * multiplier
            + value
        )

    return value


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

        gains.append(max(change, 0))
        losses.append(max(-change, 0))

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

    return 100.0 - (100.0 / (1.0 + rs))


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

    return sum(
        true_ranges[-period:]
    ) / period


def calculate_signal(candles):
    if len(candles) < 210:
        print("Not enough candles.")
        return None

    opens = [c["open"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    closes = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]

    # آخرین کندل بسته‌شده
    open_price = opens[-1]
    high = highs[-1]
    low = lows[-1]
    close = closes[-1]
    volume = volumes[-1]

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
        print("Indicator calculation failed.")
        return None

    # =========================
    # TREND
    # =========================

    long_trend = (
        close > ema200
        and ema20 > ema50
    )

    short_trend = (
        close < ema200
        and ema20 < ema50
    )

    # =========================
    # RSI
    # =========================

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

    # =========================
    # VOLUME
    # =========================

    volume_ok = volume >= volume_ma

    # =========================
    # MARKET STRUCTURE
    # =========================

    recent_high = max(highs[-7:-1])
    recent_low = min(lows[-7:-1])

    bull_break = close > recent_high
    bear_break = close < recent_low

    # =========================
    # EMA PULLBACK
    # =========================

    long_pullback = (
        low <= ema20
        and close > ema20
    )

    short_pullback = (
        high >= ema20
        and close < ema20
    )

    # =========================
    # CANDLE CONFIRMATION
    # =========================

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

    # =========================
    # VOLATILITY
    # =========================

    volatility_ok = (
        current_atr / close
    ) >= 0.002

    # =========================
    # SCORE
    # =========================

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

    print()
    print("========== ETHUSDT 4H ==========")
    print(f"Close: {close:.4f}")
    print(f"RSI: {current_rsi:.2f}")
    print(f"EMA20: {ema20:.4f}")
    print(f"EMA50: {ema50:.4f}")
    print(f"EMA200: {ema200:.4f}")
    print(f"ATR: {current_atr:.4f}")
    print(f"Volume OK: {volume_ok}")
    print(f"LONG SCORE: {long_score}/7")
    print(f"SHORT SCORE: {short_score}/7")
    print("================================")
    print()

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

    latest_candle_time = candles[-1]["time"]

    previous_candle_time = state.get(
        "last_checked_candle"
    )

    print(
        f"Latest closed 4H candle: "
        f"{latest_candle_time}"
    )

    # فقط در صورت تشکیل کندل 4H جدید بررسی کن.
    if previous_candle_time == latest_candle_time:
        print("No new closed 4H candle.")
        print("⏳ Waiting for next 4H candle.")
        return

    state["last_checked_candle"] = latest_candle_time

    signal = calculate_signal(candles)

    if signal is None:
        print("No valid signal on this 4H candle.")
        save_state(state)
        print("✅ Scan completed.")
        return

    today = datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%d")

    last_signal_day = state.get(
        "last_signal_day"
    )

    # حداکثر یک سیگنال در روز
    if last_signal_day == today:
        print("Daily signal limit already reached.")
        save_state(state)
        print("✅ Scan completed.")
        return

    direction = signal["direction"]
    score = signal["score"]
    entry = signal["price"]

    if direction == "LONG":

        tp = entry * (
            1 + TP_PERCENT / 100
        )

        sl = entry * (
            1 - SL_PERCENT / 100
        )

        message = (
            "🚨 SCORE HUNTER 4H 🚨\n\n"
            "💰 ETHUSDT\n"
            "📊 🟢 LONG\n"
            f"⭐ Score: {score}/7\n"
            f"💵 Entry: {entry:.2f}\n"
            f"🎯 TP: {tp:.2f} (+1%)\n"
            f"🛑 SL: {sl:.2f} (-0.5%)\n\n"
            "⏱ Timeframe: 4H\n"
            "⚠️ Manage risk."
        )

    else:

        tp = entry * (
            1 - TP_PERCENT / 100
        )

        sl = entry * (
            1 + SL_PERCENT / 100
        )

        message = (
            "🚨 SCORE HUNTER 4H 🚨\n\n"
            "💰 ETHUSDT\n"
            "📊 🔴 SHORT\n"
            f"⭐ Score: {score}/7\n"
            f"💵 Entry: {entry:.2f}\n"
            f"🎯 TP: {tp:.2f} (-1%)\n"
            f"🛑 SL: {sl:.2f} (+0.5%)\n\n"
            "⏱ Timeframe: 4H\n"
            "⚠️ Manage risk."
        )

    send_telegram(message)

    state["last_signal_day"] = today
    state["last_signal"] = signal
    state["signal_time"] = int(
        datetime.now(timezone.utc).timestamp()
    )

    save_state(state)

    print("🚨 SIGNAL SENT")
    print("✅ Scan completed.")


if __name__ == "__main__":
    main()
