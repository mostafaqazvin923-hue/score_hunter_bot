import os
import json
import requests
from datetime import datetime, timezone

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

KRAKEN_URL = "https://api.kraken.com/0/public/OHLC"
STATE_FILE = "state.json"

INTERVAL = 240
REQUIRED_SCORE = 5
TP_PERCENT = 1.0
SL_PERCENT = 0.50

# Fixed-TP market-space filter
TP_SPACE_LOOKBACK = 20
TP_SPACE_BUFFER_PERCENT = 0.10

COINS = {
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "BTC": "XBTUSDT",
    "ADA": "ADAUSDT",
    "LINK": "LINKUSDT",
    "DOGE": "DOGEUSDT",
}

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
        data={"chat_id": CHAT_ID, "text": message},
        timeout=20,
    )
    print("Telegram:", response.status_code)
    print(response.text)
    response.raise_for_status()

def get_4h_data(symbol):
    print(f"\nGetting {symbol} 4H candles...")
    response = requests.get(
        KRAKEN_URL,
        params={"pair": COINS[symbol], "interval": INTERVAL},
        timeout=20,
    )
    print(f"{symbol} Kraken:", response.status_code)
    response.raise_for_status()

    payload = response.json()
    if payload.get("error"):
        raise RuntimeError(f"{symbol} Kraken API error: {payload['error']}")

    result = payload.get("result", {})
    pair_key = next((key for key in result if key != "last"), None)
    if pair_key is None:
        raise RuntimeError(f"{symbol}: no candle data returned")

    candles = []
    for row in result[pair_key]:
        candles.append({
            "time": int(row[0]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[6]),
        })

    if len(candles) > 1:
        candles = candles[:-1]

    if len(candles) < 210:
        raise RuntimeError(f"{symbol}: only {len(candles)} candles available")

    print(f"{symbol}: {len(candles)} closed 4H candles")
    return candles

def ema(values, period):
    if len(values) < period:
        return None
    value = sum(values[:period]) / period
    multiplier = 2.0 / (period + 1)
    for price in values[period:]:
        value = (price - value) * multiplier + value
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
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def atr(highs, lows, closes, period=14):
    if len(closes) <= period:
        return None

    true_ranges = []
    for i in range(1, len(closes)):
        true_ranges.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))

    return sum(true_ranges[-period:]) / period

def has_tp_space(candles, direction, entry):
    # Check recent opposing structure before allowing the fixed 1% TP.
    if len(candles) < TP_SPACE_LOOKBACK + 2:
        return False

    # Exclude the current signal candle.
    lookback = candles[-(TP_SPACE_LOOKBACK + 1):-1]
    buffer = TP_SPACE_BUFFER_PERCENT / 100.0

    if direction == "LONG":
        tp = entry * (1 + TP_PERCENT / 100.0)
        resistance = max(c["high"] for c in lookback)
        return resistance >= tp * (1 + buffer)

    tp = entry * (1 - TP_PERCENT / 100.0)
    support = min(c["low"] for c in lookback)
    return support <= tp * (1 - buffer)


def calculate_signal(candles, symbol):
    if len(candles) < 210:
        return None

    opens = [c["open"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    closes = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]

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
    current_atr = atr(highs, lows, closes, 14)

    if any(x is None for x in [
        ema20, ema50, ema200, current_rsi,
        previous_rsi, volume_ma, current_atr
    ]):
        print(f"{symbol}: indicator calculation failed")
        return None

    long_trend = close > ema200 and ema20 > ema50
    short_trend = close < ema200 and ema20 < ema50

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

    long_pullback = low <= ema20 and close > ema20
    short_pullback = high >= ema20 and close < ema20

    candle_range = high - low
    if candle_range > 0:
        bull_body_ratio = (close - open_price) / candle_range
        bear_body_ratio = (open_price - close) / candle_range
    else:
        bull_body_ratio = 0.0
        bear_body_ratio = 0.0

    bull_candle = (
        close > open_price
        and candle_range > 0
        and bull_body_ratio >= 0.40
    )
    bear_candle = (
        close < open_price
        and candle_range > 0
        and bear_body_ratio >= 0.40
    )

    volatility_ok = (current_atr / close) >= 0.002

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

    print(f"\n===== {symbol} 4H =====")
    print(f"Price: {close:.8f}")
    print(f"RSI: {current_rsi:.2f}")
    print(f"EMA20: {ema20:.8f}")
    print(f"EMA50: {ema50:.8f}")
    print(f"EMA200: {ema200:.8f}")
    print(f"ATR: {current_atr:.8f}")
    print(f"Volume OK: {volume_ok}")
    print(f"LONG SCORE: {long_score}/7")
    print(f"SHORT SCORE: {short_score}/7")
    print("====================")

    if long_score >= REQUIRED_SCORE:
        if not has_tp_space(candles, "LONG", close):
            print(
                f"{symbol}: LONG rejected - insufficient space for "
                f"{TP_PERCENT}% TP"
            )
            return None

        return {"direction": "LONG", "score": long_score, "price": close}

    if short_score >= REQUIRED_SCORE:
        if not has_tp_space(candles, "SHORT", close):
            print(
                f"{symbol}: SHORT rejected - insufficient space for "
                f"{TP_PERCENT}% TP"
            )
            return None

        return {"direction": "SHORT", "score": short_score}

    return None

def create_message(symbol, signal):
    direction = signal["direction"]
    score = signal["score"]
    entry = signal["price"]

    if direction == "LONG":
        tp = entry * (1 + TP_PERCENT / 100)
        sl = entry * (1 - SL_PERCENT / 100)
        return (
            "🚨 SCORE HUNTER 4H 🚨\n\n"
            f"💰 {symbol}USDT\n"
            "📊 🟢 LONG\n"
            f"⭐ Score: {score}/7\n"
            f"💵 Entry: {entry:.8f}\n"
            f"🎯 TP: {tp:.8f} (+1%)\n"
            f"🛑 SL: {sl:.8f} (-0.5%)\n\n"
            "⏱ Timeframe: 4H\n"
            "🕯 Closed candle confirmation\n"
            "⚠️ Manage risk."
        )

    tp = entry * (1 - TP_PERCENT / 100)
    sl = entry * (1 + SL_PERCENT / 100)
    return (
        "🚨 SCORE HUNTER 4H 🚨\n\n"
        f"💰 {symbol}USDT\n"
        "📊 🔴 SHORT\n"
        f"⭐ Score: {score}/7\n"
        f"💵 Entry: {entry:.8f}\n"
        f"🎯 TP: {tp:.8f} (-1%)\n"
        f"🛑 SL: {sl:.8f} (+0.5%)\n\n"
        "⏱ Timeframe: 4H\n"
        "🕯 Closed candle confirmation\n"
        "⚠️ Manage risk."
    )

def main():
    print("🟢 SCORE HUNTER 4H MULTI-COIN SCANNING")
    print("🕯 CLOSED CANDLE ONLY - NO MID-CANDLE SIGNAL")
    print("♾️ DAILY SIGNAL LIMIT: DISABLED")
    print("📊 Coins: " + " / ".join(COINS.keys()))
    print("⏱ Timeframe: 4H")
    print(f"⭐ Minimum Score: {REQUIRED_SCORE}/7")
    print(f"🎯 TP: {TP_PERCENT}%")
    print(f"🛑 SL: {SL_PERCENT}%")
    print(f"📐 TP SPACE FILTER: ON | lookback={TP_SPACE_LOOKBACK} | buffer={TP_SPACE_BUFFER_PERCENT}%")

    state = load_state()

    for symbol in COINS:
        print(f"\n\n========== {symbol} ==========")

        try:
            candles = get_4h_data(symbol)
            latest_candle_time = candles[-1]["time"]
            coin_state = state.get(symbol, {})
            previous_candle_time = coin_state.get("last_checked_candle")

            print(
                f"{symbol} latest CLOSED 4H candle: "
                f"{latest_candle_time}"
            )

            if previous_candle_time == latest_candle_time:
                print(f"{symbol}: No new 4H candle.")
                continue

            coin_state["last_checked_candle"] = latest_candle_time

            signal = calculate_signal(candles, symbol)

            if signal is None:
                print(f"{symbol}: No valid signal.")
                state[symbol] = coin_state
                save_state(state)
                continue

            message = create_message(symbol, signal)
            send_telegram(message)

            coin_state["last_signal"] = signal
            coin_state["signal_candle"] = latest_candle_time
            coin_state["signal_time"] = int(
                datetime.now(timezone.utc).timestamp()
            )

            state[symbol] = coin_state
            save_state(state)

            print(f"🚨 {symbol}: SIGNAL SENT")

        except Exception as e:
            print(
                f"❌ {symbol} ERROR: "
                f"{type(e).__name__}: {e}"
            )
            continue

    save_state(state)
    print("\n✅ ALL COINS SCANNED")

if __name__ == "__main__":
    main()
