import os
import json
import requests
from datetime import datetime, timezone

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

KRAKEN_URL = "https://api.kraken.com/0/public/OHLC"
STATE_FILE = "state.json"

INTERVAL_1H = 60
INTERVAL_4H = 240

# SIGNAL SETTINGS
REQUIRED_SCORE = 6
MIN_STRUCTURE_SPACE_ATR = 0.25
MIN_CANDLE_BODY_RATIO = 0.50
MAX_OPPOSITE_WICK_RATIO = 0.30

# ATR-BASED RISK MANAGEMENT
SL_ATR_MULTIPLIER = 1.0
TP_ATR_MULTIPLIER = 2.0

# Fixed-TP market-space filter
# Uses the actual ATR-based TP distance.
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


def get_ohlc_data(symbol, interval, label):
    print(f"\nGetting {symbol} {label} candles...")

    response = requests.get(
        KRAKEN_URL,
        params={"pair": COINS[symbol], "interval": interval},
        timeout=20,
    )

    print(f"{symbol} Kraken {label}:", response.status_code)
    response.raise_for_status()

    payload = response.json()

    if payload.get("error"):
        raise RuntimeError(
            f"{symbol} Kraken API error: {payload['error']}"
        )

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

    # Ignore the currently forming candle.
    if len(candles) > 1:
        candles = candles[:-1]

    if len(candles) < 210:
        raise RuntimeError(
            f"{symbol}: only {len(candles)} candles available"
        )

    print(f"{symbol}: {len(candles)} closed {label} candles")
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
        avg_gain = (
            (avg_gain * (period - 1) + gains[i]) / period
        )
        avg_loss = (
            (avg_loss * (period - 1) + losses[i]) / period
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
        true_ranges.append(
            max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        )

    return sum(true_ranges[-period:]) / period


def get_risk_levels(direction, entry, current_atr):
    """
    1 ATR stop / 2 ATR target.
    Risk-reward remains 1:2.
    """

    sl_distance = current_atr * SL_ATR_MULTIPLIER
    tp_distance = current_atr * TP_ATR_MULTIPLIER

    if direction == "LONG":
        tp = entry + tp_distance
        sl = entry - sl_distance
    else:
        tp = entry - tp_distance
        sl = entry + sl_distance

    return tp, sl, tp_distance, sl_distance


def has_structure_space(candles, direction, entry, current_atr, lookback=20):
    """Require opposing structure to be at least 0.25 ATR away and leave TP room."""
    if len(candles) < lookback + 2:
        return False
    lookback_candles = candles[-(lookback + 1):-1]
    tp, _, tp_distance, _ = get_risk_levels(direction, entry, current_atr)
    min_space = current_atr * MIN_STRUCTURE_SPACE_ATR
    buffer = TP_SPACE_BUFFER_PERCENT / 100.0

    if direction == "LONG":
        resistance = max(c["high"] for c in lookback_candles)
        return (resistance - entry >= min_space) and (resistance >= tp + tp_distance * buffer)

    support = min(c["low"] for c in lookback_candles)
    return (entry - support >= min_space) and (support <= tp - tp_distance * buffer)


def get_4h_trend(candles, symbol):
    if len(candles) < 210:
        return None
    closes = [c["close"] for c in candles]
    ema50_now = ema(closes, 50)
    ema50_prev = ema(closes[:-1], 50)
    ema200_now = ema(closes, 200)
    close = closes[-1]

    if any(x is None for x in [ema50_now, ema50_prev, ema200_now]):
        return None

    long_ok = close > ema200_now and ema50_now > ema200_now and ema50_now > ema50_prev
    short_ok = close < ema200_now and ema50_now < ema200_now and ema50_now < ema50_prev

    print(f"4H trend: LONG={long_ok} | SHORT={short_ok} | Close={close:.8f} | EMA50={ema50_now:.8f} | EMA200={ema200_now:.8f}")
    if long_ok and not short_ok:
        return "LONG"
    if short_ok and not long_ok:
        return "SHORT"
    return "NEUTRAL"


def calculate_signal(candles_1h, candles_4h, symbol):
    if len(candles_1h) < 210 or len(candles_4h) < 210:
        return None

    trend_4h = get_4h_trend(candles_4h, symbol)
    if trend_4h == "NEUTRAL" or trend_4h is None:
        print(f"{symbol}: 4H trend is neutral - no entry scan")
        return None

    opens = [c["open"] for c in candles_1h]
    highs = [c["high"] for c in candles_1h]
    lows = [c["low"] for c in candles_1h]
    closes = [c["close"] for c in candles_1h]
    volumes = [c["volume"] for c in candles_1h]

    # candles_1h[-1] is the latest CLOSED 1H candle.
    open_price, high, low, close, volume = opens[-1], highs[-1], lows[-1], closes[-1], volumes[-1]

    ema20 = ema(closes, 20)
    ema50 = ema(closes, 50)
    ema200 = ema(closes, 200)
    ema50_prev = ema(closes[:-1], 50)
    current_rsi = rsi(closes, 14)
    previous_rsi = rsi(closes[:-1], 14)
    volume_ma = sma(volumes, 20)
    current_atr = atr(highs, lows, closes, 14)

    if any(x is None for x in [ema20, ema50, ema200, ema50_prev, current_rsi, previous_rsi, volume_ma, current_atr]):
        print(f"{symbol}: indicator calculation failed")
        return None

    # 1H entry score (7 components). 4H supplies the directional filter.
    long_trend = close > ema200 and ema50 > ema200 and ema50 > ema50_prev
    short_trend = close < ema200 and ema50 < ema200 and ema50 < ema50_prev

    long_rsi = current_rsi > 50 and current_rsi < 72 and current_rsi > previous_rsi
    short_rsi = current_rsi < 50 and current_rsi > 28 and current_rsi < previous_rsi

    volume_ok = volume >= volume_ma
    recent_high = max(highs[-7:-1])
    recent_low = min(lows[-7:-1])
    bull_break = close > recent_high
    bear_break = close < recent_low

    long_pullback = low <= ema20 and close > ema20
    short_pullback = high >= ema20 and close < ema20

    candle_range = high - low
    bull_body_ratio = (close - open_price) / candle_range if candle_range > 0 else 0.0
    bear_body_ratio = (open_price - close) / candle_range if candle_range > 0 else 0.0
    bull_candle = (close > open_price and candle_range > 0 and bull_body_ratio >= MIN_CANDLE_BODY_RATIO and (high - close) / candle_range <= MAX_OPPOSITE_WICK_RATIO)
    bear_candle = (close < open_price and candle_range > 0 and bear_body_ratio >= MIN_CANDLE_BODY_RATIO and (close - low) / candle_range <= MAX_OPPOSITE_WICK_RATIO)

    volatility_ok = (current_atr / close) >= 0.002

    long_score = sum([int(long_trend), int(long_rsi), int(volume_ok), int(bull_break), int(long_pullback), int(bull_candle), int(volatility_ok)])
    short_score = sum([int(short_trend), int(short_rsi), int(volume_ok), int(bear_break), int(short_pullback), int(bear_candle), int(volatility_ok)])

    atr_percent = current_atr / close * 100.0
    long_tp, long_sl, _, _ = get_risk_levels("LONG", close, current_atr)
    short_tp, short_sl, _, _ = get_risk_levels("SHORT", close, current_atr)

    print(f"\n===== {symbol} 4H + 1H =====")
    print(f"4H Direction: {trend_4h}")
    print(f"1H Price: {close:.8f}")
    print(f"1H RSI: {current_rsi:.2f}")
    print(f"1H EMA20: {ema20:.8f}")
    print(f"1H EMA50: {ema50:.8f}")
    print(f"1H EMA200: {ema200:.8f}")
    print(f"1H ATR: {current_atr:.8f} ({atr_percent:.3f}%)")
    print(f"1H Volume OK: {volume_ok}")
    print(f"LONG SCORE: {long_score}/7")
    print(f"SHORT SCORE: {short_score}/7")
    print(f"LONG TP/SL: {long_tp:.8f} / {long_sl:.8f}")
    print(f"SHORT TP/SL: {short_tp:.8f} / {short_sl:.8f}")
    print("====================")

    if trend_4h == "LONG" and long_score >= REQUIRED_SCORE:
        if not has_structure_space(candles_1h, "LONG", close, current_atr):
            print(f"{symbol}: LONG rejected - insufficient 1H structure space")
            return None
        return {"direction":"LONG", "score":long_score, "price":close, "atr":current_atr, "htf":"4H LONG", "entry_tf":"1H"}

    if trend_4h == "SHORT" and short_score >= REQUIRED_SCORE:
        if not has_structure_space(candles_1h, "SHORT", close, current_atr):
            print(f"{symbol}: SHORT rejected - insufficient 1H structure space")
            return None
        return {"direction":"SHORT", "score":short_score, "price":close, "atr":current_atr, "htf":"4H SHORT", "entry_tf":"1H"}

    return None


def create_message(symbol, signal):
    direction = signal["direction"]
    score = signal["score"]
    entry = signal["price"]
    current_atr = signal["atr"]
    tp, sl, tp_distance, sl_distance = get_risk_levels(direction, entry, current_atr)
    tp_percent = tp_distance / entry * 100.0
    sl_percent = sl_distance / entry * 100.0

    if direction == "LONG":
        direction_text = "📊 🟢 LONG"
        tp_text = f"🎯 TP: {tp:.8f} (+{tp_percent:.2f}%)"
        sl_text = f"🛑 SL: {sl:.8f} (-{sl_percent:.2f}%)"
    else:
        direction_text = "📊 🔴 SHORT"
        tp_text = f"🎯 TP: {tp:.8f} (-{tp_percent:.2f}%)"
        sl_text = f"🛑 SL: {sl:.8f} (+{sl_percent:.2f}%)"

    return ("🚨 SCORE HUNTER PRO 🚨\n\n"
            f"💰 {symbol}USDT\n{direction_text}\n"
            f"⭐ Score: {score}/7\n"
            f"💵 Entry: {entry:.8f}\n{tp_text}\n{sl_text}\n\n"
            "📊 4H Trend + 1H Entry\n"
            "🕯 Closed 1H candle confirmation\n"
            f"📐 1H ATR: {current_atr:.8f}\n"
            "🧠 4H Trend Filter: EMA50/EMA200 + slope\n"
            "📏 1H Structure Space: 0.25 ATR\n"
            "⚖️ Risk/Reward: 1:2\n"
            "⚠️ Manage risk.")

def main():
    print("🟢 SCORE HUNTER PRO | 4H TREND + 1H ENTRY")
    print("🕯 CLOSED 4H + CLOSED 1H ONLY - NO MID-CANDLE SIGNAL")
    print("🚫 PRE-SIGNAL: DISABLED")
    print("♾️ DAILY SIGNAL LIMIT: DISABLED")
    print("📊 Coins: " + " / ".join(COINS.keys()))
    print("🧭 4H: Trend direction | 1H: Entry")
    print(f"⭐ Minimum 1H Score: {REQUIRED_SCORE}/7")
    print(f"🎯 TP: {TP_ATR_MULTIPLIER} ATR (1H)")
    print(f"🛑 SL: {SL_ATR_MULTIPLIER} ATR (1H)")
    print("⚖️ Risk/Reward: 1:2")
    print(f"📏 1H Structure Space: {MIN_STRUCTURE_SPACE_ATR:.2f} ATR")
    print(f"🕯 1H Candle Filter: body >= {MIN_CANDLE_BODY_RATIO:.0%}, opposite wick <= {MAX_OPPOSITE_WICK_RATIO:.0%}")
    print("📐 TP SPACE FILTER: ON | lookback=20 | buffer=0.1%")

    state = load_state()
    for symbol in COINS:
        print(f"\n\n========== {symbol} ==========")
        try:
            candles_1h = get_ohlc_data(symbol, INTERVAL_1H, "1H")
            candles_4h = get_ohlc_data(symbol, INTERVAL_4H, "4H")
            latest_candle_time = candles_1h[-1]["time"]
            coin_state = state.get(symbol, {})
            previous_candle_time = coin_state.get("last_checked_1h_candle")
            print(f"{symbol} latest CLOSED 1H candle: {latest_candle_time}")

            if previous_candle_time == latest_candle_time:
                print(f"{symbol}: No new 1H candle.")
                continue

            coin_state["last_checked_1h_candle"] = latest_candle_time
            signal = calculate_signal(candles_1h, candles_4h, symbol)

            if signal is None:
                print(f"{symbol}: No valid signal.")
                state[symbol] = coin_state
                save_state(state)
                continue

            send_telegram(create_message(symbol, signal))
            coin_state["last_signal"] = signal
            coin_state["signal_candle_1h"] = latest_candle_time
            coin_state["signal_time"] = int(datetime.now(timezone.utc).timestamp())
            state[symbol] = coin_state
            save_state(state)
            print(f"🚨 {symbol}: SIGNAL SENT")
        except Exception as e:
            print(f"❌ {symbol} ERROR: {type(e).__name__}: {e}")
            continue

    save_state(state)
    print("\n✅ ALL COINS SCANNED")


if __name__ == "__main__":
    main()
