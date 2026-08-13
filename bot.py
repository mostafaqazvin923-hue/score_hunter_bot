import os
import json
import requests
from datetime import datetime, timezone

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# ============================================================
# LBANK PUBLIC MARKET DATA
# Official V2 public endpoint:
# /v2/kline.do
# ============================================================

LBANK_URL = "https://api.lbkex.com/v2/kline.do"
STATE_FILE = "state.json"

# ============================================================
# SETTINGS
# ============================================================

INTERVAL = "hour4"       # 4H
CANDLE_LIMIT = 500
REQUIRED_SCORE = 5
TP_PERCENT = 1.0
SL_PERCENT = 0.50

# Original four + BTC
COINS = {
    "BTC": "btc_usdt",
    "ETH": "eth_usdt",
    "SOL": "sol_usdt",
    "XRP": "xrp_usdt",
    "APT": "apt_usdt",
}

CANDLE_SECONDS = 4 * 60 * 60


# ============================================================
# STATE
# ============================================================

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


# ============================================================
# TELEGRAM
# ============================================================

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


# ============================================================
# LBANK 4H KLINES
# ============================================================

def get_4h_data(symbol):
    pair = COINS[symbol]

    print(f"\nGetting {symbol} 4H candles from LBank...")

    now_ts = int(datetime.now(timezone.utc).timestamp())

    response = requests.get(
        LBANK_URL,
        params={
            "symbol": pair,
            "size": CANDLE_LIMIT,
            "type": INTERVAL,
            "time": now_ts
        },
        timeout=20
    )

    print(f"{symbol} LBank:", response.status_code)
    response.raise_for_status()

    payload = response.json()

    if str(payload.get("result")).lower() != "true":
        raise RuntimeError(
            f"{symbol} LBank API error: {payload}"
        )

    raw_candles = payload.get("data", [])

    if not raw_candles:
        raise RuntimeError(
            f"{symbol}: no candle data returned"
        )

    candles = []

    for row in raw_candles:
        if len(row) < 6:
            continue

        candle_time = int(row[0])

        # CRITICAL:
        # LBank timestamps represent the candle start.
        # A 4H candle is usable only after its full 4 hours have elapsed.
        # Therefore the currently forming candle is NEVER analysed.
        if candle_time + CANDLE_SECONDS > now_ts:
            continue

        candles.append({
            "time": candle_time,
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5])
        })

    candles.sort(key=lambda x: x["time"])

    if len(candles) < 210:
        raise RuntimeError(
            f"{symbol}: only {len(candles)} closed 4H candles available"
        )

    print(
        f"{symbol}: received {len(candles)} CLOSED 4H candles"
    )
    print(
        f"{symbol}: latest closed 4H: "
        f"{candles[-1]['time']}"
    )

    return candles


# ============================================================
# EMA
# ============================================================

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


# ============================================================
# SMA
# ============================================================

def sma(values, period):
    if len(values) < period:
        return None

    return sum(values[-period:]) / period


# ============================================================
# RSI
# ============================================================

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

    return 100.0 - (
        100.0 / (1.0 + rs)
    )


# ============================================================
# ATR
# ============================================================

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


# ============================================================
# SCORE HUNTER
# Same 7-condition scoring logic from the supplied source
# ============================================================

def calculate_signal(candles, symbol):

    if len(candles) < 210:
        return None

    opens = [c["open"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    closes = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]

    # IMPORTANT:
    # candles[-1] is guaranteed to be a FULLY CLOSED 4H candle
    # by get_4h_data(). No live/forming candle is used here.
    open_price = opens[-1]
    high = highs[-1]
    low = lows[-1]
    close = closes[-1]
    volume = volumes[-1]

    # --------------------------------------------------------
    # INDICATORS
    # --------------------------------------------------------

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
        print(
            f"{symbol}: indicator calculation failed"
        )
        return None

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    long_trend = (
        close > ema200
        and ema20 > ema50
    )

    short_trend = (
        close < ema200
        and ema20 < ema50
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    volume_ok = (
        volume >= volume_ma
    )

    # --------------------------------------------------------
    # MARKET STRUCTURE
    # same as:
    # ta.highest(high, 6)[1]
    # ta.lowest(low, 6)[1]
    # --------------------------------------------------------

    recent_high = max(
        highs[-7:-1]
    )

    recent_low = min(
        lows[-7:-1]
    )

    bull_break = (
        close > recent_high
    )

    bear_break = (
        close < recent_low
    )

    # --------------------------------------------------------
    # EMA PULLBACK
    # --------------------------------------------------------

    long_pullback = (
        low <= ema20
        and close > ema20
    )

    short_pullback = (
        high >= ema20
        and close < ema20
    )

    # --------------------------------------------------------
    # CANDLE CONFIRMATION
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    volatility_ok = (
        current_atr / close
    ) >= 0.002

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # FINAL SIGNAL
    # Signal is created ONLY after the candle is fully closed.
    # --------------------------------------------------------

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


# ============================================================
# TELEGRAM MESSAGE
# ============================================================

def create_message(symbol, signal):

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

        return (
            "🚨 SCORE HUNTER 4H 🚨\n\n"
            f"💰 {symbol}USDT\n"
            "📊 🟢 LONG\n"
            f"⭐ Score: {score}/7\n"
            f"💵 Entry: {entry:.8f}\n"
            f"🎯 TP: {tp:.8f} (+1%)\n"
            f"🛑 SL: {sl:.8f} (-0.5%)\n\n"
            "⏱ Timeframe: 4H\n"
            "🔒 Signal generated after CLOSED candle\n"
            "⚠️ Manage risk."
        )

    tp = entry * (
        1 - TP_PERCENT / 100
    )

    sl = entry * (
        1 + SL_PERCENT / 100
    )

    return (
        "🚨 SCORE HUNTER 4H 🚨\n\n"
        f"💰 {symbol}USDT\n"
        "📊 🔴 SHORT\n"
        f"⭐ Score: {score}/7\n"
        f"💵 Entry: {entry:.8f}\n"
        f"🎯 TP: {tp:.8f} (-1%)\n"
        f"🛑 SL: {sl:.8f} (+0.5%)\n\n"
        "⏱ Timeframe: 4H\n"
        "🔒 Signal generated after CLOSED candle\n"
        "⚠️ Manage risk."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "🟢 SCORE HUNTER PRO 4H LBANK"
    )
    print(
        "🔒 CLOSED CANDLE MODE: "
        "forming 4H candle is ignored"
    )
    print(
        "📊 Source: LBank public V2 Kline API"
    )
    print(
        "⏱ Timeframe: 4H"
    )
    print(
        "🪙 Coins: BTC / ETH / SOL / XRP / APT"
    )

    state = load_state()

    for symbol in COINS:

        print(
            f"\n\n========== {symbol} =========="
        )

        try:

            candles = get_4h_data(symbol)

            latest_candle_time = (
                candles[-1]["time"]
            )

            coin_state = state.get(
                symbol,
                {}
            )

            previous_candle_time = (
                coin_state.get(
                    "last_checked_candle"
                )
            )

            print(
                f"{symbol} latest CLOSED "
                f"4H candle: "
                f"{latest_candle_time}"
            )

            # Same closed candle was already checked.
            if (
                previous_candle_time
                == latest_candle_time
            ):

                print(
                    f"{symbol}: "
                    "No new closed 4H candle."
                )

                continue

            # Record the candle that was checked.
            coin_state[
                "last_checked_candle"
            ] = latest_candle_time

            signal = calculate_signal(
                candles,
                symbol
            )

            # No signal on this newly closed candle.
            if signal is None:

                print(
                    f"{symbol}: "
                    "No valid signal."
                )

                state[symbol] = coin_state
                save_state(state)

                continue

            # ------------------------------------------------
            # DAILY LIMIT PER COIN
            # ------------------------------------------------

            today = datetime.now(
                timezone.utc
            ).strftime(
                "%Y-%m-%d"
            )

            last_signal_day = (
                coin_state.get(
                    "last_signal_day"
                )
            )

            if last_signal_day == today:

                print(
                    f"{symbol}: "
                    "Daily signal limit "
                    "already reached."
                )

                state[symbol] = coin_state
                save_state(state)

                continue

            # ------------------------------------------------
            # SEND SIGNAL
            # ------------------------------------------------

            message = create_message(
                symbol,
                signal
            )

            send_telegram(
                message
            )

            coin_state[
                "last_signal_day"
            ] = today

            coin_state[
                "last_signal"
            ] = signal

            coin_state[
                "signal_candle"
            ] = latest_candle_time

            coin_state[
                "signal_time"
            ] = int(
                datetime.now(
                    timezone.utc
                ).timestamp()
            )

            state[symbol] = coin_state
            save_state(state)

            print(
                f"🚨 {symbol}: "
                "SIGNAL SENT"
            )

        except Exception as e:

            # One coin failing must not stop
            # the other coins.
            print(
                f"❌ {symbol} ERROR: "
                f"{type(e).__name__}: {e}"
            )

            continue

    save_state(state)

    print(
        "\n✅ ALL COINS SCANNED"
    )


if __name__ == "__main__":
    main()
