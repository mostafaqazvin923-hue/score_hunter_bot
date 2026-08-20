import os
import json
import requests
from datetime import datetime, timezone

# ============================================================
# SCORE HUNTER PRO v3
# 4H MAIN TREND + 1H ENTRY
# CLOSED CANDLES ONLY
# ============================================================

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

KRAKEN_URL = "https://api.kraken.com/0/public/OHLC"
STATE_FILE = "state.json"

# ============================================================
# TIMEFRAMES
# ============================================================

INTERVAL_4H = 240
INTERVAL_1H = 60

# ============================================================
# ENTRY RULES
# ============================================================

MINIMUM_SCORE = 4
MIN_IMPORTANT_CONFIRMATIONS = 3

# ============================================================
# ATR RISK MANAGEMENT
# ============================================================

ATR_PERIOD = 14
SL_ATR_MULTIPLIER = 1.0
TP_ATR_MULTIPLIER = 2.0

# ============================================================
# TP SPACE FILTER
# ============================================================

TP_SPACE_LOOKBACK = 20
TP_SPACE_BUFFER_PERCENT = 0.10

# ============================================================
# COINS
# ============================================================

COINS = {
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "BTC": "XBTUSDT",
    "ADA": "ADAUSDT",
    "LINK": "LINKUSDT",
    "DOGE": "DOGEUSDT",
}


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
            "text": message,
        },
        timeout=20,
    )

    print("Telegram:", response.status_code)
    print(response.text)

    response.raise_for_status()


# ============================================================
# MARKET DATA
# ============================================================

def get_ohlc(symbol, interval):

    response = requests.get(
        KRAKEN_URL,
        params={
            "pair": COINS[symbol],
            "interval": interval,
        },
        timeout=20,
    )

    response.raise_for_status()

    payload = response.json()

    if payload.get("error"):
        raise RuntimeError(
            f"{symbol} Kraken API error: {payload['error']}"
        )

    result = payload.get("result", {})

    pair_key = next(
        (key for key in result if key != "last"),
        None
    )

    if pair_key is None:
        raise RuntimeError(
            f"{symbol}: no candle data returned"
        )

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

    # ========================================================
    # IMPORTANT:
    # Remove currently forming candle.
    # Only CLOSED candles are used.
    # ========================================================

    if len(candles) > 1:
        candles = candles[:-1]

    return candles


def get_4h_data(symbol):

    candles = get_ohlc(
        symbol,
        INTERVAL_4H
    )

    if len(candles) < 210:

        raise RuntimeError(
            f"{symbol}: only "
            f"{len(candles)} closed 4H candles available"
        )

    print(
        f"{symbol}: "
        f"{len(candles)} closed 4H candles"
    )

    return candles


def get_1h_data(symbol):

    candles = get_ohlc(
        symbol,
        INTERVAL_1H
    )

    if len(candles) < 210:

        raise RuntimeError(
            f"{symbol}: only "
            f"{len(candles)} closed 1H candles available"
        )

    print(
        f"{symbol}: "
        f"{len(candles)} closed 1H candles"
    )

    return candles


# ============================================================
# EMA
# ============================================================

def ema(values, period):

    if len(values) < period:
        return None

    value = sum(
        values[:period]
    ) / period

    multiplier = 2.0 / (period + 1)

    for price in values[period:]:

        value = (
            (price - value)
            * multiplier
            + value
        )

    return value


# ============================================================
# SMA
# ============================================================

def sma(values, period):

    if len(values) < period:
        return None

    return sum(
        values[-period:]
    ) / period


# ============================================================
# RSI
# ============================================================

def rsi(values, period=14):

    if len(values) <= period:
        return None

    gains = []
    losses = []

    for i in range(1, len(values)):

        change = (
            values[i]
            - values[i - 1]
        )

        gains.append(
            max(change, 0.0)
        )

        losses.append(
            max(-change, 0.0)
        )

    avg_gain = (
        sum(gains[:period])
        / period
    )

    avg_loss = (
        sum(losses[:period])
        / period
    )

    for i in range(
        period,
        len(gains)
    ):

        avg_gain = (
            (
                avg_gain
                * (period - 1)
                + gains[i]
            )
            / period
        )

        avg_loss = (
            (
                avg_loss
                * (period - 1)
                + losses[i]
            )
            / period
        )

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss

    return (
        100.0
        - (
            100.0
            / (1.0 + rs)
        )
    )


# ============================================================
# ATR
# ============================================================

def atr(
    highs,
    lows,
    closes,
    period=14
):

    if len(closes) <= period:
        return None

    true_ranges = []

    for i in range(1, len(closes)):

        true_ranges.append(
            max(
                highs[i] - lows[i],

                abs(
                    highs[i]
                    - closes[i - 1]
                ),

                abs(
                    lows[i]
                    - closes[i - 1]
                ),
            )
        )

    return (
        sum(
            true_ranges[-period:]
        )
        / period
    )


# ============================================================
# 4H MAIN TREND
# ============================================================

def get_4h_direction(candles):

    closes = [
        c["close"]
        for c in candles
    ]

    ema20 = ema(
        closes,
        20
    )

    ema50 = ema(
        closes,
        50
    )

    ema200 = ema(
        closes,
        200
    )

    info = {
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
    }

    if any(
        x is None
        for x in info.values()
    ):
        return None, info

    close = closes[-1]

    # ========================================================
    # 4H BULLISH
    # ========================================================

    if (
        close > ema200
        and ema20 > ema50 > ema200
    ):

        return "LONG", info

    # ========================================================
    # 4H BEARISH
    # ========================================================

    if (
        close < ema200
        and ema20 < ema50 < ema200
    ):

        return "SHORT", info

    # ========================================================
    # NO CLEAR TREND
    # ========================================================

    return None, info


# ============================================================
# ATR TP / SL
# ============================================================

def get_risk_levels(
    direction,
    entry,
    current_atr
):

    sl_distance = (
        current_atr
        * SL_ATR_MULTIPLIER
    )

    tp_distance = (
        current_atr
        * TP_ATR_MULTIPLIER
    )

    if direction == "LONG":

        tp = (
            entry
            + tp_distance
        )

        sl = (
            entry
            - sl_distance
        )

    else:

        tp = (
            entry
            - tp_distance
        )

        sl = (
            entry
            + sl_distance
        )

    return (
        tp,
        sl,
        tp_distance,
        sl_distance
    )


# ============================================================
# TP SPACE FILTER
# ============================================================

def has_tp_space(
    candles,
    direction,
    entry,
    current_atr
):

    if len(candles) < (
        TP_SPACE_LOOKBACK + 2
    ):
        return False, None

    # Exclude current signal candle
    lookback = candles[
        -(TP_SPACE_LOOKBACK + 1):-1
    ]

    tp, _, tp_distance, _ = (
        get_risk_levels(
            direction,
            entry,
            current_atr
        )
    )

    buffer = (
        tp_distance
        * TP_SPACE_BUFFER_PERCENT
        / 100.0
    )

    # ========================================================
    # LONG
    # ========================================================

    if direction == "LONG":

        blockers = [
            c["high"]
            for c in lookback
            if entry < c["high"] < tp
        ]

        nearest = (
            min(blockers)
            if blockers
            else None
        )

        if nearest is None:
            return True, None

        return (
            nearest < tp - buffer,
            nearest
        )

    # ========================================================
    # SHORT
    # ========================================================

    blockers = [
        c["low"]
        for c in lookback
        if tp < c["low"] < entry
    ]

    nearest = (
        max(blockers)
        if blockers
        else None
    )

    if nearest is None:
        return True, None

    return (
        nearest > tp + buffer,
        nearest
    )


# ============================================================
# 1H ENTRY
# ============================================================

def calculate_1h_signal(
    candles,
    direction,
    symbol
):

    opens = [
        c["open"]
        for c in candles
    ]

    highs = [
        c["high"]
        for c in candles
    ]

    lows = [
        c["low"]
        for c in candles
    ]

    closes = [
        c["close"]
        for c in candles
    ]

    volumes = [
        c["volume"]
        for c in candles
    ]

    open_price = opens[-1]
    high = highs[-1]
    low = lows[-1]
    close = closes[-1]

    # ========================================================
    # INDICATORS
    # ========================================================

    ema20 = ema(
        closes,
        20
    )

    ema50 = ema(
        closes,
        50
    )

    ema200 = ema(
        closes,
        200
    )

    previous_ema20 = ema(
        closes[:-1],
        20
    )

    current_rsi = rsi(
        closes,
        14
    )

    previous_rsi = rsi(
        closes[:-1],
        14
    )

    volume_ma = sma(
        volumes,
        20
    )

    current_atr = atr(
        highs,
        lows,
        closes,
        ATR_PERIOD
    )

    if any(
        x is None
        for x in [
            ema20,
            ema50,
            ema200,
            previous_ema20,
            current_rsi,
            previous_rsi,
            volume_ma,
            current_atr
        ]
    ):
        return None

    # ========================================================
    # IMPORTANT CONFIRMATION 1
    # TREND
    # ========================================================

    if direction == "LONG":

        trend_confirmation = (
            close > ema200
            and ema20 > ema50
        )

    else:

        trend_confirmation = (
            close < ema200
            and ema20 < ema50
        )

    # ========================================================
    # IMPORTANT CONFIRMATION 2
    # MOMENTUM
    # ========================================================

    if direction == "LONG":

        momentum_confirmation = (
            close > closes[-2]
            and ema20 >= previous_ema20
            and current_rsi >= previous_rsi
        )

    else:

        momentum_confirmation = (
            close < closes[-2]
            and ema20 <= previous_ema20
            and current_rsi <= previous_rsi
        )

    # ========================================================
    # IMPORTANT CONFIRMATION 3
    # MARKET STRUCTURE
    # ========================================================

    recent_high = max(
        highs[-7:-1]
    )

    recent_low = min(
        lows[-7:-1]
    )

    if direction == "LONG":

        structure_confirmation = (
            close > recent_high

            or (

                close > ema20

                and close > closes[-2]

                and highs[-1] >= highs[-2]

            )
        )

    else:

        structure_confirmation = (
            close < recent_low

            or (

                close < ema20

                and close < closes[-2]

                and lows[-1] <= lows[-2]

            )
        )

    # ========================================================
    # SCORE 4
    # RSI REGIME
    #
    # RSI > 70 DOES NOT REJECT LONG
    # RSI < 30 DOES NOT REJECT SHORT
    # ========================================================

    if direction == "LONG":

        rsi_confirmation = (
            current_rsi >= 50
            and current_rsi > previous_rsi
        )

    else:

        rsi_confirmation = (
            current_rsi <= 50
            and current_rsi < previous_rsi
        )

    # ========================================================
    # SCORE 5
    # CANDLE QUALITY
    # ========================================================

    candle_range = (
        high - low
    )

    if candle_range > 0:

        bull_body_ratio = (
            (close - open_price)
            / candle_range
        )

        bear_body_ratio = (
            (open_price - close)
            / candle_range
        )

    else:

        bull_body_ratio = 0.0
        bear_body_ratio = 0.0

    if direction == "LONG":

        candle_confirmation = (
            close > open_price
            and bull_body_ratio >= 0.35
        )

    else:

        candle_confirmation = (
            close < open_price
            and bear_body_ratio >= 0.35
        )

    # ========================================================
    # SCORE 6
    # EMA ALIGNMENT
    # ========================================================

    if direction == "LONG":

        ema_confirmation = (
            close > ema20
            and ema20 > ema50
        )

    else:

        ema_confirmation = (
            close < ema20
            and ema20 < ema50
        )

    # ========================================================
    # SCORE 7
    # VOLUME = BONUS ONLY
    # ========================================================

    volume_bonus = (
        volumes[-1]
        >= volume_ma
    )

    # ========================================================
    # PULLBACK
    #
    # OPTIONAL
    # NOT A GATE
    # ========================================================

    if direction == "LONG":

        pullback_present = (
            low <= ema20
            and close > ema20
        )

    else:

        pullback_present = (
            high >= ema20
            and close < ema20
        )

    # ========================================================
    # FINAL SCORE
    # ========================================================

    score = (

        int(
            trend_confirmation
        )

        + int(
            momentum_confirmation
        )

        + int(
            structure_confirmation
        )

        + int(
            rsi_confirmation
        )

        + int(
            candle_confirmation
        )

        + int(
            ema_confirmation
        )

        + int(
            volume_bonus
        )

    )

    # ========================================================
    # IMPORTANT CONFIRMATIONS
    # ========================================================

    important_confirmations = (

        int(
            trend_confirmation
        )

        + int(
            momentum_confirmation
        )

        + int(
            structure_confirmation
        )

    )

    # ========================================================
    # ATR
    # ========================================================

    tp, sl, tp_distance, sl_distance = (
        get_risk_levels(
            direction,
            close,
            current_atr
        )
    )

    # ========================================================
    # TP SPACE
    # ========================================================

    tp_ok, obstacle = has_tp_space(
        candles,
        direction,
        close,
        current_atr
    )

    # ========================================================
    # LOG
    # ========================================================

    print(
        f"\n===== {symbol} 1H ENTRY ====="
    )

    print(
        f"Direction from 4H: {direction}"
    )

    print(
        f"Price: {close:.8f}"
    )

    print(
        f"RSI: {current_rsi:.2f}"
    )

    print(
        f"EMA20: {ema20:.8f}"
    )

    print(
        f"EMA50: {ema50:.8f}"
    )

    print(
        f"EMA200: {ema200:.8f}"
    )

    print(
        f"Trend: {trend_confirmation}"
    )

    print(
        f"Momentum: {momentum_confirmation}"
    )

    print(
        f"Structure: {structure_confirmation}"
    )

    print(
        f"RSI regime: {rsi_confirmation}"
    )

    print(
        f"Candle quality: "
        f"{candle_confirmation}"
    )

    print(
        f"EMA alignment: "
        f"{ema_confirmation}"
    )

    print(
        f"Volume bonus: "
        f"{volume_bonus}"
    )

    print(
        f"Pullback present: "
        f"{pullback_present}"
    )

    print(
        f"IMPORTANT CONFIRMATIONS: "
        f"{important_confirmations}/3"
    )

    print(
        f"1H SCORE: {score}/7"
    )

    print(
        f"TP/SL: "
        f"{tp:.8f} / {sl:.8f}"
    )

    print(
        f"TP SPACE: {tp_ok}"
    )

    if obstacle is not None:

        print(
            f"Nearest opposing level: "
            f"{obstacle:.8f}"
        )

    print(
        "=============================="
    )

    # ========================================================
    # ENTRY GATE 1
    # SCORE >= 4
    # ========================================================

    if score < MINIMUM_SCORE:

        print(
            f"{symbol}: rejected - "
            f"score {score}/7 "
            f"< {MINIMUM_SCORE}"
        )

        return None

    # ========================================================
    # ENTRY GATE 2
    # 3 IMPORTANT CONFIRMATIONS
    # ========================================================

    if (
        important_confirmations
        < MIN_IMPORTANT_CONFIRMATIONS
    ):

        print(
            f"{symbol}: rejected - "
            f"only {important_confirmations}/3 "
            f"important confirmations"
        )

        return None

    # ========================================================
    # ENTRY GATE 3
    # TP SPACE
    # ========================================================

    if not tp_ok:

        print(
            f"{symbol}: rejected - "
            f"insufficient space for 2 ATR TP"
        )

        return None

    # ========================================================
    # VALID SIGNAL
    # ========================================================

    return {

        "direction": direction,

        "score": score,

        "important_confirmations":
            important_confirmations,

        "price": close,

        "atr": current_atr,

        "rsi": current_rsi,

        "volume_bonus":
            volume_bonus,

        "pullback_present":
            pullback_present,

        "candle_time":
            candles[-1]["time"]

    }


# ============================================================
# COMPLETE SIGNAL
# ============================================================

def calculate_signal(
    candles_4h,
    candles_1h,
    symbol
):

    direction, trend = (
        get_4h_direction(
            candles_4h
        )
    )

    close_4h = (
        candles_4h[-1]["close"]
    )

    print(
        f"\n===== {symbol} "
        f"4H DIRECTION ====="
    )

    print(
        f"Closed 4H price: "
        f"{close_4h:.8f}"
    )

    print(
        f"EMA20: "
        f"{trend['ema20']}"
    )

    print(
        f"EMA50: "
        f"{trend['ema50']}"
    )

    print(
        f"EMA200: "
        f"{trend['ema200']}"
    )

    print(
        f"4H direction: "
        f"{direction}"
    )

    print(
        "=============================="
    )

    # ========================================================
    # NO CLEAR 4H TREND = NO TRADE
    # ========================================================

    if direction is None:

        print(
            f"{symbol}: "
            f"4H trend unclear - no trade"
        )

        return None

    # ========================================================
    # 1H ENTRY
    # ========================================================

    return calculate_1h_signal(
        candles_1h,
        direction,
        symbol
    )


# ============================================================
# TELEGRAM MESSAGE
# ============================================================

def create_message(
    symbol,
    signal
):

    direction = (
        signal["direction"]
    )

    entry = (
        signal["price"]
    )

    current_atr = (
        signal["atr"]
    )

    tp, sl, tp_distance, sl_distance = (
        get_risk_levels(
            direction,
            entry,
            current_atr
        )
    )

    tp_percent = (
        tp_distance
        / entry
        * 100.0
    )

    sl_percent = (
        sl_distance
        / entry
        * 100.0
    )

    if direction == "LONG":

        side = "🟢 LONG"

        tp_text = (
            f"🎯 TP: "
            f"{tp:.8f} "
            f"(+{tp_percent:.2f}%)"
        )

        sl_text = (
            f"🛑 SL: "
            f"{sl:.8f} "
            f"(-{sl_percent:.2f}%)"
        )

    else:

        side = "🔴 SHORT"

        tp_text = (
            f"🎯 TP: "
            f"{tp:.8f} "
            f"(-{tp_percent:.2f}%)"
        )

        sl_text = (
            f"🛑 SL: "
            f"{sl:.8f} "
            f"(+{sl_percent:.2f}%)"
        )

    volume_text = (
        "✅ Bonus"
        if signal["volume_bonus"]
        else "➖ No bonus"
    )

    pullback_text = (
        "✅ Present"
        if signal["pullback_present"]
        else "➖ Not required"
    )

    return (

        "🚨 SCORE HUNTER PRO v3 🚨\n\n"

        f"💰 {symbol}USDT\n"

        f"📊 {side}\n\n"

        f"⭐ 1H Score: "
        f"{signal['score']}/7\n"

        f"🧠 Important confirmations: "
        f"{signal['important_confirmations']}/3\n\n"

        f"💵 Entry: "
        f"{entry:.8f}\n"

        f"{tp_text}\n"

        f"{sl_text}\n\n"

        "🧭 4H: Main trend direction\n"

        "⏱ 1H: Entry confirmation\n"

        "🕯 Closed 1H candle only\n"

        "🚫 No mid-candle signal\n"

        f"📊 Volume: "
        f"{volume_text}\n"

        f"↩️ Pullback: "
        f"{pullback_text}\n"

        "📐 TP: 2 ATR | SL: 1 ATR\n"

        "⚖️ Risk/Reward: 1:2\n"

        "⚠️ Manage risk."

    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n🟢 SCORE HUNTER PRO v3"
    )

    print(
        "🧭 4H TREND + 1H ENTRY"
    )

    print(
        "🕯 CLOSED 4H + CLOSED 1H ONLY"
    )

    print(
        "🚫 PRE-SIGNAL: DISABLED"
    )

    print(
        "♾️ DAILY SIGNAL LIMIT: DISABLED"
    )

    print(
        "📊 Coins: "
        + " / ".join(
            COINS.keys()
        )
    )

    print(
        "⭐ Minimum 1H Score: 4/7"
    )

    print(
        "🧠 Minimum important "
        "confirmations: 3/3"
    )

    print(
        "↩️ Pullback: OPTIONAL"
    )

    print(
        "📈 RSI >70: "
        "NOT automatic LONG rejection"
    )

    print(
        "📊 Volume: BONUS ONLY"
    )

    print(
        "🎯 TP: 2.0 ATR"
    )

    print(
        "🛑 SL: 1.0 ATR"
    )

    print(
        "⚖️ Risk/Reward: 1:2"
    )

    print(
        f"📐 TP SPACE FILTER: ON | "
        f"lookback={TP_SPACE_LOOKBACK} | "
        f"buffer="
        f"{TP_SPACE_BUFFER_PERCENT}%"
    )

    state = load_state()

    for symbol in COINS:

        print(
            f"\n\n========== "
            f"{symbol} =========="
        )

        try:

            candles_4h = (
                get_4h_data(
                    symbol
                )
            )

            candles_1h = (
                get_1h_data(
                    symbol
                )
            )

            latest_4h_time = (
                candles_4h[-1]["time"]
            )

            latest_1h_time = (
                candles_1h[-1]["time"]
            )

            coin_state = (
                state.get(
                    symbol,
                    {}
                )
            )

            previous_1h_time = (
                coin_state.get(
                    "last_checked_1h_candle"
                )
            )

            print(
                f"{symbol} latest "
                f"CLOSED 4H candle: "
                f"{latest_4h_time}"
            )

            print(
                f"{symbol} latest "
                f"CLOSED 1H candle: "
                f"{latest_1h_time}"
            )

            # =================================================
            # ONLY SCAN A NEW CLOSED 1H CANDLE
            # =================================================

            if (
                previous_1h_time
                == latest_1h_time
            ):

                print(
                    f"{symbol}: "
                    f"No new closed 1H candle."
                )

                continue

            coin_state[
                "last_checked_1h_candle"
            ] = latest_1h_time

            # =================================================
            # CALCULATE SIGNAL
            # =================================================

            signal = calculate_signal(
                candles_4h,
                candles_1h,
                symbol
            )

            if signal is None:

                print(
                    f"{symbol}: "
                    f"No valid 1H entry."
                )

                state[symbol] = (
                    coin_state
                )

                save_state(
                    state
                )

                continue

            # =================================================
            # SEND TELEGRAM
            # =================================================

            message = (
                create_message(
                    symbol,
                    signal
                )
            )

            send_telegram(
                message
            )

            # =================================================
            # SAVE SIGNAL
            # =================================================

            coin_state[
                "last_signal"
            ] = signal

            coin_state[
                "signal_candle_1h"
            ] = latest_1h_time

            coin_state[
                "signal_time"
            ] = int(
                datetime.now(
                    timezone.utc
                ).timestamp()
            )

            state[symbol] = (
                coin_state
            )

            save_state(
                state
            )

            print(
                f"🚨 {symbol}: "
                f"SIGNAL SENT"
            )

        except Exception as e:

            print(
                f"❌ {symbol} ERROR: "
                f"{type(e).__name__}: "
                f"{e}"
            )

            continue

    save_state(
        state
    )

    print(
        "\n✅ ALL COINS SCANNED"
    )


if __name__ == "__main__":
    main()
