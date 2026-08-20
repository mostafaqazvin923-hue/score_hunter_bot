import os
import json
import requests
from datetime import datetime, timezone

# ============================================================
# SCORE HUNTER PRO v4
# 4H MAIN TREND + 1H ENTRY
# CLOSED CANDLES ONLY
#
# v4:
# - Minimum Score = 4/7
# - Important Confirmations = 3/3
# - Entry requires PULLBACK or BREAKOUT
# - RSI >70 does NOT reject LONG
# - Volume = BONUS ONLY
# - TP = 2 ATR
# - SL = 1 ATR
# ============================================================

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

KRAKEN_URL = "https://api.kraken.com/0/public/OHLC"
STATE_FILE = "state.json"

INTERVAL_4H = 240
INTERVAL_1H = 60

MINIMUM_SCORE = 4
MIN_IMPORTANT_CONFIRMATIONS = 3

ATR_PERIOD = 14

SL_ATR_MULTIPLIER = 1.0
TP_ATR_MULTIPLIER = 2.0

TP_SPACE_LOOKBACK = 20
TP_SPACE_BUFFER_PERCENT = 0.10

# Pullback / Breakout settings
PULLBACK_EMA_DISTANCE_ATR = 0.35
BREAKOUT_LOOKBACK = 6

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
            "text": message
        },
        timeout=20
    )

    print("Telegram:", response.status_code)

    response.raise_for_status()


# ============================================================
# KRAKEN DATA
# ============================================================

def get_ohlc(symbol, interval):

    response = requests.get(
        KRAKEN_URL,
        params={
            "pair": COINS[symbol],
            "interval": interval
        },
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    if data.get("error"):
        raise RuntimeError(
            f"{symbol} Kraken API error: {data['error']}"
        )

    result = data.get("result", {})

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
            "volume": float(row[6])
        })

    # IMPORTANT:
    # Kraken returns the currently forming candle.
    # Remove it so ONLY CLOSED candles are used.
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
            f"{symbol}: only {len(candles)} "
            f"closed 4H candles available"
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
            f"{symbol}: only {len(candles)} "
            f"closed 1H candles available"
        )

    print(
        f"{symbol}: "
        f"{len(candles)} closed 1H candles"
    )

    return candles


# ============================================================
# INDICATORS
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
            (price - value) * multiplier
            + value
        )

    return value


def sma(values, period):

    if len(values) < period:
        return None

    return sum(
        values[-period:]
    ) / period


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
                avg_gain * (period - 1)
                + gains[i]
            )
            / period
        )

        avg_loss = (
            (
                avg_loss * (period - 1)
                + losses[i]
            )
            / period
        )

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss

    return 100.0 - (
        100.0 / (1.0 + rs)
    )


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

        tr = max(
            highs[i] - lows[i],
            abs(
                highs[i]
                - closes[i - 1]
            ),
            abs(
                lows[i]
                - closes[i - 1]
            )
        )

        true_ranges.append(tr)

    return (
        sum(true_ranges[-period:])
        / period
    )


# ============================================================
# 4H TREND
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
        "ema200": ema200
    }

    if any(
        x is None
        for x in info.values()
    ):
        return None, info

    close = closes[-1]

    # MAIN LONG TREND
    if (
        close > ema200
        and ema20 > ema50
        and ema50 > ema200
    ):
        return "LONG", info

    # MAIN SHORT TREND
    if (
        close < ema200
        and ema20 < ema50
        and ema50 < ema200
    ):
        return "SHORT", info

    return None, info


# ============================================================
# TP / SL
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

    if direction == "LONG":

        blockers = [
            c["high"]
            for c in lookback
            if (
                entry
                < c["high"]
                < tp
            )
        ]

        nearest = (
            min(blockers)
            if blockers
            else None
        )

        if nearest is None:
            return True, None

        return (
            nearest
            < tp - buffer
        ), nearest

    else:

        blockers = [
            c["low"]
            for c in lookback
            if (
                tp
                < c["low"]
                < entry
            )
        ]

        nearest = (
            max(blockers)
            if blockers
            else None
        )

        if nearest is None:
            return True, None

        return (
            nearest
            > tp + buffer
        ), nearest


# ============================================================
# v4 ENTRY SETUP
# ============================================================

def get_entry_setup(
    candles,
    direction,
    current_atr,
    ema20
):

    """
    v4 requires ONE of:

    A) Pullback + reaction from EMA20

    OR

    B) Breakout of recent 1H structure

    Volume is NOT mandatory.
    """

    if len(candles) < max(
        BREAKOUT_LOOKBACK + 2,
        25
    ):
        return (
            False,
            False,
            False
        )

    c = candles[-1]
    previous = candles[-2]

    close = c["close"]
    high = c["high"]
    low = c["low"]

    # --------------------------------------------------------
    # PULLBACK
    # --------------------------------------------------------

    distance = (
        current_atr
        * PULLBACK_EMA_DISTANCE_ATR
    )

    if direction == "LONG":

        pullback_test = (
            low
            <= ema20 + distance
            and
            low
            >= ema20 - distance
        )

        bullish_reaction = (
            close > ema20
            and
            close > c["open"]
            and
            close >= previous["close"]
        )

        pullback = (
            pullback_test
            and
            bullish_reaction
        )

    else:

        pullback_test = (
            high
            >= ema20 - distance
            and
            high
            <= ema20 + distance
        )

        bearish_reaction = (
            close < ema20
            and
            close < c["open"]
            and
            close <= previous["close"]
        )

        pullback = (
            pullback_test
            and
            bearish_reaction
        )

    # --------------------------------------------------------
    # BREAKOUT
    # --------------------------------------------------------

    structure = candles[
        -(BREAKOUT_LOOKBACK + 1):-1
    ]

    recent_high = max(
        x["high"]
        for x in structure
    )

    recent_low = min(
        x["low"]
        for x in structure
    )

    if direction == "LONG":

        breakout = (
            close > recent_high
        )

    else:

        breakout = (
            close < recent_low
        )

    entry_setup = (
        pullback
        or
        breakout
    )

    return (
        entry_setup,
        pullback,
        breakout
    )


# ============================================================
# 1H SIGNAL
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

    required = [
        ema20,
        ema50,
        ema200,
        previous_ema20,
        current_rsi,
        previous_rsi,
        volume_ma,
        current_atr
    ]

    if any(
        x is None
        for x in required
    ):
        return None

    # ========================================================
    # IMPORTANT CONFIRMATION 1
    # TREND
    # ========================================================

    if direction == "LONG":

        trend_confirmation = (
            close > ema200
            and
            ema20 > ema50
        )

    else:

        trend_confirmation = (
            close < ema200
            and
            ema20 < ema50
        )

    # ========================================================
    # IMPORTANT CONFIRMATION 2
    # MOMENTUM
    # ========================================================

    if direction == "LONG":

        momentum_confirmation = (
            close > closes[-2]
            and
            ema20 >= previous_ema20
            and
            current_rsi >= previous_rsi
        )

    else:

        momentum_confirmation = (
            close < closes[-2]
            and
            ema20 <= previous_ema20
            and
            current_rsi <= previous_rsi
        )

    # ========================================================
    # IMPORTANT CONFIRMATION 3
    # STRUCTURE
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
            or
            (
                close > ema20
                and
                close > closes[-2]
                and
                highs[-1] >= highs[-2]
            )
        )

    else:

        structure_confirmation = (
            close < recent_low
            or
            (
                close < ema20
                and
                close < closes[-2]
                and
                lows[-1] <= lows[-2]
            )
        )

    # ========================================================
    # SCORE 4
    # RSI REGIME
    #
    # RSI >70 DOES NOT REJECT LONG
    # ========================================================

    if direction == "LONG":

        rsi_confirmation = (
            current_rsi >= 50
            and
            current_rsi > previous_rsi
        )

    else:

        rsi_confirmation = (
            current_rsi <= 50
            and
            current_rsi < previous_rsi
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
            close - open_price
        ) / candle_range

        bear_body_ratio = (
            open_price - close
        ) / candle_range

    else:

        bull_body_ratio = 0
        bear_body_ratio = 0

    if direction == "LONG":

        candle_confirmation = (
            close > open_price
            and
            bull_body_ratio >= 0.35
        )

    else:

        candle_confirmation = (
            close < open_price
            and
            bear_body_ratio >= 0.35
        )

    # ========================================================
    # SCORE 6
    # EMA ALIGNMENT
    # ========================================================

    if direction == "LONG":

        ema_confirmation = (
            close > ema20
            and
            ema20 > ema50
        )

    else:

        ema_confirmation = (
            close < ema20
            and
            ema20 < ema50
        )

    # ========================================================
    # SCORE 7
    # VOLUME BONUS ONLY
    # ========================================================

    volume_bonus = (
        volumes[-1]
        >= volume_ma
    )

    # ========================================================
    # v4 ENTRY GATE
    # ========================================================

    (
        entry_setup,
        pullback,
        breakout
    ) = get_entry_setup(
        candles,
        direction,
        current_atr,
        ema20
    )

    # ========================================================
    # SCORE
    # ========================================================

    score = (
        int(trend_confirmation)
        +
        int(momentum_confirmation)
        +
        int(structure_confirmation)
        +
        int(rsi_confirmation)
        +
        int(candle_confirmation)
        +
        int(ema_confirmation)
        +
        int(volume_bonus)
    )

    important_confirmations = (
        int(trend_confirmation)
        +
        int(momentum_confirmation)
        +
        int(structure_confirmation)
    )

    # ========================================================
    # TP / SL
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
        f"\n===== {symbol} 1H ENTRY v4 ====="
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
        f"ATR: {current_atr:.8f}"
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
        f"Pullback setup: "
        f"{pullback}"
    )

    print(
        f"Breakout setup: "
        f"{breakout}"
    )

    print(
        f"ENTRY SETUP: "
        f"{entry_setup}"
    )

    print(
        f"IMPORTANT CONFIRMATIONS: "
        f"{important_confirmations}/3"
    )

    print(
        f"1H SCORE: {score}/7"
    )

    print(
        f"LONG/SHORT TP/SL: "
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
    # FILTER 1
    # SCORE
    # ========================================================

    if score < MINIMUM_SCORE:

        print(
            f"{symbol}: rejected - "
            f"score {score}/7 "
            f"< {MINIMUM_SCORE}"
        )

        return None

    # ========================================================
    # FILTER 2
    # IMPORTANT CONFIRMATIONS
    # ========================================================

    if (
        important_confirmations
        < MIN_IMPORTANT_CONFIRMATIONS
    ):

        print(
            f"{symbol}: rejected - "
            f"only "
            f"{important_confirmations}/3 "
            f"important confirmations"
        )

        return None

    # ========================================================
    # FILTER 3
    # PULLBACK OR BREAKOUT
    # ========================================================

    if not entry_setup:

        print(
            f"{symbol}: rejected - "
            f"no Pullback/Breakout setup"
        )

        return None

    # ========================================================
    # FILTER 4
    # TP SPACE
    # ========================================================

    if not tp_ok:

        print(
            f"{symbol}: rejected - "
            f"insufficient space "
            f"for 2 ATR TP"
        )

        return None

    return {
        "direction": direction,
        "score": score,
        "important_confirmations":
            important_confirmations,
        "price": close,
        "atr": current_atr,
        "rsi": current_rsi,
        "volume_bonus": volume_bonus,
        "pullback": pullback,
        "breakout": breakout,
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
        f"\n===== {symbol} 4H DIRECTION ====="
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

    if direction is None:

        print(
            f"{symbol}: "
            f"4H trend unclear - no trade"
        )

        return None

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

    (
        tp,
        sl,
        tp_distance,
        sl_distance
    ) = get_risk_levels(
        direction,
        entry,
        current_atr
    )

    tp_percent = (
        tp_distance
        / entry
        * 100
    )

    sl_percent = (
        sl_distance
        / entry
        * 100
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

    if signal["volume_bonus"]:

        volume_text = "✅ Bonus"

    else:

        volume_text = "➖ No bonus"

    if signal["pullback"]:

        setup_text = (
            "↩️ Entry Setup: PULLBACK"
        )

    elif signal["breakout"]:

        setup_text = (
            "🚀 Entry Setup: BREAKOUT"
        )

    else:

        setup_text = (
            "❌ Entry Setup: NONE"
        )

    return (
        "🚨 SCORE HUNTER PRO v4 🚨\n\n"

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

        f"{setup_text}\n"

        f"📊 Volume: "
        f"{volume_text}\n"

        "📈 RSI >70: "
        "NOT automatic rejection\n"

        "📐 TP: 2 ATR | SL: 1 ATR\n"

        "⚖️ Risk/Reward: 1:2\n"

        "⚠️ Manage risk."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n🟢 SCORE HUNTER PRO v4"
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
        "↩️ Entry: "
        "PULLBACK OR BREAKOUT"
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
        f"buffer={TP_SPACE_BUFFER_PERCENT}%"
    )

    state = load_state()

    for symbol in COINS:

        print(
            f"\n\n========== "
            f"{symbol} "
            f"=========="
        )

        try:

            candles_4h = (
                get_4h_data(symbol)
            )

            candles_1h = (
                get_1h_data(symbol)
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
                f"{symbol} latest CLOSED "
                f"4H candle: "
                f"{latest_4h_time}"
            )

            print(
                f"{symbol} latest CLOSED "
                f"1H candle: "
                f"{latest_1h_time}"
            )

            # Prevent repeated analysis
            # of the same closed candle.
            if (
                previous_1h_time
                == latest_1h_time
            ):

                print(
                    f"{symbol}: "
                    f"No new closed "
                    f"1H candle."
                )

                continue

            # Mark candle as checked
            coin_state[
                "last_checked_1h_candle"
            ] = latest_1h_time

            signal = calculate_signal(
                candles_4h,
                candles_1h,
                symbol
            )

            if signal is None:

                print(
                    f"{symbol}: "
                    f"No valid signal."
                )

                state[symbol] = (
                    coin_state
                )

                save_state(state)

                continue

            message = (
                create_message(
                    symbol,
                    signal
                )
            )

            send_telegram(
                message
            )

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

            save_state(state)

            print(
                f"🚨 {symbol}: "
                f"SIGNAL SENT"
            )

        except Exception as e:

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
