import os
import json
import time
import requests


# ============================================================
# SCORE HUNTER PRO v2 - LBANK
#
# MARKET STRUCTURE / PRICE ACTION ENGINE
#
# 4H = MAIN STRUCTURE
# 1H = ENTRY CONFIRMATION
#
# CORE:
# - HH / HL / LH / LL structure
# - BOS
# - CHoCH
# - Pullback to broken structure
# - Candle confirmation
# - EMA200 macro filter
# - RSI as secondary filter
# - Volume as secondary confirmation
# - ATR dynamic SL
# - Fixed TP = 1%
# - TP path validation
# - Minimum R:R
#
# IMPORTANT:
# This is a signal engine, NOT a guaranteed profitable system.
# Backtest and forward-test before live trading.
# ============================================================


# ============================================================
# TELEGRAM
# ============================================================

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

CHAT_ID = os.environ.get(
    "TELEGRAM_CHAT_ID",
    "2090120004"
)


# ============================================================
# LBANK
# ============================================================

LBANK_KLINE_URL = (
    "https://api.lbkex.com/v2/kline.do"
)

LBANK_TICKER_URL = (
    "https://api.lbkex.com/v2/ticker.do"
)


# ============================================================
# COINS
# ============================================================

COINS = {
    "BTC": "btc_usdt",
    "ETH": "eth_usdt",
    "SOL": "sol_usdt",
    "XRP": "xrp_usdt",
}


# ============================================================
# TIMEFRAMES
# ============================================================

MAIN_TIMEFRAME = "hour4"
ENTRY_TIMEFRAME = "hour1"

CANDLE_LIMIT_4H = 250
CANDLE_LIMIT_1H = 250


# ============================================================
# STRUCTURE
# ============================================================

SWING_LOOKBACK = 50

PIVOT_LEFT = 2
PIVOT_RIGHT = 2

BOS_LOOKBACK = 8

PULLBACK_MAX_DISTANCE_ATR = 0.60


# ============================================================
# MACRO FILTER
# ============================================================

EMA_MACRO = 200


# ============================================================
# MOMENTUM FILTERS
# ============================================================

RSI_PERIOD = 14

VOLUME_LOOKBACK = 20


# ============================================================
# SIGNAL SCORE
# ============================================================

# 4H:
#
# Structure
# BOS
# CHoCH / continuation
# Pullback
# Candle
# EMA200
# RSI
# Volume
#
# Maximum = 8

REQUIRED_SCORE_4H = 5


# 1H:
#
# Structure
# BOS / CHoCH
# Candle
# EMA200
# RSI
#
# Maximum = 5

REQUIRED_SCORE_1H = 3


# ============================================================
# ENTRY
# ============================================================

ENTRY_MAX_DISTANCE = 0.003
# 0.30%


SIGNAL_EXPIRATION_HOURS = 8


# ============================================================
# TAKE PROFIT
# ============================================================

TP_PERCENT = 0.01
# 1%


# ============================================================
# STOP LOSS
# ============================================================

ATR_PERIOD = 14

ATR_MULTIPLIER = 1.20

MIN_SL_PERCENT = 0.004
# 0.40%

MAX_SL_PERCENT = 0.025
# 2.50%


# ============================================================
# RISK / REWARD
# ============================================================

MIN_RISK_REWARD = 1.20


# ============================================================
# STRUCTURE BUFFER
# ============================================================

LEVEL_BUFFER_ATR = 0.15


# ============================================================
# STATE
# ============================================================

STATE_FILE = "state.json"


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    url = (
        f"https://api.telegram.org/"
        f"bot{TOKEN}/sendMessage"
    )

    try:

        response = requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": message,
            },
            timeout=20,
        )

        print(
            "Telegram:",
            response.status_code,
        )

        if response.status_code != 200:
            print(response.text)

        return response.ok

    except Exception as e:

        print(
            "Telegram error:",
            e
        )

        return False


# ============================================================
# STATE
# ============================================================

def empty_state():

    return {
        "last_signals": {},
        "active_trades": {},
        "completed_trades": [],
    }


def load_state():

    if not os.path.exists(STATE_FILE):
        return empty_state()

    try:

        with open(
            STATE_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            state = json.load(f)

        state.setdefault(
            "last_signals",
            {}
        )

        state.setdefault(
            "active_trades",
            {}
        )

        state.setdefault(
            "completed_trades",
            []
        )

        return state

    except Exception as e:

        print(
            "State load error:",
            e
        )

        return empty_state()


def save_state(state):

    temp_file = (
        STATE_FILE + ".tmp"
    )

    with open(
        temp_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            state,
            f,
            indent=2,
        )

    os.replace(
        temp_file,
        STATE_FILE,
    )


# ============================================================
# LBANK CANDLES
# ============================================================

def get_candles(
    symbol,
    timeframe,
    limit,
):

    print(
        f"Getting {symbol} "
        f"{timeframe} candles..."
    )

    now = int(time.time())

    hours = 1

    if timeframe == "hour4":
        hours = 4

    start_time = (
        now
        - limit
        * hours
        * 60
        * 60
    )

    response = requests.get(
        LBANK_KLINE_URL,
        params={
            "symbol": symbol,
            "size": limit,
            "type": timeframe,
            "time": start_time,
        },
        timeout=20,
    )

    print(
        "LBank:",
        response.status_code,
    )

    response.raise_for_status()

    result = response.json()

    if str(
        result.get("result")
    ).lower() != "true":

        raise RuntimeError(
            f"LBank API error: {result}"
        )

    raw_data = result.get(
        "data",
        [],
    )

    if not raw_data:

        raise RuntimeError(
            f"No candle data for {symbol}"
        )

    candles = []

    for item in raw_data:

        if len(item) < 6:
            continue

        candles.append(
            {
                "time": int(item[0]),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
            }
        )

    candles.sort(
        key=lambda x: x["time"]
    )

    print(
        f"{symbol}: received "
        f"{len(candles)} candles"
    )

    return candles


# ============================================================
# CURRENT PRICE
# ============================================================

def get_current_price(symbol):

    response = requests.get(
        LBANK_TICKER_URL,
        params={
            "symbol": symbol,
        },
        timeout=20,
    )

    response.raise_for_status()

    result = response.json()

    if str(
        result.get("result")
    ).lower() != "true":

        raise RuntimeError(
            f"LBank ticker error: {result}"
        )

    data = result.get(
        "data",
        [],
    )

    if not data:

        raise RuntimeError(
            f"No ticker data: {result}"
        )

    ticker = data[0]

    ticker_data = ticker.get(
        "ticker",
        ticker,
    )

    price = ticker_data.get(
        "latest"
    )

    if price is None:

        price = ticker_data.get(
            "last"
        )

    if price is None:

        raise RuntimeError(
            f"Price unavailable: {result}"
        )

    return float(price)


# ============================================================
# CLOSED CANDLES
# ============================================================

def get_closed_candles(candles):

    if len(candles) < 3:

        raise RuntimeError(
            "Not enough candles."
        )

    return candles[:-1]


# ============================================================
# EMA
# ============================================================

def ema_series(
    values,
    period,
):

    if len(values) < period:
        return []

    multiplier = (
        2 / (period + 1)
    )

    first = (
        sum(values[:period])
        / period
    )

    result = (
        [None] * (period - 1)
    )

    result.append(first)

    previous = first

    for price in values[period:]:

        current = (
            (
                price - previous
            )
            * multiplier
            + previous
        )

        result.append(
            current
        )

        previous = current

    return result


# ============================================================
# RSI
# ============================================================

def rsi_series(
    closes,
    period=14,
):

    if len(closes) <= period:
        return []

    gains = []
    losses = []

    for i in range(
        1,
        len(closes),
    ):

        change = (
            closes[i]
            - closes[i - 1]
        )

        gains.append(
            max(change, 0)
        )

        losses.append(
            max(-change, 0)
        )

    avg_gain = (
        sum(gains[:period])
        / period
    )

    avg_loss = (
        sum(losses[:period])
        / period
    )

    result = (
        [None] * period
    )

    if avg_loss == 0:

        result.append(100.0)

    else:

        rs = (
            avg_gain
            / avg_loss
        )

        result.append(
            100
            - 100 / (1 + rs)
        )

    for i in range(
        period,
        len(gains),
    ):

        avg_gain = (
            (
                avg_gain
                * (period - 1)
            )
            + gains[i]
        ) / period

        avg_loss = (
            (
                avg_loss
                * (period - 1)
            )
            + losses[i]
        ) / period

        if avg_loss == 0:

            value = 100.0

        else:

            rs = (
                avg_gain
                / avg_loss
            )

            value = (
                100
                - 100 / (1 + rs)
            )

        result.append(value)

    return result


# ============================================================
# ATR
# ============================================================

def atr_series(
    candles,
    period=14,
):

    if len(candles) <= period:
        return []

    true_ranges = []

    for i in range(
        1,
        len(candles),
    ):

        high = candles[i]["high"]
        low = candles[i]["low"]

        previous_close = (
            candles[i - 1]["close"]
        )

        tr = max(
            high - low,
            abs(
                high
                - previous_close
            ),
            abs(
                low
                - previous_close
            ),
        )

        true_ranges.append(tr)

    if len(true_ranges) < period:
        return []

    first_atr = (
        sum(
            true_ranges[:period]
        )
        / period
    )

    result = (
        [None] * period
    )

    result.append(first_atr)

    previous = first_atr

    for tr in true_ranges[period:]:

        current = (
            (
                previous
                * (period - 1)
            )
            + tr
        ) / period

        result.append(current)

        previous = current

    return result


# ============================================================
# PIVOTS
# ============================================================

def find_pivots(candles):

    highs = []
    lows = []

    start = PIVOT_LEFT

    end = (
        len(candles)
        - PIVOT_RIGHT
    )

    for i in range(
        start,
        end,
    ):

        high = candles[i]["high"]
        low = candles[i]["low"]

        left_highs = [
            candles[j]["high"]
            for j in range(
                i - PIVOT_LEFT,
                i,
            )
        ]

        right_highs = [
            candles[j]["high"]
            for j in range(
                i + 1,
                i + PIVOT_RIGHT + 1,
            )
        ]

        left_lows = [
            candles[j]["low"]
            for j in range(
                i - PIVOT_LEFT,
                i,
            )
        ]

        right_lows = [
            candles[j]["low"]
            for j in range(
                i + 1,
                i + PIVOT_RIGHT + 1,
            )
        ]

        if (
            high >= max(left_highs)
            and high >= max(right_highs)
        ):

            highs.append(
                {
                    "index": i,
                    "price": high,
                }
            )

        if (
            low <= min(left_lows)
            and low <= min(right_lows)
        ):

            lows.append(
                {
                    "index": i,
                    "price": low,
                }
            )

    return highs, lows


# ============================================================
# MARKET STRUCTURE
# ============================================================

def analyze_structure(
    candles,
):

    if len(candles) < SWING_LOOKBACK:
        return None

    highs, lows = find_pivots(candles)

    recent_highs = [
        x
        for x in highs
        if x["index"]
        >= len(candles)
        - SWING_LOOKBACK
    ]

    recent_lows = [
        x
        for x in lows
        if x["index"]
        >= len(candles)
        - SWING_LOOKBACK
    ]

    if len(recent_highs) < 2:
        return None

    if len(recent_lows) < 2:
        return None

    h1 = recent_highs[-2]
    h2 = recent_highs[-1]

    l1 = recent_lows[-2]
    l2 = recent_lows[-1]

    higher_high = (
        h2["price"]
        > h1["price"]
    )

    higher_low = (
        l2["price"]
        > l1["price"]
    )

    lower_high = (
        h2["price"]
        < h1["price"]
    )

    lower_low = (
        l2["price"]
        < l1["price"]
    )

    bullish_structure = (
        higher_high
        and higher_low
    )

    bearish_structure = (
        lower_high
        and lower_low
    )

    close = candles[-1]["close"]

    previous_high = h2["price"]
    previous_low = l2["price"]

    bos_long = (
        close
        > previous_high
    )

    bos_short = (
        close
        < previous_low
    )

    # Detect a structural reversal.
    choch_long = (
        bullish_structure
        and bos_long
    )

    choch_short = (
        bearish_structure
        and bos_short
    )

    return {
        "bullish_structure":
            bullish_structure,

        "bearish_structure":
            bearish_structure,

        "bos_long":
            bos_long,

        "bos_short":
            bos_short,

        "choch_long":
            choch_long,

        "choch_short":
            choch_short,

        "last_high":
            h2["price"],

        "last_low":
            l2["price"],

        "previous_high":
            h1["price"],

        "previous_low":
            l1["price"],
    }


# ============================================================
# CANDLE CONFIRMATION
# ============================================================

def candle_confirmation(candle):

    open_price = candle["open"]
    close = candle["close"]
    high = candle["high"]
    low = candle["low"]

    candle_range = (
        high - low
    )

    if candle_range <= 0:

        return {
            "bullish": False,
            "bearish": False,
            "quality": 0,
        }

    body = abs(
        close - open_price
    )

    body_ratio = (
        body / candle_range
    )

    upper_wick = (
        high
        - max(
            open_price,
            close,
        )
    )

    lower_wick = (
        min(
            open_price,
            close,
        )
        - low
    )

    bullish = (
        close > open_price
        and body_ratio >= 0.45
        and close
        >= low
        + candle_range * 0.65
    )

    bearish = (
        close < open_price
        and body_ratio >= 0.45
        and close
        <= low
        + candle_range * 0.35
    )

    # Strong rejection candles.
    bullish_rejection = (
        lower_wick
        >= body * 1.5
        and close > open_price
    )

    bearish_rejection = (
        upper_wick
        >= body * 1.5
        and close < open_price
    )

    bullish_final = (
        bullish
        or bullish_rejection
    )

    bearish_final = (
        bearish
        or bearish_rejection
    )

    quality = 0

    if bullish_final:
        quality = 1

    if bearish_final:
        quality = 1

    if (
        body_ratio >= 0.65
        and (
            bullish
            or bearish
        )
    ):
        quality = 2

    return {
        "bullish":
            bullish_final,

        "bearish":
            bearish_final,

        "quality":
            quality,
    }


# ============================================================
# VOLUME CONFIRMATION
# ============================================================

def volume_confirmation(
    candles,
):

    if len(candles) < VOLUME_LOOKBACK:
        return False

    current = candles[-1]["volume"]

    average = (
        sum(
            c["volume"]
            for c in candles[
                -VOLUME_LOOKBACK:
            ]
        )
        / VOLUME_LOOKBACK
    )

    return current >= average


# ============================================================
# PULLBACK DETECTION
# ============================================================

def pullback_confirmation(
    candles,
    direction,
    structure_level,
    atr,
):

    if atr <= 0:
        return False

    current = candles[-1]

    distance = abs(
        current["close"]
        - structure_level
    )

    near_structure = (
        distance
        <= atr
        * PULLBACK_MAX_DISTANCE_ATR
    )

    if not near_structure:
        return False

    if direction == "LONG":

        return (
            current["low"]
            <= structure_level
            + atr * LEVEL_BUFFER_ATR
            and current["close"]
            >= structure_level
        )

    return (
        current["high"]
        >= structure_level
        - atr * LEVEL_BUFFER_ATR
        and current["close"]
        <= structure_level
    )


# ============================================================
# 4H SIGNAL
# ============================================================

def calculate_4h_signal(
    candles,
):

    if len(candles) < 210:
        return None

    closes = [
        c["close"]
        for c in candles
    ]

    ema200 = ema_series(
        closes,
        EMA_MACRO,
    )

    rsi_values = rsi_series(
        closes,
        RSI_PERIOD,
    )

    atr_values = atr_series(
        candles,
        ATR_PERIOD,
    )

    structure = (
        analyze_structure(
            candles
        )
    )

    if structure is None:
        return None

    i = len(candles) - 1

    current = candles[i]

    close = current["close"]

    ema = ema200[i]

    rsi = rsi_values[-1]

    previous_rsi = (
        rsi_values[-2]
    )

    atr = atr_values[-1]

    if any(
        x is None
        for x in [
            ema,
            rsi,
            previous_rsi,
            atr,
        ]
    ):
        return None

    candle = candle_confirmation(
        current
    )

    volume_ok = (
        volume_confirmation(
            candles
        )
    )

    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    long_score = 0

    if (
        structure[
            "bullish_structure"
        ]
    ):
        long_score += 1

    if structure["bos_long"]:
        long_score += 2

    if structure["choch_long"]:
        long_score += 1

    if close > ema:
        long_score += 1

    if (
        rsi > 48
        and rsi < 72
        and rsi >= previous_rsi
    ):
        long_score += 1

    if volume_ok:
        long_score += 1

    if candle["bullish"]:
        long_score += 1

    long_pullback = (
        pullback_confirmation(
            candles,
            "LONG",
            structure["last_high"],
            atr,
        )
    )

    if long_pullback:
        long_score += 1

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    short_score = 0

    if (
        structure[
            "bearish_structure"
        ]
    ):
        short_score += 1

    if structure["bos_short"]:
        short_score += 2

    if structure["choch_short"]:
        short_score += 1

    if close < ema:
        short_score += 1

    if (
        rsi < 52
        and rsi > 28
        and rsi <= previous_rsi
    ):
        short_score += 1

    if volume_ok:
        short_score += 1

    if candle["bearish"]:
        short_score += 1

    short_pullback = (
        pullback_confirmation(
            candles,
            "SHORT",
            structure["last_low"],
            atr,
        )
    )

    if short_pullback:
        short_score += 1

    # --------------------------------------------------------
    # LONG DECISION
    # --------------------------------------------------------

    if (
        long_score >= REQUIRED_SCORE_4H
        and close > ema
        and structure[
            "bullish_structure"
        ]
        and candle["bullish"]
    ):

        return {
            "direction":
                "LONG",

            "score":
                long_score,

            "entry":
                close,

            "atr":
                atr,

            "structure_low":
                structure["last_low"],

            "structure_high":
                structure["last_high"],

            "bos":
                structure["bos_long"],

            "choch":
                structure["choch_long"],

            "pullback":
                long_pullback,

            "rsi":
                rsi,

            "candle_quality":
                candle["quality"],

            "candle_time":
                current["time"],

            "candles":
                candles,
        }

    # --------------------------------------------------------
    # SHORT DECISION
    # --------------------------------------------------------

    if (
        short_score >= REQUIRED_SCORE_4H
        and close < ema
        and structure[
            "bearish_structure"
        ]
        and candle["bearish"]
    ):

        return {
            "direction":
                "SHORT",

            "score":
                short_score,

            "entry":
                close,

            "atr":
                atr,

            "structure_low":
                structure["last_low"],

            "structure_high":
                structure["last_high"],

            "bos":
                structure["bos_short"],

            "choch":
                structure["choch_short"],

            "pullback":
                short_pullback,

            "rsi":
                rsi,

            "candle_quality":
                candle["quality"],

            "candle_time":
                current["time"],

            "candles":
                candles,
        }

    return None


# ============================================================
# 1H CONFIRMATION
# ============================================================

def calculate_1h_confirmation(
    candles,
    direction,
):

    if len(candles) < 210:

        return {
            "valid": False,
            "score": 0,
        }

    closes = [
        c["close"]
        for c in candles
    ]

    ema200 = ema_series(
        closes,
        EMA_MACRO,
    )

    rsi_values = rsi_series(
        closes,
        RSI_PERIOD,
    )

    structure = (
        analyze_structure(
            candles
        )
    )

    if structure is None:

        return {
            "valid": False,
            "score": 0,
        }

    i = len(candles) - 1

    current = candles[i]

    close = current["close"]

    ema = ema200[i]

    rsi = rsi_values[-1]

    previous_rsi = (
        rsi_values[-2]
    )

    candle = candle_confirmation(
        current
    )

    score = 0

    if direction == "LONG":

        if (
            structure[
                "bullish_structure"
            ]
        ):
            score += 1

        if (
            structure["bos_long"]
            or structure["choch_long"]
        ):
            score += 1

        if close > ema:
            score += 1

        if (
            rsi > 48
            and rsi > previous_rsi
        ):
            score += 1

        if candle["bullish"]:
            score += 1

    else:

        if (
            structure[
                "bearish_structure"
            ]
        ):
            score += 1

        if (
            structure["bos_short"]
            or structure["choch_short"]
        ):
            score += 1

        if close < ema:
            score += 1

        if (
            rsi < 52
            and rsi < previous_rsi
        ):
            score += 1

        if candle["bearish"]:
            score += 1

    return {
        "valid":
            score >= REQUIRED_SCORE_1H,

        "score":
            score,

        "rsi":
            rsi,

        "structure":
            structure,

        "candle_quality":
            candle["quality"],
    }


# ============================================================
# SIGNAL STRENGTH
# ============================================================

def signal_strength(
    score_4h,
    score_1h,
):

    if (
        score_4h >= 8
        and score_1h >= 4
    ):

        return "🔥 VERY STRONG"

    if (
        score_4h >= 7
        and score_1h >= 3
    ):

        return "💪 STRONG"

    return "🟡 VALID"


# ============================================================
# STOP LOSS
# ============================================================

def calculate_stop_loss(
    signal,
):

    direction = signal["direction"]

    entry = signal["entry"]

    atr = signal["atr"]

    structure_low = (
        signal["structure_low"]
    )

    structure_high = (
        signal["structure_high"]
    )

    structure_buffer = (
        atr
        * LEVEL_BUFFER_ATR
    )

    if direction == "LONG":

        atr_sl = (
            entry
            - ATR_MULTIPLIER * atr
        )

        structure_sl = (
            structure_low
            - structure_buffer
        )

        sl = min(
            atr_sl,
            structure_sl,
        )

        risk = (
            entry - sl
        )

    else:

        atr_sl = (
            entry
            + ATR_MULTIPLIER * atr
        )

        structure_sl = (
            structure_high
            + structure_buffer
        )

        sl = max(
            atr_sl,
            structure_sl,
        )

        risk = (
            sl - entry
        )

    min_risk = (
        entry
        * MIN_SL_PERCENT
    )

    max_risk = (
        entry
        * MAX_SL_PERCENT
    )

    risk = max(
        risk,
        min_risk,
    )

    risk = min(
        risk,
        max_risk,
    )

    if direction == "LONG":

        sl = entry - risk

    else:

        sl = entry + risk

    risk_percent = (
        abs(
            entry - sl
        )
        / entry
        * 100
    )

    return {
        "sl":
            sl,

        "risk_percent":
            risk_percent,
    }


# ============================================================
# TAKE PROFIT
# ============================================================

def calculate_tp(
    entry,
    direction,
):

    if direction == "LONG":

        return (
            entry
            * (
                1
                + TP_PERCENT
            )
        )

    return (
        entry
        * (
            1
            - TP_PERCENT
        )
    )


# ============================================================
# RESISTANCE / SUPPORT
# ============================================================

def find_resistance_levels(
    candles,
    entry,
):

    highs, _ = find_pivots(
        candles
    )

    levels = []

    start_index = max(
        0,
        len(candles)
        - STRUCTURE_LOOKBACK,
    )

    for pivot in highs:

        if pivot["index"] < start_index:
            continue

        price = pivot["price"]

        if price > entry:
            levels.append(price)

    return sorted(
        set(levels)
    )


def find_support_levels(
    candles,
    entry,
):

    _, lows = find_pivots(
        candles
    )

    levels = []

    start_index = max(
        0,
        len(candles)
        - STRUCTURE_LOOKBACK,
    )

    for pivot in lows:

        if pivot["index"] < start_index:
            continue

        price = pivot["price"]

        if price < entry:
            levels.append(price)

    return sorted(
        set(levels),
        reverse=True,
    )


# ============================================================
# TP PATH
# ============================================================

def check_tp_path(
    signal,
    tp,
):

    direction = signal["direction"]

    entry = signal["entry"]

    atr = signal["atr"]

    candles = signal["candles"]

    buffer = (
        atr
        * LEVEL_BUFFER_ATR
    )

    if direction == "LONG":

        levels = (
            find_resistance_levels(
                candles,
                entry,
            )
        )

        blocking = [
            level
            for level in levels
            if (
                level > entry
                and level
                <= tp + buffer
            )
        ]

        if blocking:

            nearest = min(
                blocking
            )

            return {
                "valid":
                    False,

                "type":
                    "RESISTANCE",

                "level":
                    nearest,
            }

    else:

        levels = (
            find_support_levels(
                candles,
                entry,
            )
        )

        blocking = [
            level
            for level in levels
            if (
                level < entry
                and level
                >= tp - buffer
            )
        ]

        if blocking:

            nearest = max(
                blocking
            )

            return {
                "valid":
                    False,

                "type":
                    "SUPPORT",

                "level":
                    nearest,
            }

    return {
        "valid": True,
        "type": None,
        "level": None,
    }


# ============================================================
# FINAL RISK VALIDATION
# ============================================================

def calculate_risk_levels(
    signal,
):

    entry = signal["entry"]

    direction = (
        signal["direction"]
    )

    stop_data = (
        calculate_stop_loss(
            signal
        )
    )

    sl = stop_data["sl"]

    risk_percent = (
        stop_data[
            "risk_percent"
        ]
    )

    tp = calculate_tp(
        entry,
        direction,
    )

    risk = abs(
        entry - sl
    )

    reward = abs(
        tp - entry
    )

    if risk <= 0:

        return {
            "valid": False,
            "reason": "Invalid risk",
        }

    rr = (
        reward / risk
    )

    if rr < MIN_RISK_REWARD:

        return {
            "valid": False,
            "reason":
                f"R:R too low 1:{rr:.2f}",
        }

    path = check_tp_path(
        signal,
        tp,
    )

    if not path["valid"]:

        return {
            "valid": False,

            "reason":
                f"{path['type']} "
                f"at "
                f"{path['level']:.8f} "
                f"blocks TP",

            "blocking_level":
                path["level"],

            "blocking_type":
                path["type"],
        }

    return {
        "valid": True,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "risk_percent": risk_percent,
        "rr": rr,
        "blocking_level": None,
        "blocking_type": None,
    }


# ============================================================
# ENTRY WINDOW
# ============================================================

def entry_is_valid(
    entry,
    current_price,
):

    if entry <= 0:
        return False

    distance = (
        abs(
            current_price
            - entry
        )
        / entry
    )

    return (
        distance
        <= ENTRY_MAX_DISTANCE
    )


# ============================================================
# EXPIRATION
# ============================================================

def signal_is_expired(
    candle_time,
):

    age_seconds = (
        int(time.time())
        - candle_time
    )

    return (
        age_seconds
        / 3600
        > SIGNAL_EXPIRATION_HOURS
    )


# ============================================================
# ACTIVE TRADE CHECK
# ============================================================

def check_active_trades(
    state,
):

    active = state.get(
        "active_trades",
        {},
    )

    if not active:
        return

    finished = []

    for trade_id, trade in list(
        active.items()
    ):

        try:

            current_price = (
                get_current_price(
                    trade[
                        "lbank_symbol"
                    ]
                )
            )

        except Exception as e:

            print(
                "Active trade error:",
                e
            )

            continue

        direction = (
            trade["direction"]
        )

        tp = trade["tp"]

        sl = trade["sl"]

        entry = trade["entry"]

        result = None

        if direction == "LONG":

            if current_price >= tp:
                result = "TP"

            elif current_price <= sl:
                result = "SL"

        else:

            if current_price <= tp:
                result = "TP"

            elif current_price >= sl:
                result = "SL"

        if result is None:
            continue

        if result == "TP":

            pnl_percent = (
                abs(
                    tp - entry
                )
                / entry
                * 100
            )

        else:

            pnl_percent = -(
                abs(
                    sl - entry
                )
                / entry
                * 100
            )

        trade["result"] = result

        trade["exit_price"] = (
            current_price
        )

        trade["pnl_percent"] = (
            pnl_percent
        )

        trade["closed_at"] = (
            int(time.time())
        )

        state.setdefault(
            "completed_trades",
            [],
        ).append(trade)

        finished.append(
            trade_id
        )

        emoji = (
            "✅"
            if result == "TP"
            else "❌"
        )

        message = (
            f"{emoji} SCORE HUNTER PRO v2\n\n"
            f"TRADE CLOSED\n\n"
            f"💰 {trade['symbol']}USDT\n"
            f"📊 {direction}\n"
            f"📌 Result: {result}\n\n"
            f"💵 Entry: {entry:.8f}\n"
            f"🏁 Exit: {current_price:.8f}\n"
            f"📈 P/L: {pnl_percent:+.2f}%\n\n"
            f"🎯 TP: 1.00%\n"
            f"🏦 LBank"
        )

        send_telegram(
            message
        )

    for trade_id in finished:

        del active[trade_id]

    state[
        "active_trades"
    ] = active


# ============================================================
# STATISTICS
# ============================================================

def get_statistics(
    state,
):

    trades = state.get(
        "completed_trades",
        [],
    )

    if not trades:
        return None

    total = len(trades)

    wins = sum(
        1
        for t in trades
        if t.get("result")
        == "TP"
    )

    losses = sum(
        1
        for t in trades
        if t.get("result")
        == "SL"
    )

    decided = (
        wins + losses
    )

    win_rate = (
        wins
        / decided
        * 100
        if decided
        else 0
    )

    total_pnl = sum(
        t.get(
            "pnl_percent",
            0,
        )
        for t in trades
    )

    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "pnl": total_pnl,
    }


# ============================================================
# TELEGRAM STATISTICS
# ============================================================

def send_statistics_if_needed(
    state,
):

    stats = get_statistics(
        state
    )

    if not stats:
        return

    if (
        stats["total"] % 10
        != 0
    ):
        return

    message = (
        "📊 SCORE HUNTER PRO v2\n\n"
        "STATISTICS\n\n"
        f"Closed: {stats['total']}\n"
        f"✅ TP: {stats['wins']}\n"
        f"❌ SL: {stats['losses']}\n"
        f"🎯 Win rate: "
        f"{stats['win_rate']:.1f}%\n"
        f"📈 P/L: "
        f"{stats['pnl']:+.2f}%\n\n"
        f"🎯 TP target: 1.00%\n"
        f"📊 Structure: 4H\n"
        f"📱 Confirmation: 1H\n"
        f"🏦 LBank"
    )

    send_telegram(
        message
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "🟢 SCORE HUNTER PRO v2"
    )

    print(
        "🧠 MARKET STRUCTURE "
        "+ PRICE ACTION"
    )

    print(
        "📊 Main structure: 4H"
    )

    print(
        "📱 Entry confirmation: 1H"
    )

    print(
        "🔎 BOS / CHoCH enabled"
    )

    print(
        "↩️ Pullback confirmation enabled"
    )

    print(
        "🕯 Candle confirmation enabled"
    )

    print(
        "🎯 Fixed TP: 1.00%"
    )

    print(
        f"🔒 Entry window: "
        f"{ENTRY_MAX_DISTANCE * 100:.2f}%"
    )

    print(
        f"🛑 ATR multiplier: "
        f"{ATR_MULTIPLIER}"
    )

    print(
        f"⚖️ Minimum R:R: "
        f"1:{MIN_RISK_REWARD}"
    )

    print(
        f"⏳ Expiration: "
        f"{SIGNAL_EXPIRATION_HOURS}h"
    )

    state = load_state()

    # --------------------------------------------------------
    # ACTIVE TRADES
    # --------------------------------------------------------

    check_active_trades(
        state
    )

    # --------------------------------------------------------
    # SCAN
    # --------------------------------------------------------

    for symbol, lbank_symbol in (
        COINS.items()
    ):

        print(
            f"\n========== "
            f"{symbol} =========="
        )

        try:

            # =================================================
            # 4H
            # =================================================

            candles_4h = (
                get_candles(
                    lbank_symbol,
                    MAIN_TIMEFRAME,
                    CANDLE_LIMIT_4H,
                )
            )

            closed_4h = (
                get_closed_candles(
                    candles_4h
                )
            )

            if len(
                closed_4h
            ) < 210:

                print(
                    f"{symbol}: "
                    "not enough 4H data"
                )

                continue

            latest_4h = (
                closed_4h[-1]
            )

            print(
                f"{symbol}: "
                f"latest closed 4H: "
                f"{latest_4h['time']}"
            )

            # =================================================
            # 4H STRUCTURE SIGNAL
            # =================================================

            signal = (
                calculate_4h_signal(
                    closed_4h
                )
            )

            if signal is None:

                print(
                    f"{symbol}: "
                    "no valid 4H "
                    "structure setup"
                )

                continue

            direction = (
                signal["direction"]
            )

            score_4h = (
                signal["score"]
            )

            entry = (
                signal["entry"]
            )

            candle_time = (
                signal["candle_time"]
            )

            print(
                f"{symbol}: "
                f"4H {direction} "
                f"score={score_4h}"
            )

            # =================================================
            # EXPIRATION
            # =================================================

            if signal_is_expired(
                candle_time
            ):

                print(
                    f"{symbol}: "
                    "signal expired"
                )

                continue

            # =================================================
            # 1H
            # =================================================

            candles_1h = (
                get_candles(
                    lbank_symbol,
                    ENTRY_TIMEFRAME,
                    CANDLE_LIMIT_1H,
                )
            )

            closed_1h = (
                get_closed_candles(
                    candles_1h
                )
            )

            confirmation = (
                calculate_1h_confirmation(
                    closed_1h,
                    direction,
                )
            )

            score_1h = (
                confirmation["score"]
            )

            if not confirmation[
                "valid"
            ]:

                print(
                    f"{symbol}: "
                    f"1H confirmation "
                    f"failed "
                    f"({score_1h}/5)"
                )

                continue

            # =================================================
            # CURRENT PRICE
            # =================================================

            current_price = (
                get_current_price(
                    lbank_symbol
                )
            )

            distance = (
                abs(
                    current_price
                    - entry
                )
                / entry
                * 100
            )

            print(
                f"{symbol}: "
                f"{direction} "
                f"4H={score_4h} "
                f"1H={score_1h} "
                f"entry={entry:.8f} "
                f"current={current_price:.8f} "
                f"distance={distance:.3f}%"
            )

            # =================================================
            # ENTRY WINDOW
            # =================================================

            if not entry_is_valid(
                entry,
                current_price,
            ):

                print(
                    f"{symbol}: "
                    "entry too far"
                )

                continue

            # =================================================
            # DUPLICATE
            # =================================================

            signal_id = (
                f"{symbol}_"
                f"{candle_time}_"
                f"{direction}"
            )

            previous_id = (
                state
                .get(
                    "last_signals",
                    {},
                )
                .get(symbol)
            )

            if (
                previous_id
                == signal_id
            ):

                print(
                    f"{symbol}: "
                    "signal already sent"
                )

                continue

            # =================================================
            # ACTIVE TRADE
            # =================================================

            existing_trade = None

            for trade in (
                state
                .get(
                    "active_trades",
                    {},
                )
                .values()
            ):

                if (
                    trade["symbol"]
                    == symbol
                ):

                    existing_trade = trade

                    break

            if existing_trade:

                print(
                    f"{symbol}: "
                    "active trade exists"
                )

                continue

            # =================================================
            # RISK
            # =================================================

            levels = (
                calculate_risk_levels(
                    signal
                )
            )

            if not levels[
                "valid"
            ]:

                print(
                    f"{symbol}: "
                    f"risk rejected: "
                    f"{levels['reason']}"
                )

                continue

            sl = levels["sl"]

            tp = levels["tp"]

            risk_percent = (
                levels["risk_percent"]
            )

            rr = levels["rr"]

            # =================================================
            # STRENGTH
            # =================================================

            strength = (
                signal_strength(
                    score_4h,
                    score_1h,
                )
            )

            # =================================================
            # FINAL SIGNAL
            # =================================================

            print(
                f"{symbol}: "
                "🔥 FINAL VALID SIGNAL"
            )

            print(
                f"Direction: {direction}"
            )

            print(
                f"4H Score: "
                f"{score_4h}"
            )

            print(
                f"1H Score: "
                f"{score_1h}"
            )

            print(
                f"BOS: "
                f"{signal['bos']}"
            )

            print(
                f"CHoCH: "
                f"{signal['choch']}"
            )

            print(
                f"Pullback: "
                f"{signal['pullback']}"
            )

            print(
                f"TP: "
                f"{tp:.8f}"
            )

            print(
                f"SL: "
                f"{sl:.8f}"
            )

            print(
                f"R:R: "
                f"1:{rr:.2f}"
            )

            # =================================================
            # SAVE TRADE
            # =================================================

            trade = {

                "trade_id":
                    signal_id,

                "symbol":
                    symbol,

                "lbank_symbol":
                    lbank_symbol,

                "direction":
                    direction,

                "score_4h":
                    score_4h,

                "score_1h":
                    score_1h,

                "strength":
                    strength,

                "entry":
                    entry,

                "current_price_at_signal":
                    current_price,

                "tp":
                    tp,

                "sl":
                    sl,

                "risk_percent":
                    risk_percent,

                "rr":
                    rr,

                "bos":
                    signal["bos"],

                "choch":
                    signal["choch"],

                "pullback":
                    signal["pullback"],

                "candle_quality":
                    signal[
                        "candle_quality"
                    ],

                "candle_time":
                    candle_time,

                "created_at":
                    int(time.time()),

                "status":
                    "ACTIVE",
            }

            state.setdefault(
                "active_trades",
                {},
            )[signal_id] = trade

            state.setdefault(
                "last_signals",
                {},
            )[symbol] = signal_id

            save_state(
                state
            )

            # =================================================
            # TELEGRAM
            # =================================================

            if direction == "LONG":

                direction_text = (
                    "🟢 LONG"
                )

                tp_percent = (
                    (
                        tp - entry
                    )
                    / entry
                    * 100
                )

                sl_percent = (
                    (
                        sl - entry
                    )
                    / entry
                    * 100
                )

            else:

                direction_text = (
                    "🔴 SHORT"
                )

                tp_percent = (
                    (
                        entry - tp
                    )
                    / entry
                    * 100
                )

                sl_percent = (
                    (
                        entry - sl
                    )
                    / entry
                    * 100
                )

            message = (

                "🚨 SCORE HUNTER PRO v2 🚨\n\n"

                f"💰 {symbol}USDT\n"

                f"📊 {direction_text}\n"

                f"{strength}\n\n"

                f"⭐ 4H Score: "
                f"{score_4h}\n"

                f"📱 1H Score: "
                f"{score_1h}\n\n"

                f"🧱 BOS: "
                f"{'YES' if signal['bos'] else 'NO'}\n"

                f"🔄 CHoCH: "
                f"{'YES' if signal['choch'] else 'NO'}\n"

                f"↩️ Pullback: "
                f"{'YES' if signal['pullback'] else 'NO'}\n"

                f"🕯 Candle Quality: "
                f"{signal['candle_quality']}\n\n"

                f"💵 Entry: "
                f"{entry:.8f}\n"

                f"📍 Current: "
                f"{current_price:.8f}\n"

                f"📏 Distance: "
                f"{distance:.2f}%\n\n"

                f"🎯 TP: "
                f"{tp:.8f} "
                f"(+{tp_percent:.2f}%)\n"

                f"🛑 SL: "
                f"{sl:.8f} "
                f"({sl_percent:+.2f}%)\n\n"

                f"⚖️ R:R: "
                f"1:{rr:.2f}\n"

                f"📐 Risk: "
                f"{risk_percent:.2f}%\n\n"

                "🧱 TP Path: CLEAR\n"

                "📊 Main Structure: 4H\n"

                "📱 Entry Confirmation: 1H\n"

                "🎯 TP Target: 1.00%\n"

                "🔒 Entry Window: ±0.30%\n\n"

                "⚠️ Signal engine only. "
                "Manage risk."
            )

            sent = send_telegram(
                message
            )

            if sent:

                print(
                    f"{symbol}: "
                    f"{direction} SENT"
                )

            else:

                print(
                    f"{symbol}: "
                    "Telegram failed"
                )

        except Exception as e:

            print(
                f"{symbol}: ERROR: {e}"
            )

    # ========================================================
    # SAVE
    # ========================================================

    save_state(
        state
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    send_statistics_if_needed(
        state
    )

    print(
        "\n✅ SCORE HUNTER PRO v2 "
        "scan completed."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
