import requests
import time
from datetime import datetime, timezone


# ============================================================
# SCORE HUNTER PRO v8
# HISTORICAL BACKTEST ENGINE
#
# FIXED VERSION
#
# 4H TREND
# +
# 1H ENTRY
# +
# PULLBACK
# +
# BREAKOUT
# +
# REVERSAL / MARKET STRUCTURE SHIFT
# +
# EMA
# +
# ADX
# +
# RSI
# +
# CLOSED CANDLE ONLY
# +
# NO LOOK-AHEAD
# +
# ATR / STRUCTURE SL
# +
# TP = 2R
#
# IMPORTANT
# ------------------------------------------------------------
# Signal is generated from the CLOSED 1H candle.
#
# The entry candle is NEVER used for TP/SL result.
#
# Result checking starts from the NEXT 1H candle.
#
# If TP and SL are both touched in one candle:
# SL FIRST (conservative assumption).
#
# One position per coin at a time.
# No overlapping trades.
# No same-candle re-entry.
# ============================================================


# ============================================================
# KRAKEN
# ============================================================

KRAKEN_URL = "https://api.kraken.com/0/public/OHLC"


# ============================================================
# TIMEFRAMES
# ============================================================

INTERVAL_4H = 240
INTERVAL_1H = 60

SECONDS_1H = 60 * 60
SECONDS_4H = 4 * 60 * 60


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
# INDICATORS
# ============================================================

EMA20 = 20
EMA50 = 50
EMA200 = 200

RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14

ADX_MIN = 20.0

PULLBACK_LOOKBACK = 8
STRUCTURE_LOOKBACK = 5

EMA_PULLBACK_ATR = 0.50

SL_ATR = 1.50
STRUCTURE_BUFFER_ATR = 0.10

TP_R_MULTIPLE = 2.0
MIN_RR = 1.50

# Prevent absurdly wide structure stops.
MAX_SL_ATR = 3.50

# Reversal
REVERSAL_LOOKBACK = 6
REVERSAL_CONFIRM_CANDLES = 2


# ============================================================
# HTTP SESSION
# ============================================================

SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent": "ScoreHunterPro-v8-Backtester/1.0"
})


# ============================================================
# DOWNLOAD OHLC
# ============================================================

def get_ohlc(symbol, interval):

    response = SESSION.get(
        KRAKEN_URL,
        params={
            "pair": COINS[symbol],
            "interval": interval
        },
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    if data.get("error"):

        raise RuntimeError(
            f"{symbol} Kraken error: "
            f"{data['error']}"
        )

    result = data.get("result", {})

    pair_keys = [
        key
        for key in result
        if key != "last"
    ]

    if not pair_keys:

        raise RuntimeError(
            f"{symbol}: Kraken returned no OHLC data."
        )

    pair_key = pair_keys[0]

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

    candles.sort(
        key=lambda x: x["time"]
    )

    # --------------------------------------------------------
    # Remove duplicate timestamps
    # --------------------------------------------------------

    unique = {}

    for candle in candles:

        unique[candle["time"]] = candle

    candles = list(
        unique.values()
    )

    candles.sort(
        key=lambda x: x["time"]
    )

    # --------------------------------------------------------
    # Remove currently forming candle.
    #
    # We use a conservative method:
    # latest candle is discarded.
    # --------------------------------------------------------

    if len(candles) > 1:

        candles = candles[:-1]

    return candles


# ============================================================
# EMA
# ============================================================

def ema(values, period):

    if len(values) < period:

        return None

    value = (
        sum(values[:period])
        / period
    )

    multiplier = (
        2.0
        / (period + 1)
    )

    for price in values[period:]:

        value = (
            (price - value)
            * multiplier
            + value
        )

    return value


# ============================================================
# ATR
# ============================================================

def atr(candles, period=14):

    if len(candles) < period + 1:

        return None

    trs = []

    for i in range(
        1,
        len(candles)
    ):

        current = candles[i]
        previous = candles[i - 1]

        tr = max(

            current["high"]
            - current["low"],

            abs(
                current["high"]
                - previous["close"]
            ),

            abs(
                current["low"]
                - previous["close"]
            )

        )

        trs.append(tr)

    if len(trs) < period:

        return None

    # Wilder-style initial average.
    value = (
        sum(trs[:period])
        / period
    )

    for tr in trs[period:]:

        value = (
            (
                value * (period - 1)
                + tr
            )
            / period
        )

    return value


# ============================================================
# RSI
# ============================================================

def rsi(candles, period=14):

    if len(candles) < period + 1:

        return None

    closes = [
        c["close"]
        for c in candles
    ]

    gains = []
    losses = []

    for i in range(
        1,
        len(closes)
    ):

        change = (
            closes[i]
            - closes[i - 1]
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

    if avg_gain == 0:

        return 0.0

    rs_value = (
        avg_gain
        / avg_loss
    )

    return (
        100.0
        - (
            100.0
            / (1.0 + rs_value)
        )
    )


# ============================================================
# ADX
# ============================================================

def adx(candles, period=14):

    if len(candles) < (
        period * 2 + 5
    ):

        return None

    tr_list = []
    plus_dm_list = []
    minus_dm_list = []

    for i in range(
        1,
        len(candles)
    ):

        current = candles[i]
        previous = candles[i - 1]

        high = current["high"]
        low = current["low"]

        prev_high = previous["high"]
        prev_low = previous["low"]
        prev_close = previous["close"]

        tr = max(

            high - low,

            abs(
                high
                - prev_close
            ),

            abs(
                low
                - prev_close
            )

        )

        up_move = (
            high
            - prev_high
        )

        down_move = (
            prev_low
            - low
        )

        plus_dm = (

            up_move

            if (
                up_move > down_move
                and up_move > 0
            )

            else 0.0

        )

        minus_dm = (

            down_move

            if (
                down_move > up_move
                and down_move > 0
            )

            else 0.0

        )

        tr_list.append(tr)
        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)

    if len(tr_list) < period * 2:

        return None

    atr_value = (
        sum(tr_list[:period])
        / period
    )

    plus_value = (
        sum(plus_dm_list[:period])
        / period
    )

    minus_value = (
        sum(minus_dm_list[:period])
        / period
    )

    dx_values = []

    for i in range(
        period,
        len(tr_list)
    ):

        atr_value = (
            (
                atr_value * (period - 1)
                + tr_list[i]
            )
            / period
        )

        plus_value = (
            (
                plus_value * (period - 1)
                + plus_dm_list[i]
            )
            / period
        )

        minus_value = (
            (
                minus_value * (period - 1)
                + minus_dm_list[i]
            )
            / period
        )

        if atr_value <= 0:

            continue

        plus_di = (
            100.0
            * plus_value
            / atr_value
        )

        minus_di = (
            100.0
            * minus_value
            / atr_value
        )

        denominator = (
            plus_di
            + minus_di
        )

        if denominator <= 0:

            dx = 0.0

        else:

            dx = (
                100.0
                * abs(
                    plus_di
                    - minus_di
                )
                / denominator
            )

        dx_values.append(dx)

    if len(dx_values) < period:

        return None

    adx_value = (
        sum(dx_values[:period])
        / period
    )

    for value in dx_values[period:]:

        adx_value = (
            (
                adx_value * (period - 1)
                + value
            )
            / period
        )

    return adx_value


# ============================================================
# 4H TREND
#
# IMPORTANT:
# This function receives ONLY fully closed 4H candles.
# ============================================================

def get_4h_direction(candles):

    if len(candles) < EMA200:

        return None

    closes = [
        c["close"]
        for c in candles
    ]

    close = closes[-1]

    ema20_value = ema(
        closes,
        EMA20
    )

    ema50_value = ema(
        closes,
        EMA50
    )

    ema200_value = ema(
        closes,
        EMA200
    )

    if (
        ema20_value is None
        or ema50_value is None
        or ema200_value is None
    ):

        return None

    if (
        close > ema200_value
        and ema20_value > ema50_value
        and ema50_value > ema200_value
    ):

        return "LONG"

    if (
        close < ema200_value
        and ema20_value < ema50_value
        and ema50_value < ema200_value
    ):

        return "SHORT"

    return None


# ============================================================
# 4H ALIGNMENT
# ============================================================

def get_4h_state(candles):

    direction = get_4h_direction(
        candles
    )

    if direction is None:

        return {
            "direction": None,
            "ema20": None,
            "ema50": None,
            "ema200": None
        }

    closes = [
        c["close"]
        for c in candles
    ]

    return {

        "direction":
            direction,

        "ema20":
            ema(closes, EMA20),

        "ema50":
            ema(closes, EMA50),

        "ema200":
            ema(closes, EMA200)

    }


# ============================================================
# STRUCTURE LEVELS
# ============================================================

def recent_structure(
    candles,
    lookback=STRUCTURE_LOOKBACK
):

    if len(candles) < lookback + 1:

        return None, None

    previous = candles[
        -lookback - 1:-1
    ]

    swing_low = min(
        c["low"]
        for c in previous
    )

    swing_high = max(
        c["high"]
        for c in previous
    )

    return swing_low, swing_high


# ============================================================
# MARKET STRUCTURE SHIFT
#
# LONG:
# close breaks previous resistance.
#
# SHORT:
# close breaks previous support.
# ============================================================

def detect_structure_shift(
    candles,
    direction,
    lookback=REVERSAL_LOOKBACK
):

    if len(candles) < lookback + 2:

        return False, None

    current = candles[-1]

    previous = candles[
        -lookback - 1:-1
    ]

    if direction == "LONG":

        resistance = max(
            c["high"]
            for c in previous
        )

        if current["close"] > resistance:

            return True, resistance

    else:

        support = min(
            c["low"]
            for c in previous
        )

        if current["close"] < support:

            return True, support

    return False, None


# ============================================================
# REVERSAL
#
# Reversal is allowed only after:
#
# 1. Structure break
# 2. Confirming candles
# 3. EMA transition
# 4. Price on correct side of EMA50
# 5. ADX
# 6. RSI
#
# It is intentionally stricter than normal setups.
# ============================================================

def detect_reversal(
    candles,
    old_direction,
    adx_value,
    rsi_value
):

    required = max(
        60,
        REVERSAL_LOOKBACK
        + REVERSAL_CONFIRM_CANDLES
        + 10
    )

    if len(candles) < required:

        return None

    closes = [
        c["close"]
        for c in candles
    ]

    ema20_now = ema(
        closes,
        EMA20
    )

    ema50_now = ema(
        closes,
        EMA50
    )

    if (
        ema20_now is None
        or ema50_now is None
    ):

        return None

    current = candles[-1]

    confirmation = candles[
        -REVERSAL_CONFIRM_CANDLES:
    ]

    # --------------------------------------------------------
    # LONG -> SHORT
    # --------------------------------------------------------

    if old_direction == "LONG":

        bearish = all(

            c["close"]
            < c["open"]

            for c in confirmation

        )

        structure_break, level = (
            detect_structure_shift(
                candles,
                "SHORT",
                REVERSAL_LOOKBACK
            )
        )

        if (

            structure_break

            and bearish

            and ema20_now < ema50_now

            and current["close"]
            < ema50_now

            and adx_value >= ADX_MIN

            and rsi_value < 50

        ):

            return {

                "direction":
                    "SHORT",

                "type":
                    "REVERSAL",

                "structure_level":
                    level

            }

    # --------------------------------------------------------
    # SHORT -> LONG
    # --------------------------------------------------------

    if old_direction == "SHORT":

        bullish = all(

            c["close"]
            > c["open"]

            for c in confirmation

        )

        structure_break, level = (
            detect_structure_shift(
                candles,
                "LONG",
                REVERSAL_LOOKBACK
            )
        )

        if (

            structure_break

            and bullish

            and ema20_now > ema50_now

            and current["close"]
            > ema50_now

            and adx_value >= ADX_MIN

            and rsi_value > 50

        ):

            return {

                "direction":
                    "LONG",

                "type":
                    "REVERSAL",

                "structure_level":
                    level

            }

    return None


# ============================================================
# PULLBACK
#
# FIX:
# A random EMA touch is NOT enough.
#
# Required:
# - EMA touch
# - candle rejects the EMA zone
# - current candle confirms direction
# ============================================================

def detect_pullback(
    candles,
    direction,
    atr_value
):

    if atr_value is None:

        return False

    minimum = (
        EMA50
        + PULLBACK_LOOKBACK
        + 5
    )

    if len(candles) < minimum:

        return False

    closes = [
        c["close"]
        for c in candles
    ]

    start = (
        len(candles)
        - PULLBACK_LOOKBACK
    )

    current = candles[-1]

    current_ema20 = ema(
        closes,
        EMA20
    )

    current_ema50 = ema(
        closes,
        EMA50
    )

    if (
        current_ema20 is None
        or current_ema50 is None
    ):

        return False

    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    if direction == "LONG":

        for i in range(
            start,
            len(candles) - 1
        ):

            candle = candles[i]

            partial = closes[:i + 1]

            e20 = ema(
                partial,
                EMA20
            )

            e50 = ema(
                partial,
                EMA50
            )

            if (
                e20 is None
                or e50 is None
            ):

                continue

            zone20 = (
                e20
                + atr_value
                * EMA_PULLBACK_ATR
            )

            zone50 = (
                e50
                + atr_value
                * EMA_PULLBACK_ATR
            )

            touched = (

                candle["low"]
                <= zone20

                or

                candle["low"]
                <= zone50

            )

            rejected = (
                candle["close"]
                > candle["open"]
            )

            closed_above = (
                candle["close"]
                >= e20
            )

            if (
                touched
                and rejected
                and closed_above
            ):

                # Current candle must continue upward.
                if (
                    current["close"]
                    > candle["close"]
                ):

                    return True

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    else:

        for i in range(
            start,
            len(candles) - 1
        ):

            candle = candles[i]

            partial = closes[:i + 1]

            e20 = ema(
                partial,
                EMA20
            )

            e50 = ema(
                partial,
                EMA50
            )

            if (
                e20 is None
                or e50 is None
            ):

                continue

            zone20 = (
                e20
                - atr_value
                * EMA_PULLBACK_ATR
            )

            zone50 = (
                e50
                - atr_value
                * EMA_PULLBACK_ATR
            )

            touched = (

                candle["high"]
                >= zone20

                or

                candle["high"]
                >= zone50

            )

            rejected = (
                candle["close"]
                < candle["open"]
            )

            closed_below = (
                candle["close"]
                <= e20
            )

            if (
                touched
                and rejected
                and closed_below
            ):

                if (
                    current["close"]
                    < candle["close"]
                ):

                    return True

    return False


# ============================================================
# BREAKOUT
#
# Current candle must CLOSE outside the previous structure.
# ============================================================

def detect_breakout(
    candles,
    direction
):

    if len(candles) < (
        STRUCTURE_LOOKBACK + 2
    ):

        return False

    current = candles[-1]

    previous = candles[
        -STRUCTURE_LOOKBACK - 1:-1
    ]

    if direction == "LONG":

        resistance = max(
            c["high"]
            for c in previous
        )

        return (
            current["close"]
            > resistance
        )

    support = min(
        c["low"]
        for c in previous
    )

    return (
        current["close"]
        < support
    )


# ============================================================
# CANDLE CONFIRMATION
# ============================================================

def candle_confirmation(
    candle,
    direction
):

    candle_range = (
        candle["high"]
        - candle["low"]
    )

    if candle_range <= 0:

        return False

    body = abs(
        candle["close"]
        - candle["open"]
    )

    body_ratio = (
        body
        / candle_range
    )

    if body_ratio < 0.45:

        return False

    if direction == "LONG":

        close_location = (
            candle["close"]
            - candle["low"]
        ) / candle_range

        return (

            candle["close"]
            > candle["open"]

            and close_location >= 0.60

        )

    close_location = (
        candle["high"]
        - candle["close"]
    ) / candle_range

    return (

        candle["close"]
        < candle["open"]

        and close_location >= 0.60

    )


# ============================================================
# MOMENTUM
# ============================================================

def momentum_confirmation(
    candles,
    direction
):

    if len(candles) < 4:

        return False

    current = candles[-1]
    previous = candles[-2]

    if direction == "LONG":

        return (
            current["close"]
            > previous["close"]
        )

    return (
        current["close"]
        < previous["close"]
    )


# ============================================================
# RSI CONFIRMATION
# ============================================================

def rsi_confirmation(
    rsi_value,
    direction
):

    if rsi_value is None:

        return False

    if direction == "LONG":

        return (
            50
            <= rsi_value
            <= 85
        )

    return (
        15
        <= rsi_value
        <= 50
    )


# ============================================================
# EMA ALIGNMENT
# ============================================================

def ema_alignment(
    candles,
    direction
):

    closes = [
        c["close"]
        for c in candles
    ]

    entry = closes[-1]

    e20 = ema(
        closes,
        EMA20
    )

    e50 = ema(
        closes,
        EMA50
    )

    e200 = ema(
        closes,
        EMA200
    )

    if (
        e20 is None
        or e50 is None
        or e200 is None
    ):

        return False

    if direction == "LONG":

        return (
            entry > e20
            and e20 > e50
            and e50 > e200
        )

    return (
        entry < e20
        and e20 < e50
        and e50 < e200
    )


# ============================================================
# RISK LEVELS
#
# FIX:
#
# LONG:
# ATR stop = entry - 1.5 ATR
# structure stop = swing low - buffer
#
# We choose the FARTHER valid stop.
# This means structure is allowed to widen risk.
#
# SHORT:
# opposite logic.
# ============================================================

def calculate_risk_levels(
    candles,
    direction,
    entry,
    atr_value
):

    if atr_value is None or atr_value <= 0:

        return None

    recent = candles[-6:]

    recent_low = min(
        c["low"]
        for c in recent
    )

    recent_high = max(
        c["high"]
        for c in recent
    )

    if direction == "LONG":

        atr_stop = (
            entry
            - atr_value * SL_ATR
        )

        structure_stop = (
            recent_low
            - atr_value
            * STRUCTURE_BUFFER_ATR
        )

        # Farther stop protects against
        # normal structural noise.
        sl = min(
            atr_stop,
            structure_stop
        )

        risk = (
            entry
            - sl
        )

        if risk <= 0:

            return None

        risk_atr = (
            risk
            / atr_value
        )

        if risk_atr > MAX_SL_ATR:

            return None

        tp = (
            entry
            + risk * TP_R_MULTIPLE
        )

    else:

        atr_stop = (
            entry
            + atr_value * SL_ATR
        )

        structure_stop = (
            recent_high
            + atr_value
            * STRUCTURE_BUFFER_ATR
        )

        sl = max(
            atr_stop,
            structure_stop
        )

        risk = (
            sl
            - entry
        )

        if risk <= 0:

            return None

        risk_atr = (
            risk
            / atr_value
        )

        if risk_atr > MAX_SL_ATR:

            return None

        tp = (
            entry
            - risk * TP_R_MULTIPLE
        )

    reward = abs(
        tp - entry
    )

    rr = (
        reward
        / risk
    )

    if rr < MIN_RR:

        return None

    return {

        "tp":
            tp,

        "sl":
            sl,

        "risk":
            risk,

        "rr":
            rr,

        "risk_atr":
            risk_atr

    }


# ============================================================
# GET USABLE 4H CANDLES
#
# CRITICAL LOOK-AHEAD FIX
#
# A 1H entry candle beginning at T closes at:
#
# T + 3600
#
# A 4H candle beginning at H closes at:
#
# H + 14400
#
# Only a 4H candle whose CLOSE is <=
# the 1H entry close may be used.
# ============================================================

def get_closed_4h_for_entry(
    candles_4h,
    entry_candle
):

    entry_close_time = (
        entry_candle["time"]
        + SECONDS_1H
    )

    usable = [

        candle

        for candle in candles_4h

        if (
            candle["time"]
            + SECONDS_4H
            <= entry_close_time
        )

    ]

    return usable


# ============================================================
# SIGNAL ANALYSIS AT HISTORICAL INDEX
# ============================================================

def analyze_at_index(
    candles_4h,
    candles_1h
):

    if len(candles_1h) < EMA200 + 10:

        return None

    if len(candles_4h) < EMA200:

        return None

    current = candles_1h[-1]

    entry = current["close"]

    # --------------------------------------------------------
    # 4H TREND
    # --------------------------------------------------------

    trend_state = get_4h_state(
        candles_4h
    )

    trend_direction = (
        trend_state["direction"]
    )

    if trend_direction is None:

        return None

    # --------------------------------------------------------
    # 1H INDICATORS
    #
    # ONLY candles up to current closed candle.
    # --------------------------------------------------------

    atr_value = atr(
        candles_1h,
        ATR_PERIOD
    )

    rsi_value = rsi(
        candles_1h,
        RSI_PERIOD
    )

    adx_value = adx(
        candles_1h,
        ADX_PERIOD
    )

    if (
        atr_value is None
        or rsi_value is None
        or adx_value is None
    ):

        return None

    # --------------------------------------------------------
    # REVERSAL
    # --------------------------------------------------------

    reversal = detect_reversal(
        candles_1h,
        trend_direction,
        adx_value,
        rsi_value
    )

    if reversal is not None:

        direction = (
            reversal["direction"]
        )

    else:

        direction = trend_direction

    # --------------------------------------------------------
    # BASIC CONDITIONS
    # --------------------------------------------------------

    pullback = detect_pullback(
        candles_1h,
        direction,
        atr_value
    )

    breakout = detect_breakout(
        candles_1h,
        direction
    )

    candle_ok = candle_confirmation(
        current,
        direction
    )

    momentum_ok = momentum_confirmation(
        candles_1h,
        direction
    )

    adx_ok = (
        adx_value
        >= ADX_MIN
    )

    rsi_ok = rsi_confirmation(
        rsi_value,
        direction
    )

    # --------------------------------------------------------
    # EMA
    #
    # For REVERSAL:
    # EMA alignment is already part of reversal.
    #
    # For normal setups:
    # strict 1H EMA20 > EMA50 > EMA200.
    # --------------------------------------------------------

    alignment_ok = ema_alignment(
        candles_1h,
        direction
    )

    # --------------------------------------------------------
    # SETUP
    # --------------------------------------------------------

    if reversal is not None:

        setup_type = "REVERSAL"

    else:

        if not alignment_ok:

            return None

        # ----------------------------------------------------
        # PULLBACK
        #
        # Pullback:
        # - real EMA interaction
        # - rejection
        # - momentum/candle confirmation
        # ----------------------------------------------------

        pullback_setup = (

            pullback

            and adx_ok

            and rsi_ok

            and (
                candle_ok
                or momentum_ok
            )

        )

        # ----------------------------------------------------
        # BREAKOUT
        #
        # Breakout requires:
        # - structure close
        # - ADX
        # - RSI
        # - strong candle
        # ----------------------------------------------------

        breakout_setup = (

            breakout

            and adx_ok

            and rsi_ok

            and candle_ok

        )

        if pullback_setup:

            setup_type = "PULLBACK"

        elif breakout_setup:

            setup_type = "BREAKOUT"

        else:

            return None

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    risk_levels = calculate_risk_levels(
        candles_1h,
        direction,
        entry,
        atr_value
    )

    if risk_levels is None:

        return None

    return {

        "direction":
            direction,

        "setup":
            setup_type,

        "entry":
            entry,

        "tp":
            risk_levels["tp"],

        "sl":
            risk_levels["sl"],

        "risk":
            risk_levels["risk"],

        "rr":
            risk_levels["rr"],

        "risk_atr":
            risk_levels["risk_atr"],

        "atr":
            atr_value,

        "adx":
            adx_value,

        "rsi":
            rsi_value,

        "entry_time":
            current["time"]

    }


# ============================================================
# CHECK TRADE RESULT
#
# ENTRY CANDLE EXCLUDED.
# ============================================================

def check_trade_result(
    candles,
    entry_index,
    position
):

    for i in range(
        entry_index + 1,
        len(candles)
    ):

        candle = candles[i]

        high = candle["high"]
        low = candle["low"]

        if position["direction"] == "LONG":

            sl_hit = (
                low
                <= position["sl"]
            )

            tp_hit = (
                high
                >= position["tp"]
            )

        else:

            sl_hit = (
                high
                >= position["sl"]
            )

            tp_hit = (
                low
                <= position["tp"]
            )

        # ----------------------------------------------------
        # BOTH
        #
        # OHLC cannot tell us which happened first.
        # Conservative = SL.
        # ----------------------------------------------------

        if sl_hit and tp_hit:

            return "SL", i

        if sl_hit:

            return "SL", i

        if tp_hit:

            return "TP", i

    return None, None


# ============================================================
# UTC
# ============================================================

def utc_time(timestamp):

    return datetime.fromtimestamp(
        timestamp,
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M"
    )


# ============================================================
# PRICE FORMAT
# ============================================================

def fmt_price(value):

    if value >= 1000:

        return f"{value:.2f}"

    if value >= 1:

        return f"{value:.5f}"

    if value >= 0.01:

        return f"{value:.6f}"

    return f"{value:.8f}"


# ============================================================
# DATA RANGE
# ============================================================

def data_range(candles):

    if not candles:

        return "N/A"

    return (
        f"{utc_time(candles[0]['time'])}"
        f" -> "
        f"{utc_time(candles[-1]['time'])}"
    )


# ============================================================
# BACKTEST ONE COIN
# ============================================================

def backtest_coin(
    symbol,
    candles_4h,
    candles_1h
):

    trades = []

    start_index = max(
        EMA200 + 10,
        80
    )

    i = start_index

    while i < len(candles_1h):

        current_candle = (
            candles_1h[i]
        )

        # ----------------------------------------------------
        # CRITICAL:
        # Only fully CLOSED 4H candles available at the
        # moment the 1H candle closes.
        # ----------------------------------------------------

        usable_4h = get_closed_4h_for_entry(
            candles_4h,
            current_candle
        )

        if len(usable_4h) < EMA200:

            i += 1

            continue

        # ----------------------------------------------------
        # SIGNAL
        #
        # candles_1h[:i+1] means:
        # current candle is CLOSED and available.
        # ----------------------------------------------------

        signal = analyze_at_index(

            usable_4h,

            candles_1h[:i + 1]

        )

        if signal is None:

            i += 1

            continue

        # ----------------------------------------------------
        # RESULT
        #
        # Starts from NEXT candle.
        # ----------------------------------------------------

        result, exit_index = (
            check_trade_result(
                candles_1h,
                i,
                signal
            )
        )

        # ----------------------------------------------------
        # Still open at end of dataset.
        # Do not count as completed trade.
        # ----------------------------------------------------

        if result is None:

            break

        exit_candle = (
            candles_1h[exit_index]
        )

        if result == "TP":

            r_result = (
                TP_R_MULTIPLE
            )

        else:

            r_result = -1.0

        trades.append({

            "symbol":
                symbol,

            "setup":
                signal["setup"],

            "direction":
                signal["direction"],

            "entry_time":
                signal["entry_time"],

            "entry":
                signal["entry"],

            "tp":
                signal["tp"],

            "sl":
                signal["sl"],

            "exit_time":
                exit_candle["time"],

            "result":
                result,

            "R":
                r_result,

            "rr":
                signal["rr"],

            "risk_atr":
                signal["risk_atr"],

            "atr":
                signal["atr"],

            "adx":
                signal["adx"],

            "rsi":
                signal["rsi"]

        })

        # ----------------------------------------------------
        # NO OVERLAPPING TRADES
        #
        # Next scan begins AFTER exit candle.
        # ----------------------------------------------------

        i = exit_index + 1

    return trades


# ============================================================
# STATISTICS
# ============================================================

def calculate_stats(trades):

    if not trades:

        return {}

    total = len(trades)

    wins = sum(
        t["result"] == "TP"
        for t in trades
    )

    losses = sum(
        t["result"] == "SL"
        for t in trades
    )

    win_rate = (
        wins
        / total
        * 100
    )

    net_r = sum(
        t["R"]
        for t in trades
    )

    gross_profit = sum(
        t["R"]
        for t in trades
        if t["R"] > 0
    )

    gross_loss = abs(
        sum(
            t["R"]
            for t in trades
            if t["R"] < 0
        )
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = float("inf")

    average_r = (
        net_r
        / total
    )

    # --------------------------------------------------------
    # EQUITY / DD
    # --------------------------------------------------------

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0

    current_win = 0
    current_loss = 0

    max_win = 0
    max_loss = 0

    for trade in trades:

        equity += trade["R"]

        peak = max(
            peak,
            equity
        )

        drawdown = (
            peak
            - equity
        )

        max_drawdown = max(
            max_drawdown,
            drawdown
        )

        if trade["result"] == "TP":

            current_win += 1
            current_loss = 0

            max_win = max(
                max_win,
                current_win
            )

        else:

            current_loss += 1
            current_win = 0

            max_loss = max(
                max_loss,
                current_loss
            )

    return {

        "total":
            total,

        "wins":
            wins,

        "losses":
            losses,

        "win_rate":
            win_rate,

        "net_r":
            net_r,

        "gross_profit":
            gross_profit,

        "gross_loss":
            gross_loss,

        "profit_factor":
            profit_factor,

        "average_r":
            average_r,

        "max_drawdown":
            max_drawdown,

        "max_win_streak":
            max_win,

        "max_loss_streak":
            max_loss

    }


# ============================================================
# REPORT
# ============================================================

def print_report(
    all_trades,
    data_ranges
):

    print(
        "\n"
        + "=" * 110
    )

    print(
        "SCORE HUNTER PRO v8"
    )

    print(
        "FIXED HISTORICAL BACKTEST"
    )

    print(
        "=" * 110
    )

    if not all_trades:

        print(
            "NO COMPLETED TRADES FOUND."
        )

        print(
            "\nDATA RANGES:"
        )

        for symbol, ranges in data_ranges.items():

            print(
                f"{symbol:5s} | "
                f"4H: {ranges['4H']} | "
                f"1H: {ranges['1H']}"
            )

        return

    stats = calculate_stats(
        all_trades
    )

    # ========================================================
    # MAIN
    # ========================================================

    print(
        f"Total Trades       : "
        f"{stats['total']}"
    )

    print(
        f"Wins               : "
        f"{stats['wins']}"
    )

    print(
        f"Losses             : "
        f"{stats['losses']}"
    )

    print(
        f"Win Rate           : "
        f"{stats['win_rate']:.2f}%"
    )

    print(
        f"Net Result         : "
        f"{stats['net_r']:+.2f}R"
    )

    print(
        f"Average R          : "
        f"{stats['average_r']:+.3f}R"
    )

    if stats["profit_factor"] == float("inf"):

        pf_text = "INF"

    else:

        pf_text = (
            f"{stats['profit_factor']:.2f}"
        )

    print(
        f"Profit Factor      : "
        f"{pf_text}"
    )

    print(
        f"Max Drawdown       : "
        f"{stats['max_drawdown']:.2f}R"
    )

    print(
        f"Max Win Streak     : "
        f"{stats['max_win_streak']}"
    )

    print(
        f"Max Loss Streak    : "
        f"{stats['max_loss_streak']}"
    )

    # ========================================================
    # DATA RANGE
    # ========================================================

    print(
        "\n"
        + "=" * 110
    )

    print(
        "ACTUAL DATA RANGE"
    )

    print(
        "=" * 110
    )

    for symbol, ranges in data_ranges.items():

        print(
            f"{symbol:5s} | "
            f"4H: {ranges['4H']} | "
            f"1H: {ranges['1H']}"
        )

    # ========================================================
    # COIN
    # ========================================================

    print(
        "\n"
        + "=" * 110
    )

    print(
        "RESULT BY COIN"
    )

    print(
        "=" * 110
    )

    for symbol in COINS:

        coin_trades = [

            t

            for t in all_trades

            if t["symbol"] == symbol

        ]

        if not coin_trades:

            print(
                f"{symbol:5s} | "
                "0 trades"
            )

            continue

        coin_stats = calculate_stats(
            coin_trades
        )

        print(
            f"{symbol:5s} | "
            f"Trades: {coin_stats['total']:3d} | "
            f"W: {coin_stats['wins']:3d} | "
            f"L: {coin_stats['losses']:3d} | "
            f"WR: {coin_stats['win_rate']:6.2f}% | "
            f"R: {coin_stats['net_r']:+7.2f} | "
            f"PF: {coin_stats['profit_factor']:.2f}"
        )

    # ========================================================
    # DIRECTION
    # ========================================================

    print(
        "\n"
        + "=" * 110
    )

    print(
        "RESULT BY DIRECTION"
    )

    print(
        "=" * 110
    )

    for direction in [
        "LONG",
        "SHORT"
    ]:

        direction_trades = [

            t

            for t in all_trades

            if t["direction"] == direction

        ]

        if not direction_trades:

            continue

        s = calculate_stats(
            direction_trades
        )

        print(
            f"{direction:5s} | "
            f"Trades: {s['total']:3d} | "
            f"W: {s['wins']:3d} | "
            f"L: {s['losses']:3d} | "
            f"WR: {s['win_rate']:6.2f}% | "
            f"R: {s['net_r']:+7.2f}"
        )

    # ========================================================
    # SETUP
    # ========================================================

    print(
        "\n"
        + "=" * 110
    )

    print(
        "RESULT BY SETUP"
    )

    print(
        "=" * 110
    )

    for setup in [
        "PULLBACK",
        "BREAKOUT",
        "REVERSAL"
    ]:

        setup_trades = [

            t

            for t in all_trades

            if t["setup"] == setup

        ]

        if not setup_trades:

            continue

        s = calculate_stats(
            setup_trades
        )

        print(
            f"{setup:10s} | "
            f"Trades: {s['total']:3d} | "
            f"W: {s['wins']:3d} | "
            f"L: {s['losses']:3d} | "
            f"WR: {s['win_rate']:6.2f}% | "
            f"R: {s['net_r']:+7.2f}"
        )

    # ========================================================
    # FULL TRADE LOG
    # ========================================================

    print(
        "\n"
        + "=" * 110
    )

    print(
        "FULL TRADE LOG"
    )

    print(
        "=" * 110
    )

    for number, trade in enumerate(
        all_trades,
        1
    ):

        print(

            f"{number:03d} | "

            f"{trade['symbol']:5s} | "

            f"{trade['direction']:5s} | "

            f"{trade['setup']:9s} | "

            f"ENTRY "
            f"{utc_time(trade['entry_time'])} | "

            f"E "
            f"{fmt_price(trade['entry'])} | "

            f"TP "
            f"{fmt_price(trade['tp'])} | "

            f"SL "
            f"{fmt_price(trade['sl'])} | "

            f"EXIT "
            f"{utc_time(trade['exit_time'])} | "

            f"{trade['result']:2s} | "

            f"{trade['R']:+.1f}R | "

            f"ADX "
            f"{trade['adx']:.1f} | "

            f"RSI "
            f"{trade['rsi']:.1f}"

        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        "=========================================================="
    )

    print(
        " SCORE HUNTER PRO v8"
    )

    print(
        " FIXED HISTORICAL BACKTEST"
    )

    print(
        "=========================================================="
    )

    print(
        "\nRules:"
    )

    print(
        "4H Trend + 1H Entry"
    )

    print(
        "Closed 4H only"
    )

    print(
        "Closed 1H only"
    )

    print(
        "NO LOOK-AHEAD"
    )

    print(
        "Pullback / Breakout / Reversal"
    )

    print(
        "Pullback = EMA interaction + rejection + confirmation"
    )

    print(
        "Breakout = structure close + confirmation"
    )

    print(
        "Reversal = structure shift + EMA transition"
    )

    print(
        "ADX >= 20"
    )

    print(
        "RSI confirmation"
    )

    print(
        "SL = 1.5 ATR / Structure"
    )

    print(
        "Maximum SL = 3.5 ATR"
    )

    print(
        "TP = 2R"
    )

    print(
        "Entry candle excluded from TP/SL"
    )

    print(
        "TP + SL same candle = SL"
    )

    print(
        "No overlapping positions"
    )

    all_trades = []

    data_ranges = {}

    # ========================================================
    # DOWNLOAD + BACKTEST
    # ========================================================

    for symbol in COINS:

        print(
            f"\nDownloading {symbol}..."
        )

        try:

            candles_4h = get_ohlc(
                symbol,
                INTERVAL_4H
            )

            time.sleep(1)

            candles_1h = get_ohlc(
                symbol,
                INTERVAL_1H
            )

            data_ranges[symbol] = {

                "4H":
                    data_range(candles_4h),

                "1H":
                    data_range(candles_1h)

            }

            print(
                f"{symbol}: "
                f"4H={len(candles_4h)} | "
                f"1H={len(candles_1h)}"
            )

            print(
                f"{symbol}: "
                f"4H range = "
                f"{data_ranges[symbol]['4H']}"
            )

            print(
                f"{symbol}: "
                f"1H range = "
                f"{data_ranges[symbol]['1H']}"
            )

            trades = backtest_coin(
                symbol,
                candles_4h,
                candles_1h
            )

            all_trades.extend(
                trades
            )

            print(
                f"{symbol}: "
                f"{len(trades)} "
                f"completed trades"
            )

        except Exception as e:

            print(
                f"{symbol} ERROR: "
                f"{type(e).__name__}: "
                f"{e}"
            )

    # ========================================================
    # SORT CHRONOLOGICALLY
    # ========================================================

    all_trades.sort(
        key=lambda x:
        x["entry_time"]
    )

    # ========================================================
    # REPORT
    # ========================================================

    print_report(
        all_trades,
        data_ranges
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
