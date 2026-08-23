# ============================================================
# SCORE HUNTER PRO v8
# 4H TREND + 1H ENTRY + REVERSAL PROTECTION
# CLOSED CANDLES ONLY
# ============================================================

from dataclasses import dataclass
from typing import Optional


# ============================================================
# SETTINGS
# ============================================================

ATR_LENGTH = 14
RSI_LENGTH = 14
ADX_LENGTH = 14

ADX_MINIMUM = 20.0

SL_ATR_MULTIPLIER = 1.5
TP_R_MULTIPLIER = 2.0

MINIMUM_RR = 1.5

STRUCTURE_LOOKBACK = 20

REVERSAL_CONFIRM_BARS = 2

MAX_EMA20_DISTANCE_ATR = 2.5


# ============================================================
# SIGNAL OBJECT
# ============================================================

@dataclass
class Signal:

    symbol: str
    direction: str
    setup: str

    entry: float
    sl: float
    tp: float

    atr: float
    adx: float
    rsi: float

    reason: str


# ============================================================
# EMA
# ============================================================

def ema(values, length):

    if len(values) < length:
        return None

    alpha = 2.0 / (length + 1.0)

    result = values[0]

    for value in values[1:]:

        result = (
            alpha * value
            + (1.0 - alpha) * result
        )

    return result


# ============================================================
# ATR
# ============================================================

def calculate_atr(
    high,
    low,
    close,
    length=ATR_LENGTH
):

    if len(close) < length + 1:
        return None

    tr = []

    for i in range(1, len(close)):

        true_range = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )

        tr.append(true_range)

    if len(tr) < length:
        return None

    return sum(tr[-length:]) / length


# ============================================================
# RSI
# ============================================================

def calculate_rsi(
    close,
    length=RSI_LENGTH
):

    if len(close) < length + 1:
        return None

    gains = []
    losses = []

    start = len(close) - length

    for i in range(start, len(close)):

        change = close[i] - close[i - 1]

        if change > 0:
            gains.append(change)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(change))

    avg_gain = sum(gains) / length
    avg_loss = sum(losses) / length

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss

    return 100.0 - (
        100.0 / (1.0 + rs)
    )


# ============================================================
# ADX
# ============================================================

def calculate_adx(
    high,
    low,
    close,
    length=ADX_LENGTH
):

    if len(close) < length * 2 + 2:
        return None

    tr = []
    plus_dm = []
    minus_dm = []

    for i in range(1, len(close)):

        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        if up_move > down_move and up_move > 0:
            p_dm = up_move
        else:
            p_dm = 0.0

        if down_move > up_move and down_move > 0:
            m_dm = down_move
        else:
            m_dm = 0.0

        true_range = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )

        tr.append(true_range)
        plus_dm.append(p_dm)
        minus_dm.append(m_dm)

    atr_sum = sum(tr[:length])
    plus_sum = sum(plus_dm[:length])
    minus_sum = sum(minus_dm[:length])

    dx_values = []

    for i in range(length, len(tr)):

        atr_sum = (
            atr_sum
            - atr_sum / length
            + tr[i]
        )

        plus_sum = (
            plus_sum
            - plus_sum / length
            + plus_dm[i]
        )

        minus_sum = (
            minus_sum
            - minus_sum / length
            + minus_dm[i]
        )

        if atr_sum == 0:
            continue

        plus_di = (
            100.0 * plus_sum / atr_sum
        )

        minus_di = (
            100.0 * minus_sum / atr_sum
        )

        denominator = plus_di + minus_di

        if denominator == 0:
            dx = 0.0
        else:
            dx = (
                100.0
                * abs(plus_di - minus_di)
                / denominator
            )

        dx_values.append(dx)

    if len(dx_values) < length:
        return None

    return sum(
        dx_values[-length:]
    ) / length


# ============================================================
# RECENT STRUCTURE
# ============================================================

def recent_high(
    high,
    lookback=STRUCTURE_LOOKBACK
):

    if len(high) < lookback + 1:
        return None

    return max(
        high[-lookback:-1]
    )


def recent_low(
    low,
    lookback=STRUCTURE_LOOKBACK
):

    if len(low) < lookback + 1:
        return None

    return min(
        low[-lookback:-1]
    )


# ============================================================
# STRUCTURE BREAK
# ============================================================

def bearish_structure_break(
    high,
    low,
    close
):

    previous_low = recent_low(
        low[:-1],
        STRUCTURE_LOOKBACK
    )

    if previous_low is None:
        return False

    return close[-1] < previous_low


def bullish_structure_break(
    high,
    low,
    close
):

    previous_high = recent_high(
        high[:-1],
        STRUCTURE_LOOKBACK
    )

    if previous_high is None:
        return False

    return close[-1] > previous_high


# ============================================================
# MOMENTUM
# ============================================================

def bearish_momentum(close):

    if len(close) < REVERSAL_CONFIRM_BARS + 2:
        return False

    count = 0

    for i in range(
        -REVERSAL_CONFIRM_BARS,
        0
    ):

        if close[i] < close[i - 1]:
            count += 1

    return count >= 1


def bullish_momentum(close):

    if len(close) < REVERSAL_CONFIRM_BARS + 2:
        return False

    count = 0

    for i in range(
        -REVERSAL_CONFIRM_BARS,
        0
    ):

        if close[i] > close[i - 1]:
            count += 1

    return count >= 1


# ============================================================
# BEARISH REVERSAL
# ============================================================

def bearish_reversal(
    high,
    low,
    close,
    ema20,
    ema50,
    adx,
    rsi
):

    structure_break = bearish_structure_break(
        high,
        low,
        close
    )

    momentum = bearish_momentum(
        close
    )

    ema_damage = (
        close[-1] < ema20
        or
        close[-1] < ema50
    )

    rsi_confirmation = (
        rsi < 60
    )

    adx_confirmation = (
        adx >= ADX_MINIMUM
    )

    confirmations = sum([
        structure_break,
        momentum,
        ema_damage,
        rsi_confirmation,
        adx_confirmation
    ])

    return (
        structure_break
        and
        confirmations >= 3
    )


# ============================================================
# BULLISH REVERSAL
# ============================================================

def bullish_reversal(
    high,
    low,
    close,
    ema20,
    ema50,
    adx,
    rsi
):

    structure_break = bullish_structure_break(
        high,
        low,
        close
    )

    momentum = bullish_momentum(
        close
    )

    ema_reclaim = (
        close[-1] > ema20
        or
        close[-1] > ema50
    )

    rsi_confirmation = (
        rsi > 40
    )

    adx_confirmation = (
        adx >= ADX_MINIMUM
    )

    confirmations = sum([
        structure_break,
        momentum,
        ema_reclaim,
        rsi_confirmation,
        adx_confirmation
    ])

    return (
        structure_break
        and
        confirmations >= 3
    )


# ============================================================
# 4H TREND
# ============================================================

def get_4h_direction(close4h):

    ema20 = ema(
        close4h,
        20
    )

    ema50 = ema(
        close4h,
        50
    )

    ema200 = ema(
        close4h,
        200
    )

    if (
        ema20 is None
        or
        ema50 is None
        or
        ema200 is None
    ):
        return None

    price = close4h[-1]

    if (
        price > ema20
        and
        ema20 > ema50
        and
        ema50 > ema200
    ):

        return "LONG"

    if (
        price < ema20
        and
        ema20 < ema50
        and
        ema50 < ema200
    ):

        return "SHORT"

    return None


# ============================================================
# SL / TP
# ============================================================

def calculate_levels(
    direction,
    entry,
    atr_value,
    structure_level
):

    if direction == "LONG":

        atr_sl = (
            entry
            -
            SL_ATR_MULTIPLIER * atr_value
        )

        if structure_level is not None:

            sl = min(
                atr_sl,
                structure_level
            )

        else:

            sl = atr_sl

        risk = entry - sl

        if risk <= 0:
            return None

        tp = (
            entry
            +
            TP_R_MULTIPLIER * risk
        )

    else:

        atr_sl = (
            entry
            +
            SL_ATR_MULTIPLIER * atr_value
        )

        if structure_level is not None:

            sl = max(
                atr_sl,
                structure_level
            )

        else:

            sl = atr_sl

        risk = sl - entry

        if risk <= 0:
            return None

        tp = (
            entry
            -
            TP_R_MULTIPLIER * risk
        )

    rr = (
        abs(tp - entry)
        /
        abs(entry - sl)
    )

    if rr < MINIMUM_RR:
        return None

    return sl, tp, rr


# ============================================================
# LONG PULLBACK
# ============================================================

def detect_long_pullback(
    close,
    ema20,
    atr_value
):

    recent = close[-5:]

    touched_ema = (
        min(recent)
        <=
        ema20 + 0.35 * atr_value
    )

    recovered = (
        close[-1] > ema20
    )

    return (
        touched_ema
        and
        recovered
    )


# ============================================================
# SHORT PULLBACK
# ============================================================

def detect_short_pullback(
    close,
    ema20,
    atr_value
):

    recent = close[-5:]

    touched_ema = (
        max(recent)
        >=
        ema20 - 0.35 * atr_value
    )

    rejected = (
        close[-1] < ema20
    )

    return (
        touched_ema
        and
        rejected
    )


# ============================================================
# LONG BREAKOUT
# ============================================================

def detect_long_breakout(
    high,
    close
):

    level = recent_high(
        high[:-1],
        STRUCTURE_LOOKBACK
    )

    if level is None:
        return False

    return close[-1] > level


# ============================================================
# SHORT BREAKOUT
# ============================================================

def detect_short_breakout(
    low,
    close
):

    level = recent_low(
        low[:-1],
        STRUCTURE_LOOKBACK
    )

    if level is None:
        return False

    return close[-1] < level


# ============================================================
# MAIN SIGNAL ENGINE
# ============================================================

def generate_signal(
    symbol,
    high1h,
    low1h,
    close1h,
    close4h
):

    if len(close1h) < 220:
        return None

    if len(close4h) < 220:
        return None


    trend4h = get_4h_direction(
        close4h
    )


    ema20 = ema(
        close1h,
        20
    )

    ema50 = ema(
        close1h,
        50
    )

    ema200 = ema(
        close1h,
        200
    )

    atr_value = calculate_atr(
        high1h,
        low1h,
        close1h
    )

    rsi_value = calculate_rsi(
        close1h
    )

    adx_value = calculate_adx(
        high1h,
        low1h,
        close1h
    )


    if (
        ema20 is None
        or
        ema50 is None
        or
        ema200 is None
        or
        atr_value is None
        or
        rsi_value is None
        or
        adx_value is None
    ):
        return None


    entry = close1h[-1]


    # ========================================================
    # REVERSAL DETECTION
    # ========================================================

    bearish_reversal_confirmed = bearish_reversal(
        high1h,
        low1h,
        close1h,
        ema20,
        ema50,
        adx_value,
        rsi_value
    )


    bullish_reversal_confirmed = bullish_reversal(
        high1h,
        low1h,
        close1h,
        ema20,
        ema50,
        adx_value,
        rsi_value
    )


    # ========================================================
    # BULLISH 4H REGIME
    # ========================================================

    if trend4h == "LONG":

        # Stop taking LONG signals when the market
        # has confirmed a bearish reversal.

        if bearish_reversal_confirmed:

            if (
                adx_value >= ADX_MINIMUM
                and
                rsi_value < 60
            ):

                structure_sl = recent_high(
                    high1h[:-1],
                    STRUCTURE_LOOKBACK
                )

                levels = calculate_levels(
                    "SHORT",
                    entry,
                    atr_value,
                    structure_sl
                )

                if levels is not None:

                    sl, tp, rr = levels

                    return Signal(
                        symbol=symbol,
                        direction="SHORT",
                        setup="REVERSAL",
                        entry=entry,
                        sl=sl,
                        tp=tp,
                        atr=atr_value,
                        adx=adx_value,
                        rsi=rsi_value,
                        reason=(
                            "Bearish structure break "
                            "confirmed against the 4H trend."
                        )
                    )

            return None


        # ====================================================
        # NORMAL LONG
        # ====================================================

        trend_ok = (
            ema20 > ema50
            and
            ema50 > ema200
        )

        extension_ok = (
            abs(entry - ema20)
            <=
            MAX_EMA20_DISTANCE_ATR * atr_value
        )

        pullback = detect_long_pullback(
            close1h,
            ema20,
            atr_value
        )

        breakout = detect_long_breakout(
            high1h,
            close1h
        )

        rsi_ok = (
            rsi_value >= 50
        )


        if not trend_ok:
            return None

        if not extension_ok:
            return None

        if not rsi_ok:
            return None

        if not (
            pullback
            or
            breakout
        ):
            return None


        structure_sl = recent_low(
            low1h[:-1],
            STRUCTURE_LOOKBACK
        )

        levels = calculate_levels(
            "LONG",
            entry,
            atr_value,
            structure_sl
        )

        if levels is None:
            return None

        sl, tp, rr = levels


        setup = (
            "PULLBACK"
            if pullback
            else
            "BREAKOUT"
        )


        return Signal(
            symbol=symbol,
            direction="LONG",
            setup=setup,
            entry=entry,
            sl=sl,
            tp=tp,
            atr=atr_value,
            adx=adx_value,
            rsi=rsi_value,
            reason=(
                "4H bullish trend + 1H confirmation."
            )
        )


    # ========================================================
    # BEARISH 4H REGIME
    # ========================================================

    if trend4h == "SHORT":

        # Stop taking SHORT signals when the market
        # has confirmed a bullish reversal.

        if bullish_reversal_confirmed:

            if (
                adx_value >= ADX_MINIMUM
                and
                rsi_value > 40
            ):

                structure_sl = recent_low(
                    low1h[:-1],
                    STRUCTURE_LOOKBACK
                )

                levels = calculate_levels(
                    "LONG",
                    entry,
                    atr_value,
                    structure_sl
                )

                if levels is not None:

                    sl, tp, rr = levels

                    return Signal(
                        symbol=symbol,
                        direction="LONG",
                        setup="REVERSAL",
                        entry=entry,
                        sl=sl,
                        tp=tp,
                        atr=atr_value,
                        adx=adx_value,
                        rsi=rsi_value,
                        reason=(
                            "Bullish structure break "
                            "confirmed against the 4H trend."
                        )
                    )

            return None


        # ====================================================
        # NORMAL SHORT
        # ====================================================

        trend_ok = (
            ema20 < ema50
            and
            ema50 < ema200
        )

        extension_ok = (
            abs(entry - ema20)
            <=
            MAX_EMA20_DISTANCE_ATR * atr_value
        )

        pullback = detect_short_pullback(
            close1h,
            ema20,
            atr_value
        )

        breakout = detect_short_breakout(
            low1h,
            close1h
        )

        rsi_ok = (
            rsi_value <= 50
        )


        if not trend_ok:
            return None

        if not extension_ok:
            return None

        if not rsi_ok:
            return None

        if not (
            pullback
            or
            breakout
        ):
            return None


        structure_sl = recent_high(
            high1h[:-1],
            STRUCTURE_LOOKBACK
        )

        levels = calculate_levels(
            "SHORT",
            entry,
            atr_value,
            structure_sl
        )

        if levels is None:
            return None

        sl, tp, rr = levels


        setup = (
            "PULLBACK"
            if pullback
            else
            "BREAKOUT"
        )


        return Signal(
            symbol=symbol,
            direction="SHORT",
            setup=setup,
            entry=entry,
            sl=sl,
            tp=tp,
            atr=atr_value,
            adx=adx_value,
            rsi=rsi_value,
            reason=(
                "4H bearish trend + 1H confirmation."
            )
        )


    return None


# ============================================================
# TELEGRAM MESSAGE FORMAT
# ============================================================

def format_signal(signal):

    if signal.direction == "LONG":

        tp_percent = (
            signal.tp / signal.entry - 1
        ) * 100

        sl_percent = (
            1 - signal.sl / signal.entry
        ) * 100

    else:

        tp_percent = (
            1 - signal.tp / signal.entry
        ) * 100

        sl_percent = (
            signal.sl / signal.entry - 1
        ) * 100


    direction = (
        "🟢 LONG"
        if signal.direction == "LONG"
        else
        "🔴 SHORT"
    )


    return f"""
🚨 SCORE HUNTER PRO v8 🚨

💰 {signal.symbol}
📊 {direction}

🧠 SETUP: {signal.setup}

💵 Entry: {signal.entry:.8f}
🎯 TP: {signal.tp:.8f} (+{tp_percent:.2f}%)
🛑 SL: {signal.sl:.8f} (-{sl_percent:.2f}%)

━━━━━━━━━━━━━━━━━━

🧭 4H: Main trend
⏱ 1H: Entry confirmation

↩️ Pullback: ON
🚀 Breakout: ON
🔄 Reversal Detection: ON

📊 ADX: {signal.adx:.2f}
📈 RSI: {signal.rsi:.2f}

🕯 Closed 1H candle only
🚫 No mid-candle signal

🛡 Reversal Protection: ACTIVE

🎯 TP: 2R
⚖️ Risk/Reward: 1:2.00

⚠️ Manage risk.
"""


# ============================================================
# DEBUG OUTPUT
# ============================================================

def print_signal(signal):

    if signal is None:

        print(
            "No valid signal."
        )

        return


    print(
        "===================================="
    )

    print(
        "SCORE HUNTER PRO v8"
    )

    print(
        "===================================="
    )

    print(
        f"Symbol: {signal.symbol}"
    )

    print(
        f"Direction: {signal.direction}"
    )

    print(
        f"Setup: {signal.setup}"
    )

    print(
        f"Entry: {signal.entry}"
    )

    print(
        f"SL: {signal.sl}"
    )

    print(
        f"TP: {signal.tp}"
    )

    print(
        f"ADX: {signal.adx:.2f}"
    )

    print(
        f"RSI: {signal.rsi:.2f}"
    )

    print(
        f"Reason: {signal.reason}"
    )

    print(
        "===================================="
    )
