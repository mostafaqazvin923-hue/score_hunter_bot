import os
import json
import requests
from datetime import datetime, timezone


# ============================================================
# SCORE HUNTER PRO v7
#
# 4H TREND
# +
# 1H PULLBACK OR BREAKOUT
# +
# ADX
# +
# MARKET STRUCTURE
# +
# CLOSED CANDLE ONLY
# +
# ATR RISK MANAGEMENT
# ============================================================


# ============================================================
# TELEGRAM
# ============================================================

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]


# ============================================================
# KRAKEN
# ============================================================

KRAKEN_URL = "https://api.kraken.com/0/public/OHLC"


# ============================================================
# TIMEFRAMES
# ============================================================

INTERVAL_4H = 240
INTERVAL_1H = 60


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

# Recent pullback search
PULLBACK_LOOKBACK = 8

# Structure breakout
STRUCTURE_LOOKBACK = 5

# Price proximity to EMA
EMA_PULLBACK_ATR = 1.0

# Risk
SL_ATR = 1.5

# Reward
TP_R_MULTIPLE = 2.0

# Minimum RR
MIN_RR = 1.5


# ============================================================
# STATE
# ============================================================

STATE_FILE = "state.json"


def load_state():

    if not os.path.exists(STATE_FILE):
        return {}

    try:

        with open(
            STATE_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception as e:

        print(
            f"⚠️ State load error: {e}"
        )

        return {}


def save_state(state):

    with open(
        STATE_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            state,
            f,
            indent=2
        )


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    url = (
        f"https://api.telegram.org/"
        f"bot{TOKEN}/sendMessage"
    )

    response = requests.post(
        url,
        data={
            "chat_id": CHAT_ID,
            "text": message
        },
        timeout=20
    )

    response.raise_for_status()


# ============================================================
# KRAKEN OHLC
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
            f"{symbol} Kraken error: "
            f"{data['error']}"
        )

    result = data.get("result", {})

    pair_key = next(
        (
            k
            for k in result
            if k != "last"
        ),
        None
    )

    if pair_key is None:
        raise RuntimeError(
            f"{symbol}: no candle data"
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

    # Remove currently forming candle.
    if len(candles) > 1:
        candles = candles[:-1]

    return candles


# ============================================================
# EMA
# ============================================================

def ema(values, period):

    if len(values) < period:
        return None

    result = (
        sum(values[:period])
        / period
    )

    multiplier = (
        2.0
        / (period + 1)
    )

    for value in values[period:]:

        result = (
            (value - result)
            * multiplier
            + result
        )

    return result


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
            )
        )

        trs.append(tr)

    return (
        sum(trs[-period:])
        / period
    )


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

    rs_value = (
        avg_gain
        / avg_loss
    )

    return (
        100
        - (
            100
            / (1 + rs_value)
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

        previous_high = (
            previous["high"]
        )

        previous_low = (
            previous["low"]
        )

        previous_close = (
            previous["close"]
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
            )
        )

        up_move = (
            high
            - previous_high
        )

        down_move = (
            previous_low
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
        sum(
            tr_list[:period]
        )
        / period
    )

    plus_value = (
        sum(
            plus_dm_list[:period]
        )
        / period
    )

    minus_value = (
        sum(
            minus_dm_list[:period]
        )
        / period
    )

    dx_values = []

    for i in range(
        period,
        len(tr_list)
    ):

        atr_value = (
            (
                atr_value
                * (period - 1)
                + tr_list[i]
            )
            / period
        )

        plus_value = (
            (
                plus_value
                * (period - 1)
                + plus_dm_list[i]
            )
            / period
        )

        minus_value = (
            (
                minus_value
                * (period - 1)
                + minus_dm_list[i]
            )
            / period
        )

        if atr_value == 0:
            continue

        plus_di = (
            100
            * plus_value
            / atr_value
        )

        minus_di = (
            100
            * minus_value
            / atr_value
        )

        denominator = (
            plus_di
            + minus_di
        )

        if denominator == 0:

            dx = 0.0

        else:

            dx = (
                100
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
        sum(
            dx_values[:period]
        )
        / period
    )

    for value in dx_values[period:]:

        adx_value = (
            (
                adx_value
                * (period - 1)
                + value
            )
            / period
        )

    return adx_value


# ============================================================
# 4H TREND
# ============================================================

def get_4h_direction(candles):

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

        return None, {}

    long_condition = (
        close > ema200_value
        and ema20_value > ema50_value
        and ema50_value > ema200_value
    )

    short_condition = (
        close < ema200_value
        and ema20_value < ema50_value
        and ema50_value < ema200_value
    )

    if long_condition:

        direction = "LONG"

    elif short_condition:

        direction = "SHORT"

    else:

        direction = None

    return direction, {
        "close": close,
        "ema20": ema20_value,
        "ema50": ema50_value,
        "ema200": ema200_value
    }


# ============================================================
# 1H EMA VALUES
# ============================================================

def get_1h_emas(candles):

    closes = [
        c["close"]
        for c in candles
    ]

    return (
        ema(closes, EMA20),
        ema(closes, EMA50),
        ema(closes, EMA200)
    )


# ============================================================
# PULLBACK DETECTION
# ============================================================

def detect_pullback(
    candles,
    direction,
    atr_value
):

    if atr_value is None:
        return False

    if len(candles) < (
        EMA50
        + PULLBACK_LOOKBACK
    ):
        return False

    recent = candles[
        -PULLBACK_LOOKBACK:
    ]

    closes = [
        c["close"]
        for c in candles
    ]

    # We don't require the final candle to touch EMA.
    # A recent candle during the setup window may have
    # reached the EMA zone.

    for i in range(
        len(candles)
        - PULLBACK_LOOKBACK,
        len(candles)
    ):

        partial = closes[:i + 1]

        ema20_value = ema(
            partial,
            EMA20
        )

        ema50_value = ema(
            partial,
            EMA50
        )

        if (
            ema20_value is None
            or ema50_value is None
        ):
            continue

        candle = candles[i]

        if direction == "LONG":

            distance20 = abs(
                candle["low"]
                - ema20_value
            )

            distance50 = abs(
                candle["low"]
                - ema50_value
            )

            touched = (
                candle["low"]
                <= ema20_value
                + atr_value
                * EMA_PULLBACK_ATR
                or
                candle["low"]
                <= ema50_value
                + atr_value
                * EMA_PULLBACK_ATR
            )

            if touched:

                return True

        elif direction == "SHORT":

            touched = (
                candle["high"]
                >= ema20_value
                - atr_value
                * EMA_PULLBACK_ATR
                or
                candle["high"]
                >= ema50_value
                - atr_value
                * EMA_PULLBACK_ATR
            )

            if touched:

                return True

    return False


# ============================================================
# STRUCTURE BREAK
# ============================================================

def detect_breakout(
    candles,
    direction
):

    if len(candles) < (
        STRUCTURE_LOOKBACK + 2
    ):
        return False, None

    current = candles[-1]

    previous = candles[
        -STRUCTURE_LOOKBACK - 1:-1
    ]

    if direction == "LONG":

        resistance = max(
            c["high"]
            for c in previous
        )

        # Closed candle must close above structure.

        if (
            current["close"]
            > resistance
        ):

            return True, resistance

    elif direction == "SHORT":

        support = min(
            c["low"]
            for c in previous
        )

        if (
            current["close"]
            < support
        ):

            return True, support

    return False, None


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

    if direction == "LONG":

        close_location = (
            candle["close"]
            - candle["low"]
        ) / candle_range

        return (
            candle["close"]
            > candle["open"]
            and body_ratio >= 0.45
            and close_location >= 0.60
        )

    if direction == "SHORT":

        close_location = (
            candle["high"]
            - candle["close"]
        ) / candle_range

        return (
            candle["close"]
            < candle["open"]
            and body_ratio >= 0.45
            and close_location >= 0.60
        )

    return False


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

    if direction == "SHORT":

        return (
            current["close"]
            < previous["close"]
        )

    return False


# ============================================================
# RSI CONFIRMATION
# ============================================================

def rsi_confirmation(
    rsi_value,
    direction
):

    if rsi_value is None:
        return False

    # RSI > 70 is NOT an automatic rejection.
    # We only reject extreme exhaustion.

    if direction == "LONG":

        return (
            50
            <= rsi_value
            <= 85
        )

    if direction == "SHORT":

        return (
            15
            <= rsi_value
            <= 50
        )

    return False


# ============================================================
# RISK LEVELS
# ============================================================

def calculate_risk_levels(
    candles,
    direction,
    entry,
    atr_value
):

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
            - (
                atr_value
                * SL_ATR
            )
        )

        structure_stop = (
            recent_low
            - (
                atr_value
                * 0.10
            )
        )

        # Keep the stop below entry while avoiding
        # an unnecessarily huge distance.

        sl = max(
            atr_stop,
            structure_stop
        )

        risk = (
            entry
            - sl
        )

        tp = (
            entry
            + risk
            * TP_R_MULTIPLE
        )

    else:

        atr_stop = (
            entry
            + (
                atr_value
                * SL_ATR
            )
        )

        structure_stop = (
            recent_high
            + (
                atr_value
                * 0.10
            )
        )

        sl = min(
            atr_stop,
            structure_stop
        )

        risk = (
            sl
            - entry
        )

        tp = (
            entry
            - risk
            * TP_R_MULTIPLE
        )

    return tp, sl, risk


# ============================================================
# ENTRY ANALYSIS
# ============================================================

def analyze(
    symbol,
    candles_4h,
    candles_1h
):

    direction, trend = (
        get_4h_direction(
            candles_4h
        )
    )

    print(
        f"\n===== "
        f"{symbol} 4H DIRECTION "
        f"====="
    )

    print(
        f"Closed 4H: "
        f"{trend.get('close')}"
    )

    print(
        f"EMA20: "
        f"{trend.get('ema20')}"
    )

    print(
        f"EMA50: "
        f"{trend.get('ema50')}"
    )

    print(
        f"EMA200: "
        f"{trend.get('ema200')}"
    )

    print(
        f"4H Direction: "
        f"{direction}"
    )

    print(
        "=============================="
    )

    if direction is None:

        print(
            f"{symbol}: "
            f"4H trend unclear."
        )

        return None

    current = candles_1h[-1]

    entry = current["close"]

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

        print(
            f"{symbol}: "
            f"Indicators unavailable."
        )

        return None

    # --------------------------------------------------------
    # 1H EMA
    # --------------------------------------------------------

    ema20_value, ema50_value, ema200_value = (
        get_1h_emas(
            candles_1h
        )
    )

    # --------------------------------------------------------
    # Pullback
    # --------------------------------------------------------

    pullback = detect_pullback(
        candles_1h,
        direction,
        atr_value
    )

    # --------------------------------------------------------
    # Breakout
    # --------------------------------------------------------

    breakout, structure_level = (
        detect_breakout(
            candles_1h,
            direction
        )
    )

    # --------------------------------------------------------
    # Candle
    # --------------------------------------------------------

    candle_ok = candle_confirmation(
        current,
        direction
    )

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    momentum_ok = momentum_confirmation(
        candles_1h,
        direction
    )

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    adx_ok = (
        adx_value
        >= ADX_MIN
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    rsi_ok = rsi_confirmation(
        rsi_value,
        direction
    )

    # --------------------------------------------------------
    # 1H EMA alignment
    # --------------------------------------------------------

    if direction == "LONG":

        ema_alignment = (
            entry > ema20_value
            and ema20_value
            > ema50_value
        )

    else:

        ema_alignment = (
            entry < ema20_value
            and ema20_value
            < ema50_value
        )

    print(
        f"\n===== "
        f"{symbol} 1H SETUP v7 "
        f"====="
    )

    print(
        f"Direction: "
        f"{direction}"
    )

    print(
        f"Entry: "
        f"{entry:.8f}"
    )

    print(
        f"RSI: "
        f"{rsi_value:.2f}"
    )

    print(
        f"ADX: "
        f"{adx_value:.2f}"
    )

    print(
        f"ATR: "
        f"{atr_value:.8f}"
    )

    print(
        f"EMA20: "
        f"{ema20_value:.8f}"
    )

    print(
        f"EMA50: "
        f"{ema50_value:.8f}"
    )

    print(
        f"Pullback: "
        f"{pullback}"
    )

    print(
        f"Breakout: "
        f"{breakout}"
    )

    print(
        f"ADX Confirmed: "
        f"{adx_ok}"
    )

    print(
        f"RSI Confirmed: "
        f"{rsi_ok}"
    )

    print(
        f"Momentum: "
        f"{momentum_ok}"
    )

    print(
        f"Candle Quality: "
        f"{candle_ok}"
    )

    print(
        f"EMA Alignment: "
        f"{ema_alignment}"
    )

    # ========================================================
    # SETUP A
    #
    # PULLBACK CONTINUATION
    #
    # Pullback
    # +
    # ADX
    # +
    # EMA alignment
    # +
    # candle OR momentum
    # ========================================================

    pullback_setup = (
        pullback
        and adx_ok
        and ema_alignment
        and (
            candle_ok
            or momentum_ok
        )
    )

    # ========================================================
    # SETUP B
    #
    # MOMENTUM BREAKOUT
    #
    # Breakout
    # +
    # ADX
    # +
    # candle confirmation
    #
    # EMA alignment is NOT mandatory here because a strong
    # breakout can temporarily extend away from EMA20.
    # ========================================================

    breakout_setup = (
        breakout
        and adx_ok
        and candle_ok
    )

    # ========================================================
    # FINAL SETUP
    #
    # PULLBACK OR BREAKOUT
    # ========================================================

    if pullback_setup:

        setup_type = "PULLBACK"

    elif breakout_setup:

        setup_type = "BREAKOUT"

    else:

        setup_type = None

    print(
        f"Pullback Setup: "
        f"{pullback_setup}"
    )

    print(
        f"Breakout Setup: "
        f"{breakout_setup}"
    )

    print(
        f"FINAL SETUP: "
        f"{setup_type}"
    )

    if setup_type is None:

        print(
            f"{symbol}: "
            f"No valid setup."
        )

        return None

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    tp, sl, risk = calculate_risk_levels(
        candles_1h,
        direction,
        entry,
        atr_value
    )

    if risk <= 0:

        print(
            f"{symbol}: "
            f"Invalid risk."
        )

        return None

    reward = abs(
        tp - entry
    )

    rr = (
        reward
        / risk
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
        f"R/R: "
        f"1:{rr:.2f}"
    )

    if rr < MIN_RR:

        print(
            f"{symbol}: "
            f"RR too low."
        )

        return None

    return {
        "direction": direction,
        "setup_type": setup_type,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "risk": risk,
        "rr": rr,
        "atr": atr_value,
        "adx": adx_value,
        "rsi": rsi_value,
        "pullback": pullback,
        "breakout": breakout,
        "candle_time": current["time"]
    }


# ============================================================
# TELEGRAM MESSAGE
# ============================================================

def signal_message(
    symbol,
    signal
):

    direction = (
        signal["direction"]
    )

    if direction == "LONG":

        emoji = "🟢 LONG"

        tp_percent = (
            (
                signal["tp"]
                - signal["entry"]
            )
            / signal["entry"]
            * 100
        )

        sl_percent = (
            (
                signal["entry"]
                - signal["sl"]
            )
            / signal["entry"]
            * 100
        )

        tp_text = (
            f"🎯 TP: "
            f"{signal['tp']:.8f} "
            f"(+{tp_percent:.2f}%)"
        )

        sl_text = (
            f"🛑 SL: "
            f"{signal['sl']:.8f} "
            f"(-{sl_percent:.2f}%)"
        )

    else:

        emoji = "🔴 SHORT"

        tp_percent = (
            (
                signal["entry"]
                - signal["tp"]
            )
            / signal["entry"]
            * 100
        )

        sl_percent = (
            (
                signal["sl"]
                - signal["entry"]
            )
            / signal["entry"]
            * 100
        )

        tp_text = (
            f"🎯 TP: "
            f"{signal['tp']:.8f} "
            f"(-{tp_percent:.2f}%)"
        )

        sl_text = (
            f"🛑 SL: "
            f"{signal['sl']:.8f} "
            f"(+{sl_percent:.2f}%)"
        )

    return (
        "🚨 SCORE HUNTER PRO v7 🚨\n\n"

        f"💰 {symbol}USDT\n"

        f"📊 {emoji}\n\n"

        f"🧠 SETUP: "
        f"{signal['setup_type']}\n\n"

        f"💵 Entry: "
        f"{signal['entry']:.8f}\n"

        f"{tp_text}\n"

        f"{sl_text}\n\n"

        "━━━━━━━━━━━━━━━━━━\n"

        "🧭 4H: Main trend\n"

        "⏱ 1H: Entry confirmation\n"

        f"↩️ Pullback: "
        f"{'✅' if signal['pullback'] else '➖'}\n"

        f"🚀 Breakout: "
        f"{'✅' if signal['breakout'] else '➖'}\n"

        f"📊 ADX: "
        f"{signal['adx']:.2f}\n"

        f"📈 RSI: "
        f"{signal['rsi']:.2f}\n\n"

        "🕯 Closed 1H candle only\n"

        "🚫 No mid-candle signal\n"

        "📊 Volume: Not required\n\n"

        f"🎯 TP: 2R\n"

        f"⚖️ Risk/Reward: "
        f"1:{signal['rr']:.2f}\n\n"

        "⚠️ Manage risk."
    )


# ============================================================
# POSITION MANAGEMENT
# ============================================================

def check_position(
    candles,
    position
):

    direction = (
        position["direction"]
    )

    candle = candles[-1]

    close = candle["close"]

    tp = position["tp"]
    sl = position["sl"]

    if direction == "LONG":

        if close <= sl:

            return "SL", sl

        if close >= tp:

            return "TP", tp

    else:

        if close >= sl:

            return "SL", sl

        if close <= tp:

            return "TP", tp

    return None, None


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n🟢 SCORE HUNTER PRO v7"
    )

    print(
        "🧭 4H TREND + 1H ENTRY"
    )

    print(
        "🕯 CLOSED 4H + CLOSED 1H ONLY"
    )

    print(
        "↩️ PULLBACK OR BREAKOUT"
    )

    print(
        "📊 ADX MINIMUM: "
        f"{ADX_MIN}"
    )

    print(
        "📈 RSI: CONFIRMATION ONLY"
    )

    print(
        "🚫 RSI >70 NOT automatic rejection"
    )

    print(
        "📊 VOLUME: NOT REQUIRED"
    )

    print(
        f"🛑 SL: "
        f"{SL_ATR} ATR / structure"
    )

    print(
        "🎯 TP: 2R"
    )

    print(
        f"⚖️ MINIMUM RR: "
        f"1:{MIN_RR}"
    )

    print(
        "🚫 NO OLD SCORE SYSTEM"
    )

    print(
        "🚫 NO MID-CANDLE SIGNAL"
    )

    state = load_state()

    for symbol in COINS:

        print(
            f"\n\n========== "
            f"{symbol} "
            f"=========="
        )

        try:

            candles_4h = get_ohlc(
                symbol,
                INTERVAL_4H
            )

            candles_1h = get_ohlc(
                symbol,
                INTERVAL_1H
            )

            print(
                f"{symbol}: "
                f"{len(candles_4h)} "
                f"closed 4H candles"
            )

            print(
                f"{symbol}: "
                f"{len(candles_1h)} "
                f"closed 1H candles"
            )

            if len(candles_4h) < 210:

                print(
                    f"{symbol}: "
                    f"Not enough 4H candles."
                )

                continue

            if len(candles_1h) < 80:

                print(
                    f"{symbol}: "
                    f"Not enough 1H candles."
                )

                continue

            latest_4h = (
                candles_4h[-1]["time"]
            )

            latest_1h = (
                candles_1h[-1]["time"]
            )

            print(
                f"{symbol} latest CLOSED "
                f"4H candle: "
                f"{latest_4h}"
            )

            print(
                f"{symbol} latest CLOSED "
                f"1H candle: "
                f"{latest_1h}"
            )

            coin_state = state.get(
                symbol,
                {}
            )

            last_checked = (
                coin_state.get(
                    "last_checked_1h"
                )
            )

            # ------------------------------------------------
            # Only scan once per new CLOSED 1H candle.
            # ------------------------------------------------

            if last_checked == latest_1h:

                print(
                    f"{symbol}: "
                    f"No new closed 1H candle."
                )

                continue

            coin_state[
                "last_checked_1h"
            ] = latest_1h

            # =================================================
            # EXISTING POSITION
            # =================================================

            position = (
                coin_state.get(
                    "position"
                )
            )

            if position:

                print(
                    f"{symbol}: "
                    f"Existing position: "
                    f"{position['direction']}"
                )

                result, level = (
                    check_position(
                        candles_1h,
                        position
                    )
                )

                if result is None:

                    print(
                        f"{symbol}: "
                        f"Position active."
                    )

                    state[symbol] = (
                        coin_state
                    )

                    save_state(state)

                    continue

                if result == "TP":

                    message = (
                        "🎯 SCORE HUNTER PRO v7\n\n"
                        f"💰 {symbol}USDT\n"
                        f"📊 {position['direction']}\n\n"
                        "✅ TAKE PROFIT REACHED\n\n"
                        f"Closed price: "
                        f"{level:.8f}\n"
                        "🕯 Closed 1H candle."
                    )

                else:

                    message = (
                        "🛑 SCORE HUNTER PRO v7\n\n"
                        f"💰 {symbol}USDT\n"
                        f"📊 {position['direction']}\n\n"
                        "❌ STOP LOSS REACHED\n\n"
                        f"Closed price: "
                        f"{level:.8f}\n"
                        "🕯 Closed 1H candle."
                    )

                send_telegram(
                    message
                )

                coin_state[
                    "position"
                ] = None

                state[symbol] = (
                    coin_state
                )

                save_state(state)

                continue

            # =================================================
            # NEW SIGNAL
            # =================================================

            signal = analyze(
                symbol,
                candles_4h,
                candles_1h
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

            message = signal_message(
                symbol,
                signal
            )

            send_telegram(
                message
            )

            # ------------------------------------------------
            # Save position
            # ------------------------------------------------

            coin_state[
                "position"
            ] = {

                "direction":
                    signal["direction"],

                "setup_type":
                    signal["setup_type"],

                "entry":
                    signal["entry"],

                "tp":
                    signal["tp"],

                "sl":
                    signal["sl"],

                "atr":
                    signal["atr"],

                "adx":
                    signal["adx"],

                "rsi":
                    signal["rsi"],

                "entry_candle":
                    signal["candle_time"]
            }

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
                f"NEW SIGNAL SENT"
            )

        except Exception as e:

            print(
                f"❌ {symbol} ERROR: "
                f"{type(e).__name__}: "
                f"{e}"
            )

            continue

    save_state(state)

    print(
        "\n✅ ALL COINS SCANNED"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
