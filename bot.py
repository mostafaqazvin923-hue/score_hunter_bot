import os
import json
import requests
from datetime import datetime, timezone


# ============================================================
# SCORE HUNTER PRO v6
#
# 4H TREND
# +
# 1H PULLBACK
# +
# ADX TREND STRENGTH
# +
# MARKET STRUCTURE BREAK
# +
# ATR RISK MANAGEMENT
#
# CLOSED CANDLES ONLY
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
# SYMBOLS
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
# INDICATOR SETTINGS
# ============================================================

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

ADX_PERIOD = 14
ADX_MIN = 20.0

RSI_PERIOD = 14

ATR_PERIOD = 14

# Pullback must have happened recently
PULLBACK_LOOKBACK = 8

# Market structure
STRUCTURE_LOOKBACK = 3

# SL / TP
SL_ATR_MULTIPLIER = 1.5
TP_ATR_MULTIPLIER = 2.5

# Minimum TP space
MIN_RR = 1.50


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

        print(f"⚠️ State load error: {e}")
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
            f"{symbol} Kraken error: "
            f"{data['error']}"
        )

    result = data.get("result", {})

    pair_key = next(
        (
            key
            for key in result
            if key != "last"
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

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Kraken normally returns the currently forming candle
    # as the final candle.
    #
    # Remove it.
    # --------------------------------------------------------

    if len(candles) > 1:
        candles = candles[:-1]

    return candles


# ============================================================
# EMA
# ============================================================

def calculate_ema(values, period):

    if len(values) < period:
        return None

    ema = (
        sum(values[:period])
        / period
    )

    multiplier = (
        2.0
        / (period + 1)
    )

    for value in values[period:]:

        ema = (
            (value - ema)
            * multiplier
            + ema
        )

    return ema


# ============================================================
# ATR
# ============================================================

def calculate_atr(
    candles,
    period=14
):

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

    if len(trs) < period:
        return None

    return (
        sum(trs[-period:])
        / period
    )


# ============================================================
# RSI
# ============================================================

def calculate_rsi(
    candles,
    period=14
):

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

    rs = (
        avg_gain
        / avg_loss
    )

    return (
        100
        - (
            100
            / (1 + rs)
        )
    )


# ============================================================
# ADX
# ============================================================

def calculate_adx(
    candles,
    period=14
):

    if len(candles) < (
        period * 2 + 5
    ):
        return None

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

    tr_values = []
    plus_dm = []
    minus_dm = []

    for i in range(
        1,
        len(candles)
    ):

        high = highs[i]
        low = lows[i]

        prev_high = highs[i - 1]
        prev_low = lows[i - 1]
        prev_close = closes[i - 1]

        tr = max(
            high - low,
            abs(
                high - prev_close
            ),
            abs(
                low - prev_close
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

        if (
            up_move > down_move
            and up_move > 0
        ):
            pdm = up_move
        else:
            pdm = 0.0

        if (
            down_move > up_move
            and down_move > 0
        ):
            mdm = down_move
        else:
            mdm = 0.0

        tr_values.append(tr)
        plus_dm.append(pdm)
        minus_dm.append(mdm)

    if len(tr_values) < period * 2:
        return None

    atr = (
        sum(
            tr_values[:period]
        )
        / period
    )

    plus = (
        sum(
            plus_dm[:period]
        )
        / period
    )

    minus = (
        sum(
            minus_dm[:period]
        )
        / period
    )

    dx_values = []

    for i in range(
        period,
        len(tr_values)
    ):

        atr = (
            (
                atr * (period - 1)
                + tr_values[i]
            )
            / period
        )

        plus = (
            (
                plus * (period - 1)
                + plus_dm[i]
            )
            / period
        )

        minus = (
            (
                minus * (period - 1)
                + minus_dm[i]
            )
            / period
        )

        if atr == 0:
            continue

        plus_di = (
            100
            * plus
            / atr
        )

        minus_di = (
            100
            * minus
            / atr
        )

        denominator = (
            plus_di
            + minus_di
        )

        if denominator == 0:
            dx = 0
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

    adx = (
        sum(
            dx_values[:period]
        )
        / period
    )

    for value in dx_values[period:]:

        adx = (
            (
                adx * (period - 1)
                + value
            )
            / period
        )

    return adx


# ============================================================
# 4H TREND
# ============================================================

def get_4h_trend(candles):

    closes = [
        c["close"]
        for c in candles
    ]

    ema20 = calculate_ema(
        closes,
        EMA_FAST
    )

    ema50 = calculate_ema(
        closes,
        EMA_MID
    )

    ema200 = calculate_ema(
        closes,
        EMA_SLOW
    )

    close = closes[-1]

    if (
        ema20 is None
        or ema50 is None
        or ema200 is None
    ):
        return None, {}

    # --------------------------------------------------------
    # STRONG LONG TREND
    # --------------------------------------------------------

    long_trend = (
        close > ema200
        and ema20 > ema50
        and ema50 > ema200
    )

    # --------------------------------------------------------
    # STRONG SHORT TREND
    # --------------------------------------------------------

    short_trend = (
        close < ema200
        and ema20 < ema50
        and ema50 < ema200
    )

    if long_trend:

        direction = "LONG"

    elif short_trend:

        direction = "SHORT"

    else:

        direction = None

    return direction, {
        "close": close,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200
    }


# ============================================================
# PULLBACK DETECTION
# ============================================================

def detect_pullback(
    candles,
    direction
):

    if len(candles) < (
        PULLBACK_LOOKBACK
        + EMA_MID
    ):
        return False, None

    recent = candles[
        -PULLBACK_LOOKBACK:
    ]

    closes = [
        c["close"]
        for c in candles
    ]

    ema20_series = []

    ema50_series = []

    # --------------------------------------------------------
    # Build EMA values for recent candles
    # --------------------------------------------------------

    for i in range(
        len(candles)
        - PULLBACK_LOOKBACK,
        len(candles)
    ):

        partial = closes[:i + 1]

        ema20 = calculate_ema(
            partial,
            EMA_FAST
        )

        ema50 = calculate_ema(
            partial,
            EMA_MID
        )

        ema20_series.append(
            ema20
        )

        ema50_series.append(
            ema50
        )

    # --------------------------------------------------------
    # LONG PULLBACK
    #
    # Price must have touched/reached EMA20 or EMA50
    # during the recent pullback.
    # --------------------------------------------------------

    if direction == "LONG":

        touched = False
        touch_index = None

        for i, candle in enumerate(
            recent
        ):

            ema20 = (
                ema20_series[i]
            )

            ema50 = (
                ema50_series[i]
            )

            if (
                ema20 is None
                or ema50 is None
            ):
                continue

            lower_zone = min(
                ema20,
                ema50
            )

            upper_zone = max(
                ema20,
                ema50
            )

            if (
                candle["low"]
                <= upper_zone
                and
                candle["high"]
                >= lower_zone
            ):

                touched = True
                touch_index = i

        return touched, touch_index

    # --------------------------------------------------------
    # SHORT PULLBACK
    # --------------------------------------------------------

    if direction == "SHORT":

        touched = False
        touch_index = None

        for i, candle in enumerate(
            recent
        ):

            ema20 = (
                ema20_series[i]
            )

            ema50 = (
                ema50_series[i]
            )

            if (
                ema20 is None
                or ema50 is None
            ):
                continue

            lower_zone = min(
                ema20,
                ema50
            )

            upper_zone = max(
                ema20,
                ema50
            )

            if (
                candle["high"]
                >= lower_zone
                and
                candle["low"]
                <= upper_zone
            ):

                touched = True
                touch_index = i

        return touched, touch_index

    return False, None


# ============================================================
# MARKET STRUCTURE
# ============================================================

def detect_structure_break(
    candles,
    direction
):

    if len(candles) < (
        STRUCTURE_LOOKBACK + 2
    ):
        return False, None

    current = candles[-1]

    previous = candles[
        -(STRUCTURE_LOOKBACK + 1):-1
    ]

    if direction == "LONG":

        structure_high = max(
            c["high"]
            for c in previous
        )

        # Current CLOSED candle must break
        # recent structure high.

        if (
            current["close"]
            > structure_high
        ):

            return True, structure_high

    if direction == "SHORT":

        structure_low = min(
            c["low"]
            for c in previous
        )

        if (
            current["close"]
            < structure_low
        ):

            return True, structure_low

    return False, None


# ============================================================
# CANDLE QUALITY
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

        close_position = (
            candle["close"]
            - candle["low"]
        ) / candle_range

        return (
            candle["close"]
            > candle["open"]
            and body_ratio >= 0.45
            and close_position >= 0.65
        )

    if direction == "SHORT":

        close_position = (
            candle["high"]
            - candle["close"]
        ) / candle_range

        return (
            candle["close"]
            < candle["open"]
            and body_ratio >= 0.45
            and close_position >= 0.65
        )

    return False


# ============================================================
# RSI CONFIRMATION
# ============================================================

def rsi_confirmation(
    rsi,
    direction
):

    if rsi is None:
        return False

    # --------------------------------------------------------
    # RSI is NOT used as:
    #
    # RSI > 70 = reject LONG
    #
    # Instead we want momentum to support the direction.
    # --------------------------------------------------------

    if direction == "LONG":

        return (
            50 <= rsi <= 78
        )

    if direction == "SHORT":

        return (
            22 <= rsi <= 50
        )

    return False


# ============================================================
# ATR LEVELS
# ============================================================

def calculate_levels(
    candles,
    direction,
    entry,
    atr
):

    # --------------------------------------------------------
    # Structure-based SL
    # --------------------------------------------------------

    recent = candles[-5:]

    recent_low = min(
        c["low"]
        for c in recent
    )

    recent_high = max(
        c["high"]
        for c in recent
    )

    if direction == "LONG":

        atr_sl = (
            entry
            - (
                atr
                * SL_ATR_MULTIPLIER
            )
        )

        structure_sl = (
            recent_low
            - (
                atr * 0.15
            )
        )

        # Use the tighter reasonable stop
        # while ensuring it is below entry.

        sl = max(
            atr_sl,
            structure_sl
        )

        risk = (
            entry
            - sl
        )

        tp = (
            entry
            + (
                risk
                * 2.0
            )
        )

    else:

        atr_sl = (
            entry
            + (
                atr
                * SL_ATR_MULTIPLIER
            )
        )

        structure_sl = (
            recent_high
            + (
                atr * 0.15
            )
        )

        sl = min(
            atr_sl,
            structure_sl
        )

        risk = (
            sl
            - entry
        )

        tp = (
            entry
            - (
                risk
                * 2.0
            )
        )

    return tp, sl, risk


# ============================================================
# MAIN ENTRY ANALYSIS
# ============================================================

def analyze_entry(
    symbol,
    candles_4h,
    candles_1h
):

    direction, trend = (
        get_4h_trend(
            candles_4h
        )
    )

    print(
        f"\n===== {symbol} 4H TREND ====="
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
            f"4H trend unclear"
        )

        return None

    current = candles_1h[-1]

    entry = current["close"]

    atr = calculate_atr(
        candles_1h,
        ATR_PERIOD
    )

    adx = calculate_adx(
        candles_1h,
        ADX_PERIOD
    )

    rsi = calculate_rsi(
        candles_1h,
        RSI_PERIOD
    )

    # --------------------------------------------------------
    # PULLBACK
    # --------------------------------------------------------

    pullback, pullback_index = (
        detect_pullback(
            candles_1h,
            direction
        )
    )

    # --------------------------------------------------------
    # STRUCTURE BREAK
    # --------------------------------------------------------

    structure_break, structure_level = (
        detect_structure_break(
            candles_1h,
            direction
        )
    )

    # --------------------------------------------------------
    # CANDLE CONFIRMATION
    # --------------------------------------------------------

    candle_ok = (
        candle_confirmation(
            current,
            direction
        )
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    rsi_ok = (
        rsi_confirmation(
            rsi,
            direction
        )
    )

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    adx_ok = (
        adx is not None
        and adx >= ADX_MIN
    )

    print(
        f"\n===== {symbol} 1H SETUP v6 ====="
    )

    print(
        f"Direction: {direction}"
    )

    print(
        f"Entry: {entry:.8f}"
    )

    print(
        f"RSI: "
        f"{rsi:.2f}"
        if rsi is not None
        else "RSI: None"
    )

    print(
        f"ADX: "
        f"{adx:.2f}"
        if adx is not None
        else "ADX: None"
    )

    print(
        f"ATR: "
        f"{atr:.8f}"
        if atr is not None
        else "ATR: None"
    )

    print(
        f"Pullback: "
        f"{pullback}"
    )

    print(
        f"Structure Break: "
        f"{structure_break}"
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
        f"Candle Quality: "
        f"{candle_ok}"
    )

    if (
        atr is None
        or adx is None
    ):

        print(
            f"{symbol}: "
            f"Indicators unavailable"
        )

        return None

    # ========================================================
    # IMPORTANT ENTRY LOGIC
    #
    # We require:
    #
    # 1. Pullback
    # 2. Structure break
    #
    # AND at least ONE of:
    #
    # 3. ADX confirmation
    # 4. RSI confirmation
    # 5. Candle confirmation
    #
    # This prevents a raw breakout from generating signals.
    # ========================================================

    confirmations = sum([
        bool(adx_ok),
        bool(rsi_ok),
        bool(candle_ok)
    ])

    entry_setup = (
        pullback
        and
        structure_break
        and
        confirmations >= 1
    )

    print(
        f"Secondary confirmations: "
        f"{confirmations}/3"
    )

    print(
        f"ENTRY SETUP: "
        f"{entry_setup}"
    )

    if not entry_setup:

        print(
            f"{symbol}: "
            f"No valid pullback + "
            f"structure setup."
        )

        return None

    # ========================================================
    # RISK LEVELS
    # ========================================================

    tp, sl, risk = (
        calculate_levels(
            candles_1h,
            direction,
            entry,
            atr
        )
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
        f"Entry: {entry:.8f}"
    )

    print(
        f"SL: {sl:.8f}"
    )

    print(
        f"TP: {tp:.8f}"
    )

    print(
        f"Risk/Reward: 1:{rr:.2f}"
    )

    # --------------------------------------------------------
    # Minimum RR
    # --------------------------------------------------------

    if rr < MIN_RR:

        print(
            f"{symbol}: "
            f"RR too low."
        )

        return None

    # --------------------------------------------------------
    # Return signal
    # --------------------------------------------------------

    return {
        "direction": direction,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "atr": atr,
        "adx": adx,
        "rsi": rsi,
        "pullback": pullback,
        "structure_break": structure_break,
        "structure_level": structure_level,
        "secondary_confirmations": confirmations,
        "rr": rr,
        "candle_time": current["time"]
    }


# ============================================================
# TELEGRAM ENTRY MESSAGE
# ============================================================

def create_entry_message(
    symbol,
    signal
):

    direction = (
        signal["direction"]
    )

    if direction == "LONG":

        side = "🟢 LONG"

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

        side = "🔴 SHORT"

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
        "🚨 SCORE HUNTER PRO v6 🚨\n\n"

        f"💰 {symbol}USDT\n"

        f"📊 {side}\n\n"

        "🧠 TREND + PULLBACK SETUP\n\n"

        f"💵 Entry: "
        f"{signal['entry']:.8f}\n"

        f"{tp_text}\n"

        f"{sl_text}\n\n"

        "━━━━━━━━━━━━━━━━━━\n"

        "🧭 4H: Main trend\n"

        "↩️ 1H: Pullback confirmed\n"

        "📐 Structure: Break confirmed\n"

        f"📊 ADX: "
        f"{signal['adx']:.2f}\n"

        f"📈 RSI: "
        f"{signal['rsi']:.2f}\n"

        f"🧠 Secondary confirmations: "
        f"{signal['secondary_confirmations']}/3\n\n"

        "🕯 Closed 1H candle only\n"

        "🚫 No mid-candle signal\n\n"

        "🛑 SL: Structure + ATR\n"

        "🎯 TP: 2R\n"

        f"⚖️ R/R: 1:{signal['rr']:.2f}\n\n"

        "⚠️ Manage risk."
    )


# ============================================================
# POSITION MANAGEMENT
# ============================================================

def manage_position(
    candles,
    position
):

    direction = (
        position["direction"]
    )

    latest = candles[-1]

    close = latest["close"]

    tp = position["tp"]
    sl = position["sl"]

    # --------------------------------------------------------
    # CLOSED CANDLE EXIT
    # --------------------------------------------------------

    if direction == "LONG":

        if close <= sl:

            return "SL", sl

        if close >= tp:

            return "TP", tp

    elif direction == "SHORT":

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
        "\n🟢 SCORE HUNTER PRO v6"
    )

    print(
        "🧠 TREND + PULLBACK + "
        "MARKET STRUCTURE"
    )

    print(
        "🧭 4H TREND + 1H ENTRY"
    )

    print(
        "🕯 CLOSED 4H + CLOSED 1H ONLY"
    )

    print(
        "↩️ PULLBACK REQUIRED"
    )

    print(
        "📐 MARKET STRUCTURE BREAK REQUIRED"
    )

    print(
        f"📊 ADX MINIMUM: {ADX_MIN}"
    )

    print(
        "📈 RSI: CONFIRMATION ONLY"
    )

    print(
        "📊 VOLUME: NOT REQUIRED"
    )

    print(
        f"🛑 SL: "
        f"{SL_ATR_MULTIPLIER} ATR + structure"
    )

    print(
        "🎯 TP: 2R"
    )

    print(
        "⚖️ MINIMUM R/R: "
        f"1:{MIN_RR}"
    )

    print(
        "🚫 NO SCORE SYSTEM"
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
                    f"Not enough 4H data"
                )

                continue

            if len(candles_1h) < 80:

                print(
                    f"{symbol}: "
                    f"Not enough 1H data"
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

            # =================================================
            # NEW CLOSED CANDLE CHECK
            # =================================================

            last_checked = (
                coin_state.get(
                    "last_checked_1h"
                )
            )

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
            # OPEN POSITION
            # =================================================

            position = (
                coin_state.get(
                    "position"
                )
            )

            if position:

                print(
                    f"{symbol}: "
                    f"Existing "
                    f"{position['direction']} "
                    f"position."
                )

                result, level = (
                    manage_position(
                        candles_1h,
                        position
                    )
                )

                if result is None:

                    print(
                        f"{symbol}: "
                        f"Position still active."
                    )

                    state[symbol] = (
                        coin_state
                    )

                    save_state(state)

                    continue

                if result == "TP":

                    message = (
                        "🎯 SCORE HUNTER PRO v6\n\n"
                        f"💰 {symbol}USDT\n"
                        f"📊 {position['direction']}\n\n"
                        "✅ TAKE PROFIT REACHED\n\n"
                        f"Price: {level:.8f}\n"
                        "🕯 Closed 1H candle."
                    )

                else:

                    message = (
                        "🛑 SCORE HUNTER PRO v6\n\n"
                        f"💰 {symbol}USDT\n"
                        f"📊 {position['direction']}\n\n"
                        "❌ STOP LOSS REACHED\n\n"
                        f"Price: {level:.8f}\n"
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
            # FIND NEW ENTRY
            # =================================================

            signal = analyze_entry(
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

            # =================================================
            # SEND SIGNAL
            # =================================================

            message = (
                create_entry_message(
                    symbol,
                    signal
                )
            )

            send_telegram(
                message
            )

            # =================================================
            # SAVE POSITION
            # =================================================

            coin_state[
                "position"
            ] = {
                "direction":
                    signal["direction"],

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

                "structure_level":
                    signal[
                        "structure_level"
                    ],

                "entry_candle":
                    signal[
                        "candle_time"
                    ]
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
