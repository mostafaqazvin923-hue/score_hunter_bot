import requests
import time
from datetime import datetime, timezone, timedelta


# ============================================================
# SCORE HUNTER PRO v8.5
# ROBUST LONG + SHORT BACKTEST
#
# DATA ENGINE:
#   1) BYBIT LINEAR FUTURES
#   2) OKX SWAP FALLBACK
#
# 4H TREND + 1H ENTRY
# LONG + SHORT
# BREAKOUT + STRICT REVERSAL
# NO PULLBACK
# CLOSED CANDLE ONLY
# NO LOOK-AHEAD
# ADX >= 20
# RSI CONFIRMATION
# STRONG BREAKOUT CANDLE
# REAL STRUCTURE BREAK
# 4H REGIME FILTER
# 1H EMA TRANSITION FOR REVERSAL
# SL = 1.5 ATR / STRUCTURE
# MAX SL = 3.5 ATR
# TP = 2R
# ENTRY CANDLE EXCLUDED
# TP + SL SAME CANDLE = SL
# NO OVERLAPPING POSITIONS
#
# HISTORY = 365 DAYS
# OOS = 90 DAYS
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

HISTORY_DAYS = 365
OOS_DAYS = 90

INTERVAL_1H = "60"
INTERVAL_4H = "240"

SECONDS_1H = 3600
SECONDS_4H = 14400

EMA20 = 20
EMA50 = 50
EMA200 = 200

RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14

ADX_MIN = 20.0

STRUCTURE_LOOKBACK = 5
REVERSAL_LOOKBACK = 6

SL_ATR = 1.50
STRUCTURE_BUFFER_ATR = 0.10
MAX_SL_ATR = 3.50

TP_R_MULTIPLE = 2.0
MIN_RR = 1.50

MIN_BODY_RATIO = 0.55
MIN_CLOSE_LOCATION = 0.70

# More strict than v8.1
BREAKOUT_MIN_ATR = 0.20

# ============================================================
# SYMBOLS
# ============================================================

COINS = {
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "BTC": "BTCUSDT",
    "ADA": "ADAUSDT",
    "LINK": "LINKUSDT",
    "DOGE": "DOGEUSDT",
}


# ============================================================
# API
# ============================================================

BYBIT_URL = "https://api.bybit.com/v5/market/kline"

OKX_URL = "https://www.okx.com/api/v5/market/history-candles"

SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent":
        "ScoreHunterPro-v8.5-Backtester/1.0",
    "Accept":
        "application/json"
})


# ============================================================
# TIME
# ============================================================

def utc_time(timestamp):

    return datetime.fromtimestamp(
        timestamp,
        tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M")


def now_timestamp():

    return int(
        datetime.now(
            timezone.utc
        ).timestamp()
    )


# ============================================================
# REQUEST
# ============================================================

def safe_get(url, params):

    last_error = None

    for attempt in range(3):

        try:

            response = SESSION.get(
                url,
                params=params,
                timeout=30
            )

            response.raise_for_status()

            return response.json()

        except Exception as exc:

            last_error = exc

            if attempt < 2:
                time.sleep(1.5)

    raise last_error


# ============================================================
# BYBIT DATA
# ============================================================

def get_bybit_ohlc(
    symbol,
    interval,
    start_ts,
    end_ts
):

    candles = {}

    cursor = start_ts

    while cursor < end_ts:

        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "start": cursor * 1000,
            "end": end_ts * 1000,
            "limit": 1000
        }

        data = safe_get(
            BYBIT_URL,
            params
        )

        if data.get("retCode") != 0:

            raise RuntimeError(
                f"Bybit error: "
                f"{data.get('retMsg')}"
            )

        result = data.get(
            "result",
            {}
        )

        rows = result.get(
            "list",
            []
        )

        if not rows:
            break

        for row in rows:

            ts = int(
                int(row[0])
                / 1000
            )

            if (
                ts >= start_ts
                and ts <= end_ts
            ):

                candles[ts] = {
                    "time": ts,
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5])
                }

        timestamps = [
            int(row[0]) // 1000
            for row in rows
        ]

        oldest = min(
            timestamps
        )

        newest = max(
            timestamps
        )

        # Move forward.
        if newest >= cursor:
            cursor = newest + (
                SECONDS_1H
                if interval == INTERVAL_1H
                else SECONDS_4H
            )
        else:
            break

        # Safety.
        if oldest < start_ts:
            break

        if len(rows) < 1000:
            break

    result = list(
        candles.values()
    )

    result.sort(
        key=lambda x: x["time"]
    )

    return result


# ============================================================
# OKX DATA
# ============================================================

def get_okx_ohlc(
    symbol,
    interval,
    start_ts,
    end_ts
):

    inst_id = (
        symbol.replace(
            "USDT",
            "-USDT-SWAP"
        )
    )

    bar = (
        "1H"
        if interval == INTERVAL_1H
        else "4H"
    )

    candles = {}

    # OKX history endpoint works backwards.
    before = end_ts * 1000

    safety = 0

    while before > start_ts * 1000:

        safety += 1

        if safety > 100:

            break

        params = {
            "instId": inst_id,
            "bar": bar,
            "limit": 300,
            "after": before
        }

        data = safe_get(
            OKX_URL,
            params
        )

        if data.get("code") != "0":

            raise RuntimeError(
                f"OKX error: "
                f"{data.get('msg')}"
            )

        rows = data.get(
            "data",
            []
        )

        if not rows:
            break

        for row in rows:

            ts = int(
                int(row[0])
                / 1000
            )

            if (
                start_ts
                <= ts
                <= end_ts
            ):

                candles[ts] = {
                    "time": ts,
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5])
                }

        oldest = min(
            int(row[0])
            for row in rows
        )

        before = oldest - 1

        time.sleep(0.15)

    result = list(
        candles.values()
    )

    result.sort(
        key=lambda x: x["time"]
    )

    return result


# ============================================================
# ROBUST DATA LOADER
# ============================================================

def get_ohlc(
    symbol,
    interval,
    start_ts,
    end_ts
):

    errors = []

    # --------------------------------------------------------
    # BYBIT
    # --------------------------------------------------------

    try:

        data = get_bybit_ohlc(
            symbol,
            interval,
            start_ts,
            end_ts
        )

        if len(data) > 100:

            return data, "BYBIT"

        errors.append(
            "Bybit returned insufficient candles."
        )

    except Exception as exc:

        errors.append(
            f"Bybit: {exc}"
        )

    # --------------------------------------------------------
    # OKX FALLBACK
    # --------------------------------------------------------

    try:

        data = get_okx_ohlc(
            symbol,
            interval,
            start_ts,
            end_ts
        )

        if len(data) > 100:

            return data, "OKX"

        errors.append(
            "OKX returned insufficient candles."
        )

    except Exception as exc:

        errors.append(
            f"OKX: {exc}"
        )

    raise RuntimeError(
        " | ".join(errors)
    )


# ============================================================
# REMOVE FORMING CANDLE
# ============================================================

def remove_forming_candle(
    candles,
    interval_seconds
):

    if not candles:
        return candles

    current_ts = now_timestamp()

    result = []

    for candle in candles:

        candle_close = (
            candle["time"]
            + interval_seconds
        )

        if candle_close <= current_ts:
            result.append(candle)

    return result


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

def atr(
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

    value = (
        sum(trs[:period])
        / period
    )

    for tr in trs[period:]:

        value = (
            value * (period - 1)
            + tr
        ) / period

    return value


# ============================================================
# RSI
# ============================================================

def rsi(
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
            avg_gain
            * (period - 1)
            + gains[i]
        ) / period

        avg_loss = (
            avg_loss
            * (period - 1)
            + losses[i]
        ) / period

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

def adx(
    candles,
    period=14
):

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
            abs(high - prev_close),
            abs(low - prev_close)
        )

        up_move = (
            high - prev_high
        )

        down_move = (
            prev_low - low
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
        plus_dm_list.append(
            plus_dm
        )
        minus_dm_list.append(
            minus_dm
        )

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
            atr_value * (period - 1)
            + tr_list[i]
        ) / period

        plus_value = (
            plus_value * (period - 1)
            + plus_dm_list[i]
        ) / period

        minus_value = (
            minus_value * (period - 1)
            + minus_dm_list[i]
        ) / period

        if atr_value <= 0:
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

        if denominator <= 0:
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
        sum(dx_values[:period])
        / period
    )

    for value in dx_values[period:]:

        adx_value = (
            adx_value * (period - 1)
            + value
        ) / period

    return adx_value


# ============================================================
# 4H REGIME
# ============================================================

def get_4h_regime(
    candles
):

    if len(candles) < EMA200:
        return None

    closes = [
        c["close"]
        for c in candles
    ]

    close = closes[-1]

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
        return None

    if (
        close > e200
        and e20 > e50
        and e50 > e200
    ):

        return "LONG"

    if (
        close < e200
        and e20 < e50
        and e50 < e200
    ):

        return "SHORT"

    return None


# ============================================================
# CLOSED 4H AVAILABLE AT ENTRY
# ============================================================

def get_closed_4h_for_entry(
    candles_4h,
    entry_candle
):

    entry_close = (
        entry_candle["time"]
        + SECONDS_1H
    )

    return [
        candle
        for candle in candles_4h
        if (
            candle["time"]
            + SECONDS_4H
            <= entry_close
        )
    ]


# ============================================================
# EMA ALIGNMENT
# ============================================================

def ema_alignment_long(
    candles
):

    if len(candles) < EMA200:
        return False

    closes = [
        c["close"]
        for c in candles
    ]

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

    return (
        e20 is not None
        and e50 is not None
        and e200 is not None
        and closes[-1] > e200
        and e20 > e50
        and e50 > e200
    )


def ema_alignment_short(
    candles
):

    if len(candles) < EMA200:
        return False

    closes = [
        c["close"]
        for c in candles
    ]

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

    return (
        e20 is not None
        and e50 is not None
        and e200 is not None
        and closes[-1] < e200
        and e20 < e50
        and e50 < e200
    )


# ============================================================
# BREAKOUT LONG
# ============================================================

def detect_breakout_long(
    candles,
    atr_value
):

    if len(candles) < (
        STRUCTURE_LOOKBACK + 2
    ):
        return False, None

    current = candles[-1]

    previous = candles[
        -STRUCTURE_LOOKBACK - 1:-1
    ]

    resistance = max(
        c["high"]
        for c in previous
    )

    if current["close"] <= resistance:
        return False, None

    candle_range = (
        current["high"]
        - current["low"]
    )

    if candle_range <= 0:
        return False, None

    body = abs(
        current["close"]
        - current["open"]
    )

    body_ratio = (
        body
        / candle_range
    )

    close_location = (
        current["close"]
        - current["low"]
    ) / candle_range

    if current["close"] <= current["open"]:
        return False, None

    if body_ratio < MIN_BODY_RATIO:
        return False, None

    if close_location < MIN_CLOSE_LOCATION:
        return False, None

    # Real break distance.
    if (
        current["close"]
        - resistance
    ) < (
        atr_value
        * BREAKOUT_MIN_ATR
    ):
        return False, None

    if (
        current["close"]
        <= candles[-2]["close"]
    ):
        return False, None

    return True, resistance


# ============================================================
# BREAKOUT SHORT
# ============================================================

def detect_breakout_short(
    candles,
    atr_value
):

    if len(candles) < (
        STRUCTURE_LOOKBACK + 2
    ):
        return False, None

    current = candles[-1]

    previous = candles[
        -STRUCTURE_LOOKBACK - 1:-1
    ]

    support = min(
        c["low"]
        for c in previous
    )

    if current["close"] >= support:
        return False, None

    candle_range = (
        current["high"]
        - current["low"]
    )

    if candle_range <= 0:
        return False, None

    body = abs(
        current["close"]
        - current["open"]
    )

    body_ratio = (
        body
        / candle_range
    )

    close_location = (
        current["high"]
        - current["close"]
    ) / candle_range

    if current["close"] >= current["open"]:
        return False, None

    if body_ratio < MIN_BODY_RATIO:
        return False, None

    if close_location < MIN_CLOSE_LOCATION:
        return False, None

    if (
        support
        - current["close"]
    ) < (
        atr_value
        * BREAKOUT_MIN_ATR
    ):
        return False, None

    if (
        current["close"]
        >= candles[-2]["close"]
    ):
        return False, None

    return True, support


# ============================================================
# REVERSAL LONG
# ============================================================

def detect_reversal_long(
    candles,
    trend_direction,
    adx_value,
    rsi_value
):

    if trend_direction != "SHORT":
        return None

    required = max(
        60,
        REVERSAL_LOOKBACK + 12
    )

    if len(candles) < required:
        return None

    current = candles[-1]

    previous = candles[
        -REVERSAL_LOOKBACK - 1:-1
    ]

    resistance = max(
        c["high"]
        for c in previous
    )

    if current["close"] <= resistance:
        return None

    if current["close"] <= current["open"]:
        return None

    candle_range = (
        current["high"]
        - current["low"]
    )

    if candle_range <= 0:
        return None

    body_ratio = (
        abs(
            current["close"]
            - current["open"]
        )
        / candle_range
    )

    if body_ratio < MIN_BODY_RATIO:
        return None

    close_location = (
        current["close"]
        - current["low"]
    ) / candle_range

    if close_location < MIN_CLOSE_LOCATION:
        return None

    closes = [
        c["close"]
        for c in candles
    ]

    e20 = ema(
        closes,
        EMA20
    )

    e50 = ema(
        closes,
        EMA50
    )

    if (
        e20 is None
        or e50 is None
    ):
        return None

    if e20 <= e50:
        return None

    if current["close"] <= e50:
        return None

    if adx_value < ADX_MIN:
        return None

    if rsi_value < 50:
        return None

    if (
        candles[-2]["close"]
        <= candles[-2]["open"]
    ):
        return None

    return {
        "structure_level":
            resistance
    }


# ============================================================
# REVERSAL SHORT
# ============================================================

def detect_reversal_short(
    candles,
    trend_direction,
    adx_value,
    rsi_value
):

    if trend_direction != "LONG":
        return None

    required = max(
        60,
        REVERSAL_LOOKBACK + 12
    )

    if len(candles) < required:
        return None

    current = candles[-1]

    previous = candles[
        -REVERSAL_LOOKBACK - 1:-1
    ]

    support = min(
        c["low"]
        for c in previous
    )

    if current["close"] >= support:
        return None

    if current["close"] >= current["open"]:
        return None

    candle_range = (
        current["high"]
        - current["low"]
    )

    if candle_range <= 0:
        return None

    body_ratio = (
        abs(
            current["close"]
            - current["open"]
        )
        / candle_range
    )

    if body_ratio < MIN_BODY_RATIO:
        return None

    close_location = (
        current["high"]
        - current["close"]
    ) / candle_range

    if close_location < MIN_CLOSE_LOCATION:
        return None

    closes = [
        c["close"]
        for c in candles
    ]

    e20 = ema(
        closes,
        EMA20
    )

    e50 = ema(
        closes,
        EMA50
    )

    if (
        e20 is None
        or e50 is None
    ):
        return None

    if e20 >= e50:
        return None

    if current["close"] >= e50:
        return None

    if adx_value < ADX_MIN:
        return None

    if rsi_value > 50:
        return None

    if (
        candles[-2]["close"]
        >= candles[-2]["open"]
    ):
        return None

    return {
        "structure_level":
            support
    }


# ============================================================
# RSI
# ============================================================

def rsi_long_ok(value):

    return (
        value is not None
        and 50 <= value <= 85
    )


def rsi_short_ok(value):

    return (
        value is not None
        and 15 <= value <= 50
    )


# ============================================================
# LEVELS LONG
# ============================================================

def calculate_long_levels(
    candles,
    entry,
    atr_value
):

    if (
        atr_value is None
        or atr_value <= 0
    ):
        return None

    recent = candles[
        -STRUCTURE_LOOKBACK - 1:-1
    ]

    recent_low = min(
        c["low"]
        for c in recent
    )

    atr_stop = (
        entry
        - atr_value * SL_ATR
    )

    structure_stop = (
        recent_low
        - atr_value
        * STRUCTURE_BUFFER_ATR
    )

    sl = min(
        atr_stop,
        structure_stop
    )

    risk = (
        entry - sl
    )

    if risk <= 0:
        return None

    risk_atr = (
        risk / atr_value
    )

    if risk_atr > MAX_SL_ATR:
        return None

    tp = (
        entry
        + risk * TP_R_MULTIPLE
    )

    rr = (
        tp - entry
    ) / risk

    if rr < MIN_RR:
        return None

    return {
        "tp": tp,
        "sl": sl,
        "risk": risk,
        "rr": rr,
        "risk_atr": risk_atr
    }


# ============================================================
# LEVELS SHORT
# ============================================================

def calculate_short_levels(
    candles,
    entry,
    atr_value
):

    if (
        atr_value is None
        or atr_value <= 0
    ):
        return None

    recent = candles[
        -STRUCTURE_LOOKBACK - 1:-1
    ]

    recent_high = max(
        c["high"]
        for c in recent
    )

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
        sl - entry
    )

    if risk <= 0:
        return None

    risk_atr = (
        risk / atr_value
    )

    if risk_atr > MAX_SL_ATR:
        return None

    tp = (
        entry
        - risk * TP_R_MULTIPLE
    )

    rr = (
        entry - tp
    ) / risk

    if rr < MIN_RR:
        return None

    return {
        "tp": tp,
        "sl": sl,
        "risk": risk,
        "rr": rr,
        "risk_atr": risk_atr
    }


# ============================================================
# SIGNAL
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

    trend = get_4h_regime(
        candles_4h
    )

    if trend is None:
        return None

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

    if adx_value < ADX_MIN:
        return None

    # ========================================================
    # LONG
    # ========================================================

    if trend == "LONG":

        if not rsi_long_ok(
            rsi_value
        ):
            return None

        if not ema_alignment_long(
            candles_1h
        ):
            return None

        breakout, level = (
            detect_breakout_long(
                candles_1h,
                atr_value
            )
        )

        if not breakout:
            return None

        levels = (
            calculate_long_levels(
                candles_1h,
                entry,
                atr_value
            )
        )

        if levels is None:
            return None

        return {
            "direction": "LONG",
            "setup": "BREAKOUT",
            "entry_time":
                current["time"],
            "entry":
                entry,
            "tp":
                levels["tp"],
            "sl":
                levels["sl"],
            "risk":
                levels["risk"],
            "rr":
                levels["rr"],
            "risk_atr":
                levels["risk_atr"],
            "atr":
                atr_value,
            "adx":
                adx_value,
            "rsi":
                rsi_value,
            "regime":
                "LONG"
        }

    # ========================================================
    # SHORT
    # ========================================================

    if trend == "SHORT":

        if not rsi_short_ok(
            rsi_value
        ):
            return None

        if not ema_alignment_short(
            candles_1h
        ):
            return None

        breakout, level = (
            detect_breakout_short(
                candles_1h,
                atr_value
            )
        )

        if not breakout:
            return None

        levels = (
            calculate_short_levels(
                candles_1h,
                entry,
                atr_value
            )
        )

        if levels is None:
            return None

        return {
            "direction": "SHORT",
            "setup": "BREAKOUT",
            "entry_time":
                current["time"],
            "entry":
                entry,
            "tp":
                levels["tp"],
            "sl":
                levels["sl"],
            "risk":
                levels["risk"],
            "rr":
                levels["rr"],
            "risk_atr":
                levels["risk_atr"],
            "atr":
                atr_value,
            "adx":
                adx_value,
            "rsi":
                rsi_value,
            "regime":
                "SHORT"
        }

    return None


# ============================================================
# RESULT
# ============================================================

def check_trade_result(
    candles,
    entry_index,
    signal
):

    tp = signal["tp"]
    sl = signal["sl"]

    for i in range(
        entry_index + 1,
        len(candles)
    ):

        candle = candles[i]

        if signal["direction"] == "LONG":

            hit_tp = (
                candle["high"]
                >= tp
            )

            hit_sl = (
                candle["low"]
                <= sl
            )

        else:

            hit_tp = (
                candle["low"]
                <= tp
            )

            hit_sl = (
                candle["high"]
                >= sl
            )

        # Conservative:
        # same candle TP + SL = SL
        if hit_tp and hit_sl:
            return "SL", i

        if hit_sl:
            return "SL", i

        if hit_tp:
            return "TP", i

    return None, None


# ============================================================
# BACKTEST COIN
# ============================================================

def backtest_coin(
    symbol,
    candles_4h,
    candles_1h,
    oos_start_ts
):

    trades = []

    i = EMA200 + 10

    while i < len(candles_1h):

        entry_candle = candles_1h[i]

        usable_4h = (
            get_closed_4h_for_entry(
                candles_4h,
                entry_candle
            )
        )

        if len(usable_4h) < EMA200:
            i += 1
            continue

        signal = analyze_at_index(
            usable_4h,
            candles_1h[:i + 1]
        )

        if signal is None:
            i += 1
            continue

        result, exit_index = (
            check_trade_result(
                candles_1h,
                i,
                signal
            )
        )

        if result is None:
            break

        exit_candle = (
            candles_1h[
                exit_index
            ]
        )

        r_result = (
            TP_R_MULTIPLE
            if result == "TP"
            else -1.0
        )

        trades.append({

            "symbol":
                symbol,

            "setup":
                signal["setup"],

            "direction":
                signal["direction"],

            "regime":
                signal["regime"],

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
                signal["rsi"],

            "oos":
                signal["entry_time"]
                >= oos_start_ts
        })

        # No overlapping positions.
        i = exit_index + 1

    return trades


# ============================================================
# STATISTICS
# ============================================================

def calculate_stats(
    trades
):

    if not trades:

        return {
            "total": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "net_r": 0.0,
            "average_r": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
            "max_win_streak": 0,
            "max_loss_streak": 0
        }

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

    average_r = (
        net_r
        / total
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

    expectancy = average_r

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

        "average_r":
            average_r,

        "profit_factor":
            profit_factor,

        "expectancy":
            expectancy,

        "max_drawdown":
            max_drawdown,

        "max_win_streak":
            max_win,

        "max_loss_streak":
            max_loss
    }


# ============================================================
# PRINT STATS
# ============================================================

def print_stats(
    title,
    trades
):

    s = calculate_stats(
        trades
    )

    print()
    print("=" * 110)
    print(title)
    print("=" * 110)

    print(
        f"Trades             : "
        f"{s['total']}"
    )

    print(
        f"Wins               : "
        f"{s['wins']}"
    )

    print(
        f"Losses             : "
        f"{s['losses']}"
    )

    print(
        f"Win Rate           : "
        f"{s['win_rate']:.2f}%"
    )

    print(
        f"Net Result         : "
        f"{s['net_r']:+.2f}R"
    )

    print(
        f"Average R          : "
        f"{s['average_r']:+.3f}R"
    )

    pf = s["profit_factor"]

    pf_text = (
        "INF"
        if pf == float("inf")
        else f"{pf:.2f}"
    )

    print(
        f"Profit Factor      : "
        f"{pf_text}"
    )

    print(
        f"Expectancy         : "
        f"{s['expectancy']:+.3f}R"
    )

    print(
        f"Max Drawdown       : "
        f"{s['max_drawdown']:.2f}R"
    )

    print(
        f"Max Win Streak     : "
        f"{s['max_win_streak']}"
    )

    print(
        f"Max Loss Streak    : "
        f"{s['max_loss_streak']}"
    )


# ============================================================
# PRICE FORMAT
# ============================================================

def fmt_price(
    price
):

    if price >= 1000:
        return f"{price:.2f}"

    if price >= 1:
        return f"{price:.5f}"

    return f"{price:.8f}"


# ============================================================
# MAIN
# ============================================================

def main():

    now = datetime.now(
        timezone.utc
    )

    end_ts = int(
        now.timestamp()
    )

    start_dt = (
        now
        - timedelta(
            days=HISTORY_DAYS
        )
    )

    oos_dt = (
        now
        - timedelta(
            days=OOS_DAYS
        )
    )

    start_ts = int(
        start_dt.timestamp()
    )

    oos_start_ts = int(
        oos_dt.timestamp()
    )

    print()
    print("=" * 110)
    print(
        " SCORE HUNTER PRO v8.5"
    )
    print(
        " ROBUST LONG + SHORT BACKTEST"
    )
    print("=" * 110)

    print()
    print("Rules:")
    print("4H Trend + 1H Entry")
    print("LONG + SHORT")
    print("BREAKOUT + STRICT REVERSAL")
    print("NO PULLBACK")
    print("Closed 4H only")
    print("Closed 1H only")
    print("NO LOOK-AHEAD")
    print("ADX >= 20")
    print("RSI confirmation")
    print("Strong breakout candle")
    print("Real structure break")
    print("4H regime filter")
    print("1H EMA transition for reversal")
    print("SL = 1.5 ATR / Structure")
    print("Maximum SL = 3.5 ATR")
    print("TP = 2R")
    print("Entry candle excluded")
    print("TP + SL same candle = SL")
    print("No overlapping positions")

    print()
    print(
        f"History: "
        f"{HISTORY_DAYS} days"
    )

    print(
        f"OOS period: "
        f"{OOS_DAYS} days"
    )

    print(
        "Target sample: 100+ trades"
    )

    print()
    print(
        f"Backtest start: "
        f"{utc_time(start_ts)}"
    )

    print(
        f"Backtest end  : "
        f"{utc_time(end_ts)}"
    )

    print(
        f"OOS starts     : "
        f"{utc_time(oos_start_ts)}"
    )

    all_trades = []

    sources = {}

    # ========================================================
    # DOWNLOAD
    # ========================================================

    for symbol, api_symbol in COINS.items():

        print()
        print(
            f"Downloading {symbol}..."
        )

        try:

            candles_4h, source_4h = (
                get_ohlc(
                    api_symbol,
                    INTERVAL_4H,
                    start_ts,
                    end_ts
                )
            )

            candles_1h, source_1h = (
                get_ohlc(
                    api_symbol,
                    INTERVAL_1H,
                    start_ts,
                    end_ts
                )
            )

            candles_4h = (
                remove_forming_candle(
                    candles_4h,
                    SECONDS_4H
                )
            )

            candles_1h = (
                remove_forming_candle(
                    candles_1h,
                    SECONDS_1H
                )
            )

            print(
                f"{symbol}: "
                f"4H={len(candles_4h)} | "
                f"1H={len(candles_1h)} | "
                f"Source="
                f"{source_4h}/{source_1h}"
            )

            if candles_4h:

                print(
                    f"{symbol}: "
                    f"4H range = "
                    f"{utc_time(candles_4h[0]['time'])}"
                    f" -> "
                    f"{utc_time(candles_4h[-1]['time'])}"
                )

            if candles_1h:

                print(
                    f"{symbol}: "
                    f"1H range = "
                    f"{utc_time(candles_1h[0]['time'])}"
                    f" -> "
                    f"{utc_time(candles_1h[-1]['time'])}"
                )

            trades = backtest_coin(
                symbol,
                candles_4h,
                candles_1h,
                oos_start_ts
            )

            print(
                f"{symbol}: "
                f"{len(trades)} completed trades"
            )

            all_trades.extend(
                trades
            )

            sources[symbol] = (
                source_4h,
                source_1h
            )

        except Exception as exc:

            print(
                f"{symbol}: ERROR"
            )

            print(
                f"    {exc}"
            )

            print(
                "    Skipping this coin."
            )

    # ========================================================
    # CHECK DATA
    # ========================================================

    if not all_trades:

        print()
        print("=" * 110)
        print("BACKTEST ABORTED")
        print("=" * 110)

        print(
            "No completed trades were produced."
        )

        print(
            "This is a DATA problem, "
            "not a 0% win-rate result."
        )

        return

    # ========================================================
    # SORT ALL TRADES CHRONOLOGICALLY
    # ========================================================

    all_trades.sort(
        key=lambda x:
            x["entry_time"]
    )

    # ========================================================
    # OVERALL
    # ========================================================

    print_stats(
        "OVERALL RESULT",
        all_trades
    )

    # ========================================================
    # LONG
    # ========================================================

    long_trades = [
        t for t in all_trades
        if t["direction"] == "LONG"
    ]

    print_stats(
        "LONG ONLY",
        long_trades
    )

    # ========================================================
    # SHORT
    # ========================================================

    short_trades = [
        t for t in all_trades
        if t["direction"] == "SHORT"
    ]

    print_stats(
        "SHORT ONLY",
        short_trades
    )

    # ========================================================
    # IN SAMPLE
    # ========================================================

    is_trades = [
        t for t in all_trades
        if not t["oos"]
    ]

    print_stats(
        "IN-SAMPLE RESULT",
        is_trades
    )

    # ========================================================
    # OOS
    # ========================================================

    oos_trades = [
        t for t in all_trades
        if t["oos"]
    ]

    print_stats(
        "OUT-OF-SAMPLE RESULT",
        oos_trades
    )

    # ========================================================
    # BY COIN
    # ========================================================

    print()
    print("=" * 110)
    print("RESULT BY COIN")
    print("=" * 110)

    for symbol in COINS:

        coin_trades = [
            t for t in all_trades
            if t["symbol"] == symbol
        ]

        if not coin_trades:

            print(
                f"{symbol:5s} | 0 trades"
            )

            continue

        s = calculate_stats(
            coin_trades
        )

        pf = s["profit_factor"]

        pf_text = (
            "INF"
            if pf == float("inf")
            else f"{pf:.2f}"
        )

        print(
            f"{symbol:5s} | "
            f"Trades: {s['total']:3d} | "
            f"W: {s['wins']:3d} | "
            f"L: {s['losses']:3d} | "
            f"WR: {s['win_rate']:6.2f}% | "
            f"R: {s['net_r']:+7.2f} | "
            f"PF: {pf_text}"
        )

    # ========================================================
    # SETUP
    # ========================================================

    print()
    print("=" * 110)
    print("RESULT BY SETUP")
    print("=" * 110)

    for setup in [
        "BREAKOUT",
        "REVERSAL"
    ]:

        setup_trades = [
            t for t in all_trades
            if t["setup"] == setup
        ]

        if not setup_trades:

            print(
                f"{setup:10s} | 0 trades"
            )

            continue

        s = calculate_stats(
            setup_trades
        )

        pf = s["profit_factor"]

        pf_text = (
            "INF"
            if pf == float("inf")
            else f"{pf:.2f}"
        )

        print(
            f"{setup:10s} | "
            f"Trades: {s['total']:3d} | "
            f"W: {s['wins']:3d} | "
            f"L: {s['losses']:3d} | "
            f"WR: {s['win_rate']:6.2f}% | "
            f"R: {s['net_r']:+7.2f} | "
            f"PF: {pf_text}"
        )

    # ========================================================
    # 4H REGIME
    # ========================================================

    print()
    print("=" * 110)
    print("RESULT BY 4H REGIME")
    print("=" * 110)

    for regime in [
        "LONG",
        "SHORT"
    ]:

        regime_trades = [
            t for t in all_trades
            if t["regime"] == regime
        ]

        if not regime_trades:

            print(
                f"{regime:6s} | 0 trades"
            )

            continue

        s = calculate_stats(
            regime_trades
        )

        print(
            f"{regime:6s} | "
            f"Trades: {s['total']:3d} | "
            f"W: {s['wins']:3d} | "
            f"L: {s['losses']:3d} | "
            f"WR: {s['win_rate']:6.2f}% | "
            f"R: {s['net_r']:+7.2f}"
        )

    # ========================================================
    # OOS BY COIN
    # ========================================================

    print()
    print("=" * 110)
    print("OUT-OF-SAMPLE BY COIN")
    print("=" * 110)

    for symbol in COINS:

        coin_oos = [
            t for t in oos_trades
            if t["symbol"] == symbol
        ]

        if not coin_oos:

            print(
                f"{symbol:5s} | 0 trades"
            )

            continue

        s = calculate_stats(
            coin_oos
        )

        print(
            f"{symbol:5s} | "
            f"Trades: {s['total']:3d} | "
            f"W: {s['wins']:3d} | "
            f"L: {s['losses']:3d} | "
            f"WR: {s['win_rate']:6.2f}% | "
            f"R: {s['net_r']:+7.2f}"
        )

    # ========================================================
    # FULL LOG
    # ========================================================

    print()
    print("=" * 110)
    print("FULL TRADE LOG")
    print("=" * 110)

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
            f"{trade['rsi']:.1f} | "
            f"{'OOS' if trade['oos'] else 'IS'}"
        )

    # ========================================================
    # ROBUSTNESS
    # ========================================================

    print()
    print("=" * 110)
    print("ROBUSTNESS CHECK")
    print("=" * 110)

    print(
        f"Total completed trades : "
        f"{len(all_trades)}"
    )

    print(
        "Target sample          : "
        "100+"
    )

    if len(all_trades) >= 100:

        print(
            "Sample size status     : "
            "OK - 100+ trades"
        )

    else:

        print(
            "Sample size status     : "
            "NOT YET - fewer than 100 trades"
        )

    print(
        f"OOS trades              : "
        f"{len(oos_trades)}"
    )

    if len(oos_trades) >= 30:

        print(
            "OOS sample status      : "
            "ACCEPTABLE"
        )

    else:

        print(
            "OOS sample status      : "
            "WEAK - more OOS trades needed"
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Do NOT optimize parameters "
        "using the OOS results."
    )

    print(
        "OOS is reserved for judging "
        "whether the rules generalize."
    )

    print()
    print("=" * 110)
    print(
        "BACKTEST FINISHED"
    )
    print("=" * 110)


if __name__ == "__main__":
    main()
