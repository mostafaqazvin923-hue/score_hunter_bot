import requests
import time
from datetime import datetime, timezone, timedelta

# ============================================================
# SCORE HUNTER PRO v8.4
# ROBUST LONG + SHORT BACKTEST
#
# DATA SOURCE:
# OKX PUBLIC MARKET API
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
#
# SL = 1.5 ATR / STRUCTURE
# MAX SL = 3.5 ATR
# TP = 2R
#
# ENTRY CANDLE EXCLUDED FROM TP/SL
# TP + SL SAME CANDLE = SL
# NO OVERLAPPING POSITIONS
#
# 365 DAYS HISTORY
# 90 DAYS OOS
# ============================================================


# ============================================================
# CONFIG
# ============================================================

OKX_URL = "https://www.okx.com/api/v5/market/history-candles"

BAR_1H = "1H"
BAR_4H = "4H"

COINS = {
    "ETH": "ETH-USDT-SWAP",
    "SOL": "SOL-USDT-SWAP",
    "XRP": "XRP-USDT-SWAP",
    "BTC": "BTC-USDT-SWAP",
    "ADA": "ADA-USDT-SWAP",
    "LINK": "LINK-USDT-SWAP",
    "DOGE": "DOGE-USDT-SWAP",
}


# ============================================================
# BACKTEST PERIOD
# ============================================================

HISTORY_DAYS = 365
OOS_DAYS = 90

TARGET_TRADES = 100

END_TIME = datetime.now(
    timezone.utc
)

START_TIME = (
    END_TIME
    - timedelta(days=HISTORY_DAYS)
)

OOS_START_TIME = (
    END_TIME
    - timedelta(days=OOS_DAYS)
)


# ============================================================
# INDICATORS
# ============================================================

EMA20 = 20
EMA50 = 50
EMA200 = 200

RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14


# ============================================================
# STRATEGY
# ============================================================

ADX_MIN = 20.0

STRUCTURE_LOOKBACK = 5
REVERSAL_LOOKBACK = 6

SL_ATR = 1.50
STRUCTURE_BUFFER_ATR = 0.10
MAX_SL_ATR = 3.50

TP_R_MULTIPLE = 2.0
MIN_RR = 1.50


# ============================================================
# BREAKOUT QUALITY
# ============================================================

MIN_BODY_RATIO = 0.55
MIN_CLOSE_LOCATION = 0.70


# ============================================================
# REVERSAL QUALITY
# ============================================================

REVERSAL_MIN_BODY_RATIO = 0.55
REVERSAL_CLOSE_LOCATION = 0.70


# ============================================================
# HTTP SESSION
# ============================================================

SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent":
        "ScoreHunterPro-v8.4-Backtester/1.0",
    "Accept":
        "application/json"
})


# ============================================================
# HELPERS
# ============================================================

def dt_to_ms(dt):
    return int(
        dt.timestamp() * 1000
    )


def ms_to_time(ms):

    return datetime.fromtimestamp(
        int(ms) / 1000,
        tz=timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M"
    )


def utc_time(timestamp):

    return datetime.fromtimestamp(
        timestamp,
        tz=timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M"
    )


def fmt_price(price):

    if price >= 1000:
        return f"{price:.2f}"

    if price >= 1:
        return f"{price:.5f}"

    return f"{price:.8f}"


# ============================================================
# OKX DATA
# ============================================================

def get_okx_candles(
    symbol,
    bar,
    start_ms,
    end_ms
):

    all_rows = []

    current_end = end_ms

    max_pages = 100

    for page in range(max_pages):

        params = {
            "instId": symbol,
            "bar": bar,
            "limit": "300",
            "before": str(current_end)
        }

        response = SESSION.get(
            OKX_URL,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        if data.get("code") != "0":

            raise RuntimeError(
                f"OKX error for {symbol} "
                f"{bar}: "
                f"{data}"
            )

        rows = data.get(
            "data",
            []
        )

        if not rows:
            break

        all_rows.extend(rows)

        oldest = min(
            int(row[0])
            for row in rows
        )

        if oldest <= start_ms:
            break

        next_end = oldest - 1

        if next_end >= current_end:
            break

        current_end = next_end

        time.sleep(0.08)

    if not all_rows:

        raise RuntimeError(
            f"{symbol} {bar}: "
            f"OKX returned no candles."
        )

    candles = []

    for row in all_rows:

        # OKX candle:
        # [timestamp, open, high, low, close,
        #  volume, volumeCurrency,
        #  volumeCurrencyQuote, confirm]

        timestamp = int(row[0])

        if timestamp < start_ms:
            continue

        if timestamp > end_ms:
            continue

        # Only completed candles.
        confirm = (
            str(row[8])
            if len(row) > 8
            else "1"
        )

        if confirm != "1":
            continue

        candles.append({
            "time": timestamp // 1000,
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5])
        })

    unique = {}

    for candle in candles:

        unique[
            candle["time"]
        ] = candle

    candles = list(
        unique.values()
    )

    candles.sort(
        key=lambda x: x["time"]
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

    value = sum(
        trs[:period]
    ) / period

    for tr in trs[period:]:

        value = (
            (
                value
                * (period - 1)
            )
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

    if avg_gain == 0:
        return 0.0

    rs = (
        avg_gain
        / avg_loss
    )

    return (
        100.0
        - (
            100.0
            / (1.0 + rs)
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
            abs(
                high - prev_close
            ),
            abs(
                low - prev_close
            )
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

    if len(tr_list) < (
        period * 2
    ):
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
# EMA VALUES
# ============================================================

def get_ema_values(candles):

    if len(candles) < EMA200:
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

    return {
        "close": closes[-1],
        "ema20": e20,
        "ema50": e50,
        "ema200": e200
    }


# ============================================================
# 4H REGIME
# ============================================================

def get_4h_regime(candles):

    values = get_ema_values(
        candles
    )

    if values is None:
        return None

    close = values["close"]
    e20 = values["ema20"]
    e50 = values["ema50"]
    e200 = values["ema200"]

    if (
        close > e200
        and e20 > e50
        and e50 > e200
    ):
        return "BULL"

    if (
        close < e200
        and e20 < e50
        and e50 < e200
    ):
        return "BEAR"

    return "NEUTRAL"


# ============================================================
# CLOSED 4H CANDLES AVAILABLE
# AT 1H ENTRY
# ============================================================

def get_closed_4h_for_entry(
    candles_4h,
    entry_candle
):

    entry_close_time = (
        entry_candle["time"]
        + 3600
    )

    result = []

    for candle in candles_4h:

        candle_close = (
            candle["time"]
            + 14400
        )

        if (
            candle_close
            <= entry_close_time
        ):
            result.append(candle)

    return result


# ============================================================
# BREAKOUT LONG
# ============================================================

def detect_breakout_long(
    candles
):

    if len(candles) < (
        STRUCTURE_LOOKBACK + 2
    ):
        return None

    current = candles[-1]

    previous = candles[
        -STRUCTURE_LOOKBACK - 1:-1
    ]

    resistance = max(
        c["high"]
        for c in previous
    )

    if current["close"] <= resistance:
        return None

    candle_range = (
        current["high"]
        - current["low"]
    )

    if candle_range <= 0:
        return None

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
        return None

    if body_ratio < MIN_BODY_RATIO:
        return None

    if (
        close_location
        < MIN_CLOSE_LOCATION
    ):
        return None

    if (
        current["close"]
        <= candles[-2]["close"]
    ):
        return None

    return {
        "structure_level":
            resistance
    }


# ============================================================
# BREAKOUT SHORT
# ============================================================

def detect_breakout_short(
    candles
):

    if len(candles) < (
        STRUCTURE_LOOKBACK + 2
    ):
        return None

    current = candles[-1]

    previous = candles[
        -STRUCTURE_LOOKBACK - 1:-1
    ]

    support = min(
        c["low"]
        for c in previous
    )

    if current["close"] >= support:
        return None

    candle_range = (
        current["high"]
        - current["low"]
    )

    if candle_range <= 0:
        return None

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
        return None

    if body_ratio < MIN_BODY_RATIO:
        return None

    if (
        close_location
        < MIN_CLOSE_LOCATION
    ):
        return None

    if (
        current["close"]
        >= candles[-2]["close"]
    ):
        return None

    return {
        "structure_level":
            support
    }


# ============================================================
# STRICT REVERSAL LONG
# ============================================================

def detect_reversal_long(
    candles,
    regime,
    adx_value,
    rsi_value
):

    if regime != "BEAR":
        return None

    required = max(
        60,
        REVERSAL_LOOKBACK + 15
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

    # Real structure break.
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

    if (
        body_ratio
        < REVERSAL_MIN_BODY_RATIO
    ):
        return None

    close_location = (
        current["close"]
        - current["low"]
    ) / candle_range

    if (
        close_location
        < REVERSAL_CLOSE_LOCATION
    ):
        return None

    values = get_ema_values(
        candles
    )

    if values is None:
        return None

    # 1H EMA transition.
    if values["ema20"] <= values["ema50"]:
        return None

    if values["close"] <= values["ema50"]:
        return None

    # Previous candle also bullish.
    previous_candle = candles[-2]

    if (
        previous_candle["close"]
        <= previous_candle["open"]
    ):
        return None

    # Momentum confirmation.
    if adx_value < ADX_MIN:
        return None

    if rsi_value < 50:
        return None

    return {
        "structure_level":
            resistance
    }


# ============================================================
# STRICT REVERSAL SHORT
# ============================================================

def detect_reversal_short(
    candles,
    regime,
    adx_value,
    rsi_value
):

    if regime != "BULL":
        return None

    required = max(
        60,
        REVERSAL_LOOKBACK + 15
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

    # Real structure break.
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

    if (
        body_ratio
        < REVERSAL_MIN_BODY_RATIO
    ):
        return None

    close_location = (
        current["high"]
        - current["close"]
    ) / candle_range

    if (
        close_location
        < REVERSAL_CLOSE_LOCATION
    ):
        return None

    values = get_ema_values(
        candles
    )

    if values is None:
        return None

    # 1H EMA transition.
    if values["ema20"] >= values["ema50"]:
        return None

    if values["close"] >= values["ema50"]:
        return None

    previous_candle = candles[-2]

    if (
        previous_candle["close"]
        >= previous_candle["open"]
    ):
        return None

    if adx_value < ADX_MIN:
        return None

    if rsi_value > 50:
        return None

    return {
        "structure_level":
            support
    }


# ============================================================
# RSI
# ============================================================

def rsi_confirmation_long(
    value
):

    if value is None:
        return False

    return (
        50 <= value <= 85
    )


def rsi_confirmation_short(
    value
):

    if value is None:
        return False

    return (
        15 <= value <= 50
    )


# ============================================================
# LONG LEVELS
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

    if not recent:
        return None

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
        risk
        / atr_value
    )

    if (
        risk_atr
        > MAX_SL_ATR
    ):
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
# SHORT LEVELS
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

    if not recent:
        return None

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
        risk
        / atr_value
    )

    if (
        risk_atr
        > MAX_SL_ATR
    ):
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
# ANALYZE
# ============================================================

def analyze_at_index(
    candles_4h,
    candles_1h
):

    if len(candles_4h) < EMA200:
        return None

    if len(candles_1h) < (
        EMA200 + 20
    ):
        return None

    current = candles_1h[-1]

    entry = current["close"]

    regime = get_4h_regime(
        candles_4h
    )

    if regime is None:
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

    if rsi_confirmation_long(
        rsi_value
    ):

        reversal = detect_reversal_long(
            candles_1h,
            regime,
            adx_value,
            rsi_value
        )

        if reversal is not None:

            levels = calculate_long_levels(
                candles_1h,
                entry,
                atr_value
            )

            if levels is not None:

                return {
                    "direction":
                        "LONG",

                    "setup":
                        "REVERSAL",

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
                        regime
                }

        # Normal breakout only with bull 4H regime.
        if regime == "BULL":

            values = get_ema_values(
                candles_1h
            )

            if values is not None:

                ema_long = (
                    values["close"]
                    > values["ema200"]
                    and values["ema20"]
                    > values["ema50"]
                    and values["ema50"]
                    > values["ema200"]
                )

                if ema_long:

                    breakout = (
                        detect_breakout_long(
                            candles_1h
                        )
                    )

                    if breakout is not None:

                        levels = (
                            calculate_long_levels(
                                candles_1h,
                                entry,
                                atr_value
                            )
                        )

                        if levels is not None:

                            return {
                                "direction":
                                    "LONG",

                                "setup":
                                    "BREAKOUT",

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
                                    regime
                            }

    # ========================================================
    # SHORT
    # ========================================================

    if rsi_confirmation_short(
        rsi_value
    ):

        reversal = detect_reversal_short(
            candles_1h,
            regime,
            adx_value,
            rsi_value
        )

        if reversal is not None:

            levels = (
                calculate_short_levels(
                    candles_1h,
                    entry,
                    atr_value
                )
            )

            if levels is not None:

                return {
                    "direction":
                        "SHORT",

                    "setup":
                        "REVERSAL",

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
                        regime
                }

        # Normal breakout only with bear 4H regime.
        if regime == "BEAR":

            values = get_ema_values(
                candles_1h
            )

            if values is not None:

                ema_short = (
                    values["close"]
                    < values["ema200"]
                    and values["ema20"]
                    < values["ema50"]
                    and values["ema50"]
                    < values["ema200"]
                )

                if ema_short:

                    breakout = (
                        detect_breakout_short(
                            candles_1h
                        )
                    )

                    if breakout is not None:

                        levels = (
                            calculate_short_levels(
                                candles_1h,
                                entry,
                                atr_value
                            )
                        )

                        if levels is not None:

                            return {
                                "direction":
                                    "SHORT",

                                "setup":
                                    "BREAKOUT",

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
                                    regime
                            }

    return None


# ============================================================
# CHECK TRADE
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
        # If both happen inside same candle,
        # count SL.
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
    candles_1h
):

    trades = []

    i = EMA200 + 20

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

        signal["symbol"] = symbol

        result, exit_index = (
            check_trade_result(
                candles_1h,
                i,
                signal
            )
        )

        if result is None:

            # Open position at end.
            # Do not count incomplete trade.
            break

        exit_candle = candles_1h[
            exit_index
        ]

        if result == "TP":
            r_result = TP_R_MULTIPLE
        else:
            r_result = -1.0

        trades.append({

            "symbol":
                symbol,

            "direction":
                signal["direction"],

            "setup":
                signal["setup"],

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
                signal["rsi"]
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

    expectancy = (
        net_r
        / total
    )

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
        "total": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "net_r": net_r,
        "average_r": average_r,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "max_drawdown": max_drawdown,
        "max_win_streak": max_win,
        "max_loss_streak": max_loss
    }


# ============================================================
# PRINT STATS
# ============================================================

def print_stats(
    title,
    trades
):

    stats = calculate_stats(
        trades
    )

    print()
    print("=" * 110)
    print(title)
    print("=" * 110)

    print(
        f"Trades             : "
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

    pf = stats["profit_factor"]

    if pf == float("inf"):
        pf_text = "INF"
    else:
        pf_text = f"{pf:.2f}"

    print(
        f"Profit Factor      : "
        f"{pf_text}"
    )

    print(
        f"Expectancy         : "
        f"{stats['expectancy']:+.3f}R"
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

    return stats


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 110)
    print(" SCORE HUNTER PRO v8.4")
    print(" ROBUST LONG + SHORT BACKTEST")
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
        f"Target sample: "
        f"{TARGET_TRADES}+ trades"
    )

    print()
    print(
        f"Backtest start: "
        f"{START_TIME.strftime('%Y-%m-%d %H:%M')}"
    )

    print(
        f"Backtest end  : "
        f"{END_TIME.strftime('%Y-%m-%d %H:%M')}"
    )

    print(
        f"OOS starts     : "
        f"{OOS_START_TIME.strftime('%Y-%m-%d %H:%M')}"
    )

    all_trades = []

    for symbol, okx_symbol in COINS.items():

        print()
        print(
            f"Downloading {symbol}..."
        )

        try:

            start_ms = dt_to_ms(
                START_TIME
            )

            end_ms = dt_to_ms(
                END_TIME
            )

            candles_4h = get_okx_candles(
                okx_symbol,
                BAR_4H,
                start_ms,
                end_ms
            )

            candles_1h = get_okx_candles(
                okx_symbol,
                BAR_1H,
                start_ms,
                end_ms
            )

            print(
                f"{symbol}: "
                f"4H={len(candles_4h)} | "
                f"1H={len(candles_1h)}"
            )

            if not candles_4h:
                raise RuntimeError(
                    "4H data empty."
                )

            if not candles_1h:
                raise RuntimeError(
                    "1H data empty."
                )

            print(
                f"{symbol}: "
                f"4H range = "
                f"{utc_time(candles_4h[0]['time'])}"
                f" -> "
                f"{utc_time(candles_4h[-1]['time'])}"
            )

            print(
                f"{symbol}: "
                f"1H range = "
                f"{utc_time(candles_1h[0]['time'])}"
                f" -> "
                f"{utc_time(candles_1h[-1]['time'])}"
            )

            # Minimum data sanity check.
            expected_1h = (
                HISTORY_DAYS * 24
            )

            expected_4h = (
                HISTORY_DAYS * 6
            )

            if len(candles_1h) < (
                expected_1h * 0.85
            ):

                raise RuntimeError(
                    f"{symbol}: "
                    f"1H data incomplete "
                    f"({len(candles_1h)} "
                    f"/ expected ~{expected_1h})"
                )

            if len(candles_4h) < (
                expected_4h * 0.85
            ):

                raise RuntimeError(
                    f"{symbol}: "
                    f"4H data incomplete "
                    f"({len(candles_4h)} "
                    f"/ expected ~{expected_4h})"
                )

            trades = backtest_coin(
                symbol,
                candles_4h,
                candles_1h
            )

            print(
                f"{symbol}: "
                f"{len(trades)} "
                f"completed trades"
            )

            all_trades.extend(
                trades
            )

        except Exception as e:

            print(
                f"{symbol}: ERROR"
            )

            print(
                f"    {e}"
            )

            print()
            print(
                "BACKTEST ABORTED."
            )

            print(
                "Reason: "
                "data could not be loaded "
                "reliably."
            )

            return

    # ========================================================
    # OVERALL
    # ========================================================

    overall = print_stats(
        "OVERALL RESULT",
        all_trades
    )

    # ========================================================
    # LONG ONLY
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
    # SHORT ONLY
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
    # IS / OOS
    # ========================================================

    oos_timestamp = (
        int(
            OOS_START_TIME.timestamp()
        )
    )

    is_trades = [
        t for t in all_trades
        if t["entry_time"]
        < oos_timestamp
    ]

    oos_trades = [
        t for t in all_trades
        if t["entry_time"]
        >= oos_timestamp
    ]

    print_stats(
        "IN-SAMPLE RESULT",
        is_trades
    )

    print_stats(
        "OUT-OF-SAMPLE RESULT",
        oos_trades
    )

    # ========================================================
    # RESULT BY COIN
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
    # RESULT BY SETUP
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
    # RESULT BY 4H REGIME
    # ========================================================

    print()
    print("=" * 110)
    print("RESULT BY 4H REGIME")
    print("=" * 110)

    for regime in [
        "BULL",
        "BEAR"
    ]:

        regime_trades = [
            t for t in all_trades
            if t["regime"] == regime
        ]

        if not regime_trades:

            print(
                f"{regime:10s} | 0 trades"
            )

            continue

        s = calculate_stats(
            regime_trades
        )

        pf = s["profit_factor"]

        pf_text = (
            "INF"
            if pf == float("inf")
            else f"{pf:.2f}"
        )

        print(
            f"{regime:10s} | "
            f"Trades: {s['total']:3d} | "
            f"W: {s['wins']:3d} | "
            f"L: {s['losses']:3d} | "
            f"WR: {s['win_rate']:6.2f}% | "
            f"R: {s['net_r']:+7.2f} | "
            f"PF: {pf_text}"
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
    # FULL TRADE LOG
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
            f"{trade['regime']:5s} | "
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

    # ========================================================
    # ROBUSTNESS CHECK
    # ========================================================

    print()
    print("=" * 110)
    print("ROBUSTNESS CHECK")
    print("=" * 110)

    total = len(all_trades)
    oos_total = len(oos_trades)

    print(
        f"Total completed trades : "
        f"{total}"
    )

    print(
        f"Target sample          : "
        f"{TARGET_TRADES}+"
    )

    if total >= TARGET_TRADES:

        print(
            "Sample size status     : "
            "PASS"
        )

    else:

        print(
            "Sample size status     : "
            "NOT YET - fewer than 100 trades"
        )

    print(
        f"OOS trades              : "
        f"{oos_total}"
    )

    if oos_total >= 30:

        print(
            "OOS sample status      : "
            "GOOD"
        )

    elif oos_total >= 15:

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
    print("BACKTEST FINISHED")
    print("=" * 110)


if __name__ == "__main__":
    main()
