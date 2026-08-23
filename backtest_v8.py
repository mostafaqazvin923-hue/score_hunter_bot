import requests
import time
from datetime import datetime, timezone, timedelta


# ============================================================
# SCORE HUNTER PRO v8.7
# ROBUST LONG + SHORT BACKTEST
#
# DATA:
#   LBank -> Kraken fallback
#
# STRATEGY:
#   4H TREND + 1H ENTRY
#   LONG + SHORT
#   BREAKOUT + STRICT REVERSAL
#   NO PULLBACK
#   CLOSED CANDLE ONLY
#   NO LOOK-AHEAD
#
# VALIDATION:
#   365 DAYS HISTORY
#   90 DAYS OOS
#   IS / OOS SEPARATION
#
# RISK:
#   SL = 1.5 ATR / STRUCTURE
#   MAX SL = 3.5 ATR
#   TP = 2R
#   SAME CANDLE TP+SL = SL
#   NO OVERLAPPING POSITIONS
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

COINS = {
    "ETH": {
        "lbank": "eth_usdt",
        "kraken": "ETHUSDT"
    },
    "SOL": {
        "lbank": "sol_usdt",
        "kraken": "SOLUSDT"
    },
    "XRP": {
        "lbank": "xrp_usdt",
        "kraken": "XRPUSDT"
    },
    "BTC": {
        "lbank": "btc_usdt",
        "kraken": "XBTUSDT"
    },
    "ADA": {
        "lbank": "ada_usdt",
        "kraken": "ADAUSDT"
    },
    "LINK": {
        "lbank": "link_usdt",
        "kraken": "LINKUSDT"
    },
    "DOGE": {
        "lbank": "doge_usdt",
        "kraken": "DOGEUSDT"
    }
}


HISTORY_DAYS = 365
OOS_DAYS = 90

INTERVAL_1H = 60
INTERVAL_4H = 240

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

REQUEST_TIMEOUT = 20
MAX_RETRIES = 3

MIN_REQUIRED_1H = 250
MIN_REQUIRED_4H = 250


# ============================================================
# HTTP SESSION
# ============================================================

SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent":
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36",
    "Accept": "application/json",
    "Connection": "keep-alive"
})


# ============================================================
# TIME
# ============================================================

def utc_now():
    return datetime.now(timezone.utc)


def timestamp_ms(dt):
    return int(dt.timestamp() * 1000)


def utc_time(timestamp):
    return datetime.fromtimestamp(
        timestamp,
        tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M")


# ============================================================
# REQUEST HELPER
# ============================================================

def safe_get(
    url,
    params=None,
    retries=MAX_RETRIES
):

    last_error = None

    for attempt in range(retries):

        try:

            response = SESSION.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT
            )

            response.raise_for_status()

            return response

        except Exception as exc:

            last_error = exc

            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(
        f"API failed after {retries} attempts: "
        f"{last_error}"
    )


# ============================================================
# CANDLE NORMALIZATION
# ============================================================

def normalize_candle(
    timestamp_value,
    open_price,
    high_price,
    low_price,
    close_price,
    volume
):

    try:

        timestamp_value = float(timestamp_value)

        # Some APIs return milliseconds.
        if timestamp_value > 10_000_000_000:
            timestamp_value /= 1000.0

        return {
            "time": int(timestamp_value),
            "open": float(open_price),
            "high": float(high_price),
            "low": float(low_price),
            "close": float(close_price),
            "volume": float(volume)
        }

    except Exception:

        return None


# ============================================================
# LBANK
#
# LBank API formats can change.
# This parser accepts several common structures.
# ============================================================

LBANK_URLS = [
    "https://api.lbkex.com/v2/kline.do",
    "https://api.lbank.info/v2/kline.do"
]


def parse_lbank_response(data):

    # ---------------------------------------------
    # Format A:
    # {
    #   "data": [
    #       [time, open, high, low, close, volume]
    #   ]
    # }
    # ---------------------------------------------

    if isinstance(data, dict):

        raw = data.get("data")

        if isinstance(raw, list):
            return raw

        raw = data.get("result")

        if isinstance(raw, list):
            return raw

    # ---------------------------------------------
    # Direct list
    # ---------------------------------------------

    if isinstance(data, list):
        return data

    return None


def get_lbank_klines(
    symbol,
    interval,
    start_dt,
    end_dt
):

    pair = COINS[symbol]["lbank"]

    interval_map = {
        60: "hour1",
        240: "hour4"
    }

    lbank_interval = interval_map[interval]

    last_error = None

    for base_url in LBANK_URLS:

        try:

            params = {
                "symbol": pair,
                "type": lbank_interval,
                "size": 1000
            }

            response = safe_get(
                base_url,
                params=params
            )

            data = response.json()

            raw = parse_lbank_response(data)

            if not raw:
                raise RuntimeError(
                    "LBank returned no candle data."
                )

            candles = []

            for row in raw:

                if isinstance(row, dict):

                    t = (
                        row.get("timestamp")
                        or row.get("time")
                        or row.get("ts")
                    )

                    o = (
                        row.get("open")
                        or row.get("o")
                    )

                    h = (
                        row.get("high")
                        or row.get("h")
                    )

                    l = (
                        row.get("low")
                        or row.get("l")
                    )

                    c = (
                        row.get("close")
                        or row.get("c")
                    )

                    v = (
                        row.get("volume")
                        or row.get("vol")
                        or 0
                    )

                elif isinstance(row, (list, tuple)):

                    if len(row) < 6:
                        continue

                    t = row[0]
                    o = row[1]
                    h = row[2]
                    l = row[3]
                    c = row[4]
                    v = row[5]

                else:
                    continue

                candle = normalize_candle(
                    t,
                    o,
                    h,
                    l,
                    c,
                    v
                )

                if candle is None:
                    continue

                if (
                    candle["time"]
                    < int(start_dt.timestamp())
                ):
                    continue

                if (
                    candle["time"]
                    > int(end_dt.timestamp())
                ):
                    continue

                candles.append(candle)

            candles = deduplicate(
                candles
            )

            if len(candles) < 50:

                raise RuntimeError(
                    "LBank returned too few candles."
                )

            return candles

        except Exception as exc:

            last_error = exc

    raise RuntimeError(
        f"{symbol}: LBank failed: "
        f"{last_error}"
    )


# ============================================================
# KRAKEN FALLBACK
#
# Kraken public OHLC has a limited recent window.
# We use it only when it provides enough data.
# ============================================================

KRAKEN_URL = (
    "https://api.kraken.com/0/public/OHLC"
)


def get_kraken_klines(
    symbol,
    interval
):

    pair = COINS[symbol]["kraken"]

    response = safe_get(
        KRAKEN_URL,
        params={
            "pair": pair,
            "interval": interval
        }
    )

    data = response.json()

    if data.get("error"):
        raise RuntimeError(
            f"Kraken error: {data['error']}"
        )

    result = data.get(
        "result",
        {}
    )

    keys = [
        key
        for key in result
        if key != "last"
    ]

    if not keys:
        raise RuntimeError(
            "Kraken returned no pair."
        )

    rows = result[keys[0]]

    candles = []

    for row in rows:

        if len(row) < 7:
            continue

        candle = normalize_candle(
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[6]
        )

        if candle:
            candles.append(candle)

    return deduplicate(candles)


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate(candles):

    unique = {}

    for candle in candles:

        unique[candle["time"]] = candle

    candles = list(
        unique.values()
    )

    candles.sort(
        key=lambda x: x["time"]
    )

    return candles


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_candles(
    candles,
    interval,
    minimum
):

    if not candles:
        return False

    if len(candles) < minimum:
        return False

    expected_gap = interval * 60

    gaps = []

    for i in range(
        1,
        len(candles)
    ):

        gap = (
            candles[i]["time"]
            - candles[i - 1]["time"]
        )

        gaps.append(gap)

    # We allow some missing candles,
    # but reject completely broken data.
    if gaps:

        huge_gaps = sum(
            gap > expected_gap * 10
            for gap in gaps
        )

        if huge_gaps > len(gaps) * 0.20:
            return False

    return True


# ============================================================
# DATA LOADER
# ============================================================

def load_market_data(
    symbol,
    start_dt,
    end_dt
):

    print()
    print(
        f"Downloading {symbol}..."
    )

    # --------------------------------------------------------
    # LBank
    # --------------------------------------------------------

    try:

        candles_4h = get_lbank_klines(
            symbol,
            INTERVAL_4H,
            start_dt,
            end_dt
        )

        candles_1h = get_lbank_klines(
            symbol,
            INTERVAL_1H,
            start_dt,
            end_dt
        )

        if (
            validate_candles(
                candles_4h,
                INTERVAL_4H,
                MIN_REQUIRED_4H
            )
            and
            validate_candles(
                candles_1h,
                INTERVAL_1H,
                MIN_REQUIRED_1H
            )
        ):

            print(
                f"{symbol}: "
                f"LBank OK"
            )

            return (
                candles_4h,
                candles_1h,
                "LBANK"
            )

        print(
            f"{symbol}: "
            f"LBank data insufficient."
        )

    except Exception as exc:

        print(
            f"{symbol}: "
            f"LBank failed:"
        )

        print(
            f"    {exc}"
        )

    # --------------------------------------------------------
    # Kraken fallback
    # --------------------------------------------------------

    try:

        print(
            f"{symbol}: "
            f"Trying Kraken fallback..."
        )

        candles_4h = get_kraken_klines(
            symbol,
            INTERVAL_4H
        )

        candles_1h = get_kraken_klines(
            symbol,
            INTERVAL_1H
        )

        if (
            validate_candles(
                candles_4h,
                INTERVAL_4H,
                250
            )
            and
            validate_candles(
                candles_1h,
                INTERVAL_1H,
                250
            )
        ):

            print(
                f"{symbol}: "
                f"Kraken fallback OK"
            )

            return (
                candles_4h,
                candles_1h,
                "KRAKEN"
            )

        raise RuntimeError(
            "Kraken data insufficient."
        )

    except Exception as exc:

        print(
            f"{symbol}: "
            f"FAILED"
        )

        print(
            f"    {exc}"
        )

        return (
            None,
            None,
            "FAILED"
        )


# ============================================================
# EMA
# ============================================================

def ema(
    values,
    period
):

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
        -
        (
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

    if len(candles) < period * 2 + 5:
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
# ============================================================

def get_4h_direction(
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
# CLOSED 4H CANDLES AVAILABLE AT 1H ENTRY
# ============================================================

def get_closed_4h_for_entry(
    candles_4h,
    entry_candle
):

    entry_close_time = (
        entry_candle["time"]
        + SECONDS_1H
    )

    usable = []

    for candle in candles_4h:

        if (
            candle["time"]
            + SECONDS_4H
            <= entry_close_time
        ):

            usable.append(candle)

    return usable


# ============================================================
# STRONG BREAKOUT LONG
# ============================================================

def detect_breakout_long(
    candles
):

    if (
        len(candles)
        < STRUCTURE_LOOKBACK + 2
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

    if (
        current["close"]
        <= candles[-2]["close"]
    ):
        return False, None

    return True, resistance


# ============================================================
# STRONG BREAKOUT SHORT
# ============================================================

def detect_breakout_short(
    candles
):

    if (
        len(candles)
        < STRUCTURE_LOOKBACK + 2
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

    if e20 is None or e50 is None:
        return None

    if e20 <= e50:
        return None

    if current["close"] <= e50:
        return None

    if adx_value < ADX_MIN:
        return None

    if rsi_value < 50:
        return None

    if len(candles) >= 2:

        previous_candle = candles[-2]

        if (
            previous_candle["close"]
            <= previous_candle["open"]
        ):
            return None

    return {
        "structure_level": resistance
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

    if e20 is None or e50 is None:
        return None

    if e20 >= e50:
        return None

    if current["close"] >= e50:
        return None

    if adx_value < ADX_MIN:
        return None

    if rsi_value > 50:
        return None

    if len(candles) >= 2:

        previous_candle = candles[-2]

        if (
            previous_candle["close"]
            >= previous_candle["open"]
        ):
            return None

    return {
        "structure_level": support
    }


# ============================================================
# EMA ALIGNMENT LONG
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


# ============================================================
# EMA ALIGNMENT SHORT
# ============================================================

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
# RSI
# ============================================================

def rsi_confirmation_long(
    value
):

    return (
        value is not None
        and 50 <= value <= 85
    )


def rsi_confirmation_short(
    value
):

    return (
        value is not None
        and 15 <= value <= 50
    )


# ============================================================
# LONG LEVELS
# ============================================================

def calculate_long_levels(
    candles,
    entry,
    atr_value
):

    if atr_value is None or atr_value <= 0:
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

    if risk_atr > MAX_SL_ATR:
        return None

    tp = (
        entry
        + risk * TP_R_MULTIPLE
    )

    rr = (
        (tp - entry)
        / risk
    )

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

    if atr_value is None or atr_value <= 0:
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

    if risk_atr > MAX_SL_ATR:
        return None

    tp = (
        entry
        - risk * TP_R_MULTIPLE
    )

    rr = (
        (entry - tp)
        / risk
    )

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

    if len(candles_1h) < EMA200 + 10:
        return None

    if len(candles_4h) < EMA200:
        return None

    current = candles_1h[-1]

    entry = current["close"]

    trend_direction = get_4h_direction(
        candles_4h
    )

    if trend_direction is None:
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

    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    if trend_direction == "LONG":

        # Normal breakout
        if ema_alignment_long(
            candles_1h
        ):

            if rsi_confirmation_long(
                rsi_value
            ):

                breakout, _ = (
                    detect_breakout_long(
                        candles_1h
                    )
                )

                if breakout:

                    levels = (
                        calculate_long_levels(
                            candles_1h,
                            entry,
                            atr_value
                        )
                    )

                    if levels:

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
                                rsi_value
                        }

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    if trend_direction == "SHORT":

        if ema_alignment_short(
            candles_1h
        ):

            if rsi_confirmation_short(
                rsi_value
            ):

                breakout, _ = (
                    detect_breakout_short(
                        candles_1h
                    )
                )

                if breakout:

                    levels = (
                        calculate_short_levels(
                            candles_1h,
                            entry,
                            atr_value
                        )
                    )

                    if levels:

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
                                rsi_value
                        }

    # --------------------------------------------------------
    # REVERSAL LONG
    # --------------------------------------------------------

    reversal_long = (
        detect_reversal_long(
            candles_1h,
            trend_direction,
            adx_value,
            rsi_value
        )
    )

    if reversal_long:

        levels = (
            calculate_long_levels(
                candles_1h,
                entry,
                atr_value
            )
        )

        if levels:

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
                    rsi_value
            }

    # --------------------------------------------------------
    # REVERSAL SHORT
    # --------------------------------------------------------

    reversal_short = (
        detect_reversal_short(
            candles_1h,
            trend_direction,
            adx_value,
            rsi_value
        )
    )

    if reversal_short:

        levels = (
            calculate_short_levels(
                candles_1h,
                entry,
                atr_value
            )
        )

        if levels:

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
                    rsi_value
            }

    return None


# ============================================================
# TRADE RESULT
# ============================================================

def check_trade_result(
    candles,
    entry_index,
    signal
):

    direction = signal["direction"]

    tp = signal["tp"]
    sl = signal["sl"]

    # Entry candle excluded.
    for i in range(
        entry_index + 1,
        len(candles)
    ):

        candle = candles[i]

        if direction == "LONG":

            hit_tp = (
                candle["high"] >= tp
            )

            hit_sl = (
                candle["low"] <= sl
            )

        else:

            hit_tp = (
                candle["low"] <= tp
            )

            hit_sl = (
                candle["high"] >= sl
            )

        # Conservative assumption.
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
    oos_start
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

        signal["symbol"] = symbol

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
            candles_1h[exit_index]
        )

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

            "period":
                (
                    "OOS"
                    if signal["entry_time"]
                    >= int(
                        oos_start.timestamp()
                    )
                    else "IS"
                )
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
# PRINT STAT
# ============================================================

def print_stats(
    title,
    stats
):

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


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 110)
    print(
        " SCORE HUNTER PRO v8.7"
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
    print("1H EMA transition for reversal")
    print("SL = 1.5 ATR / Structure")
    print("Maximum SL = 3.5 ATR")
    print("TP = 2R")
    print("Entry candle excluded")
    print("TP + SL same candle = SL")
    print("No overlapping positions")

    print()
    print(
        f"History: {HISTORY_DAYS} days"
    )

    print(
        f"OOS period: {OOS_DAYS} days"
    )

    print(
        "Target sample: 100+ trades"
    )

    end_dt = utc_now()

    start_dt = (
        end_dt
        - timedelta(
            days=HISTORY_DAYS
        )
    )

    oos_start = (
        end_dt
        - timedelta(
            days=OOS_DAYS
        )
    )

    print()
    print(
        "Backtest start: "
        f"{utc_time(start_dt.timestamp())}"
    )

    print(
        "Backtest end  : "
        f"{utc_time(end_dt.timestamp())}"
    )

    print(
        "OOS starts     : "
        f"{utc_time(oos_start.timestamp())}"
    )

    all_trades = []

    data_status = {}

    # ========================================================
    # DOWNLOAD
    # ========================================================

    for symbol in COINS:

        candles_4h, candles_1h, source = (
            load_market_data(
                symbol,
                start_dt,
                end_dt
            )
        )

        if (
            candles_4h is None
            or candles_1h is None
        ):

            data_status[symbol] = (
                "FAILED"
            )

            continue

        # Remove currently forming candles.
        if len(candles_4h) > 1:
            candles_4h = candles_4h[:-1]

        if len(candles_1h) > 1:
            candles_1h = candles_1h[:-1]

        print(
            f"{symbol}: "
            f"4H={len(candles_4h)} | "
            f"1H={len(candles_1h)} | "
            f"SOURCE={source}"
        )

        if candles_4h:

            print(
                f"{symbol}: 4H "
                f"{utc_time(candles_4h[0]['time'])}"
                f" -> "
                f"{utc_time(candles_4h[-1]['time'])}"
            )

        if candles_1h:

            print(
                f"{symbol}: 1H "
                f"{utc_time(candles_1h[0]['time'])}"
                f" -> "
                f"{utc_time(candles_1h[-1]['time'])}"
            )

        data_status[symbol] = source

        trades = backtest_coin(
            symbol,
            candles_4h,
            candles_1h,
            oos_start
        )

        print(
            f"{symbol}: "
            f"{len(trades)} completed trades"
        )

        all_trades.extend(
            trades
        )

    # ========================================================
    # DATA SUMMARY
    # ========================================================

    print()
    print("=" * 110)
    print("DATA SOURCE SUMMARY")
    print("=" * 110)

    for symbol in COINS:

        print(
            f"{symbol:5s} | "
            f"{data_status.get(symbol, 'FAILED')}"
        )

    # ========================================================
    # OVERALL
    # ========================================================

    overall = calculate_stats(
        all_trades
    )

    print_stats(
        "OVERALL RESULT",
        overall
    )

    # ========================================================
    # LONG
    # ========================================================

    long_trades = [
        t
        for t in all_trades
        if t["direction"] == "LONG"
    ]

    long_stats = calculate_stats(
        long_trades
    )

    print_stats(
        "LONG ONLY",
        long_stats
    )

    # ========================================================
    # SHORT
    # ========================================================

    short_trades = [
        t
        for t in all_trades
        if t["direction"] == "SHORT"
    ]

    short_stats = calculate_stats(
        short_trades
    )

    print_stats(
        "SHORT ONLY",
        short_stats
    )

    # ========================================================
    # IN SAMPLE
    # ========================================================

    is_trades = [
        t
        for t in all_trades
        if t["period"] == "IS"
    ]

    is_stats = calculate_stats(
        is_trades
    )

    print_stats(
        "IN-SAMPLE RESULT",
        is_stats
    )

    # ========================================================
    # OOS
    # ========================================================

    oos_trades = [
        t
        for t in all_trades
        if t["period"] == "OOS"
    ]

    oos_stats = calculate_stats(
        oos_trades
    )

    print_stats(
        "OUT-OF-SAMPLE RESULT",
        oos_stats
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
            t
            for t in all_trades
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
    # BY SETUP
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
            t
            for t in all_trades
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
    # OOS BY COIN
    # ========================================================

    print()
    print("=" * 110)
    print("OUT-OF-SAMPLE BY COIN")
    print("=" * 110)

    for symbol in COINS:

        coin_oos = [
            t
            for t in oos_trades
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
            f"{trade['period']:3s} | "
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
            "PASS - 100+ trades"
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
            "GOOD - 30+ OOS trades"
        )

    elif len(oos_trades) >= 15:

        print(
            "OOS sample status      : "
            "MEDIUM - more OOS trades preferred"
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
        "using OOS results."
    )

    print(
        "OOS is reserved for judging "
        "whether the rules generalize."
    )

    # ========================================================
    # ABORT WARNING
    # ========================================================

    if not all_trades:

        failed = [
            symbol
            for symbol in COINS
            if data_status.get(symbol)
            == "FAILED"
        ]

        print()
        print("=" * 110)
        print("WARNING")
        print("=" * 110)

        if len(failed) == len(COINS):

            print(
                "No market data was loaded."
            )

            print(
                "This is NOT a 0% win-rate result."
            )

            print(
                "The backtest is invalid "
                "until market data is available."
            )

        else:

            print(
                "No completed trades were produced."
            )

            print(
                "Available market data was "
                "insufficient for this strategy."
            )

    print()
    print("=" * 110)
    print("BACKTEST FINISHED")
    print("=" * 110)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
