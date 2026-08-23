import requests
import time
from datetime import datetime, timezone, timedelta


# ============================================================
# SCORE HUNTER PRO v8.6
# ROBUST LONG + SHORT BACKTEST
#
# PRIMARY DATA:
# LBank Futures
#
# FALLBACK:
# Kraken
#
# IMPORTANT:
# - No infinite loop
# - Limited retries
# - Chunked historical download
# - Data validation
# - No partial-data backtest
# - Closed candles only
# - No look-ahead
# ============================================================


# ============================================================
# CONFIG
# ============================================================

VERSION = "v8.6"

HISTORY_DAYS = 365
OOS_DAYS = 90

TARGET_TRADES = 100

REQUEST_TIMEOUT = 12
MAX_RETRIES = 3

# Do not wait forever between failed requests.
RETRY_SLEEP = 1.5

# Number of candles requested per chunk.
# 1H: 500 candles ~= 20.8 days
# 4H: 500 candles ~= 83 days
CHUNK_SIZE = 500


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
# TIMEFRAMES
# ============================================================

INTERVAL_1H = 3600
INTERVAL_4H = 14400

SECONDS_1H = 3600
SECONDS_4H = 14400


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

STRUCTURE_LOOKBACK = 5
REVERSAL_LOOKBACK = 6

SL_ATR = 1.50
STRUCTURE_BUFFER_ATR = 0.10
MAX_SL_ATR = 3.50

TP_R_MULTIPLE = 2.0
MIN_RR = 1.50

MIN_BODY_RATIO = 0.55
MIN_CLOSE_LOCATION = 0.70


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
        "Chrome/126 Safari/537.36",
    "Accept": "application/json",
    "Connection": "keep-alive"
})


# ============================================================
# DATA SOURCE CONFIG
# ============================================================

# LBank official API infrastructure has separate
# spot/contract systems.
#
# The contract host is intentionally isolated here so that
# changing the LBank API endpoint later does NOT require
# rewriting the backtester.

LBANK_CONTRACT_BASE = (
    "https://lbkperp.lbank.com"
)

# Known LBank contract API namespace used by current
# exchange integrations.
LBANK_KLINE_PATH = (
    "/cfd/openApi/v1/pub/marketData"
)

# NOTE:
# If LBank changes its public contract endpoint,
# only this function needs to be changed:
# lbank_get_klines()


# ============================================================
# KRAKEN FALLBACK
# ============================================================

KRAKEN_URL = (
    "https://api.kraken.com/0/public/OHLC"
)

KRAKEN_SYMBOLS = {
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "BTC": "XBTUSDT",
    "ADA": "ADAUSDT",
    "LINK": "LINKUSDT",
    "DOGE": "DOGEUSDT",
}


# ============================================================
# UTILITIES
# ============================================================

def utc_now():

    return datetime.now(
        timezone.utc
    )


def timestamp_ms(dt):

    return int(
        dt.timestamp() * 1000
    )


def timestamp_sec(dt):

    return int(
        dt.timestamp()
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
# HTTP GET WITH RETRIES
# ============================================================

def safe_get(
    url,
    params=None,
    label=""
):

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            response = SESSION.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 200:

                return response

            last_error = (
                f"HTTP {response.status_code}"
            )

        except Exception as exc:

            last_error = str(exc)

        if attempt < MAX_RETRIES:

            time.sleep(
                RETRY_SLEEP
            )

    raise RuntimeError(
        f"{label} failed after "
        f"{MAX_RETRIES} attempts: "
        f"{last_error}"
    )


# ============================================================
# NORMALIZE CANDLE
# ============================================================

def normalize_candle(
    timestamp,
    open_price,
    high_price,
    low_price,
    close_price,
    volume
):

    return {
        "time": int(timestamp),
        "open": float(open_price),
        "high": float(high_price),
        "low": float(low_price),
        "close": float(close_price),
        "volume": float(volume)
    }


# ============================================================
# VALIDATE CANDLES
# ============================================================

def validate_candles(
    candles,
    expected_interval,
    symbol,
    timeframe
):

    if not candles:

        raise RuntimeError(
            f"{symbol} {timeframe}: "
            f"empty candle set."
        )

    candles = sorted(
        candles,
        key=lambda x: x["time"]
    )

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

    # Validate OHLC.
    for candle in candles:

        o = candle["open"]
        h = candle["high"]
        l = candle["low"]
        c = candle["close"]

        if (
            h < l
            or h < o
            or h < c
            or l > o
            or l > c
        ):

            raise RuntimeError(
                f"{symbol} {timeframe}: "
                f"invalid OHLC."
            )

        if (
            o <= 0
            or h <= 0
            or l <= 0
            or c <= 0
        ):

            raise RuntimeError(
                f"{symbol} {timeframe}: "
                f"invalid price."
            )

    # Check major gaps.
    gaps = []

    for i in range(
        1,
        len(candles)
    ):

        delta = (
            candles[i]["time"]
            - candles[i - 1]["time"]
        )

        if delta > expected_interval * 2:

            gaps.append(
                (
                    candles[i - 1]["time"],
                    candles[i]["time"]
                )
            )

    if gaps:

        raise RuntimeError(
            f"{symbol} {timeframe}: "
            f"large candle gaps detected."
        )

    return candles


# ============================================================
# LBANK CONTRACT KLINES
# ============================================================

def parse_lbank_response(
    data,
    symbol,
    interval
):

    if not isinstance(
        data,
        dict
    ):

        raise RuntimeError(
            "LBank response is not JSON object."
        )

    # Common error fields.
    if (
        data.get("error_code")
        not in (None, 0, "0")
    ):

        raise RuntimeError(
            f"LBank error: "
            f"{data.get('error_code')}"
        )

    raw = data.get("data")

    if raw is None:

        raw = data.get("result")

    if raw is None:

        raw = data.get("rows")

    if raw is None:

        raise RuntimeError(
            "LBank response has no candle data."
        )

    candles = []

    if isinstance(
        raw,
        list
    ):

        for row in raw:

            # ------------------------------------------------
            # List format
            # ------------------------------------------------

            if isinstance(
                row,
                list
            ):

                if len(row) < 6:
                    continue

                try:

                    ts = int(
                        float(row[0])
                    )

                    # LBank-style OHLC arrays can vary.
                    #
                    # Most common:
                    # timestamp, open, high, low,
                    # close, volume
                    candles.append(
                        normalize_candle(
                            ts,
                            row[1],
                            row[2],
                            row[3],
                            row[4],
                            row[5]
                        )
                    )

                except Exception:
                    continue

            # ------------------------------------------------
            # Dict format
            # ------------------------------------------------

            elif isinstance(
                row,
                dict
            ):

                try:

                    ts = (
                        row.get("time")
                        or row.get("timestamp")
                        or row.get("ts")
                    )

                    op = (
                        row.get("open")
                        or row.get("o")
                    )

                    hi = (
                        row.get("high")
                        or row.get("h")
                    )

                    lo = (
                        row.get("low")
                        or row.get("l")
                    )

                    cl = (
                        row.get("close")
                        or row.get("c")
                    )

                    vol = (
                        row.get("volume")
                        or row.get("vol")
                        or row.get("v")
                        or 0
                    )

                    if None in (
                        ts,
                        op,
                        hi,
                        lo,
                        cl
                    ):
                        continue

                    ts = int(
                        float(ts)
                    )

                    # Convert milliseconds.
                    if ts > 10_000_000_000:
                        ts //= 1000

                    candles.append(
                        normalize_candle(
                            ts,
                            op,
                            hi,
                            lo,
                            cl,
                            vol
                        )
                    )

                except Exception:
                    continue

    if not candles:

        raise RuntimeError(
            f"{symbol}: LBank returned "
            f"unrecognized candle format."
        )

    return candles


def lbank_get_klines(
    symbol,
    interval_seconds,
    start_dt,
    end_dt
):

    # --------------------------------------------------------
    # LBank contract market API.
    #
    # We use a deliberately small page size and validate
    # every response.
    # --------------------------------------------------------

    interval_name = {
        3600: "hour1",
        14400: "hour4"
    }.get(
        interval_seconds
    )

    if interval_name is None:

        raise RuntimeError(
            "Unsupported LBank interval."
        )

    url = (
        LBANK_CONTRACT_BASE
        + LBANK_KLINE_PATH
    )

    params = {
        "symbol": COINS[symbol],
        "productGroup": "SwapU",
        "interval": interval_name,
        "startTime": timestamp_ms(
            start_dt
        ),
        "endTime": timestamp_ms(
            end_dt
        ),
        "limit": CHUNK_SIZE
    }

    response = safe_get(
        url,
        params=params,
        label=(
            f"LBank {symbol} "
            f"{interval_name}"
        )
    )

    try:

        data = response.json()

    except Exception:

        raise RuntimeError(
            f"LBank {symbol}: "
            f"invalid JSON response."
        )

    return parse_lbank_response(
        data,
        symbol,
        interval_name
    )


# ============================================================
# LBANK PAGINATED DOWNLOAD
# ============================================================

def download_lbank(
    symbol,
    interval_seconds,
    start_dt,
    end_dt
):

    candles = []

    cursor = start_dt

    max_requests = 100

    requests_used = 0

    while (
        cursor < end_dt
        and requests_used < max_requests
    ):

        requests_used += 1

        # Limit each request to avoid giant API windows.
        chunk_end = min(
            end_dt,
            cursor
            + timedelta(
                seconds=(
                    interval_seconds
                    * CHUNK_SIZE
                )
            )
        )

        batch = lbank_get_klines(
            symbol,
            interval_seconds,
            cursor,
            chunk_end
        )

        if not batch:

            break

        candles.extend(
            batch
        )

        latest = max(
            c["time"]
            for c in batch
        )

        next_cursor = (
            datetime.fromtimestamp(
                latest,
                tz=timezone.utc
            )
            + timedelta(
                seconds=interval_seconds
            )
        )

        if next_cursor <= cursor:

            break

        cursor = next_cursor

        # If the server returned less than a full page,
        # the next request will determine whether more data
        # exists.
        if len(batch) < CHUNK_SIZE:

            if cursor >= end_dt:
                break

        time.sleep(
            0.15
        )

    if requests_used >= max_requests:

        raise RuntimeError(
            f"{symbol}: LBank pagination "
            f"limit reached."
        )

    return validate_candles(
        candles,
        interval_seconds,
        symbol,
        (
            "1H"
            if interval_seconds == INTERVAL_1H
            else "4H"
        )
    )


# ============================================================
# KRAKEN FALLBACK
# ============================================================

def kraken_get_ohlc(
    symbol,
    interval,
    since
):

    params = {
        "pair":
            KRAKEN_SYMBOLS[symbol],
        "interval":
            interval,
        "since":
            since
    }

    response = safe_get(
        KRAKEN_URL,
        params=params,
        label=(
            f"Kraken {symbol} "
            f"{interval}"
        )
    )

    data = response.json()

    if data.get("error"):

        raise RuntimeError(
            f"Kraken error: "
            f"{data['error']}"
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
            f"{symbol}: "
            f"Kraken returned no candles."
        )

    pair = keys[0]

    candles = []

    for row in result[pair]:

        if len(row) < 7:
            continue

        candles.append(
            normalize_candle(
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                row[6]
            )
        )

    return candles


def download_kraken(
    symbol,
    interval_seconds,
    start_dt,
    end_dt
):

    # Kraken public OHLC has limited history.
    # We try to collect what is available.
    #
    # IMPORTANT:
    # If the required 365-day range cannot be obtained,
    # this function raises instead of silently producing
    # an incomplete backtest.

    interval_minutes = (
        interval_seconds // 60
    )

    since = timestamp_sec(
        start_dt
    )

    candles = []

    max_requests = 100

    for _ in range(
        max_requests
    ):

        batch = kraken_get_ohlc(
            symbol,
            interval_minutes,
            since
        )

        if not batch:
            break

        candles.extend(
            batch
        )

        latest = max(
            c["time"]
            for c in batch
        )

        if latest <= since:
            break

        since = (
            latest
            + interval_seconds
        )

        if latest >= timestamp_sec(
            end_dt
        ):
            break

        time.sleep(
            0.25
        )

        # Kraken's public OHLC history can be limited.
        # Stop if the API repeats the same old range.
        if len(batch) < 2:
            break

    candles = [
        c
        for c in candles
        if (
            c["time"]
            >= timestamp_sec(
                start_dt
            )
            and
            c["time"]
            <= timestamp_sec(
                end_dt
            )
        )
    ]

    candles = validate_candles(
        candles,
        interval_seconds,
        symbol,
        (
            "1H"
            if interval_seconds == INTERVAL_1H
            else "4H"
        )
    )

    required_start = timestamp_sec(
        start_dt
    )

    required_end = timestamp_sec(
        end_dt
    )

    if (
        candles[0]["time"]
        > required_start
        + interval_seconds * 10
    ):

        raise RuntimeError(
            f"{symbol}: Kraken history "
            f"is incomplete."
        )

    if (
        candles[-1]["time"]
        < required_end
        - interval_seconds * 10
    ):

        raise RuntimeError(
            f"{symbol}: Kraken data "
            f"does not reach end date."
        )

    return candles


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
    # PRIMARY: LBANK FUTURES
    # --------------------------------------------------------

    try:

        candles_4h = download_lbank(
            symbol,
            INTERVAL_4H,
            start_dt,
            end_dt
        )

        candles_1h = download_lbank(
            symbol,
            INTERVAL_1H,
            start_dt,
            end_dt
        )

        print(
            f"{symbol}: "
            f"LBank Futures OK"
        )

        print(
            f"{symbol}: "
            f"4H={len(candles_4h)} | "
            f"1H={len(candles_1h)}"
        )

        return (
            candles_4h,
            candles_1h,
            "LBANK"
        )

    except Exception as lbank_error:

        print(
            f"{symbol}: "
            f"LBank failed:"
        )

        print(
            f"    {lbank_error}"
        )

        print(
            f"{symbol}: "
            f"Trying Kraken fallback..."
        )

    # --------------------------------------------------------
    # FALLBACK: KRAKEN
    # --------------------------------------------------------

    try:

        candles_4h = download_kraken(
            symbol,
            INTERVAL_4H,
            start_dt,
            end_dt
        )

        candles_1h = download_kraken(
            symbol,
            INTERVAL_1H,
            start_dt,
            end_dt
        )

        print(
            f"{symbol}: "
            f"Kraken fallback OK"
        )

        print(
            f"{symbol}: "
            f"4H={len(candles_4h)} | "
            f"1H={len(candles_1h)}"
        )

        return (
            candles_4h,
            candles_1h,
            "KRAKEN"
        )

    except Exception as kraken_error:

        raise RuntimeError(
            f"{symbol}: "
            f"LBank and Kraken failed.\n"
            f"LBank: {lbank_error}\n"
            f"Kraken: {kraken_error}"
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

    value = (
        sum(
            values[:period]
        )
        / period
    )

    multiplier = (
        2.0
        / (period + 1)
    )

    for price in values[period:]:

        value = (
            (
                price - value
            )
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

        trs.append(
            tr
        )

    if len(trs) < period:
        return None

    value = (
        sum(
            trs[:period]
        )
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
            max(
                change,
                0.0
            )
        )

        losses.append(
            max(
                -change,
                0.0
            )
        )

    avg_gain = (
        sum(
            gains[:period]
        )
        / period
    )

    avg_loss = (
        sum(
            losses[:period]
        )
        / period
    )

    for i in range(
        period,
        len(gains)
    ):

        avg_gain = (
            (
                avg_gain * (
                    period - 1
                )
                + gains[i]
            )
            / period
        )

        avg_loss = (
            (
                avg_loss * (
                    period - 1
                )
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
            / (
                1.0
                + rs_value
            )
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

        tr_list.append(
            tr
        )

        plus_dm_list.append(
            plus_dm
        )

        minus_dm_list.append(
            minus_dm
        )

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

        dx_values.append(
            dx
        )

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
# CLOSED 4H DATA
# ============================================================

def get_closed_4h_for_entry(
    candles_4h,
    entry_candle
):

    entry_close_time = (
        entry_candle["time"]
        + SECONDS_1H
    )

    return [
        candle
        for candle in candles_4h
        if (
            candle["time"]
            + SECONDS_4H
            <= entry_close_time
        )
    ]


# ============================================================
# LONG BREAKOUT
# ============================================================

def detect_breakout_long(
    candles
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

    if (
        close_location
        < MIN_CLOSE_LOCATION
    ):
        return False, None

    if len(candles) >= 2:

        if (
            current["close"]
            <= candles[-2]["close"]
        ):
            return False, None

    return True, resistance


# ============================================================
# SHORT BREAKOUT
# ============================================================

def detect_breakout_short(
    candles
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

    if (
        close_location
        < MIN_CLOSE_LOCATION
    ):
        return False, None

    if len(candles) >= 2:

        if (
            current["close"]
            >= candles[-2]["close"]
        ):
            return False, None

    return True, support


# ============================================================
# LONG REVERSAL
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

    if (
        close_location
        < MIN_CLOSE_LOCATION
    ):
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

    if len(candles) >= 2:

        previous_candle = candles[-2]

        if (
            previous_candle["close"]
            <= previous_candle["open"]
        ):
            return None

    return {
        "structure_level":
            resistance
    }


# ============================================================
# SHORT REVERSAL
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

    if (
        close_location
        < MIN_CLOSE_LOCATION
    ):
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

    if len(candles) >= 2:

        previous_candle = candles[-2]

        if (
            previous_candle["close"]
            >= previous_candle["open"]
        ):
            return None

    return {
        "structure_level":
            support
    }


# ============================================================
# LONG EMA ALIGNMENT
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

    if (
        e20 is None
        or e50 is None
        or e200 is None
    ):
        return False

    current_close = closes[-1]

    return (
        current_close > e200
        and e20 > e50
        and e50 > e200
    )


# ============================================================
# SHORT EMA ALIGNMENT
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

    if (
        e20 is None
        or e50 is None
        or e200 is None
    ):
        return False

    current_close = closes[-1]

    return (
        current_close < e200
        and e20 < e50
        and e50 < e200
    )


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

    reward = (
        tp - entry
    )

    if reward <= 0:
        return None

    rr = (
        reward
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

    reward = (
        entry - tp
    )

    if reward <= 0:
        return None

    rr = (
        reward
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
# ANALYSIS
# ============================================================

def analyze_at_index(
    candles_4h,
    candles_1h
):

    if len(candles_1h) < (
        EMA200 + 10
    ):
        return None

    if len(candles_4h) < EMA200:
        return None

    current = candles_1h[-1]

    entry = current["close"]

    trend_direction = (
        get_4h_direction(
            candles_4h
        )
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

    # ========================================================
    # LONG
    # ========================================================

    if rsi_confirmation_long(
        rsi_value
    ):

        reversal = (
            detect_reversal_long(
                candles_1h,
                trend_direction,
                adx_value,
                rsi_value
            )
        )

        if reversal is not None:

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

        if trend_direction == "LONG":

            if ema_alignment_long(
                candles_1h
            ):

                breakout, level = (
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
                                rsi_value
                        }

    # ========================================================
    # SHORT
    # ========================================================

    if rsi_confirmation_short(
        rsi_value
    ):

        reversal = (
            detect_reversal_short(
                candles_1h,
                trend_direction,
                adx_value,
                rsi_value
            )
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
                        rsi_value
                }

        if trend_direction == "SHORT":

            if ema_alignment_short(
                candles_1h
            ):

                breakout, level = (
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

    tp = signal["tp"]
    sl = signal["sl"]

    direction = (
        signal["direction"]
    )

    for i in range(
        entry_index + 1,
        len(candles)
    ):

        candle = candles[i]

        if direction == "LONG":

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

        # Conservative assumption:
        # If both are touched inside one candle,
        # count SL first.
        if hit_tp and hit_sl:

            return (
                "SL",
                i
            )

        if hit_sl:

            return (
                "SL",
                i
            )

        if hit_tp:

            return (
                "TP",
                i
            )

    return (
        None,
        None
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

    i = EMA200 + 10

    while i < len(
        candles_1h
    ):

        entry_candle = (
            candles_1h[i]
        )

        usable_4h = (
            get_closed_4h_for_entry(
                candles_4h,
                entry_candle
            )
        )

        if len(usable_4h) < EMA200:

            i += 1
            continue

        signal = (
            analyze_at_index(
                usable_4h,
                candles_1h[
                    :i + 1
                ]
            )
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

            # Open trade at the end of data.
            # Do not count it.
            break

        if result == "TP":

            r_result = (
                TP_R_MULTIPLE
            )

        else:

            r_result = -1.0

        exit_candle = (
            candles_1h[
                exit_index
            ]
        )

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
                signal["rsi"]
        })

        # No overlapping trades.
        i = (
            exit_index
            + 1
        )

    return trades


# ============================================================
# STATS
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

    total = len(
        trades
    )

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

        profit_factor = float(
            "inf"
        )

    average_r = (
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
            average_r,

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

    stats = calculate_stats(
        trades
    )

    print()
    print(
        "=" * 110
    )

    print(title)

    print(
        "=" * 110
    )

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

    pf = (
        stats["profit_factor"]
    )

    if pf == float("inf"):

        pf_text = "INF"

    else:

        pf_text = (
            f"{pf:.2f}"
        )

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

    now = utc_now()

    start_dt = (
        now
        - timedelta(
            days=HISTORY_DAYS
        )
    )

    end_dt = now

    oos_start = (
        end_dt
        - timedelta(
            days=OOS_DAYS
        )
    )

    print()

    print(
        "=" * 110
    )

    print(
        f" SCORE HUNTER PRO {VERSION}"
    )

    print(
        " ROBUST LONG + SHORT BACKTEST"
    )

    print(
        "=" * 110
    )

    print()

    print("Rules:")

    print(
        "4H Trend + 1H Entry"
    )

    print(
        "LONG + SHORT"
    )

    print(
        "BREAKOUT + STRICT REVERSAL"
    )

    print(
        "NO PULLBACK"
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
        "ADX >= 20"
    )

    print(
        "RSI confirmation"
    )

    print(
        "Strong breakout candle"
    )

    print(
        "Real structure break"
    )

    print(
        "1H EMA transition for reversal"
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
        "Entry candle excluded"
    )

    print(
        "TP + SL same candle = SL"
    )

    print(
        "No overlapping positions"
    )

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
        f"{utc_time(timestamp_sec(start_dt))}"
    )

    print(
        f"Backtest end  : "
        f"{utc_time(timestamp_sec(end_dt))}"
    )

    print(
        f"OOS starts     : "
        f"{utc_time(timestamp_sec(oos_start))}"
    )

    all_trades = []

    failed_coins = []

    sources_used = {}

    data_ranges = {}

    # ========================================================
    # DOWNLOAD
    # ========================================================

    for symbol in COINS:

        try:

            (
                candles_4h,
                candles_1h,
                source
            ) = load_market_data(
                symbol,
                start_dt,
                end_dt
            )

            data_ranges[
                symbol
            ] = (
                candles_4h,
                candles_1h
            )

            sources_used[
                symbol
            ] = source

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

            trades = (
                backtest_coin(
                    symbol,
                    candles_4h,
                    candles_1h
                )
            )

            print(
                f"{symbol}: "
                f"{len(trades)} "
                f"completed trades"
            )

            all_trades.extend(
                trades
            )

        except Exception as exc:

            failed_coins.append(
                symbol
            )

            print()

            print(
                f"{symbol}: "
                f"FAILED"
            )

            print(
                f"    {exc}"
            )

    # ========================================================
    # SAFETY
    # ========================================================

    print()

    print(
        "=" * 110
    )

    print(
        "DATA SOURCE SUMMARY"
    )

    print(
        "=" * 110
    )

    for symbol in COINS:

        source = (
            sources_used.get(
                symbol,
                "FAILED"
            )
        )

        print(
            f"{symbol:5s} | "
            f"{source}"
        )

    # --------------------------------------------------------
    # Do NOT create a fake 0-trade backtest.
    # --------------------------------------------------------

    if not all_trades:

        print()

        print(
            "=" * 110
        )

        print(
            "BACKTEST ABORTED"
        )

        print(
            "=" * 110
        )

        print(
            "No completed trades were produced."
        )

        print(
            "This is NOT a valid 0% win-rate result."
        )

        if failed_coins:

            print(
                "Failed coins: "
                + ", ".join(
                    failed_coins
                )
            )

        print()

        print(
            "Reason: market data could not "
            "be loaded reliably."
        )

        print(
            "The program will terminate normally."
        )

        print(
            "=" * 110
        )

        return

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
        t
        for t in all_trades
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
        t
        for t in all_trades
        if t["direction"] == "SHORT"
    ]

    print_stats(
        "SHORT ONLY",
        short_trades
    )

    # ========================================================
    # IN SAMPLE / OOS
    # ========================================================

    in_sample = []

    oos = []

    oos_timestamp = timestamp_sec(
        oos_start
    )

    for trade in all_trades:

        if (
            trade["entry_time"]
            >= oos_timestamp
        ):

            oos.append(
                trade
            )

        else:

            in_sample.append(
                trade
            )

    print_stats(
        "IN-SAMPLE RESULT",
        in_sample
    )

    print_stats(
        "OUT-OF-SAMPLE RESULT",
        oos
    )

    # ========================================================
    # RESULT BY COIN
    # ========================================================

    print()

    print(
        "=" * 110
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
                f"0 trades"
            )

            continue

        s = calculate_stats(
            coin_trades
        )

        pf = (
            s["profit_factor"]
        )

        if pf == float("inf"):

            pf_text = "INF"

        else:

            pf_text = (
                f"{pf:.2f}"
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

    print(
        "=" * 110
    )

    print(
        "RESULT BY SETUP"
    )

    print(
        "=" * 110
    )

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
                f"{setup:10s} | "
                f"0 trades"
            )

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
    # LONG / SHORT SETUP COMBINATION
    # ========================================================

    print()

    print(
        "=" * 110
    )

    print(
        "RESULT BY DIRECTION + SETUP"
    )

    print(
        "=" * 110
    )

    combinations = [
        ("LONG", "BREAKOUT"),
        ("LONG", "REVERSAL"),
        ("SHORT", "BREAKOUT"),
        ("SHORT", "REVERSAL")
    ]

    for direction, setup in combinations:

        selected = [
            t
            for t in all_trades
            if (
                t["direction"]
                == direction
                and
                t["setup"]
                == setup
            )
        ]

        if not selected:

            print(
                f"{direction:5s} "
                f"{setup:9s} | "
                f"0 trades"
            )

            continue

        s = calculate_stats(
            selected
        )

        print(
            f"{direction:5s} "
            f"{setup:9s} | "
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

    print(
        "=" * 110
    )

    print(
        "OUT-OF-SAMPLE BY COIN"
    )

    print(
        "=" * 110
    )

    for symbol in COINS:

        selected = [
            t
            for t in oos
            if t["symbol"] == symbol
        ]

        if not selected:

            print(
                f"{symbol:5s} | "
                f"0 trades"
            )

            continue

        s = calculate_stats(
            selected
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
    # FULL TRADE LOG
    # ========================================================

    print()

    print(
        "=" * 110
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

    # ========================================================
    # ROBUSTNESS
    # ========================================================

    stats = calculate_stats(
        all_trades
    )

    oos_stats = calculate_stats(
        oos
    )

    print()

    print(
        "=" * 110
    )

    print(
        "ROBUSTNESS CHECK"
    )

    print(
        "=" * 110
    )

    print(
        f"Total completed trades : "
        f"{stats['total']}"
    )

    print(
        f"Target sample          : "
        f"{TARGET_TRADES}+"
    )

    if stats["total"] >= TARGET_TRADES:

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
        f"{oos_stats['total']}"
    )

    if oos_stats["total"] >= 20:

        print(
            "OOS sample status       : "
            "REASONABLE"
        )

    else:

        print(
            "OOS sample status       : "
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

    print(
        "=" * 110
    )

    print(
        "BACKTEST FINISHED"
    )

    print(
        "=" * 110
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()

        print(
            "BACKTEST STOPPED BY USER."
        )

    except Exception as exc:

        print()

        print(
            "=" * 110
        )

        print(
            "BACKTEST ERROR"
        )

        print(
            "=" * 110
        )

        print(
            str(exc)
        )

        print(
            "The program terminated normally."
        )

        print(
            "=" * 110
        )
