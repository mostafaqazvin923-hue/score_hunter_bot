import requests
import time
from datetime import datetime, timezone, timedelta


# ============================================================
# SCORE HUNTER PRO v8.12
#
# REAL 365-DAY HISTORICAL DATA BACKTEST
#
# DATA ENGINE FULL REBUILD
#
# FIXES:
#   1. LBank time parameter = SECONDS
#   2. Correct forward pagination
#   3. Start -> End historical downloading
#   4. No backward-anchor bug
#   5. No repeated-page loop
#   6. No zero-progress loop
#   7. Maximum request protection
#   8. Strict historical coverage validation
#   9. Current candle removal
#  10. Exact candle deduplication
#  11. Correct 1H / 4H mapping
#  12. No fake Kraken 365-day fallback
#
# STRATEGY:
#   4H TREND + 1H ENTRY
#   LONG + SHORT
#   BREAKOUT + STRICT REVERSAL
#   NO PULLBACK
#   CLOSED CANDLE ONLY
#   NO LOOK-AHEAD
#
# SL = 1.5 ATR / STRUCTURE
# MAX SL = 3.5 ATR
# TP = 2R
# SAME CANDLE TP+SL = SL
# NO OVERLAPPING POSITIONS
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

COINS = {
    "ETH": "eth_usdt",
    "SOL": "sol_usdt",
    "XRP": "xrp_usdt",
    "BTC": "btc_usdt",
    "ADA": "ada_usdt",
    "LINK": "link_usdt",
    "DOGE": "doge_usdt",
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

LBANK_MAX_SIZE = 2000

WARMUP_DAYS = 3

MIN_REQUIRED_1H = 250
MIN_REQUIRED_4H = 250

# 365 days:
#
# 1H  = about 8760 candles
# 4H  = about 2190 candles
#
# With max 2000 candles/request:
# 1H requires about 5 requests
# 4H requires about 2 requests
#
# Extra safety:
MAX_PAGINATION_CHUNKS_1H = 10
MAX_PAGINATION_CHUNKS_4H = 5


# ============================================================
# LBANK
#
# Official current endpoint.
# LBank documentation specifies:
#
# time = timestamp in SECONDS
# size = 1..2000
#
# ============================================================

LBANK_URL = (
    "https://api.lbkex.com/v2/kline.do"
)


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


def timestamp_seconds(dt):
    return int(dt.timestamp())


def utc_time(timestamp):
    return datetime.fromtimestamp(
        int(timestamp),
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

    for attempt in range(1, retries + 1):

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

            print(
                f"      request retry "
                f"{attempt}/{retries}: {exc}"
            )

            if attempt < retries:

                time.sleep(
                    float(attempt)
                )

    raise RuntimeError(
        f"API failed after {retries} attempts: "
        f"{last_error}"
    )


# ============================================================
# NORMALIZE CANDLE
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

        if (
            timestamp_value is None
            or open_price is None
            or high_price is None
            or low_price is None
            or close_price is None
        ):
            return None

        timestamp_value = float(
            timestamp_value
        )

        # ----------------------------------------------------
        # LBank Kline timestamps are documented
        # in SECONDS.
        #
        # Still protect against accidental milliseconds.
        # ----------------------------------------------------

        if timestamp_value > 10_000_000_000:

            timestamp_value /= 1000.0

        o = float(open_price)
        h = float(high_price)
        l = float(low_price)
        c = float(close_price)

        if volume is None:

            v = 0.0

        else:

            v = float(volume)

        if (
            o <= 0
            or h <= 0
            or l <= 0
            or c <= 0
            or h < l
        ):

            return None

        return {
            "time": int(timestamp_value),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v
        }

    except Exception:

        return None


# ============================================================
# DEDUPLICATE
# ============================================================

def deduplicate(candles):

    unique = {}

    for candle in candles:

        if candle is None:
            continue

        unique[
            candle["time"]
        ] = candle

    result = list(
        unique.values()
    )

    result.sort(
        key=lambda x: x["time"]
    )

    return result


# ============================================================
# PARSE LBANK
# ============================================================

def parse_lbank_response(data):

    if not isinstance(data, dict):

        return None

    raw = data.get("data")

    if isinstance(raw, list):

        return raw

    return None


# ============================================================
# PARSE ONE LBANK PAGE
# ============================================================

def lbank_request_page(
    symbol,
    interval,
    start_timestamp,
    size=LBANK_MAX_SIZE
):

    if interval == INTERVAL_1H:

        lbank_type = "hour1"

    elif interval == INTERVAL_4H:

        lbank_type = "hour4"

    else:

        raise RuntimeError(
            f"Unsupported interval: {interval}"
        )

    params = {
        "symbol": COINS[symbol],
        "size": min(
            int(size),
            LBANK_MAX_SIZE
        ),
        "type": lbank_type,

        # IMPORTANT:
        # LBank requires SECONDS here.
        "time": int(start_timestamp)
    }

    response = safe_get(
        LBANK_URL,
        params=params
    )

    try:

        data = response.json()

    except Exception as exc:

        raise RuntimeError(
            f"Invalid JSON from LBank: {exc}"
        )

    if (
        isinstance(data, dict)
        and data.get("error_code")
        not in (None, 0, "0")
    ):

        raise RuntimeError(
            f"LBank error: "
            f"{data.get('error_code')} "
            f"{data.get('msg')}"
        )

    raw = parse_lbank_response(
        data
    )

    if not raw:

        raise RuntimeError(
            "LBank returned empty K-line data."
        )

    candles = []

    for row in raw:

        if isinstance(
            row,
            (list, tuple)
        ):

            if len(row) < 6:
                continue

            candle = normalize_candle(
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                row[5]
            )

            if candle:

                candles.append(
                    candle
                )

        elif isinstance(
            row,
            dict
        ):

            t = (
                row.get("timestamp")
                if row.get("timestamp")
                is not None
                else row.get("time")
            )

            if t is None:

                t = row.get("ts")

            o = (
                row.get("open")
                if row.get("open")
                is not None
                else row.get("o")
            )

            h = (
                row.get("high")
                if row.get("high")
                is not None
                else row.get("h")
            )

            l = (
                row.get("low")
                if row.get("low")
                is not None
                else row.get("l")
            )

            c = (
                row.get("close")
                if row.get("close")
                is not None
                else row.get("c")
            )

            v = (
                row.get("volume")
                if row.get("volume")
                is not None
                else row.get("vol")
            )

            candle = normalize_candle(
                t,
                o,
                h,
                l,
                c,
                v
            )

            if candle:

                candles.append(
                    candle
                )

    candles = deduplicate(
        candles
    )

    if not candles:

        raise RuntimeError(
            "LBank parser produced zero candles."
        )

    return candles


# ============================================================
# LBANK FULL HISTORY
#
# CRITICAL:
#
# LBank "time" is used as the beginning timestamp.
#
# Therefore:
#
#     START
#       ↓
#     PAGE
#       ↓
#     NEWEST CANDLE
#       ↓
#     NEWEST + INTERVAL
#       ↓
#     NEXT PAGE
#
# NOT backwards.
# ============================================================

def get_lbank_klines(
    symbol,
    interval,
    start_dt,
    end_dt
):

    start_ts = timestamp_seconds(
        start_dt
    )

    end_ts = timestamp_seconds(
        end_dt
    )

    interval_seconds = (
        interval * 60
    )

    if interval == INTERVAL_1H:

        max_chunks = (
            MAX_PAGINATION_CHUNKS_1H
        )

    else:

        max_chunks = (
            MAX_PAGINATION_CHUNKS_4H
        )

    all_candles = []

    # --------------------------------------------------------
    # START FROM THE BEGINNING.
    # --------------------------------------------------------

    cursor = start_ts

    previous_cursor = None
    previous_newest = None

    for chunk_number in range(
        1,
        max_chunks + 1
    ):

        print(
            f"    {symbol} "
            f"{'1H' if interval == INTERVAL_1H else '4H'} "
            f"request {chunk_number}/"
            f"{max_chunks}"
        )

        print(
            f"      cursor: "
            f"{utc_time(cursor)}"
        )

        # ----------------------------------------------------
        # Cursor progress protection.
        # ----------------------------------------------------

        if previous_cursor is not None:

            if cursor <= previous_cursor:

                raise RuntimeError(
                    "Pagination cursor did not "
                    "move forward."
                )

        previous_cursor = cursor

        candles = lbank_request_page(
            symbol,
            interval,
            cursor,
            LBANK_MAX_SIZE
        )

        if not candles:

            raise RuntimeError(
                "Empty LBank page."
            )

        page_oldest = min(
            c["time"]
            for c in candles
        )

        page_newest = max(
            c["time"]
            for c in candles
        )

        print(
            f"      received: "
            f"{len(candles)} candles"
        )

        print(
            f"      page: "
            f"{utc_time(page_oldest)}"
            f" -> "
            f"{utc_time(page_newest)}"
        )

        # ----------------------------------------------------
        # Repeated newest protection.
        # ----------------------------------------------------

        if previous_newest is not None:

            if page_newest <= previous_newest:

                raise RuntimeError(
                    "Pagination returned no new "
                    "historical candles."
                )

        previous_newest = page_newest

        # ----------------------------------------------------
        # Only requested range.
        # ----------------------------------------------------

        for candle in candles:

            if (
                start_ts
                <= candle["time"]
                <= end_ts
            ):

                all_candles.append(
                    candle
                )

        # ----------------------------------------------------
        # End reached.
        # ----------------------------------------------------

        if page_newest >= end_ts:

            print(
                "      END REACHED"
            )

            break

        # ----------------------------------------------------
        # If page returned fewer than max,
        # it can mean there are no more candles.
        #
        # But don't immediately fail if the end
        # wasn't reached; move forward and verify.
        # ----------------------------------------------------

        next_cursor = (
            page_newest
            + interval_seconds
        )

        # ----------------------------------------------------
        # Absolute progress protection.
        # ----------------------------------------------------

        if next_cursor <= cursor:

            raise RuntimeError(
                "Pagination made zero progress."
            )

        cursor = next_cursor

        time.sleep(0.15)

    else:

        raise RuntimeError(
            f"{symbol}: maximum pagination "
            f"chunks reached before requested "
            f"end date."
        )

    result = deduplicate(
        all_candles
    )

    result = [
        c
        for c in result
        if (
            start_ts
            <= c["time"]
            <= end_ts
        )
    ]

    return result


# ============================================================
# REMOVE FORMING CANDLE
# ============================================================

def remove_forming_candle(
    candles,
    interval
):

    if not candles:

        return []

    now_ts = timestamp_seconds(
        utc_now()
    )

    interval_seconds = (
        interval * 60
    )

    closed = []

    for candle in candles:

        candle_close = (
            candle["time"]
            + interval_seconds
        )

        if candle_close <= now_ts:

            closed.append(
                candle
            )

    return deduplicate(
        closed
    )


# ============================================================
# COVERAGE REPORT
# ============================================================

def coverage_report(
    candles,
    start_dt,
    end_dt,
    interval
):

    if not candles:

        return {
            "valid": False,
            "coverage": 0.0,
            "count_ratio": 0.0,
            "actual_start": None,
            "actual_end": None,
            "expected_count": 0,
            "actual_count": 0,
            "large_gaps": 0
        }

    candles = deduplicate(
        candles
    )

    required_start = timestamp_seconds(
        start_dt
    )

    required_end = timestamp_seconds(
        end_dt
    )

    actual_start = candles[0]["time"]
    actual_end = candles[-1]["time"]

    interval_seconds = (
        interval * 60
    )

    total_required = (
        required_end
        - required_start
    )

    covered = max(
        0,
        min(
            actual_end,
            required_end
        )
        -
        max(
            actual_start,
            required_start
        )
    )

    coverage = (
        covered
        / total_required
        * 100.0
        if total_required > 0
        else 0.0
    )

    expected_count = (
        int(
            total_required
            / interval_seconds
        )
        + 1
    )

    actual_count = len(
        candles
    )

    count_ratio = (
        actual_count
        / expected_count
        * 100.0
        if expected_count > 0
        else 0.0
    )

    large_gaps = 0

    for i in range(
        1,
        len(candles)
    ):

        gap = (
            candles[i]["time"]
            - candles[i - 1]["time"]
        )

        if gap > (
            interval_seconds * 2
        ):

            large_gaps += 1

    # --------------------------------------------------------
    # Allow one interval boundary tolerance.
    # --------------------------------------------------------

    start_ok = (
        actual_start
        <= required_start
        + interval_seconds
    )

    end_ok = (
        actual_end
        >= required_end
        - interval_seconds
    )

    valid = (
        start_ok
        and end_ok
        and coverage >= 99.0
        and count_ratio >= 99.0
        and large_gaps == 0
    )

    return {
        "valid": valid,
        "coverage": coverage,
        "count_ratio": count_ratio,
        "actual_start": actual_start,
        "actual_end": actual_end,
        "expected_count": expected_count,
        "actual_count": actual_count,
        "large_gaps": large_gaps
    }


# ============================================================
# CANDLE VALIDATION
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

    expected_gap = (
        interval * 60
    )

    huge_gaps = 0

    for i in range(
        1,
        len(candles)
    ):

        gap = (
            candles[i]["time"]
            - candles[i - 1]["time"]
        )

        if gap > (
            expected_gap * 10
        ):

            huge_gaps += 1

    total_gaps = max(
        1,
        len(candles) - 1
    )

    if huge_gaps > (
        total_gaps * 0.20
    ):

        return False

    return True


# ============================================================
# LOAD MARKET DATA
# ============================================================

def load_market_data(
    symbol,
    start_dt,
    end_dt
):

    print()
    print("=" * 100)
    print(
        f"DOWNLOADING {symbol}"
    )
    print("=" * 100)

    try:

        # ----------------------------------------------------
        # DOWNLOAD 4H
        # ----------------------------------------------------

        candles_4h = get_lbank_klines(
            symbol,
            INTERVAL_4H,
            start_dt,
            end_dt
        )

        # ----------------------------------------------------
        # DOWNLOAD 1H
        # ----------------------------------------------------

        candles_1h = get_lbank_klines(
            symbol,
            INTERVAL_1H,
            start_dt,
            end_dt
        )

        # ----------------------------------------------------
        # Remove currently forming candles.
        # ----------------------------------------------------

        candles_4h = remove_forming_candle(
            candles_4h,
            INTERVAL_4H
        )

        candles_1h = remove_forming_candle(
            candles_1h,
            INTERVAL_1H
        )

        # ----------------------------------------------------
        # Coverage
        # ----------------------------------------------------

        report_4h = coverage_report(
            candles_4h,
            start_dt,
            end_dt,
            INTERVAL_4H
        )

        report_1h = coverage_report(
            candles_1h,
            start_dt,
            end_dt,
            INTERVAL_1H
        )

        print()
        print(
            f"{symbol} 4H coverage : "
            f"{report_4h['coverage']:.2f}%"
        )

        print(
            f"{symbol} 4H count    : "
            f"{report_4h['count_ratio']:.2f}%"
        )

        print(
            f"{symbol} 4H gaps     : "
            f"{report_4h['large_gaps']}"
        )

        print(
            f"{symbol} 1H coverage : "
            f"{report_1h['coverage']:.2f}%"
        )

        print(
            f"{symbol} 1H count    : "
            f"{report_1h['count_ratio']:.2f}%"
        )

        print(
            f"{symbol} 1H gaps     : "
            f"{report_1h['large_gaps']}"
        )

        if report_4h["actual_start"]:

            print(
                f"{symbol} 4H actual: "
                f"{utc_time(report_4h['actual_start'])}"
                f" -> "
                f"{utc_time(report_4h['actual_end'])}"
            )

        if report_1h["actual_start"]:

            print(
                f"{symbol} 1H actual: "
                f"{utc_time(report_1h['actual_start'])}"
                f" -> "
                f"{utc_time(report_1h['actual_end'])}"
            )

        valid_4h = (
            report_4h["valid"]
            and
            validate_candles(
                candles_4h,
                INTERVAL_4H,
                MIN_REQUIRED_4H
            )
        )

        valid_1h = (
            report_1h["valid"]
            and
            validate_candles(
                candles_1h,
                INTERVAL_1H,
                MIN_REQUIRED_1H
            )
        )

        if (
            valid_4h
            and valid_1h
        ):

            print()
            print(
                f"{symbol}: "
                f"LBANK VERIFIED FULL HISTORY"
            )

            return (
                candles_4h,
                candles_1h,
                "LBANK"
            )

        print()
        print(
            f"{symbol}: "
            f"STRICT HISTORY VALIDATION FAILED"
        )

    except Exception as exc:

        print()
        print(
            f"{symbol}: DATA DOWNLOAD FAILED"
        )

        print(
            f"    ERROR: {exc}"
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

    if len(candles) < (
        period + 1
    ):

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
        sum(
            trs[:period]
        )
        / period
    )

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

    if len(candles) < (
        period + 1
    ):

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

    rs_value = (
        avg_gain
        / avg_loss
    )

    return (
        100.0
        -
        (
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

        tr_list.append(tr)
        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)

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
# CLOSED 4H FOR ENTRY
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

        candle_close = (
            candle["time"]
            + SECONDS_4H
        )

        if (
            candle_close
            <= entry_close_time
        ):

            usable.append(
                candle
            )

    return usable


# ============================================================
# BREAKOUT LONG
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

    if close_location < MIN_CLOSE_LOCATION:

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
# ANALYZE
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
    # LONG BREAKOUT
    # ========================================================

    if trend_direction == "LONG":

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

    # ========================================================
    # SHORT BREAKOUT
    # ========================================================

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

    # ========================================================
    # REVERSAL LONG
    # ========================================================

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

    # ========================================================
    # REVERSAL SHORT
    # ========================================================

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

        # ----------------------------------------------------
        # Conservative rule:
        # same candle TP + SL = SL
        # ----------------------------------------------------

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
    strategy_start,
    oos_start
):

    trades = []

    strategy_start_ts = (
        timestamp_seconds(
            strategy_start
        )
    )

    oos_start_ts = (
        timestamp_seconds(
            oos_start
        )
    )

    i = EMA200 + 10

    while i < len(candles_1h):

        entry_candle = candles_1h[i]

        if (
            entry_candle["time"]
            < strategy_start_ts
        ):

            i += 1
            continue

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
            candles_1h[
                exit_index
            ]
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
                    >= oos_start_ts
                    else "IS"
                )
        })

        # ----------------------------------------------------
        # NO OVERLAPPING POSITIONS
        # ----------------------------------------------------

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
        * 100.0
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

    pf = stats[
        "profit_factor"
    ]

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
        " SCORE HUNTER PRO v8.12"
    )
    print(
        " REAL 365-DAY DATA BACKTEST"
    )
    print(
        " LBANK FORWARD PAGINATION FIXED"
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
    print("Entry candle excluded from result")
    print("TP + SL same candle = SL")
    print("No overlapping positions")
    print()
    print("DATA ENGINE:")
    print("LBank time parameter = SECONDS")
    print("Pagination = FORWARD")
    print("Maximum page size = 2000")
    print("Current candle = REMOVED")

    # ========================================================
    # DATES
    # ========================================================

    end_dt = utc_now()

    strategy_start = (
        end_dt
        - timedelta(
            days=HISTORY_DAYS
        )
    )

    data_start = (
        strategy_start
        - timedelta(
            days=WARMUP_DAYS
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
        f"History requested : "
        f"{HISTORY_DAYS} days"
    )

    print(
        f"OOS period        : "
        f"{OOS_DAYS} days"
    )

    print(
        "Requested strategy start: "
        f"{utc_time(timestamp_seconds(strategy_start))}"
    )

    print(
        "Data download start: "
        f"{utc_time(timestamp_seconds(data_start))}"
    )

    print(
        "Backtest end: "
        f"{utc_time(timestamp_seconds(end_dt))}"
    )

    print(
        "OOS starts: "
        f"{utc_time(timestamp_seconds(oos_start))}"
    )

    all_trades = []

    data_status = {}

    coverage_status = {}

    # ========================================================
    # DOWNLOAD ALL COINS
    # ========================================================

    for symbol in COINS:

        candles_4h, candles_1h, source = (
            load_market_data(
                symbol,
                data_start,
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

            coverage_status[symbol] = (
                "INCOMPLETE / FAILED"
            )

            print()
            print(
                f"{symbol}: EXCLUDED"
            )

            continue

        # ----------------------------------------------------
        # Final safety: deduplicate again.
        # ----------------------------------------------------

        candles_4h = deduplicate(
            candles_4h
        )

        candles_1h = deduplicate(
            candles_1h
        )

        print()
        print(
            f"{symbol}: "
            f"4H={len(candles_4h)} | "
            f"1H={len(candles_1h)} | "
            f"SOURCE={source}"
        )

        if candles_4h:

            print(
                f"{symbol} 4H: "
                f"{utc_time(candles_4h[0]['time'])}"
                f" -> "
                f"{utc_time(candles_4h[-1]['time'])}"
            )

        if candles_1h:

            print(
                f"{symbol} 1H: "
                f"{utc_time(candles_1h[0]['time'])}"
                f" -> "
                f"{utc_time(candles_1h[-1]['time'])}"
            )

        data_status[symbol] = source

        coverage_status[symbol] = (
            "VERIFIED"
        )

        trades = backtest_coin(
            symbol,
            candles_4h,
            candles_1h,
            strategy_start,
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
    # DATA SOURCE SUMMARY
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
    # COVERAGE
    # ========================================================

    print()
    print("=" * 110)
    print("365-DAY DATA COVERAGE CHECK")
    print("=" * 110)

    full_count = 0

    for symbol in COINS:

        status = coverage_status.get(
            symbol,
            "INCOMPLETE / FAILED"
        )

        print(
            f"{symbol:5s} | "
            f"{status}"
        )

        if status == "VERIFIED":

            full_count += 1

    print()
    print(
        f"Symbols with full data: "
        f"{full_count}/{len(COINS)}"
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

    print_stats(
        "LONG ONLY",
        calculate_stats(
            long_trades
        )
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
        calculate_stats(
            short_trades
        )
    )

    # ========================================================
    # IS
    # ========================================================

    is_trades = [
        t
        for t in all_trades
        if t["period"] == "IS"
    ]

    print_stats(
        "IN-SAMPLE RESULT",
        calculate_stats(
            is_trades
        )
    )

    # ========================================================
    # OOS
    # ========================================================

    oos_trades = [
        t
        for t in all_trades
        if t["period"] == "OOS"
    ]

    print_stats(
        "OUT-OF-SAMPLE RESULT",
        calculate_stats(
            oos_trades
        )
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

    for setup in (
        "BREAKOUT",
        "REVERSAL"
    ):

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
        f"OOS trades             : "
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
    # DATA WARNING
    # ========================================================

    if full_count < len(COINS):

        print()
        print("=" * 110)
        print("DATA WARNING")
        print("=" * 110)

        print(
            "Not every symbol has verified "
            "full historical data."
        )

        print(
            "Unverified symbols were excluded."
        )

    # ========================================================
    # NO TRADES
    # ========================================================

    if not all_trades:

        print()
        print("=" * 110)
        print("NO COMPLETED TRADES")
        print("=" * 110)

        if full_count == 0:

            print(
                "No symbol passed historical "
                "data validation."
            )

            print(
                "This is NOT a 0% win-rate result."
            )

            print(
                "The backtest is invalid until "
                "complete historical market data "
                "is available."
            )

        else:

            print(
                "Historical data loaded, but "
                "the strategy generated no "
                "completed trades."
            )

    # ========================================================
    # FINISH
    # ========================================================

    print()
    print("=" * 110)
    print(
        "SCORE HUNTER PRO v8.12 "
        "BACKTEST FINISHED"
    )
    print("=" * 110)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
