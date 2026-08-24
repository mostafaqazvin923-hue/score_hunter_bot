import requests
import time
from datetime import datetime, timezone, timedelta


# ============================================================
# SCORE HUNTER PRO v8.16
#
# 4H TREND + 1H ENTRY
# REAL 365-DAY HISTORICAL BACKTEST
#
# v8.15 DATA ENGINE PRESERVED
#
# v8.16 CHANGES:
#   1. Full performance statistics
#   2. Average Win R
#   3. Average Loss R
#   4. Gross Profit / Gross Loss
#   5. Profit Factor
#   6. Expectancy
#   7. Max Drawdown in R
#   8. Max Drawdown in %
#   9. Largest Win / Loss
#  10. IS detailed statistics
#  11. OOS detailed statistics
#  12. OOS LONG / SHORT
#  13. OOS BREAKOUT / REVERSAL
#  14. IS vs OOS comparison
#  15. Separate DATA VALID from PERFORMANCE VALID
#  16. Performance verdict
#
# STRATEGY:
#   4H TREND + 1H ENTRY
#   LONG + SHORT
#   BREAKOUT + STRICT REVERSAL
#   NO PULLBACK
#
# RISK:
#   SL = 1.5 ATR / STRUCTURE
#   MAX SL = 3.5 ATR
#   TP = 2R
#   MIN RR = 1.5
#
# TRADE:
#   CLOSED CANDLE ONLY
#   ENTRY AT 1H CLOSE
#   SAME CANDLE TP + SL = SL
#   NO OVERLAPPING POSITIONS
#
# IMPORTANT:
#   OOS RESULTS ARE NEVER USED FOR OPTIMIZATION.
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
WARMUP_DAYS = 60

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

MAX_PAGES_HARD = 100


# ============================================================
# PERFORMANCE SETTINGS v8.16
# ============================================================

# فقط برای تبدیل R به درصد Drawdown.
#
# مثال:
# اگر 1R = 1% ریسک باشد:
# +2R = +2%
# -1R = -1%
#
# این مقدار هیچ اثری روی سیگنال یا نتیجه بک‌تست ندارد.

RISK_PER_TRADE_PERCENT = 1.0


# Minimum sample requirements

MIN_TOTAL_TRADES = 100
MIN_OOS_TRADES = 30


# OOS performance evaluation thresholds
#
# اینها فقط برای Verdict هستند
# و پارامترهای استراتژی نیستند.

MIN_OOS_NET_R = 0.0
MIN_OOS_PROFIT_FACTOR = 1.05
MIN_OOS_WIN_RATE = 35.0
MAX_OOS_DRAWDOWN_R = 15.0


# ============================================================
# LBANK
# ============================================================

LBANK_URL = "https://api.lbkex.com/v2/kline.do"


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

    return datetime.now(
        timezone.utc
    )


def timestamp_seconds(dt):

    return int(
        dt.timestamp()
    )


def utc_time(timestamp):

    return datetime.fromtimestamp(
        int(timestamp),
        tz=timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M"
    )


def candle_close_timestamp(
    candle,
    interval
):

    return (
        candle["time"]
        + interval * 60
    )


# ============================================================
# REQUEST
# ============================================================

def safe_get(
    url,
    params=None,
    retries=MAX_RETRIES
):

    last_error = None

    for attempt in range(
        1,
        retries + 1
    ):

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
                f"{attempt}/{retries}: "
                f"{exc}"
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
# NORMALIZE
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


        if timestamp_value > 10_000_000_000:

            timestamp_value /= 1000.0


        o = float(open_price)
        h = float(high_price)
        l = float(low_price)
        c = float(close_price)

        v = (
            float(volume)
            if volume is not None
            else 0.0
        )


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
# DEDUP
# ============================================================

def deduplicate(
    candles
):

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
        key=lambda x:
            x["time"]
    )


    return result


# ============================================================
# PARSE RESPONSE
# ============================================================

def parse_lbank_response(
    data
):

    if not isinstance(
        data,
        dict
    ):

        raise RuntimeError(
            "LBank response is not a JSON object."
        )


    error_code = data.get(
        "error_code"
    )


    if (
        error_code
        not in (
            None,
            0,
            "0"
        )
    ):

        raise RuntimeError(
            f"LBank error: "
            f"{error_code} "
            f"{data.get('msg')}"
        )


    raw = data.get(
        "data"
    )


    if not isinstance(
        raw,
        list
    ):

        raise RuntimeError(
            "LBank response contains no candle list."
        )


    return raw


# ============================================================
# PARSE CANDLE ROW
# ============================================================

def parse_row(
    row
):

    if isinstance(
        row,
        (list, tuple)
    ):

        if len(row) < 6:

            return None


        return normalize_candle(
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5]
        )


    if isinstance(
        row,
        dict
    ):

        t = (
            row.get("timestamp")
            if row.get("timestamp") is not None
            else row.get("time")
        )


        if t is None:

            t = row.get("ts")


        o = (
            row.get("open")
            if row.get("open") is not None
            else row.get("o")
        )


        h = (
            row.get("high")
            if row.get("high") is not None
            else row.get("h")
        )


        l = (
            row.get("low")
            if row.get("low") is not None
            else row.get("l")
        )


        c = (
            row.get("close")
            if row.get("close") is not None
            else row.get("c")
        )


        v = (
            row.get("volume")
            if row.get("volume") is not None
            else row.get("vol")
        )


        return normalize_candle(
            t,
            o,
            h,
            l,
            c,
            v
        )


    return None


# ============================================================
# REQUEST PAGE
# ============================================================

def lbank_request_page(
    symbol,
    interval,
    cursor
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

        "symbol":
            COINS[symbol],

        "size":
            LBANK_MAX_SIZE,

        "type":
            lbank_type,

        "time":
            int(cursor)
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


    raw = parse_lbank_response(
        data
    )


    candles = []


    for row in raw:

        candle = parse_row(
            row
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
            "LBank returned zero valid candles."
        )


    return candles


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

        close_ts = (
            candle["time"]
            + interval_seconds
        )


        if close_ts <= now_ts:

            closed.append(
                candle
            )


    return deduplicate(
        closed
    )


# ============================================================
# GET LAST CLOSED TIMESTAMP
# ============================================================

def get_last_closed_open(
    now_ts,
    interval_seconds
):

    return (
        (now_ts // interval_seconds)
        * interval_seconds
        - interval_seconds
    )


# ============================================================
# PAGINATION PAGE MERGE
# ============================================================

def merge_page(
    all_candles,
    page
):

    before = len(
        all_candles
    )


    all_candles.extend(
        page
    )


    merged = deduplicate(
        all_candles
    )


    after = len(
        merged
    )


    return merged, (
        after - before
    )


# ============================================================
# GET FULL HISTORY
# ============================================================

def get_lbank_klines(
    symbol,
    interval,
    start_dt,
    end_dt
):

    requested_start = timestamp_seconds(
        start_dt
    )

    requested_end = timestamp_seconds(
        end_dt
    )


    interval_seconds = (
        interval * 60
    )


    now_ts = timestamp_seconds(
        utc_now()
    )


    last_closed_open = (
        get_last_closed_open(
            now_ts,
            interval_seconds
        )
    )


    effective_end = min(
        requested_end,
        last_closed_open
    )


    if effective_end <= requested_start:

        raise RuntimeError(
            "Invalid historical range."
        )


    expected_candles = (
        (
            effective_end
            - requested_start
        )
        // interval_seconds
    ) + 1


    estimated_pages = (
        expected_candles
        + LBANK_MAX_SIZE
        - 1
    ) // LBANK_MAX_SIZE


    max_pages = min(
        MAX_PAGES_HARD,
        estimated_pages + 5
    )


    tf_name = (
        "1H"
        if interval == INTERVAL_1H
        else "4H"
    )


    print()
    print(
        f"    {symbol} {tf_name}"
    )


    print(
        f"    requested: "
        f"{utc_time(requested_start)}"
        f" -> "
        f"{utc_time(effective_end)}"
    )


    print(
        f"    expected candles: "
        f"{expected_candles}"
    )


    print(
        f"    estimated pages: "
        f"{estimated_pages}"
    )


    # ========================================================
    # FORWARD
    # ========================================================

    all_candles = []

    cursor = requested_start

    seen_cursors = set()

    previous_page_newest = None

    forward_failed = False


    for page_number in range(
        1,
        max_pages + 1
    ):

        if cursor in seen_cursors:

            forward_failed = True

            print(
                "      forward cursor repeated"
            )

            break


        seen_cursors.add(
            cursor
        )


        print(
            f"      FORWARD page "
            f"{page_number}/{max_pages} "
            f"cursor={utc_time(cursor)}"
        )


        try:

            page = lbank_request_page(
                symbol,
                interval,
                cursor
            )

        except Exception as exc:

            print(
                f"      forward request failed: "
                f"{exc}"
            )

            forward_failed = True

            break


        page_oldest = min(
            c["time"]
            for c in page
        )


        page_newest = max(
            c["time"]
            for c in page
        )


        print(
            f"      received={len(page)} "
            f"range="
            f"{utc_time(page_oldest)}"
            f" -> "
            f"{utc_time(page_newest)}"
        )


        all_candles, added = merge_page(
            all_candles,
            page
        )


        if (
            previous_page_newest is not None
            and page_newest
            <= previous_page_newest
        ):

            print(
                "      forward made no chronological progress"
            )

            forward_failed = True

            break


        previous_page_newest = page_newest


        if page_newest >= effective_end:

            print(
                "      FORWARD END REACHED"
            )

            break


        next_cursor = (
            page_newest
            + interval_seconds
        )


        if next_cursor <= cursor:

            print(
                "      forward zero progress"
            )

            forward_failed = True

            break


        cursor = next_cursor

        time.sleep(
            0.15
        )


    # ========================================================
    # PRELIMINARY COVERAGE
    # ========================================================

    preliminary = deduplicate(
        all_candles
    )


    preliminary = [
        c
        for c in preliminary
        if (
            requested_start
            <= c["time"]
            <= effective_end
        )
    ]


    preliminary = remove_forming_candle(
        preliminary,
        interval
    )


    if not preliminary:

        forward_coverage = 0.0

    else:

        actual_start = preliminary[0]["time"]
        actual_end = preliminary[-1]["time"]


        covered = max(
            0,
            min(
                actual_end,
                effective_end
            )
            -
            max(
                actual_start,
                requested_start
            )
        )


        required = (
            effective_end
            - requested_start
        )


        forward_coverage = (
            covered
            / required
            * 100.0
        )


    # ========================================================
    # REVERSE FALLBACK
    # ========================================================

    if (
        forward_failed
        or forward_coverage < 98.0
    ):

        print()
        print(
            "      FORWARD COVERAGE INSUFFICIENT"
        )

        print(
            "      TRYING REVERSE PAGINATION"
        )


        reverse_candles = []

        reverse_cursor = effective_end

        reverse_seen = set()

        previous_page_oldest = None


        for page_number in range(
            1,
            max_pages + 1
        ):

            if reverse_cursor in reverse_seen:

                print(
                    "      reverse cursor repeated"
                )

                break


            reverse_seen.add(
                reverse_cursor
            )


            print(
                f"      REVERSE page "
                f"{page_number}/{max_pages} "
                f"cursor="
                f"{utc_time(reverse_cursor)}"
            )


            try:

                page = lbank_request_page(
                    symbol,
                    interval,
                    reverse_cursor
                )

            except Exception as exc:

                print(
                    f"      reverse request failed: "
                    f"{exc}"
                )

                break


            page_oldest = min(
                c["time"]
                for c in page
            )


            page_newest = max(
                c["time"]
                for c in page
            )


            print(
                f"      received={len(page)} "
                f"range="
                f"{utc_time(page_oldest)}"
                f" -> "
                f"{utc_time(page_newest)}"
            )


            reverse_candles.extend(
                page
            )


            reverse_candles = deduplicate(
                reverse_candles
            )


            if (
                previous_page_oldest
                is not None
                and page_oldest
                >= previous_page_oldest
            ):

                print(
                    "      reverse made no chronological progress"
                )

                break


            previous_page_oldest = page_oldest


            if page_oldest <= requested_start:

                print(
                    "      REVERSE START REACHED"
                )

                break


            next_cursor = (
                page_oldest
                - interval_seconds
            )


            if next_cursor >= reverse_cursor:

                print(
                    "      reverse zero progress"
                )

                break


            reverse_cursor = next_cursor

            time.sleep(
                0.15
            )


        all_candles.extend(
            reverse_candles
        )


    # ========================================================
    # FINAL FILTER
    # ========================================================

    result = deduplicate(
        all_candles
    )


    result = [
        c
        for c in result
        if (
            requested_start
            <= c["time"]
            <= effective_end
        )
    ]


    result = remove_forming_candle(
        result,
        interval
    )


    result = deduplicate(
        result
    )


    if not result:

        raise RuntimeError(
            "No candles survived final filtering."
        )


    print(
        f"      FINAL candles: "
        f"{len(result)}"
    )


    print(
        f"      FINAL range: "
        f"{utc_time(result[0]['time'])}"
        f" -> "
        f"{utc_time(result[-1]['time'])}"
    )


    return result


# ============================================================
# COVERAGE REPORT
# ============================================================

def coverage_report(
    candles,
    start_dt,
    end_dt,
    interval
):

    candles = deduplicate(
        candles
    )


    if not candles:

        return {
            "valid": False,
            "coverage": 0.0,
            "count_ratio": 0.0,
            "actual_start": None,
            "actual_end": None,
            "expected_count": 0,
            "actual_count": 0,
            "missing_candles": 0,
            "large_gaps": 0
        }


    interval_seconds = (
        interval * 60
    )


    required_start = timestamp_seconds(
        start_dt
    )


    requested_end = timestamp_seconds(
        end_dt
    )


    now_ts = timestamp_seconds(
        utc_now()
    )


    effective_end = min(
        requested_end,
        get_last_closed_open(
            now_ts,
            interval_seconds
        )
    )


    candles = [
        c
        for c in candles
        if (
            required_start
            <= c["time"]
            <= effective_end
        )
    ]


    if not candles:

        return {
            "valid": False,
            "coverage": 0.0,
            "count_ratio": 0.0,
            "actual_start": None,
            "actual_end": None,
            "expected_count": 0,
            "actual_count": 0,
            "missing_candles": 0,
            "large_gaps": 0
        }


    actual_start = candles[0]["time"]
    actual_end = candles[-1]["time"]


    required_span = (
        effective_end
        - required_start
    )


    covered_span = max(
        0,
        min(
            actual_end,
            effective_end
        )
        -
        max(
            actual_start,
            required_start
        )
    )


    if required_span > 0:

        coverage = (
            covered_span
            / required_span
            * 100.0
        )

    else:

        coverage = 100.0


    expected_count = (
        (
            effective_end
            - required_start
        )
        // interval_seconds
    ) + 1


    actual_count = len(
        candles
    )


    count_ratio = (
        actual_count
        / expected_count
        * 100.0
    )


    missing_candles = 0
    large_gaps = 0


    for i in range(
        1,
        len(candles)
    ):

        gap = (
            candles[i]["time"]
            - candles[i - 1]["time"]
        )


        if gap > interval_seconds:

            missing = (
                gap
                // interval_seconds
            ) - 1


            if missing > 0:

                missing_candles += missing


        if gap > (
            interval_seconds * 2
        ):

            large_gaps += 1


    start_ok = (
        actual_start
        <= (
            required_start
            + interval_seconds
        )
    )


    end_ok = (
        actual_end
        >= (
            effective_end
            - interval_seconds
        )
    )


    missing_ratio = (
        missing_candles
        / max(
            1,
            expected_count
        )
        * 100.0
    )


    valid = (
        start_ok
        and end_ok
        and coverage >= 99.0
        and count_ratio >= 99.0
        and missing_ratio <= 1.0
        and large_gaps <= 3
    )


    return {
        "valid":
            valid,

        "coverage":
            coverage,

        "count_ratio":
            count_ratio,

        "actual_start":
            actual_start,

        "actual_end":
            actual_end,

        "expected_count":
            expected_count,

        "actual_count":
            actual_count,

        "missing_candles":
            missing_candles,

        "large_gaps":
            large_gaps
    }


# ============================================================
# VALIDATE CANDLES
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


    severe_gaps = 0


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

            severe_gaps += 1


    total_gaps = max(
        1,
        len(candles) - 1
    )


    if severe_gaps > (
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
    print("=" * 110)
    print(
        f"DOWNLOADING {symbol}"
    )
    print("=" * 110)


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


        candles_4h = remove_forming_candle(
            candles_4h,
            INTERVAL_4H
        )


        candles_1h = remove_forming_candle(
            candles_1h,
            INTERVAL_1H
        )


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
            f"{report_4h['coverage']:.3f}%"
        )


        print(
            f"{symbol} 4H count    : "
            f"{report_4h['count_ratio']:.3f}%"
        )


        print(
            f"{symbol} 4H missing  : "
            f"{report_4h['missing_candles']}"
        )


        print(
            f"{symbol} 4H gaps     : "
            f"{report_4h['large_gaps']}"
        )


        print()
        print(
            f"{symbol} 1H coverage : "
            f"{report_1h['coverage']:.3f}%"
        )


        print(
            f"{symbol} 1H count    : "
            f"{report_1h['count_ratio']:.3f}%"
        )


        print(
            f"{symbol} 1H missing  : "
            f"{report_1h['missing_candles']}"
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
                250
            )
        )


        valid_1h = (
            report_1h["valid"]
            and
            validate_candles(
                candles_1h,
                INTERVAL_1H,
                250
            )
        )


        print()
        print(
            f"{symbol} 4H validation: "
            f"{'PASS' if valid_4h else 'FAIL'}"
        )


        print(
            f"{symbol} 1H validation: "
            f"{'PASS' if valid_1h else 'FAIL'}"
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


        return (
            None,
            None,
            "FAILED"
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


        tr_list.append(
            tr
        )

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
# CLOSED 4H CONTEXT
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


        if candle_close <= entry_close_time:

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

        return False


    current = candles[-1]


    previous = candles[
        -STRUCTURE_LOOKBACK - 1:-1
    ]


    resistance = max(
        c["high"]
        for c in previous
    )


    if current["close"] <= resistance:

        return False


    candle_range = (
        current["high"]
        - current["low"]
    )


    if candle_range <= 0:

        return False


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

        return False


    if body_ratio < MIN_BODY_RATIO:

        return False


    if close_location < MIN_CLOSE_LOCATION:

        return False


    if (
        current["close"]
        <= candles[-2]["close"]
    ):

        return False


    return True


# ============================================================
# BREAKOUT SHORT
# ============================================================

def detect_breakout_short(
    candles
):

    if len(candles) < (
        STRUCTURE_LOOKBACK + 2
    ):

        return False


    current = candles[-1]


    previous = candles[
        -STRUCTURE_LOOKBACK - 1:-1
    ]


    support = min(
        c["low"]
        for c in previous
    )


    if current["close"] >= support:

        return False


    candle_range = (
        current["high"]
        - current["low"]
    )


    if candle_range <= 0:

        return False


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

        return False


    if body_ratio < MIN_BODY_RATIO:

        return False


    if close_location < MIN_CLOSE_LOCATION:

        return False


    if (
        current["close"]
        >= candles[-2]["close"]
    ):

        return False


    return True


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

        return False


    required = max(
        60,
        REVERSAL_LOOKBACK + 12
    )


    if len(candles) < required:

        return False


    current = candles[-1]


    previous = candles[
        -REVERSAL_LOOKBACK - 1:-1
    ]


    resistance = max(
        c["high"]
        for c in previous
    )


    if current["close"] <= resistance:

        return False


    if current["close"] <= current["open"]:

        return False


    candle_range = (
        current["high"]
        - current["low"]
    )


    if candle_range <= 0:

        return False


    body_ratio = (
        abs(
            current["close"]
            - current["open"]
        )
        / candle_range
    )


    if body_ratio < MIN_BODY_RATIO:

        return False


    close_location = (
        current["close"]
        - current["low"]
    ) / candle_range


    if close_location < MIN_CLOSE_LOCATION:

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


    if (
        e20 is None
        or e50 is None
    ):

        return False


    if e20 <= e50:

        return False


    if current["close"] <= e50:

        return False


    if adx_value < ADX_MIN:

        return False


    if rsi_value < 50:

        return False


    previous_candle = candles[-2]


    if (
        previous_candle["close"]
        <= previous_candle["open"]
    ):

        return False


    return True


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

        return False


    required = max(
        60,
        REVERSAL_LOOKBACK + 12
    )


    if len(candles) < required:

        return False


    current = candles[-1]


    previous = candles[
        -REVERSAL_LOOKBACK - 1:-1
    ]


    support = min(
        c["low"]
        for c in previous
    )


    if current["close"] >= support:

        return False


    if current["close"] >= current["open"]:

        return False


    candle_range = (
        current["high"]
        - current["low"]
    )


    if candle_range <= 0:

        return False


    body_ratio = (
        abs(
            current["close"]
            - current["open"]
        )
        / candle_range
    )


    if body_ratio < MIN_BODY_RATIO:

        return False


    close_location = (
        current["high"]
        - current["close"]
    ) / candle_range


    if close_location < MIN_CLOSE_LOCATION:

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


    if (
        e20 is None
        or e50 is None
    ):

        return False


    if e20 >= e50:

        return False


    if current["close"] >= e50:

        return False


    if adx_value < ADX_MIN:

        return False


    if rsi_value > 50:

        return False


    previous_candle = candles[-2]


    if (
        previous_candle["close"]
        >= previous_candle["open"]
    ):

        return False


    return True


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


    # ========================================================
    # LONG BREAKOUT
    # ========================================================

    if trend_direction == "LONG":

        if ema_alignment_long(
            candles_1h
        ):

            if (
                50
                <= rsi_value
                <= 85
            ):

                if detect_breakout_long(
                    candles_1h
                ):

                    levels = calculate_long_levels(
                        candles_1h,
                        entry,
                        atr_value
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

            if (
                15
                <= rsi_value
                <= 50
            ):

                if detect_breakout_short(
                    candles_1h
                ):

                    levels = calculate_short_levels(
                        candles_1h,
                        entry,
                        atr_value
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

    if detect_reversal_long(
        candles_1h,
        trend_direction,
        adx_value,
        rsi_value
    ):

        levels = calculate_long_levels(
            candles_1h,
            entry,
            atr_value
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

    if detect_reversal_short(
        candles_1h,
        trend_direction,
        adx_value,
        rsi_value
    ):

        levels = calculate_short_levels(
            candles_1h,
            entry,
            atr_value
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


        # Conservative:
        # same candle = SL

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


    strategy_start_ts = timestamp_seconds(
        strategy_start
    )


    oos_start_ts = timestamp_seconds(
        oos_start
    )


    i = 0


    while i < len(candles_1h):

        entry_candle = candles_1h[i]


        if (
            entry_candle["time"]
            < strategy_start_ts
        ):

            i += 1

            continue


        usable_4h = get_closed_4h_for_entry(
            candles_4h,
            entry_candle
        )


        if len(usable_4h) < EMA200:

            i += 1

            continue


        usable_1h = candles_1h[
            :i + 1
        ]


        signal = analyze_at_index(
            usable_4h,
            usable_1h
        )


        if signal is None:

            i += 1

            continue


        signal["symbol"] = symbol


        result, exit_index = check_trade_result(
            candles_1h,
            i,
            signal
        )


        if result is None:

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


        i = exit_index + 1


    return trades


# ============================================================
# STATISTICS v8.16
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
            "average_win_r": 0.0,
            "average_loss_r": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_percent": 0.0,
            "max_win_streak": 0,
            "max_loss_streak": 0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "final_equity_r": 0.0,
            "peak_equity_r": 0.0
        }


    total = len(
        trades
    )


    wins = sum(
        1
        for t in trades
        if t["result"] == "TP"
    )


    losses = sum(
        1
        for t in trades
        if t["result"] == "SL"
    )


    win_rate = (
        wins
        / total
        * 100.0
    )


    r_values = [
        float(t["R"])
        for t in trades
    ]


    net_r = sum(
        r_values
    )


    average_r = (
        net_r
        / total
    )


    winning_r = [
        r
        for r in r_values
        if r > 0
    ]


    losing_r = [
        r
        for r in r_values
        if r < 0
    ]


    average_win_r = (
        sum(winning_r)
        / len(winning_r)
        if winning_r
        else 0.0
    )


    average_loss_r = (
        sum(losing_r)
        / len(losing_r)
        if losing_r
        else 0.0
    )


    gross_profit = sum(
        winning_r
    )


    gross_loss = abs(
        sum(losing_r)
    )


    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    elif gross_profit > 0:

        profit_factor = float(
            "inf"
        )

    else:

        profit_factor = 0.0


    expectancy = average_r


    # ========================================================
    # EQUITY / DRAWDOWN
    # ========================================================

    equity = 0.0
    peak = 0.0

    max_drawdown = 0.0

    current_win = 0
    current_loss = 0

    max_win = 0
    max_loss = 0

    largest_win = 0.0
    largest_loss = 0.0


    for trade in trades:

        r_value = float(
            trade["R"]
        )


        equity += r_value


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


        largest_win = max(
            largest_win,
            r_value
        )


        largest_loss = min(
            largest_loss,
            r_value
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


    max_drawdown_percent = (
        max_drawdown
        * RISK_PER_TRADE_PERCENT
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

        "average_win_r":
            average_win_r,

        "average_loss_r":
            average_loss_r,

        "gross_profit":
            gross_profit,

        "gross_loss":
            gross_loss,

        "profit_factor":
            profit_factor,

        "expectancy":
            expectancy,

        "max_drawdown":
            max_drawdown,

        "max_drawdown_percent":
            max_drawdown_percent,

        "max_win_streak":
            max_win,

        "max_loss_streak":
            max_loss,

        "largest_win":
            largest_win,

        "largest_loss":
            largest_loss,

        "final_equity_r":
            equity,

        "peak_equity_r":
            peak
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
# PRINT STATISTICS
# ============================================================

def print_stats(
    title,
    stats
):

    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


    if stats["total"] == 0:

        print(
            "No completed trades."
        )

        return


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


    print(
        f"Average Win        : "
        f"{stats['average_win_r']:+.3f}R"
    )


    print(
        f"Average Loss       : "
        f"{stats['average_loss_r']:+.3f}R"
    )


    pf = stats["profit_factor"]


    if pf == float("inf"):

        pf_text = "INF"

    else:

        pf_text = f"{pf:.3f}"


    print(
        f"Profit Factor      : "
        f"{pf_text}"
    )


    print(
        f"Expectancy         : "
        f"{stats['expectancy']:+.3f}R"
    )


    print(
        f"Gross Profit       : "
        f"{stats['gross_profit']:+.2f}R"
    )


    print(
        f"Gross Loss         : "
        f"-{stats['gross_loss']:.2f}R"
    )


    print(
        f"Max Drawdown       : "
        f"{stats['max_drawdown']:.2f}R"
    )


    print(
        f"Max Drawdown       : "
        f"{stats['max_drawdown_percent']:.2f}% "
        f"(1R={RISK_PER_TRADE_PERCENT:.2f}%)"
    )


    print(
        f"Max Win Streak     : "
        f"{stats['max_win_streak']}"
    )


    print(
        f"Max Loss Streak    : "
        f"{stats['max_loss_streak']}"
    )


    print(
        f"Largest Win        : "
        f"{stats['largest_win']:+.2f}R"
    )


    print(
        f"Largest Loss       : "
        f"{stats['largest_loss']:+.2f}R"
    )


    print(
        f"Final Equity       : "
        f"{stats['final_equity_r']:+.2f}R"
    )


# ============================================================
# SIMPLE ROW STAT
# ============================================================

def print_compact_stats(
    label,
    stats
):

    if stats["total"] == 0:

        print(
            f"{label} | 0 trades"
        )

        return


    pf = stats["profit_factor"]


    if pf == float("inf"):

        pf_text = "INF"

    else:

        pf_text = f"{pf:.2f}"


    print(
        f"{label} | "
        f"Trades: {stats['total']:3d} | "
        f"W: {stats['wins']:3d} | "
        f"L: {stats['losses']:3d} | "
        f"WR: {stats['win_rate']:6.2f}% | "
        f"R: {stats['net_r']:+8.2f} | "
        f"PF: {pf_text:>6} | "
        f"DD: {stats['max_drawdown']:.2f}R"
    )


# ============================================================
# PERFORMANCE VERDICT
# ============================================================

def performance_verdict(
    overall_stats,
    is_stats,
    oos_stats
):

    print()
    print("=" * 110)
    print(
        "PERFORMANCE VERDICT"
    )
    print("=" * 110)


    total_ok = (
        overall_stats["total"]
        >= MIN_TOTAL_TRADES
    )


    oos_sample_ok = (
        oos_stats["total"]
        >= MIN_OOS_TRADES
    )


    print(
        f"Total sample        : "
        f"{overall_stats['total']}"
    )


    print(
        f"Required total      : "
        f"{MIN_TOTAL_TRADES}"
    )


    print(
        "Total sample status : "
        f"{'PASS' if total_ok else 'FAIL'}"
    )


    print()


    print(
        f"OOS sample          : "
        f"{oos_stats['total']}"
    )


    print(
        f"Required OOS        : "
        f"{MIN_OOS_TRADES}"
    )


    print(
        "OOS sample status   : "
        f"{'PASS' if oos_sample_ok else 'FAIL'}"
    )


    if oos_stats["total"] == 0:

        print()
        print(
            "FINAL VERDICT       : "
            "OOS FAILED"
        )

        return "OOS FAILED"


    pf = oos_stats[
        "profit_factor"
    ]


    pf_ok = (
        pf >= MIN_OOS_PROFIT_FACTOR
        if pf != float("inf")
        else True
    )


    net_ok = (
        oos_stats["net_r"]
        > MIN_OOS_NET_R
    )


    wr_ok = (
        oos_stats["win_rate"]
        >= MIN_OOS_WIN_RATE
    )


    dd_ok = (
        oos_stats["max_drawdown"]
        <= MAX_OOS_DRAWDOWN_R
    )


    print()
    print(
        "OOS PERFORMANCE CHECKS"
    )


    print(
        f"OOS Win Rate       : "
        f"{oos_stats['win_rate']:.2f}% "
        f"-> "
        f"{'PASS' if wr_ok else 'FAIL'}"
    )


    print(
        f"OOS Net R          : "
        f"{oos_stats['net_r']:+.2f}R "
        f"-> "
        f"{'PASS' if net_ok else 'FAIL'}"
    )


    if pf == float("inf"):

        pf_display = "INF"

    else:

        pf_display = f"{pf:.3f}"


    print(
        f"OOS Profit Factor  : "
        f"{pf_display} "
        f"-> "
        f"{'PASS' if pf_ok else 'FAIL'}"
    )


    print(
        f"OOS Max Drawdown   : "
        f"{oos_stats['max_drawdown']:.2f}R "
        f"-> "
        f"{'PASS' if dd_ok else 'FAIL'}"
    )


    print()
    print(
        "IS / OOS COMPARISON"
    )


    print(
        f"IS  Trades         : "
        f"{is_stats['total']}"
    )


    print(
        f"OOS Trades         : "
        f"{oos_stats['total']}"
    )


    print(
        f"IS  Win Rate       : "
        f"{is_stats['win_rate']:.2f}%"
    )


    print(
        f"OOS Win Rate       : "
        f"{oos_stats['win_rate']:.2f}%"
    )


    print(
        f"IS  Net R          : "
        f"{is_stats['net_r']:+.2f}R"
    )


    print(
        f"OOS Net R          : "
        f"{oos_stats['net_r']:+.2f}R"
    )


    if is_stats["profit_factor"] == float("inf"):

        is_pf_text = "INF"

    else:

        is_pf_text = (
            f"{is_stats['profit_factor']:.3f}"
        )


    print(
        f"IS  Profit Factor  : "
        f"{is_pf_text}"
    )


    print(
        f"OOS Profit Factor  : "
        f"{pf_display}"
    )


    print(
        f"IS  Max DD         : "
        f"{is_stats['max_drawdown']:.2f}R"
    )


    print(
        f"OOS Max DD         : "
        f"{oos_stats['max_drawdown']:.2f}R"
    )


    # ========================================================
    # FINAL CLASSIFICATION
    # ========================================================

    if not total_ok:

        verdict = (
            "INSUFFICIENT SAMPLE"
        )

    elif not oos_sample_ok:

        verdict = (
            "OOS FAILED - SAMPLE TOO SMALL"
        )

    elif (
        net_ok
        and pf_ok
        and wr_ok
        and dd_ok
    ):

        verdict = (
            "PROFITABLE - OOS PASSED"
        )

    elif (
        net_ok
        and pf_ok
    ):

        verdict = (
            "MARGINAL - OOS POSITIVE BUT NEEDS REVIEW"
        )

    else:

        verdict = (
            "UNPROFITABLE - OOS FAILED"
        )


    print()
    print("=" * 110)
    print(
        f"FINAL VERDICT       : "
        f"{verdict}"
    )
    print("=" * 110)


    return verdict


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 110)
    print(
        " SCORE HUNTER PRO v8.16"
    )
    print(
        " REAL 365-DAY HISTORICAL BACKTEST"
    )
    print(
        " LBANK ROBUST PAGINATION ENGINE"
    )
    print(
        " FULL PERFORMANCE ANALYSIS"
    )
    print("=" * 110)


    print()
    print("RULES:")
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
    print("LBank legacy KLINE API")
    print("Timestamp = seconds")
    print("Adaptive pagination")
    print("Forward + reverse fallback")
    print("Direction auto-detection")
    print("Exact timestamp deduplication")
    print("Forming candle removed BEFORE validation")
    print("Closed-data coverage validation")
    print("Isolated missing candle tolerance")
    print("Large-gap detection")
    print("Warmup = 60 days")


    print()
    print("PERFORMANCE ENGINE:")
    print(
        f"Risk model           : "
        f"1R = {RISK_PER_TRADE_PERCENT:.2f}%"
    )
    print(
        f"Minimum total sample : "
        f"{MIN_TOTAL_TRADES}"
    )
    print(
        f"Minimum OOS sample   : "
        f"{MIN_OOS_TRADES}"
    )
    print(
        f"Minimum OOS PF       : "
        f"{MIN_OOS_PROFIT_FACTOR}"
    )
    print(
        f"Minimum OOS WR       : "
        f"{MIN_OOS_WIN_RATE:.1f}%"
    )
    print(
        f"Maximum OOS DD       : "
        f"{MAX_OOS_DRAWDOWN_R:.2f}R"
    )


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
        f"Warmup            : "
        f"{WARMUP_DAYS} days"
    )


    print(
        f"OOS period        : "
        f"{OOS_DAYS} days"
    )


    print(
        "Strategy start: "
        f"{utc_time(timestamp_seconds(strategy_start))}"
    )


    print(
        "Data start: "
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
    # DOWNLOAD
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

            data_status[symbol] = "FAILED"

            coverage_status[symbol] = (
                "INCOMPLETE / FAILED"
            )


            print()
            print(
                f"{symbol}: EXCLUDED"
            )


            continue


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

        coverage_status[symbol] = "VERIFIED"


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
    print(
        "DATA SOURCE SUMMARY"
    )
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
    print(
        "HISTORICAL DATA COVERAGE"
    )
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

    overall_stats = calculate_stats(
        all_trades
    )


    print_stats(
        "OVERALL RESULT",
        overall_stats
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
    # IS
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
    # OOS LONG
    # ========================================================

    oos_long = [
        t
        for t in oos_trades
        if t["direction"] == "LONG"
    ]


    print_stats(
        "OOS LONG",
        calculate_stats(
            oos_long
        )
    )


    # ========================================================
    # OOS SHORT
    # ========================================================

    oos_short = [
        t
        for t in oos_trades
        if t["direction"] == "SHORT"
    ]


    print_stats(
        "OOS SHORT",
        calculate_stats(
            oos_short
        )
    )


    # ========================================================
    # BY COIN
    # ========================================================

    print()
    print("=" * 110)
    print(
        "RESULT BY COIN"
    )
    print("=" * 110)


    for symbol in COINS:

        coin_trades = [
            t
            for t in all_trades
            if t["symbol"] == symbol
        ]


        coin_stats = calculate_stats(
            coin_trades
        )


        print_compact_stats(
            f"{symbol:5s}",
            coin_stats
        )


    # ========================================================
    # BY SETUP
    # ========================================================

    print()
    print("=" * 110)
    print(
        "RESULT BY SETUP"
    )
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


        print_compact_stats(
            f"{setup:10s}",
            calculate_stats(
                setup_trades
            )
        )


    # ========================================================
    # OOS BY COIN
    # ========================================================

    print()
    print("=" * 110)
    print(
        "OUT-OF-SAMPLE BY COIN"
    )
    print("=" * 110)


    for symbol in COINS:

        coin_oos = [
            t
            for t in oos_trades
            if t["symbol"] == symbol
        ]


        print_compact_stats(
            f"{symbol:5s}",
            calculate_stats(
                coin_oos
            )
        )


    # ========================================================
    # OOS BY SETUP
    # ========================================================

    print()
    print("=" * 110)
    print(
        "OUT-OF-SAMPLE BY SETUP"
    )
    print("=" * 110)


    for setup in (
        "BREAKOUT",
        "REVERSAL"
    ):

        setup_oos = [
            t
            for t in oos_trades
            if t["setup"] == setup
        ]


        print_compact_stats(
            f"{setup:10s}",
            calculate_stats(
                setup_oos
            )
        )


    # ========================================================
    # FULL TRADE LOG
    # ========================================================

    print()
    print("=" * 110)
    print(
        "FULL TRADE LOG"
    )
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
    print(
        "ROBUSTNESS CHECK"
    )
    print("=" * 110)


    print(
        f"Total completed trades : "
        f"{len(all_trades)}"
    )


    print(
        f"Target sample          : "
        f"{MIN_TOTAL_TRADES}+"
    )


    if len(all_trades) >= MIN_TOTAL_TRADES:

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


    if len(oos_trades) >= MIN_OOS_TRADES:

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
        print(
            "DATA WARNING"
        )
        print("=" * 110)


        print(
            "Not every symbol has verified "
            "full historical data."
        )


        print(
            "Unverified symbols were excluded."
        )


    # ========================================================
    # DATA VALIDITY
    # ========================================================

    print()
    print("=" * 110)
    print(
        "DATA VALIDITY"
    )
    print("=" * 110)


    if full_count == len(COINS):

        print(
            "DATA STATUS          : "
            "VALID"
        )


        print(
            f"Verified symbols     : "
            f"{full_count}/{len(COINS)}"
        )

    elif full_count > 0:

        print(
            "DATA STATUS          : "
            "PARTIAL"
        )


        print(
            f"Verified symbols     : "
            f"{full_count}/{len(COINS)}"
        )

    else:

        print(
            "DATA STATUS          : "
            "INVALID"
        )


    # ========================================================
    # PERFORMANCE VERDICT
    # ========================================================

    verdict = performance_verdict(
        overall_stats,
        is_stats,
        oos_stats
    )


    # ========================================================
    # FINAL BACKTEST STATUS
    # ========================================================

    print()
    print("=" * 110)
    print(
        "FINAL BACKTEST STATUS"
    )
    print("=" * 110)


    if full_count == 0:

        print(
            "BACKTEST INVALID"
        )


        print(
            "No symbol passed historical "
            "data validation."
        )


        print(
            "This is NOT a 0% win-rate result."
        )


        print(
            "The problem is historical data."
        )


    elif not all_trades:

        print(
            "DATA VALID"
        )


        print(
            "Historical data loaded successfully,"
        )


        print(
            "but the strategy generated "
            "no completed trades."
        )


    else:

        print(
            "DATA VALID"
        )


        print(
            f"Completed trades: "
            f"{len(all_trades)}"
        )


        print(
            f"OOS trades: "
            f"{len(oos_trades)}"
        )


        print(
            f"PERFORMANCE: "
            f"{verdict}"
        )


    # ========================================================
    # FINISH
    # ========================================================

    print()
    print("=" * 110)
    print(
        "SCORE HUNTER PRO v8.16 "
        "BACKTEST FINISHED"
    )
    print("=" * 110)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
