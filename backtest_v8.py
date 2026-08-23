import requests
import time
from datetime import datetime, timezone, timedelta


# ============================================================
# SCORE HUNTER PRO v8.10
#
# 4H TREND + 1H ENTRY
# LONG + SHORT
# BREAKOUT + STRICT REVERSAL
# CLOSED CANDLE ONLY
# NO LOOK-AHEAD
#
# FIXED:
#   - LBank pagination cannot loop forever
#   - Detect repeated/invalid pages
#   - Hard request/chunk limits
#   - Correct historical boundaries
#   - Remove incomplete current candles
#   - Strict 365-day coverage validation
#   - No fake backtest if history is incomplete
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

REQUEST_TIMEOUT = 15
MAX_RETRIES = 3

LBANK_MAX_SIZE = 2000

WARMUP_DAYS = 5

MIN_REQUIRED_1H = 250
MIN_REQUIRED_4H = 250

# جلوگیری قطعی از گیر کردن pagination
MAX_LBANK_CHUNKS_1H = 20
MAX_LBANK_CHUNKS_4H = 10


# ============================================================
# SESSION
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
# HTTP
# ============================================================

def safe_get(url, params=None, retries=MAX_RETRIES):

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
                time.sleep(
                    1.0 + attempt
                )

    raise RuntimeError(
        f"HTTP failed: {last_error}"
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

        if timestamp_value is None:
            return None

        timestamp_value = float(
            timestamp_value
        )

        if timestamp_value > 10_000_000_000:
            timestamp_value /= 1000.0

        candle = {
            "time": int(timestamp_value),
            "open": float(open_price),
            "high": float(high_price),
            "low": float(low_price),
            "close": float(close_price),
            "volume": float(volume)
        }

        if candle["high"] < candle["low"]:
            return None

        if candle["open"] <= 0:
            return None

        if candle["high"] <= 0:
            return None

        if candle["low"] <= 0:
            return None

        if candle["close"] <= 0:
            return None

        return candle

    except Exception:

        return None


# ============================================================
# PARSE LBANK
# ============================================================

def parse_lbank_response(data):

    if isinstance(data, dict):

        raw = data.get("data")

        if isinstance(raw, list):
            return raw

        raw = data.get("result")

        if isinstance(raw, list):
            return raw

        # بعض پاسخ‌های LBank ممکن است data را
        # به شکل دیکشنری برگردانند.
        if isinstance(raw, dict):

            for key in (
                "data",
                "kline",
                "klines",
                "result"
            ):

                value = raw.get(key)

                if isinstance(value, list):
                    return value

    if isinstance(data, list):
        return data

    return None


# ============================================================
# DEDUP
# ============================================================

def deduplicate(candles):

    unique = {}

    for candle in candles:

        unique[candle["time"]] = candle

    result = list(
        unique.values()
    )

    result.sort(
        key=lambda x: x["time"]
    )

    return result


# ============================================================
# LBANK SINGLE REQUEST
# ============================================================

LBANK_URLS = [
    "https://api.lbkex.com/v2/kline.do",
    "https://api.lbank.info/v2/kline.do"
]


def lbank_single_request(
    symbol,
    interval,
    anchor_timestamp,
    size=LBANK_MAX_SIZE
):

    pair = COINS[symbol]["lbank"]

    interval_map = {
        60: "hour1",
        240: "hour4"
    }

    lbank_interval = interval_map[
        interval
    ]

    last_error = None

    for base_url in LBANK_URLS:

        try:

            params = {
                "symbol": pair,
                "size": min(
                    int(size),
                    LBANK_MAX_SIZE
                ),
                "type": lbank_interval,

                # LBank time is milliseconds
                "time": int(
                    anchor_timestamp
                )
            }

            response = safe_get(
                base_url,
                params=params
            )

            data = response.json()

            raw = parse_lbank_response(
                data
            )

            if not raw:
                raise RuntimeError(
                    "Empty LBank response."
                )

            candles = []

            for row in raw:

                if isinstance(row, dict):

                    t = (
                        row.get("timestamp")
                        if row.get("timestamp") is not None
                        else row.get("time")
                    )

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
                        else row.get("vol", 0)
                    )

                elif isinstance(
                    row,
                    (list, tuple)
                ):

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

                if candle:
                    candles.append(
                        candle
                    )

            candles = deduplicate(
                candles
            )

            if not candles:
                raise RuntimeError(
                    "Parser returned zero candles."
                )

            return candles

        except Exception as exc:

            last_error = exc

    raise RuntimeError(
        f"LBank request failed: {last_error}"
    )


# ============================================================
# LBANK HISTORICAL DOWNLOAD
#
# IMPORTANT:
# We use a STRICT backward-progress check.
#
# If LBank returns the same page twice:
#   STOP
#
# If the oldest timestamp does not move backward:
#   STOP
#
# Therefore the program can NEVER spin forever.
# ============================================================

def get_lbank_klines(
    symbol,
    interval,
    start_dt,
    end_dt
):

    interval_seconds = (
        interval * 60
    )

    start_ts = int(
        start_dt.timestamp()
    )

    end_ts = int(
        end_dt.timestamp()
    )

    max_chunks = (
        MAX_LBANK_CHUNKS_1H
        if interval == INTERVAL_1H
        else MAX_LBANK_CHUNKS_4H
    )

    all_candles = []

    # کمی عقب‌تر از پایان درخواست می‌کنیم
    # تا candle فعلی باعث مشکل نشود.
    anchor = end_ts

    previous_oldest = None

    previous_signature = None

    for chunk in range(
        1,
        max_chunks + 1
    ):

        print(
            f"    [{symbol}] "
            f"{'1H' if interval == 60 else '4H'} "
            f"request {chunk}/{max_chunks}..."
        )

        candles = lbank_single_request(
            symbol,
            interval,
            anchor,
            LBANK_MAX_SIZE
        )

        if not candles:

            print(
                "    Empty page -> STOP"
            )

            break

        # ====================================================
        # Signature
        # ====================================================

        page_times = tuple(
            c["time"]
            for c in candles
        )

        signature = (
            len(candles),
            page_times[0],
            page_times[-1]
        )

        # ====================================================
        # Same page returned
        # ====================================================

        if signature == previous_signature:

            print(
                "    SAME PAGE DETECTED -> STOP"
            )

            break

        previous_signature = signature

        # ====================================================
        # Add only target-range candles
        # ====================================================

        in_range = [
            c
            for c in candles
            if (
                start_ts
                <= c["time"]
                <= end_ts
            )
        ]

        if in_range:

            all_candles.extend(
                in_range
            )

        oldest_api = min(
            c["time"]
            for c in candles
        )

        newest_api = max(
            c["time"]
            for c in candles
        )

        print(
            f"    received "
            f"{len(candles)} | "
            f"API range "
            f"{utc_time(oldest_api)} -> "
            f"{utc_time(newest_api)}"
        )

        # ====================================================
        # Reached beginning
        # ====================================================

        if oldest_api <= start_ts:

            print(
                "    Historical start reached."
            )

            break

        # ====================================================
        # CRITICAL LOOP PROTECTION
        # ====================================================

        if (
            previous_oldest is not None
            and oldest_api >= previous_oldest
        ):

            print(
                "    BACKWARD PROGRESS FAILED -> STOP"
            )

            break

        previous_oldest = oldest_api

        # ====================================================
        # Move one candle before oldest
        # ====================================================

        new_anchor = (
            oldest_api
            - interval_seconds
        )

        if new_anchor >= anchor:

            print(
                "    Anchor did not move backward -> STOP"
            )

            break

        anchor = new_anchor

        # ====================================================
        # Small delay
        # ====================================================

        time.sleep(0.15)

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

    print(
        f"    FINAL "
        f"{'1H' if interval == 60 else '4H'}: "
        f"{len(result)} candles"
    )

    return result


# ============================================================
# REMOVE FORMING CANDLES
# ============================================================

def remove_forming_candles(
    candles,
    interval
):

    if not candles:
        return []

    now_ts = int(
        utc_now().timestamp()
    )

    interval_seconds = (
        interval * 60
    )

    result = []

    for candle in candles:

        candle_close = (
            candle["time"]
            + interval_seconds
        )

        if candle_close <= now_ts:
            result.append(
                candle
            )

    return result


# ============================================================
# KRAKEN
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
            str(data["error"])
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

    rows = result[
        keys[0]
    ]

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
            candles.append(
                candle
            )

    return deduplicate(
        candles
    )


# ============================================================
# COVERAGE
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

    required_start = int(
        start_dt.timestamp()
    )

    required_end = int(
        end_dt.timestamp()
    )

    actual_start = candles[0]["time"]
    actual_end = candles[-1]["time"]

    candle_seconds = (
        interval * 60
    )

    expected_count = int(
        (
            required_end
            - required_start
        )
        / candle_seconds
    ) + 1

    actual_count = len(
        candles
    )

    total_range = (
        required_end
        - required_start
    )

    covered_range = max(
        0,
        min(
            actual_end,
            required_end
        )
        - max(
            actual_start,
            required_start
        )
    )

    coverage = (
        covered_range
        / total_range
        * 100
        if total_range > 0
        else 0
    )

    count_ratio = (
        actual_count
        / expected_count
        * 100
        if expected_count > 0
        else 0
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

        if gap > candle_seconds * 2:
            large_gaps += 1

    start_ok = (
        actual_start
        <= required_start
        + candle_seconds
    )

    end_ok = (
        actual_end
        >= required_end
        - candle_seconds * 2
    )

    valid = (
        start_ok
        and end_ok
        and coverage >= 99.0
        and count_ratio >= 99.0
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
# VALIDATE
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

        if gap > expected_gap * 10:
            huge_gaps += 1

    if huge_gaps > len(candles) * 0.20:
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
    print("=" * 80)
    print(
        f"DOWNLOADING {symbol}"
    )
    print("=" * 80)

    # ========================================================
    # LBANK
    # ========================================================

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

        # Remove forming candles
        candles_4h = (
            remove_forming_candles(
                candles_4h,
                INTERVAL_4H
            )
        )

        candles_1h = (
            remove_forming_candles(
                candles_1h,
                INTERVAL_1H
            )
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
            f"{symbol} 4H coverage: "
            f"{report_4h['coverage']:.2f}%"
        )

        print(
            f"{symbol} 1H coverage: "
            f"{report_1h['coverage']:.2f}%"
        )

        print(
            f"{symbol} 4H count: "
            f"{len(candles_4h)}"
        )

        print(
            f"{symbol} 1H count: "
            f"{len(candles_1h)}"
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
            and
            report_4h["valid"]
            and
            report_1h["valid"]
        ):

            print(
                f"{symbol}: "
                f"LBANK VERIFIED"
            )

            return (
                candles_4h,
                candles_1h,
                "LBANK"
            )

        print(
            f"{symbol}: "
            f"LBank full-history validation FAILED."
        )

    except Exception as exc:

        print(
            f"{symbol}: "
            f"LBank ERROR -> {exc}"
        )

    # ========================================================
    # KRAKEN FALLBACK
    # ========================================================

    try:

        print(
            f"{symbol}: "
            f"Trying Kraken..."
        )

        candles_4h = get_kraken_klines(
            symbol,
            INTERVAL_4H
        )

        candles_1h = get_kraken_klines(
            symbol,
            INTERVAL_1H
        )

        candles_4h = [
            c
            for c in candles_4h
            if (
                int(start_dt.timestamp())
                <= c["time"]
                <= int(end_dt.timestamp())
            )
        ]

        candles_1h = [
            c
            for c in candles_1h
            if (
                int(start_dt.timestamp())
                <= c["time"]
                <= int(end_dt.timestamp())
            )
        ]

        candles_4h = remove_forming_candles(
            candles_4h,
            INTERVAL_4H
        )

        candles_1h = remove_forming_candles(
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

        if (
            report_4h["valid"]
            and report_1h["valid"]
        ):

            print(
                f"{symbol}: "
                f"KRAKEN VERIFIED"
            )

            return (
                candles_4h,
                candles_1h,
                "KRAKEN"
            )

        print(
            f"{symbol}: "
            f"Kraken does not have "
            f"verified 365-day coverage."
        )

    except Exception as exc:

        print(
            f"{symbol}: "
            f"Kraken ERROR -> {exc}"
        )

    return None, None, "FAILED"


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
        100.0
        / (1.0 + rs_value)
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
# CLOSED 4H
# ============================================================

def get_4h_direction_at_entry(
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

    if len(usable) < EMA200:
        return None

    return get_4h_direction(
        usable
    )


# ============================================================
# BREAKOUT LONG
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

    if current["close"] <= candles[-2]["close"]:
        return False, None

    return True, resistance


# ============================================================
# BREAKOUT SHORT
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

    if current["close"] >= candles[-2]["close"]:
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

    e20 = ema(closes, EMA20)
    e50 = ema(closes, EMA50)
    e200 = ema(closes, EMA200)

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

    e20 = ema(closes, EMA20)
    e50 = ema(closes, EMA50)
    e200 = ema(closes, EMA200)

    return (
        e20 is not None
        and e50 is not None
        and e200 is not None
        and closes[-1] < e200
        and e20 < e50
        and e50 < e200
    )


# ============================================================
# RSI CONFIRMATION
# ============================================================

def rsi_confirmation_long(value):

    return (
        value is not None
        and 50 <= value <= 85
    )


def rsi_confirmation_short(value):

    return (
        value is not None
        and 15 <= value <= 50
    )


# ============================================================
# LEVELS
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

    risk = entry - sl

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

    risk = sl - entry

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

    if len(candles_1h) < EMA200 + 10:
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
                            "direction": "LONG",
                            "setup": "BREAKOUT",
                            "entry_time": current["time"],
                            "entry": entry,
                            "tp": levels["tp"],
                            "sl": levels["sl"],
                            "risk": levels["risk"],
                            "rr": levels["rr"],
                            "risk_atr": levels["risk_atr"],
                            "atr": atr_value,
                            "adx": adx_value,
                            "rsi": rsi_value
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
                            "direction": "SHORT",
                            "setup": "BREAKOUT",
                            "entry_time": current["time"],
                            "entry": entry,
                            "tp": levels["tp"],
                            "sl": levels["sl"],
                            "risk": levels["risk"],
                            "rr": levels["rr"],
                            "risk_atr": levels["risk_atr"],
                            "atr": atr_value,
                            "adx": adx_value,
                            "rsi": rsi_value
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
                "direction": "LONG",
                "setup": "REVERSAL",
                "entry_time": current["time"],
                "entry": entry,
                "tp": levels["tp"],
                "sl": levels["sl"],
                "risk": levels["risk"],
                "rr": levels["rr"],
                "risk_atr": levels["risk_atr"],
                "atr": atr_value,
                "adx": adx_value,
                "rsi": rsi_value
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
                "direction": "SHORT",
                "setup": "REVERSAL",
                "entry_time": current["time"],
                "entry": entry,
                "tp": levels["tp"],
                "sl": levels["sl"],
                "risk": levels["risk"],
                "rr": levels["rr"],
                "risk_atr": levels["risk_atr"],
                "atr": atr_value,
                "adx": adx_value,
                "rsi": rsi_value
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

        # Same candle TP + SL = SL
        if hit_tp and hit_sl:
            return "SL", i

        if hit_sl:
            return "SL", i

        if hit_tp:
            return "TP", i

    return None, None


# ============================================================
# BACKTEST
# ============================================================

def backtest_coin(
    symbol,
    candles_4h,
    candles_1h,
    strategy_start,
    oos_start
):

    trades = []

    strategy_start_ts = int(
        strategy_start.timestamp()
    )

    oos_start_ts = int(
        oos_start.timestamp()
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

        usable_4h = [
            candle
            for candle in candles_4h
            if (
                candle["time"]
                + SECONDS_4H
                <= (
                    entry_candle["time"]
                    + SECONDS_1H
                )
            )
        ]

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

        r_result = (
            TP_R_MULTIPLE
            if result == "TP"
            else -1.0
        )

        trades.append({

            "symbol": symbol,

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

        # No overlapping positions
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

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else float("inf")
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

        max_drawdown = max(
            max_drawdown,
            peak - equity
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
        "expectancy": average_r,
        "max_drawdown": max_drawdown,
        "max_win_streak": max_win,
        "max_loss_streak": max_loss
    }


# ============================================================
# FORMAT
# ============================================================

def fmt_price(price):

    if price >= 1000:
        return f"{price:.2f}"

    if price >= 1:
        return f"{price:.5f}"

    return f"{price:.8f}"


def print_stats(
    title,
    stats
):

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    print(
        f"Trades             : {stats['total']}"
    )

    print(
        f"Wins               : {stats['wins']}"
    )

    print(
        f"Losses             : {stats['losses']}"
    )

    print(
        f"Win Rate           : {stats['win_rate']:.2f}%"
    )

    print(
        f"Net Result         : {stats['net_r']:+.2f}R"
    )

    print(
        f"Average R          : {stats['average_r']:+.3f}R"
    )

    pf = stats["profit_factor"]

    pf_text = (
        "INF"
        if pf == float("inf")
        else f"{pf:.2f}"
    )

    print(
        f"Profit Factor      : {pf_text}"
    )

    print(
        f"Expectancy         : {stats['expectancy']:+.3f}R"
    )

    print(
        f"Max Drawdown       : {stats['max_drawdown']:.2f}R"
    )

    print(
        f"Max Win Streak     : {stats['max_win_streak']}"
    )

    print(
        f"Max Loss Streak    : {stats['max_loss_streak']}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 100)
    print(" SCORE HUNTER PRO v8.10")
    print(" REAL 365-DAY HISTORICAL BACKTEST")
    print("=" * 100)

    print()
    print("4H Trend + 1H Entry")
    print("LONG + SHORT")
    print("BREAKOUT + STRICT REVERSAL")
    print("NO PULLBACK")
    print("CLOSED CANDLE ONLY")
    print("NO LOOK-AHEAD")
    print("ADX >= 20")
    print("RSI confirmation")
    print("SL = 1.5 ATR / Structure")
    print("MAX SL = 3.5 ATR")
    print("TP = 2R")
    print("TP + SL same candle = SL")
    print("NO OVERLAPPING POSITIONS")
    print("PARAMETERS UNCHANGED")

    print()
    print(
        f"History: {HISTORY_DAYS} days"
    )

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
        "Strategy start :",
        utc_time(
            strategy_start.timestamp()
        )
    )

    print(
        "Data start     :",
        utc_time(
            data_start.timestamp()
        )
    )

    print(
        "End            :",
        utc_time(
            end_dt.timestamp()
        )
    )

    print(
        "OOS start      :",
        utc_time(
            oos_start.timestamp()
        )
    )

    all_trades = []

    data_status = {}

    # ========================================================
    # DOWNLOAD EACH COIN
    # ========================================================

    for symbol in COINS:

        try:

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

                print(
                    f"{symbol}: EXCLUDED"
                )

                continue

            print()
            print(
                f"{symbol}: "
                f"4H={len(candles_4h)} | "
                f"1H={len(candles_1h)} | "
                f"SOURCE={source}"
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

            data_status[symbol] = source

            all_trades.extend(
                trades
            )

        except KeyboardInterrupt:

            print()
            print(
                "STOPPED BY USER."
            )

            return

        except Exception as exc:

            print()
            print(
                f"{symbol}: "
                f"FATAL ERROR -> {exc}"
            )

            data_status[symbol] = "FAILED"

    # ========================================================
    # DATA SUMMARY
    # ========================================================

    print()
    print("=" * 100)
    print("DATA SOURCE SUMMARY")
    print("=" * 100)

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
    print("=" * 100)
    print("RESULT BY COIN")
    print("=" * 100)

    for symbol in COINS:

        coin_trades = [
            t
            for t in all_trades
            if t["symbol"] == symbol
        ]

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
            f"Trades {s['total']:3d} | "
            f"W {s['wins']:3d} | "
            f"L {s['losses']:3d} | "
            f"WR {s['win_rate']:6.2f}% | "
            f"R {s['net_r']:+7.2f} | "
            f"PF {pf_text}"
        )

    # ========================================================
    # BY SETUP
    # ========================================================

    print()
    print("=" * 100)
    print("RESULT BY SETUP")
    print("=" * 100)

    for setup in (
        "BREAKOUT",
        "REVERSAL"
    ):

        setup_trades = [
            t
            for t in all_trades
            if t["setup"] == setup
        ]

        s = calculate_stats(
            setup_trades
        )

        print(
            f"{setup:10s} | "
            f"Trades {s['total']:3d} | "
            f"W {s['wins']:3d} | "
            f"L {s['losses']:3d} | "
            f"WR {s['win_rate']:6.2f}% | "
            f"R {s['net_r']:+7.2f}"
        )

    # ========================================================
    # OOS BY COIN
    # ========================================================

    print()
    print("=" * 100)
    print("OOS BY COIN")
    print("=" * 100)

    for symbol in COINS:

        coin_oos = [
            t
            for t in oos_trades
            if t["symbol"] == symbol
        ]

        s = calculate_stats(
            coin_oos
        )

        print(
            f"{symbol:5s} | "
            f"Trades {s['total']:3d} | "
            f"W {s['wins']:3d} | "
            f"L {s['losses']:3d} | "
            f"WR {s['win_rate']:6.2f}% | "
            f"R {s['net_r']:+7.2f}"
        )

    # ========================================================
    # FULL LOG
    # ========================================================

    print()
    print("=" * 100)
    print("FULL TRADE LOG")
    print("=" * 100)

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
            f"{utc_time(trade['entry_time'])} | "
            f"E {fmt_price(trade['entry'])} | "
            f"TP {fmt_price(trade['tp'])} | "
            f"SL {fmt_price(trade['sl'])} | "
            f"{trade['result']:2s} | "
            f"{trade['R']:+.1f}R | "
            f"ADX {trade['adx']:.1f} | "
            f"RSI {trade['rsi']:.1f}"
        )

    # ========================================================
    # ROBUSTNESS
    # ========================================================

    print()
    print("=" * 100)
    print("ROBUSTNESS")
    print("=" * 100)

    print(
        f"Total trades : {len(all_trades)}"
    )

    print(
        f"OOS trades   : {len(oos_trades)}"
    )

    if len(all_trades) >= 100:

        print(
            "Sample size  : PASS"
        )

    else:

        print(
            "Sample size  : < 100"
        )

    if len(oos_trades) >= 30:

        print(
            "OOS sample   : GOOD"
        )

    elif len(oos_trades) >= 15:

        print(
            "OOS sample   : MEDIUM"
        )

    else:

        print(
            "OOS sample   : WEAK"
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "OOS results must NOT be used "
        "to optimize the strategy."
    )

    print()
    print("=" * 100)
    print(
        "SCORE HUNTER PRO v8.10 FINISHED"
    )
    print("=" * 100)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
