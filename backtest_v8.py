"""
WHALE FLOW PRO
1H Bias + 15M Precision Entry
Binance USD-M Futures
1-Year Historical Backtest

Symbols:
BTCUSDT
ETHUSDT
SOLUSDT
XRPUSDT

NO pandas required.
Uses Binance public REST API.

Strategy:
1H:
    EMA20 > EMA50 > EMA200  -> LONG bias
    EMA20 < EMA50 < EMA200  -> SHORT bias
    ADX >= 20
    RSI filter
    VWAP filter

15M:
    Liquidity Sweep
    Market Structure Shift
    Displacement
    FVG
    Volume confirmation
    ATR-based SL
    TP = 2R

Risk:
    1R = SL distance
    TP = 2R
    Max 4 trades/day
    Max daily loss = -2R
    Max 2 correlated positions
    No overlapping positions
"""

import requests
import time
import math
import csv
from datetime import datetime, timedelta, timezone
from collections import defaultdict


# ============================================================
# CONFIG
# ============================================================

BASE_URL = "https://fapi.binance.com"

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT"
]

DAYS = 365

# Indicators
EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14

ADX_MIN = 20.0

# RSI
LONG_RSI_MIN = 52.0
LONG_RSI_MAX = 68.0

SHORT_RSI_MIN = 32.0
SHORT_RSI_MAX = 48.0

# Liquidity
SWEEP_LOOKBACK = 10

# Structure
MSS_LOOKBACK = 8

# Displacement
DISPLACEMENT_ATR = 1.15

# Volume
VOLUME_LOOKBACK = 20
VOLUME_MULTIPLIER = 1.15

# FVG
FVG_MIN_ATR = 0.10

# SL
MIN_SL_ATR = 0.80
MAX_SL_ATR = 1.30

# Risk / reward
RR = 2.0

# Trading limits
MAX_TRADES_PER_DAY = 4
MAX_DAILY_LOSS_R = -2.0

# To avoid taking all highly correlated coins
MAX_SIMULTANEOUS_DIRECTION = 2

# Fees / slippage assumptions
TAKER_FEE = 0.0005
SLIPPAGE = 0.0002

# Save files
TRADES_FILE = "whale_flow_trades.csv"
SUMMARY_FILE = "whale_flow_summary.csv"


# ============================================================
# HTTP
# ============================================================

session = requests.Session()


def get_json(url, params, retries=5):

    for attempt in range(retries):

        try:
            r = session.get(
                url,
                params=params,
                timeout=20
            )

            if r.status_code == 200:
                return r.json()

            if r.status_code == 429:
                time.sleep(2 + attempt * 2)
                continue

            print("HTTP ERROR:", r.status_code, r.text[:200])

        except Exception as e:
            print("REQUEST ERROR:", e)

        time.sleep(1 + attempt)

    raise RuntimeError("Binance API request failed")


# ============================================================
# BINANCE KLINES
# ============================================================

def get_klines(symbol, interval, start_ms, end_ms):

    data = []

    current = start_ms

    while current < end_ms:

        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current,
            "endTime": end_ms,
            "limit": 1500
        }

        batch = get_json(
            BASE_URL + "/fapi/v1/klines",
            params
        )

        if not batch:
            break

        for k in batch:

            ts = int(k[0])

            if ts >= start_ms and ts <= end_ms:
                data.append({
                    "time": ts,
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5])
                })

        last_ts = int(batch[-1][0])

        next_ts = last_ts + 1

        if next_ts <= current:
            break

        current = next_ts

        if len(batch) < 1500:
            break

        time.sleep(0.05)

    # remove duplicates
    unique = {}

    for x in data:
        unique[x["time"]] = x

    return [
        unique[k]
        for k in sorted(unique)
    ]


# ============================================================
# INDICATORS
# ============================================================

def sma(values, period):

    if len(values) < period:
        return None

    return sum(values[-period:]) / period


def ema_series(values, period):

    if len(values) < period:
        return [None] * len(values)

    result = [None] * len(values)

    seed = sum(values[:period]) / period

    result[period - 1] = seed

    multiplier = 2.0 / (period + 1)

    prev = seed

    for i in range(period, len(values)):

        prev = (
            values[i] - prev
        ) * multiplier + prev

        result[i] = prev

    return result


def rsi_series(closes, period=14):

    result = [None] * len(closes)

    if len(closes) <= period:
        return result

    gains = []
    losses = []

    for i in range(1, len(closes)):

        change = closes[i] - closes[i - 1]

        gains.append(max(change, 0))
        losses.append(max(-change, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100 - (100 / (1 + rs))

    for i in range(period + 1, len(closes)):

        gain = gains[i - 1]
        loss = losses[i - 1]

        avg_gain = (
            (avg_gain * (period - 1)) + gain
        ) / period

        avg_loss = (
            (avg_loss * (period - 1)) + loss
        ) / period

        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100 - (100 / (1 + rs))

    return result


def atr_series(candles, period=14):

    tr = [None] * len(candles)

    for i in range(len(candles)):

        if i == 0:
            tr[i] = (
                candles[i]["high"]
                - candles[i]["low"]
            )
        else:

            h = candles[i]["high"]
            l = candles[i]["low"]
            pc = candles[i - 1]["close"]

            tr[i] = max(
                h - l,
                abs(h - pc),
                abs(l - pc)
            )

    result = [None] * len(candles)

    if len(candles) <= period:
        return result

    first = sum(
        tr[1:period + 1]
    ) / period

    result[period] = first

    prev = first

    for i in range(period + 1, len(candles)):

        prev = (
            (prev * (period - 1))
            + tr[i]
        ) / period

        result[i] = prev

    return result


def adx_series(candles, period=14):

    n = len(candles)

    if n < period * 2 + 2:
        return [None] * n

    tr = [0.0] * n
    plus_dm = [0.0] * n
    minus_dm = [0.0] * n

    for i in range(1, n):

        h = candles[i]["high"]
        l = candles[i]["low"]

        ph = candles[i - 1]["high"]
        pl = candles[i - 1]["low"]
        pc = candles[i - 1]["close"]

        tr[i] = max(
            h - l,
            abs(h - pc),
            abs(l - pc)
        )

        up = h - ph
        down = pl - l

        if up > down and up > 0:
            plus_dm[i] = up

        if down > up and down > 0:
            minus_dm[i] = down

    atr = [None] * n
    p_dm = [None] * n
    m_dm = [None] * n

    atr[period] = sum(
        tr[1:period + 1]
    )

    p_dm[period] = sum(
        plus_dm[1:period + 1]
    )

    m_dm[period] = sum(
        minus_dm[1:period + 1]
    )

    dx = [None] * n

    for i in range(period, n):

        if i > period:

            atr[i] = (
                atr[i - 1]
                - atr[i - 1] / period
                + tr[i]
            )

            p_dm[i] = (
                p_dm[i - 1]
                - p_dm[i - 1] / period
                + plus_dm[i]
            )

            m_dm[i] = (
                m_dm[i - 1]
                - m_dm[i - 1] / period
                + minus_dm[i]
            )

        if atr[i] == 0:
            continue

        pdi = 100 * p_dm[i] / atr[i]
        mdi = 100 * m_dm[i] / atr[i]

        denominator = pdi + mdi

        if denominator != 0:

            dx[i] = (
                100
                * abs(pdi - mdi)
                / denominator
            )

    adx = [None] * n

    valid_dx = [
        x for x in dx
        if x is not None
    ]

    if len(valid_dx) < period:
        return adx

    first_adx = sum(
        valid_dx[:period]
    ) / period

    first_index = None

    count = 0

    for i in range(n):

        if dx[i] is not None:

            count += 1

            if count == period:
                first_index = i
                break

    if first_index is None:
        return adx

    adx[first_index] = first_adx

    prev = first_adx

    for i in range(first_index + 1, n):

        if dx[i] is not None:

            prev = (
                (prev * (period - 1))
                + dx[i]
            ) / period

            adx[i] = prev

    return adx


# ============================================================
# VWAP
# ============================================================

def add_daily_vwap(candles):

    cumulative_pv = 0.0
    cumulative_volume = 0.0
    current_day = None

    for c in candles:

        dt = datetime.fromtimestamp(
            c["time"] / 1000,
            tz=timezone.utc
        )

        day = dt.date()

        if day != current_day:

            current_day = day
            cumulative_pv = 0.0
            cumulative_volume = 0.0

        typical = (
            c["high"]
            + c["low"]
            + c["close"]
        ) / 3

        cumulative_pv += (
            typical * c["volume"]
        )

        cumulative_volume += c["volume"]

        if cumulative_volume > 0:
            c["vwap"] = (
                cumulative_pv
                / cumulative_volume
            )
        else:
            c["vwap"] = c["close"]


# ============================================================
# INDICATOR PREPARATION
# ============================================================

def prepare_1h(candles):

    closes = [
        c["close"]
        for c in candles
    ]

    ema20 = ema_series(
        closes,
        EMA_FAST
    )

    ema50 = ema_series(
        closes,
        EMA_MID
    )

    ema200 = ema_series(
        closes,
        EMA_SLOW
    )

    rsi = rsi_series(
        closes,
        RSI_PERIOD
    )

    atr = atr_series(
        candles,
        ATR_PERIOD
    )

    adx = adx_series(
        candles,
        ADX_PERIOD
    )

    add_daily_vwap(candles)

    for i, c in enumerate(candles):

        c["ema20"] = ema20[i]
        c["ema50"] = ema50[i]
        c["ema200"] = ema200[i]
        c["rsi"] = rsi[i]
        c["atr"] = atr[i]
        c["adx"] = adx[i]


def prepare_15m(candles):

    atr = atr_series(
        candles,
        ATR_PERIOD
    )

    avg_vol = [None] * len(candles)

    for i in range(
        VOLUME_LOOKBACK,
        len(candles)
    ):

        vals = [
            candles[j]["volume"]
            for j in range(
                i - VOLUME_LOOKBACK,
                i
            )
        ]

        avg_vol[i] = (
            sum(vals) / len(vals)
        )

    for i, c in enumerate(candles):

        c["atr"] = atr[i]
        c["avg_volume"] = avg_vol[i]


# ============================================================
# TIME MAPPING
# ============================================================

def find_1h_context(candles_1h, timestamp):

    # latest 1H candle that has CLOSED
    lo = 0
    hi = len(candles_1h) - 1

    answer = None

    while lo <= hi:

        mid = (lo + hi) // 2

        if candles_1h[mid]["time"] <= timestamp:
            answer = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return answer


# ============================================================
# MARKET STRUCTURE
# ============================================================

def previous_swing_high(candles, i, lookback):

    start = max(
        0,
        i - lookback
    )

    highs = [
        candles[j]["high"]
        for j in range(start, i)
    ]

    if not highs:
        return None

    return max(highs)


def previous_swing_low(candles, i, lookback):

    start = max(
        0,
        i - lookback
    )

    lows = [
        candles[j]["low"]
        for j in range(start, i)
    ]

    if not lows:
        return None

    return min(lows)


# ============================================================
# SIGNAL DETECTION
# ============================================================

def long_signal(candles, i):

    if i < max(
        SWEEP_LOOKBACK,
        MSS_LOOKBACK,
        VOLUME_LOOKBACK
    ) + 3:
        return None

    c = candles[i]

    atr = c["atr"]

    if atr is None:
        return None

    # --------------------------------------------------------
    # 1. Liquidity Sweep
    # Current candle takes previous low,
    # but closes back above it.
    # --------------------------------------------------------

    prior_low = previous_swing_low(
        candles,
        i,
        SWEEP_LOOKBACK
    )

    if prior_low is None:
        return None

    sweep = (
        c["low"] < prior_low
        and c["close"] > prior_low
    )

    if not sweep:
        return None

    # --------------------------------------------------------
    # 2. MSS
    # After sweep, current close must break
    # a recent lower high.
    # --------------------------------------------------------

    mss_level = previous_swing_high(
        candles,
        i,
        MSS_LOOKBACK
    )

    if mss_level is None:
        return None

    # We require the displacement candle itself
    # to close above structure.
    if c["close"] <= mss_level:
        return None

    # --------------------------------------------------------
    # 3. Displacement
    # --------------------------------------------------------

    body = abs(
        c["close"] - c["open"]
    )

    if body < DISPLACEMENT_ATR * atr:
        return None

    if c["close"] <= c["open"]:
        return None

    # --------------------------------------------------------
    # 4. Volume
    # --------------------------------------------------------

    if c["avg_volume"] is None:
        return None

    if c["volume"] < (
        c["avg_volume"]
        * VOLUME_MULTIPLIER
    ):
        return None

    # --------------------------------------------------------
    # 5. FVG
    #
    # Bullish FVG:
    # current candle low > candle two bars ago high
    # --------------------------------------------------------

    if i < 2:
        return None

    fvg_low = candles[i - 2]["high"]
    fvg_high = c["low"]

    gap = fvg_high - fvg_low

    if gap < FVG_MIN_ATR * atr:
        return None

    # --------------------------------------------------------
    # Entry = 50% FVG
    # --------------------------------------------------------

    entry = (
        fvg_low + fvg_high
    ) / 2

    sweep_low = c["low"]

    sl_distance = (
        entry - sweep_low
    )

    if sl_distance <= 0:
        return None

    sl_atr = sl_distance / atr

    if sl_atr < MIN_SL_ATR:
        return None

    if sl_atr > MAX_SL_ATR:
        return None

    stop = sweep_low

    target = (
        entry
        + sl_distance * RR
    )

    return {
        "side": "LONG",
        "entry": entry,
        "stop": stop,
        "target": target,
        "atr": atr,
        "score": 0
    }


def short_signal(candles, i):

    if i < max(
        SWEEP_LOOKBACK,
        MSS_LOOKBACK,
        VOLUME_LOOKBACK
    ) + 3:
        return None

    c = candles[i]

    atr = c["atr"]

    if atr is None:
        return None

    # --------------------------------------------------------
    # 1. Liquidity Sweep
    # --------------------------------------------------------

    prior_high = previous_swing_high(
        candles,
        i,
        SWEEP_LOOKBACK
    )

    if prior_high is None:
        return None

    sweep = (
        c["high"] > prior_high
        and c["close"] < prior_high
    )

    if not sweep:
        return None

    # --------------------------------------------------------
    # 2. MSS
    # --------------------------------------------------------

    mss_level = previous_swing_low(
        candles,
        i,
        MSS_LOOKBACK
    )

    if mss_level is None:
        return None

    if c["close"] >= mss_level:
        return None

    # --------------------------------------------------------
    # 3. Displacement
    # --------------------------------------------------------

    body = abs(
        c["close"] - c["open"]
    )

    if body < DISPLACEMENT_ATR * atr:
        return None

    if c["close"] >= c["open"]:
        return None

    # --------------------------------------------------------
    # 4. Volume
    # --------------------------------------------------------

    if c["avg_volume"] is None:
        return None

    if c["volume"] < (
        c["avg_volume"]
        * VOLUME_MULTIPLIER
    ):
        return None

    # --------------------------------------------------------
    # 5. Bearish FVG
    # current high < candle two bars ago low
    # --------------------------------------------------------

    if i < 2:
        return None

    fvg_high = candles[i - 2]["low"]
    fvg_low = c["high"]

    gap = fvg_high - fvg_low

    if gap < FVG_MIN_ATR * atr:
        return None

    entry = (
        fvg_low + fvg_high
    ) / 2

    sweep_high = c["high"]

    sl_distance = (
        sweep_high - entry
    )

    if sl_distance <= 0:
        return None

    sl_atr = sl_distance / atr

    if sl_atr < MIN_SL_ATR:
        return None

    if sl_atr > MAX_SL_ATR:
        return None

    stop = sweep_high

    target = (
        entry
        - sl_distance * RR
    )

    return {
        "side": "SHORT",
        "entry": entry,
        "stop": stop,
        "target": target,
        "atr": atr,
        "score": 0
    }


# ============================================================
# 1H FILTER
# ============================================================

def get_bias(c):

    if (
        c["ema20"] is None
        or c["ema50"] is None
        or c["ema200"] is None
        or c["rsi"] is None
        or c["adx"] is None
        or c["vwap"] is None
    ):
        return None

    # LONG
    if (
        c["ema20"] > c["ema50"]
        and c["ema50"] > c["ema200"]
        and c["close"] > c["ema20"]
        and c["adx"] >= ADX_MIN
        and LONG_RSI_MIN <= c["rsi"] <= LONG_RSI_MAX
        and c["close"] > c["vwap"]
    ):
        return "LONG"

    # SHORT
    if (
        c["ema20"] < c["ema50"]
        and c["ema50"] < c["ema200"]
        and c["close"] < c["ema20"]
        and c["adx"] >= ADX_MIN
        and SHORT_RSI_MIN <= c["rsi"] <= SHORT_RSI_MAX
        and c["close"] < c["vwap"]
    ):
        return "SHORT"

    return None


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(
    candles,
    start_index,
    signal
):

    side = signal["side"]

    entry = signal["entry"]
    stop = signal["stop"]
    target = signal["target"]

    # We enter only when price returns to FVG.
    # Search future candles for entry touch.

    for j in range(
        start_index + 1,
        len(candles)
    ):

        c = candles[j]

        high = c["high"]
        low = c["low"]

        # ----------------------------------------------------
        # LONG ENTRY
        # ----------------------------------------------------

        if side == "LONG":

            if low <= entry <= high:

                # Apply slippage
                actual_entry = (
                    entry * (1 + SLIPPAGE)
                )

                risk = (
                    actual_entry - stop
                )

                actual_target = (
                    actual_entry
                    + risk * RR
                )

                for k in range(
                    j,
                    len(candles)
                ):

                    bar = candles[k]

                    hit_sl = (
                        bar["low"]
                        <= stop
                    )

                    hit_tp = (
                        bar["high"]
                        >= actual_target
                    )

                    # Conservative rule:
                    # if both occur same candle -> SL
                    if hit_sl and hit_tp:

                        return {
                            "result": "LOSS",
                            "r": -1.0,
                            "entry": actual_entry,
                            "exit": stop,
                            "entry_index": j,
                            "exit_index": k
                        }

                    if hit_sl:

                        return {
                            "result": "LOSS",
                            "r": -1.0,
                            "entry": actual_entry,
                            "exit": stop,
                            "entry_index": j,
                            "exit_index": k
                        }

                    if hit_tp:

                        return {
                            "result": "WIN",
                            "r": RR,
                            "entry": actual_entry,
                            "exit": actual_target,
                            "entry_index": j,
                            "exit_index": k
                        }

                return None

        # ----------------------------------------------------
        # SHORT ENTRY
        # ----------------------------------------------------

        else:

            if low <= entry <= high:

                actual_entry = (
                    entry * (1 - SLIPPAGE)
                )

                risk = (
                    stop - actual_entry
                )

                actual_target = (
                    actual_entry
                    - risk * RR
                )

                for k in range(
                    j,
                    len(candles)
                ):

                    bar = candles[k]

                    hit_sl = (
                        bar["high"]
                        >= stop
                    )

                    hit_tp = (
                        bar["low"]
                        <= actual_target
                    )

                    if hit_sl and hit_tp:

                        return {
                            "result": "LOSS",
                            "r": -1.0,
                            "entry": actual_entry,
                            "exit": stop,
                            "entry_index": j,
                            "exit_index": k
                        }

                    if hit_sl:

                        return {
                            "result": "LOSS",
                            "r": -1.0,
                            "entry": actual_entry,
                            "exit": stop,
                            "entry_index": j,
                            "exit_index": k
                        }

                    if hit_tp:

                        return {
                            "result": "WIN",
                            "r": RR,
                            "entry": actual_entry,
                            "exit": actual_target,
                            "entry_index": j,
                            "exit_index": k
                        }

                return None

    return None


# ============================================================
# BACKTEST ONE SYMBOL
# ============================================================

def backtest_symbol(symbol, start_ms, end_ms):

    print("\n" + "=" * 70)
    print("DOWNLOADING:", symbol)
    print("=" * 70)

    candles_1h = get_klines(
        symbol,
        "1h",
        start_ms,
        end_ms
    )

    candles_15m = get_klines(
        symbol,
        "15m",
        start_ms,
        end_ms
    )

    print(
        "1H candles:",
        len(candles_1h)
    )

    print(
        "15M candles:",
        len(candles_15m)
    )

    if len(candles_1h) < 300:
        print("Not enough 1H data.")
        return []

    if len(candles_15m) < 1000:
        print("Not enough 15M data.")
        return []

    print("Calculating indicators...")

    prepare_1h(candles_1h)
    prepare_15m(candles_15m)

    trades = []

    daily_trades = defaultdict(int)
    daily_r = defaultdict(float)

    i = 250

    while i < len(candles_15m) - 10:

        c15 = candles_15m[i]

        timestamp = c15["time"]

        dt = datetime.fromtimestamp(
            timestamp / 1000,
            tz=timezone.utc
        )

        day = dt.date()

        # Daily limits
        if daily_trades[day] >= MAX_TRADES_PER_DAY:
            i += 1
            continue

        if daily_r[day] <= MAX_DAILY_LOSS_R:
            i += 1
            continue

        # ----------------------------------------------------
        # 1H FILTER
        # ----------------------------------------------------

        idx_1h = find_1h_context(
            candles_1h,
            timestamp
        )

        if idx_1h is None:
            i += 1
            continue

        c1h = candles_1h[idx_1h]

        bias = get_bias(c1h)

        if bias is None:
            i += 1
            continue

        # ----------------------------------------------------
        # SIGNAL
        # ----------------------------------------------------

        if bias == "LONG":

            signal = long_signal(
                candles_15m,
                i
            )

        else:

            signal = short_signal(
                candles_15m,
                i
            )

        if signal is None:
            i += 1
            continue

        # ----------------------------------------------------
        # SIMULATE
        # ----------------------------------------------------

        result = simulate_trade(
            candles_15m,
            i,
            signal
        )

        if result is None:
            i += 1
            continue

        entry_index = result["entry_index"]
        exit_index = result["exit_index"]

        entry_candle = candles_15m[
            entry_index
        ]

        exit_candle = candles_15m[
            exit_index
        ]

        entry_time = datetime.fromtimestamp(
            entry_candle["time"] / 1000,
            tz=timezone.utc
        )

        exit_time = datetime.fromtimestamp(
            exit_candle["time"] / 1000,
            tz=timezone.utc
        )

        # Fees in R terms.
        # Approximate round-trip fee based on price movement.
        fee_r = (
            2 * TAKER_FEE
        ) / max(
            abs(
                result["entry"]
                - signal["stop"]
            )
            / result["entry"],
            1e-9
        )

        net_r = result["r"] - fee_r

        trade = {
            "symbol": symbol,
            "side": signal["side"],
            "signal_time": dt.isoformat(),
            "entry_time": entry_time.isoformat(),
            "exit_time": exit_time.isoformat(),
            "entry": result["entry"],
            "stop": signal["stop"],
            "target": signal["target"],
            "exit": result["exit"],
            "gross_r": result["r"],
            "fee_r": fee_r,
            "net_r": net_r,
            "result": (
                "WIN"
                if net_r > 0
                else "LOSS"
            )
        }

        trades.append(trade)

        daily_trades[day] += 1
        daily_r[day] += net_r

        # Jump to exit candle to avoid overlapping positions
        i = max(
            i + 1,
            exit_index + 1
        )

    return trades


# ============================================================
# STATISTICS
# ============================================================

def calculate_stats(trades):

    if not trades:
        return {
            "trades": 0
        }

    wins = [
        t for t in trades
        if t["net_r"] > 0
    ]

    losses = [
        t for t in trades
        if t["net_r"] <= 0
    ]

    total_r = sum(
        t["net_r"]
        for t in trades
    )

    win_rate = (
        len(wins)
        / len(trades)
        * 100
    )

    gross_profit = sum(
        t["net_r"]
        for t in wins
    )

    gross_loss = abs(
        sum(
            t["net_r"]
            for t in losses
        )
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit
            / gross_loss
        )
    else:
        profit_factor = float("inf")

    # Equity curve in R
    equity = 0.0
    peak = 0.0
    max_dd = 0.0

    for t in trades:

        equity += t["net_r"]

        if equity > peak:
            peak = equity

        dd = peak - equity

        if dd > max_dd:
            max_dd = dd

    # Consecutive losses
    max_consecutive_losses = 0
    current_losses = 0

    for t in trades:

        if t["net_r"] <= 0:

            current_losses += 1

            max_consecutive_losses = max(
                max_consecutive_losses,
                current_losses
            )

        else:
            current_losses = 0

    # Average R
    avg_r = (
        total_r
        / len(trades)
    )

    # Average trades/day
    days = set()

    for t in trades:

        d = t["entry_time"][:10]
        days.add(d)

    avg_trades_day = (
        len(trades)
        / max(len(days), 1)
    )

    # Expectancy
    expectancy = (
        (win_rate / 100) * RR
        - ((100 - win_rate) / 100)
    )

    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "total_r": total_r,
        "profit_factor": profit_factor,
        "max_drawdown_r": max_dd,
        "max_consecutive_losses":
            max_consecutive_losses,
        "average_r": avg_r,
        "average_trades_day":
            avg_trades_day,
        "expectancy_r":
            expectancy
    }


# ============================================================
# PRINT STATS
# ============================================================

def print_stats(title, stats):

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)

    if stats["trades"] == 0:

        print("NO TRADES")

        return

    print(
        "Trades:",
        stats["trades"]
    )

    print(
        "Wins:",
        stats["wins"]
    )

    print(
        "Losses:",
        stats["losses"]
    )

    print(
        "Win Rate:",
        f"{stats['win_rate']:.2f}%"
    )

    print(
        "Total R:",
        f"{stats['total_r']:.2f}"
    )

    print(
        "Profit Factor:",
        f"{stats['profit_factor']:.2f}"
    )

    print(
        "Max Drawdown:",
        f"{stats['max_drawdown_r']:.2f}R"
    )

    print(
        "Max Consecutive Losses:",
        stats["max_consecutive_losses"]
    )

    print(
        "Average R:",
        f"{stats['average_r']:.4f}"
    )

    print(
        "Average Trades/Day:",
        f"{stats['average_trades_day']:.2f}"
    )

    print(
        "Theoretical Expectancy:",
        f"{stats['expectancy_r']:.4f}R"
    )


# ============================================================
# SAVE CSV
# ============================================================

def save_trades(trades):

    if not trades:
        return

    fields = list(
        trades[0].keys()
    )

    with open(
        TRADES_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(trades)


def save_summary(rows):

    if not rows:
        return

    fields = list(
        rows[0].keys()
    )

    with open(
        SUMMARY_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    now = datetime.now(
        timezone.utc
    )

    end = now

    start = now - timedelta(
        days=DAYS
    )

    start_ms = int(
        start.timestamp() * 1000
    )

    end_ms = int(
        end.timestamp() * 1000
    )

    print("\n")
    print("#" * 70)
    print("WHALE FLOW PRO")
    print("BINANCE FUTURES 1-YEAR BACKTEST")
    print("#" * 70)

    print(
        "Period:",
        start.isoformat(),
        "->",
        end.isoformat()
    )

    print(
        "Symbols:",
        ", ".join(SYMBOLS)
    )

    print(
        "RR:",
        f"1:{RR}"
    )

    print(
        "Max trades/day:",
        MAX_TRADES_PER_DAY
    )

    print(
        "Max daily loss:",
        MAX_DAILY_LOSS_R,
        "R"
    )

    all_trades = []
    summary_rows = []

    # --------------------------------------------------------
    # EACH SYMBOL
    # --------------------------------------------------------

    for symbol in SYMBOLS:

        try:

            trades = backtest_symbol(
                symbol,
                start_ms,
                end_ms
            )

            all_trades.extend(
                trades
            )

            stats = calculate_stats(
                trades
            )

            print_stats(
                symbol,
                stats
            )

            row = {
                "symbol": symbol,
                **stats
            }

            summary_rows.append(row)

        except Exception as e:

            print(
                "\nERROR IN",
                symbol,
                ":",
                e
            )

    # --------------------------------------------------------
    # COMBINED
    # --------------------------------------------------------

    combined = calculate_stats(
        all_trades
    )

    print_stats(
        "ALL 4 COINS COMBINED",
        combined
    )

    summary_rows.append({
        "symbol": "ALL",
        **combined
    })

    # --------------------------------------------------------
    # LONG / SHORT
    # --------------------------------------------------------

    longs = [
        t for t in all_trades
        if t["side"] == "LONG"
    ]

    shorts = [
        t for t in all_trades
        if t["side"] == "SHORT"
    ]

    print_stats(
        "LONG ONLY",
        calculate_stats(longs)
    )

    print_stats(
        "SHORT ONLY",
        calculate_stats(shorts)
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_trades(
        all_trades
    )

    save_summary(
        summary_rows
    )

    print("\n")
    print("=" * 70)
    print("BACKTEST FINISHED")
    print("=" * 70)

    print(
        "Trades file:",
        TRADES_FILE
    )

    print(
        "Summary file:",
        SUMMARY_FILE
    )

    print("\nIMPORTANT:")
    print(
        "A 70% win rate is NOT assumed."
    )

    print(
        "The backtest must prove whether the"
        " strategy reaches it."
    )


if __name__ == "__main__":
    main()
