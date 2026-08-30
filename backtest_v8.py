import json
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone


# ============================================================
# SCORE HUNTER v11
#
# REAL LBANK DATA
#
# STRATEGY:
#   1H REGIME / TREND
#   +
#   15M BREAKOUT
#   +
#   RETEST / MOMENTUM CONFIRMATION
#
# NO PANDAS
# NO NUMPY
# CLOSED CANDLES ONLY
#
# FEATURES:
#   - Real LBank historical K-lines
#   - Pagination
#   - 1H trend filter
#   - EMA 50 / 200
#   - ADX trend-strength filter
#   - 15M structure breakout
#   - Retest confirmation
#   - Relative volume
#   - ATR stop
#   - Dynamic structural stop
#   - TP = configurable R
#   - Fees
#   - Slippage
#   - Cooldown after trade
#   - One position per symbol
#   - In-Sample / Out-of-Sample
#   - Detailed statistics
# ============================================================


# ============================================================
# CONFIG
# ============================================================

INITIAL_CAPITAL = 1000.0

RISK_PER_TRADE = 0.01

# Reward/risk target.
# We start at 1.7R instead of forcing 2R.
RR = 1.70

# ATR stop multiplier
ATR_SL_MULT = 1.40

# Structure stop buffer
STRUCTURE_BUFFER_ATR = 0.15

# Minimum acceptable reward/risk
MIN_RR = 1.40

# EMA
EMA_FAST = 50
EMA_SLOW = 200

# ADX
ADX_PERIOD = 14
ADX_MIN = 18.0

# RSI
RSI_PERIOD = 14

# 15M breakout structure
BREAKOUT_LOOKBACK = 20

# Retest window after breakout
RETEST_LOOKBACK = 5

# Volume
VOLUME_LOOKBACK = 20
RELATIVE_VOLUME_MIN = 1.15

# Candle strength
MIN_BODY_RATIO = 0.45

# Momentum
MIN_ATR_BODY = 0.20

# Cooldown after a completed trade
COOLDOWN_BARS = 4

# Maximum holding period.
# Prevents dead trades staying open forever.
MAX_HOLDING_BARS = 32

# Trading costs.
# These are deliberately configurable.
FEE_RATE = 0.0010       # 0.10% per side
SLIPPAGE_RATE = 0.0003  # 0.03% per side

# Data
LBANK_BASE = "https://api.lbank.info"

# LBank K-line supports max 2000 bars/request.
PAGE_SIZE = 2000

# Number of pages.
# 15M * 2000 * 12 = roughly 416 days.
# 1H * 2000 * 12 = roughly 500 days.
PAGES = 12

# Symbols
SYMBOLS = [
    "btc_usdt",
    "eth_usdt",
    "sol_usdt",
    "xrp_usdt",
    "ada_usdt",
    "link_usdt",
]

# OOS percentage
OOS_PERCENT = 0.25


# ============================================================
# HTTP
# ============================================================

def http_get(url, params):

    query = urllib.parse.urlencode(params)

    full_url = url + "?" + query

    request = urllib.request.Request(
        full_url,
        headers={
            "User-Agent": "SCORE-HUNTER-V11"
        }
    )

    with urllib.request.urlopen(
        request,
        timeout=30
    ) as response:

        return json.loads(
            response.read().decode("utf-8")
        )


# ============================================================
# LBANK KLINE
# ============================================================

def get_lbank_klines(
    symbol,
    candle_type,
    pages=PAGES
):

    print(
        f"Downloading {symbol} "
        f"{candle_type}..."
    )

    all_data = []

    # Start from current time.
    end_time = int(
        time.time()
    )

    for page in range(pages):

        try:

            data = http_get(
                LBANK_BASE + "/v2/kline.do",
                {
                    "symbol": symbol,
                    "size": PAGE_SIZE,
                    "type": candle_type,
                    "time": end_time
                }
            )

        except Exception as e:

            print(
                f"{symbol} "
                f"{candle_type} ERROR: "
                f"{type(e).__name__}: {e}"
            )

            break

        rows = data.get(
            "data",
            []
        )

        if not rows:
            break

        for row in rows:

            if len(row) < 6:
                continue

            all_data.append({
                "time": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5])
            })

        print(
            f"{symbol} "
            f"{candle_type}: "
            f"{len(all_data)} candles"
        )

        oldest = min(
            int(row[0])
            for row in rows
        )

        # Move backwards.
        new_end = oldest - 1

        if new_end >= end_time:
            break

        end_time = new_end

        if len(rows) < PAGE_SIZE:
            break

        time.sleep(0.15)

    # Remove duplicates
    unique = {}

    for candle in all_data:

        unique[
            candle["time"]
        ] = candle

    candles = list(
        unique.values()
    )

    candles.sort(
        key=lambda x: x["time"]
    )

    # Remove the newest candle because
    # it may still be forming.
    if len(candles) > 1:

        candles = candles[:-1]

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
        2.0 / (period + 1.0)
    )

    for x in values[period:]:

        value = (
            (x - value)
            * multiplier
            + value
        )

    return value


def ema_series(candles, period):

    result = [
        None
        for _ in candles
    ]

    closes = [
        c["close"]
        for c in candles
    ]

    if len(closes) < period:
        return result

    value = sum(
        closes[:period]
    ) / period

    result[period - 1] = value

    multiplier = (
        2.0 / (period + 1.0)
    )

    for i in range(
        period,
        len(closes)
    ):

        value = (
            (closes[i] - value)
            * multiplier
            + value
        )

        result[i] = value

    return result


# ============================================================
# ATR
# ============================================================

def atr_series(
    candles,
    period=14
):

    result = [
        None
        for _ in candles
    ]

    if len(candles) <= period:
        return result

    trs = []

    for i in range(
        len(candles)
    ):

        if i == 0:

            tr = (
                candles[i]["high"]
                - candles[i]["low"]
            )

        else:

            high = candles[i]["high"]
            low = candles[i]["low"]
            prev_close = (
                candles[i - 1]["close"]
            )

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

        trs.append(tr)

    value = sum(
        trs[1:period + 1]
    ) / period

    result[period] = value

    for i in range(
        period + 1,
        len(candles)
    ):

        value = (
            (
                value
                * (period - 1)
            )
            + trs[i]
        ) / period

        result[i] = value

    return result


# ============================================================
# RSI
# ============================================================

def rsi_series(
    candles,
    period=14
):

    result = [
        None
        for _ in candles
    ]

    if len(candles) <= period:
        return result

    gains = []
    losses = []

    for i in range(
        1,
        len(candles)
    ):

        change = (
            candles[i]["close"]
            - candles[i - 1]["close"]
        )

        gains.append(
            max(change, 0.0)
        )

        losses.append(
            max(-change, 0.0)
        )

    avg_gain = sum(
        gains[:period]
    ) / period

    avg_loss = sum(
        losses[:period]
    ) / period

    def calculate():
        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss

        return 100.0 - (
            100.0 / (1.0 + rs)
        )

    result[period] = calculate()

    for i in range(
        period,
        len(gains)
    ):

        avg_gain = (
            (
                avg_gain
                * (period - 1)
            )
            + gains[i]
        ) / period

        avg_loss = (
            (
                avg_loss
                * (period - 1)
            )
            + losses[i]
        ) / period

        index = i + 1

        result[index] = calculate()

    return result


# ============================================================
# ADX
# ============================================================

def adx_series(
    candles,
    period=14
):

    result = [
        None
        for _ in candles
    ]

    if len(candles) < (
        period * 2 + 2
    ):
        return result

    tr = []
    plus_dm = []
    minus_dm = []

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

        true_range = max(
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

        pdm = (
            up_move
            if (
                up_move > down_move
                and up_move > 0
            )
            else 0.0
        )

        mdm = (
            down_move
            if (
                down_move > up_move
                and down_move > 0
            )
            else 0.0
        )

        tr.append(true_range)
        plus_dm.append(pdm)
        minus_dm.append(mdm)

    if len(tr) < period * 2:
        return result

    atr_value = sum(
        tr[:period]
    ) / period

    plus_value = sum(
        plus_dm[:period]
    ) / period

    minus_value = sum(
        minus_dm[:period]
    ) / period

    dx = []

    for i in range(
        period,
        len(tr)
    ):

        atr_value = (
            (
                atr_value
                * (period - 1)
            )
            + tr[i]
        ) / period

        plus_value = (
            (
                plus_value
                * (period - 1)
            )
            + plus_dm[i]
        ) / period

        minus_value = (
            (
                minus_value
                * (period - 1)
            )
            + minus_dm[i]
        ) / period

        if atr_value == 0:

            dx.append(0.0)
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

        denom = (
            plus_di
            + minus_di
        )

        if denom == 0:

            dx.append(0.0)

        else:

            dx.append(
                100.0
                * abs(
                    plus_di
                    - minus_di
                )
                / denom
            )

    if len(dx) < period:
        return result

    adx_value = sum(
        dx[:period]
    ) / period

    first_index = (
        period * 2
    )

    if first_index < len(result):

        result[
            first_index
        ] = adx_value

    for j in range(
        period,
        len(dx)
    ):

        adx_value = (
            (
                adx_value
                * (period - 1)
            )
            + dx[j]
        ) / period

        index = (
            j
            + period
            + 1
        )

        if index < len(result):

            result[index] = adx_value

    return result


# ============================================================
# CANDLE HELPERS
# ============================================================

def body_ratio(candle):

    rng = (
        candle["high"]
        - candle["low"]
    )

    if rng <= 0:
        return 0.0

    return abs(
        candle["close"]
        - candle["open"]
    ) / rng


def bullish_candle(candle):

    return (
        candle["close"]
        > candle["open"]
    )


def bearish_candle(candle):

    return (
        candle["close"]
        < candle["open"]
    )


# ============================================================
# VOLUME
# ============================================================

def relative_volume(
    candles,
    index,
    lookback
):

    if index < lookback:
        return None

    values = [
        candles[j]["volume"]
        for j in range(
            index - lookback,
            index
        )
    ]

    avg = sum(values) / len(values)

    if avg <= 0:
        return 0.0

    return (
        candles[index]["volume"]
        / avg
    )


# ============================================================
# MARKET STRUCTURE
# ============================================================

def highest_high(
    candles,
    start,
    end
):

    return max(
        candles[i]["high"]
        for i in range(
            start,
            end
        )
    )


def lowest_low(
    candles,
    start,
    end
):

    return min(
        candles[i]["low"]
        for i in range(
            start,
            end
        )
    )


# ============================================================
# RESAMPLE 15M -> 1H
# ============================================================

def build_hourly_from_15m(
    candles
):

    buckets = {}

    for candle in candles:

        hour = (
            candle["time"]
            // 3600
        ) * 3600

        if hour not in buckets:

            buckets[hour] = {
                "time": hour,
                "open": candle["open"],
                "high": candle["high"],
                "low": candle["low"],
                "close": candle["close"],
                "volume": candle["volume"],
                "count": 1
            }

        else:

            x = buckets[hour]

            x["high"] = max(
                x["high"],
                candle["high"]
            )

            x["low"] = min(
                x["low"],
                candle["low"]
            )

            x["close"] = candle["close"]

            x["volume"] += (
                candle["volume"]
            )

            x["count"] += 1

    result = []

    for key in sorted(buckets):

        x = buckets[key]

        # A complete 1H candle must contain
        # four 15M candles.
        if x["count"] == 4:

            result.append({
                "time": x["time"],
                "open": x["open"],
                "high": x["high"],
                "low": x["low"],
                "close": x["close"],
                "volume": x["volume"]
            })

    return result


# ============================================================
# MAP 1H INDICATORS TO 15M
# ============================================================

def map_hourly_values(
    candles_15m,
    candles_1h,
    values
):

    result = [
        None
        for _ in candles_15m
    ]

    lookup = {}

    for i, candle in enumerate(
        candles_1h
    ):

        lookup[
            candle["time"]
        ] = values[i]

    for i, candle in enumerate(
        candles_15m
    ):

        hour = (
            candle["time"]
            // 3600
        ) * 3600

        result[i] = lookup.get(
            hour
        )

    return result


# ============================================================
# 1H TREND
# ============================================================

def get_hourly_trend(
    i,
    candles_1h,
    ema50,
    ema200,
    adx
):

    if i < 2:
        return None

    e50 = ema50[i]
    e200 = ema200[i]
    a = adx[i]

    if (
        e50 is None
        or e200 is None
        or a is None
    ):
        return None

    close = candles_1h[i]["close"]

    previous_e50 = ema50[i - 1]
    previous_e200 = ema200[i - 1]

    if (
        previous_e50 is None
        or previous_e200 is None
    ):
        return None

    bullish = (
        close > e50
        and e50 > e200
        and e50 > previous_e50
        and e200 >= previous_e200
        and a >= ADX_MIN
    )

    bearish = (
        close < e50
        and e50 < e200
        and e50 < previous_e50
        and e200 <= previous_e200
        and a >= ADX_MIN
    )

    if bullish:
        return "LONG"

    if bearish:
        return "SHORT"

    return None


# ============================================================
# ENTRY SIGNAL
# ============================================================

def find_signal(
    i,
    candles,
    trend,
    atr_values,
    rsi_values
):

    if trend is None:
        return None

    if i < (
        BREAKOUT_LOOKBACK
        + VOLUME_LOOKBACK
        + 5
    ):
        return None

    current = candles[i]
    previous = candles[i - 1]

    atr_value = atr_values[i]
    rsi_value = rsi_values[i]

    if (
        atr_value is None
        or rsi_value is None
        or atr_value <= 0
    ):
        return None

    # --------------------------------------------------------
    # Structure level BEFORE current candle
    # --------------------------------------------------------

    start = (
        i - BREAKOUT_LOOKBACK
    )

    resistance = highest_high(
        candles,
        start,
        i
    )

    support = lowest_low(
        candles,
        start,
        i
    )

    # --------------------------------------------------------
    # Relative volume
    # --------------------------------------------------------

    rv = relative_volume(
        candles,
        i,
        VOLUME_LOOKBACK
    )

    if rv is None:
        return None

    # --------------------------------------------------------
    # Current candle strength
    # --------------------------------------------------------

    ratio = body_ratio(
        current
    )

    if ratio < MIN_BODY_RATIO:
        return None

    # --------------------------------------------------------
    # ATR body momentum
    # --------------------------------------------------------

    body = abs(
        current["close"]
        - current["open"]
    )

    if (
        body
        < atr_value * MIN_ATR_BODY
    ):
        return None

    # ========================================================
    # LONG
    # ========================================================

    if trend == "LONG":

        breakout = (
            current["close"]
            > resistance
            and previous["close"]
            <= resistance
        )

        # Retest:
        # Either current candle tested the old level
        # OR one of the previous few candles did.
        retest = False

        for j in range(
            max(
                0,
                i - RETEST_LOOKBACK
            ),
            i + 1
        ):

            candle = candles[j]

            if (
                candle["low"]
                <= resistance
                and candle["close"]
                > resistance
            ):

                retest = True
                break

        momentum = (
            current["close"]
            > current["open"]
            and current["close"]
            > previous["close"]
        )

        rsi_ok = (
            52.0
            <= rsi_value
            <= 72.0
        )

        volume_ok = (
            rv
            >= RELATIVE_VOLUME_MIN
        )

        if (
            breakout
            and retest
            and momentum
            and rsi_ok
            and volume_ok
        ):

            return {
                "direction": "LONG",
                "level": resistance,
                "atr": atr_value,
                "rsi": rsi_value,
                "relative_volume": rv
            }

    # ========================================================
    # SHORT
    # ========================================================

    if trend == "SHORT":

        breakout = (
            current["close"]
            < support
            and previous["close"]
            >= support
        )

        retest = False

        for j in range(
            max(
                0,
                i - RETEST_LOOKBACK
            ),
            i + 1
        ):

            candle = candles[j]

            if (
                candle["high"]
                >= support
                and candle["close"]
                < support
            ):

                retest = True
                break

        momentum = (
            current["close"]
            < current["open"]
            and current["close"]
            < previous["close"]
        )

        rsi_ok = (
            28.0
            <= rsi_value
            <= 48.0
        )

        volume_ok = (
            rv
            >= RELATIVE_VOLUME_MIN
        )

        if (
            breakout
            and retest
            and momentum
            and rsi_ok
            and volume_ok
        ):

            return {
                "direction": "SHORT",
                "level": support,
                "atr": atr_value,
                "rsi": rsi_value,
                "relative_volume": rv
            }

    return None


# ============================================================
# ENTRY / SL / TP
# ============================================================

def create_trade(
    signal,
    entry_candle,
    candles,
    index
):

    direction = signal[
        "direction"
    ]

    entry = (
        entry_candle["open"]
    )

    atr_value = signal["atr"]

    # --------------------------------------------------------
    # Apply slippage
    # --------------------------------------------------------

    if direction == "LONG":

        entry *= (
            1.0
            + SLIPPAGE_RATE
        )

    else:

        entry *= (
            1.0
            - SLIPPAGE_RATE
        )

    recent_start = max(
        0,
        index - 10
    )

    if direction == "LONG":

        swing_low = min(
            candles[j]["low"]
            for j in range(
                recent_start,
                index
            )
        )

        atr_stop = (
            entry
            - atr_value
            * ATR_SL_MULT
        )

        structure_stop = (
            swing_low
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

        tp = (
            entry
            + risk * RR
        )

    else:

        swing_high = max(
            candles[j]["high"]
            for j in range(
                recent_start,
                index
            )
        )

        atr_stop = (
            entry
            + atr_value
            * ATR_SL_MULT
        )

        structure_stop = (
            swing_high
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

        tp = (
            entry
            - risk * RR
        )

    if risk <= 0:
        return None

    actual_rr = abs(
        tp - entry
    ) / risk

    if actual_rr < MIN_RR:
        return None

    return {
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "risk_price": risk,
        "signal_time": entry_candle["time"],
        "bars": 0,
        "rsi": signal["rsi"],
        "relative_volume":
            signal["relative_volume"]
    }


# ============================================================
# COST
# ============================================================

def apply_entry_fee(
    capital,
    risk_amount
):

    # Risk sizing is based on capital.
    # Fees are deducted separately at exit.
    return risk_amount


# ============================================================
# SIMULATE TRADE
# ============================================================

def simulate_trade(
    candles,
    start_index,
    trade,
    capital
):

    direction = trade[
        "direction"
    ]

    entry = trade[
        "entry"
    ]

    sl = trade["sl"]
    tp = trade["tp"]

    risk_price = trade[
        "risk_price"
    ]

    risk_amount = (
        capital
        * RISK_PER_TRADE
    )

    quantity = (
        risk_amount
        / risk_price
    )

    max_index = min(
        len(candles) - 1,
        start_index
        + MAX_HOLDING_BARS
    )

    for i in range(
        start_index,
        max_index + 1
    ):

        candle = candles[i]

        high = candle["high"]
        low = candle["low"]

        # ----------------------------------------------------
        # Conservative rule:
        # If SL and TP are both touched in same candle,
        # assume SL first.
        # ----------------------------------------------------

        if direction == "LONG":

            hit_sl = (
                low <= sl
            )

            hit_tp = (
                high >= tp
            )

            if hit_sl:

                exit_price = sl

                gross_r = (
                    exit_price
                    - entry
                ) / risk_price

                reason = "SL"

                return (
                    gross_r,
                    reason,
                    i,
                    exit_price
                )

            if hit_tp:

                exit_price = tp

                gross_r = (
                    exit_price
                    - entry
                ) / risk_price

                reason = "TP"

                return (
                    gross_r,
                    reason,
                    i,
                    exit_price
                )

        else:

            hit_sl = (
                high >= sl
            )

            hit_tp = (
                low <= tp
            )

            if hit_sl:

                exit_price = sl

                gross_r = (
                    entry
                    - exit_price
                ) / risk_price

                reason = "SL"

                return (
                    gross_r,
                    reason,
                    i,
                    exit_price
                )

            if hit_tp:

                exit_price = tp

                gross_r = (
                    entry
                    - exit_price
                ) / risk_price

                reason = "TP"

                return (
                    gross_r,
                    reason,
                    i,
                    exit_price
                )

    # --------------------------------------------------------
    # Time exit
    # --------------------------------------------------------

    exit_candle = candles[
        max_index
    ]

    exit_price = (
        exit_candle["close"]
    )

    if direction == "LONG":

        gross_r = (
            exit_price
            - entry
        ) / risk_price

    else:

        gross_r = (
            entry
            - exit_price
        ) / risk_price

    return (
        gross_r,
        "TIME",
        max_index,
        exit_price
    )


# ============================================================
# STATISTICS
# ============================================================

def calculate_stats(
    trades,
    initial_capital
):

    if not trades:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit_factor": None,
            "net_r": 0.0,
            "average_r": 0.0,
            "max_drawdown": 0.0,
            "final_balance":
                initial_capital
        }

    wins = [
        t for t in trades
        if t["net_r"] > 0
    ]

    losses = [
        t for t in trades
        if t["net_r"] <= 0
    ]

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

        pf = (
            gross_profit
            / gross_loss
        )

    else:

        pf = None

    balance = (
        initial_capital
    )

    peak = balance
    max_dd = 0.0

    for t in trades:

        balance *= (
            1.0
            + (
                t["net_r"]
                * RISK_PER_TRADE
            )
        )

        if balance > peak:

            peak = balance

        dd = (
            peak - balance
        ) / peak * 100.0

        max_dd = max(
            max_dd,
            dd
        )

    net_r = sum(
        t["net_r"]
        for t in trades
    )

    avg_r = (
        net_r
        / len(trades)
    )

    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate":
            len(wins)
            / len(trades)
            * 100.0,
        "profit_factor": pf,
        "net_r": net_r,
        "average_r": avg_r,
        "max_drawdown": max_dd,
        "final_balance": balance
    }


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    symbol,
    candles_15m,
    candles_1h
):

    if len(candles_15m) < 1000:

        print(
            f"{symbol}: "
            f"not enough 15M candles"
        )

        return [], [], []

    if len(candles_1h) < 250:

        print(
            f"{symbol}: "
            f"not enough 1H candles"
        )

        return [], [], []

    # --------------------------------------------------------
    # Indicators
    # --------------------------------------------------------

    ema50_1h = ema_series(
        candles_1h,
        EMA_FAST
    )

    ema200_1h = ema_series(
        candles_1h,
        EMA_SLOW
    )

    adx_1h = adx_series(
        candles_1h,
        ADX_PERIOD
    )

    atr_15m = atr_series(
        candles_15m,
        ATR_SL_MULT
    )

    rsi_15m = rsi_series(
        candles_15m,
        RSI_PERIOD
    )

    # --------------------------------------------------------
    # Map 1H trend values to 15M.
    #
    # IMPORTANT:
    # We only use the completed hourly candle belonging
    # to the previous hour. This prevents look-ahead.
    # --------------------------------------------------------

    trend_map = [
        None
        for _ in candles_15m
    ]

    hourly_lookup = {}

    for i in range(
        len(candles_1h)
    ):

        hourly_lookup[
            candles_1h[i]["time"]
        ] = get_hourly_trend(
            i,
            candles_1h,
            ema50_1h,
            ema200_1h,
            adx_1h
        )

    for i in range(
        len(candles_15m)
    ):

        hour = (
            candles_15m[i]["time"]
            // 3600
        ) * 3600

        previous_hour = (
            hour - 3600
        )

        trend_map[i] = (
            hourly_lookup.get(
                previous_hour
            )
        )

    # --------------------------------------------------------
    # IS / OOS split
    # --------------------------------------------------------

    split_index = int(
        len(candles_15m)
        * (1.0 - OOS_PERCENT)
    )

    trades = []

    capital = (
        INITIAL_CAPITAL
    )

    cooldown_until = -1

    i = max(
        300,
        BREAKOUT_LOOKBACK
        + VOLUME_LOOKBACK
        + 20
    )

    while i < (
        len(candles_15m) - 2
    ):

        if i <= cooldown_until:

            i += 1
            continue

        trend = trend_map[i]

        if trend is None:

            i += 1
            continue

        signal = find_signal(
            i,
            candles_15m,
            trend,
            atr_15m,
            rsi_15m
        )

        if signal is None:

            i += 1
            continue

        # ----------------------------------------------------
        # Entry on NEXT 15M candle open
        # ----------------------------------------------------

        entry_index = i + 1

        entry_candle = (
            candles_15m[
                entry_index
            ]
        )

        trade = create_trade(
            signal,
            entry_candle,
            candles_15m,
            entry_index
        )

        if trade is None:

            i += 1
            continue

        (
            gross_r,
            reason,
            exit_index,
            exit_price
        ) = simulate_trade(
            candles_15m,
            entry_index,
            trade,
            capital
        )

        # ----------------------------------------------------
        # Fees:
        # Entry + exit.
        #
        # Approximate fee in R terms.
        # ----------------------------------------------------

        entry_fee_price = (
            trade["entry"]
            * FEE_RATE
        )

        exit_fee_price = (
            exit_price
            * FEE_RATE
        )

        total_fee_price = (
            entry_fee_price
            + exit_fee_price
        )

        fee_r = (
            total_fee_price
            / trade["risk_price"]
        )

        net_r = (
            gross_r
            - fee_r
        )

        # ----------------------------------------------------
        # Slippage is already applied at entry.
        # Apply exit slippage too.
        # ----------------------------------------------------

        exit_slippage_price = (
            exit_price
            * SLIPPAGE_RATE
        )

        slippage_r = (
            exit_slippage_price
            / trade["risk_price"]
        )

        net_r -= slippage_r

        trade_record = {
            "symbol": symbol,
            "direction":
                trade["direction"],
            "entry_time":
                trade["signal_time"],
            "entry":
                trade["entry"],
            "sl":
                trade["sl"],
            "tp":
                trade["tp"],
            "exit_time":
                candles_15m[
                    exit_index
                ]["time"],
            "exit":
                exit_price,
            "reason":
                reason,
            "gross_r":
                gross_r,
            "net_r":
                net_r,
            "rsi":
                trade["rsi"],
            "relative_volume":
                trade[
                    "relative_volume"
                ],
            "sample":
                (
                    "IS"
                    if entry_index
                    < split_index
                    else "OOS"
                )
        }

        trades.append(
            trade_record
        )

        capital *= (
            1.0
            + (
                net_r
                * RISK_PER_TRADE
            )
        )

        cooldown_until = (
            exit_index
            + COOLDOWN_BARS
        )

        i = (
            exit_index
            + 1
        )

    is_trades = [
        t for t in trades
        if t["sample"] == "IS"
    ]

    oos_trades = [
        t for t in trades
        if t["sample"] == "OOS"
    ]

    return (
        trades,
        is_trades,
        oos_trades
    )


# ============================================================
# FORMAT
# ============================================================

def print_stats(
    title,
    stats
):

    print(
        "\n"
        + "=" * 64
    )

    print(title)

    print(
        "=" * 64
    )

    print(
        f"Trades       : "
        f"{stats['trades']}"
    )

    print(
        f"Wins         : "
        f"{stats['wins']}"
    )

    print(
        f"Losses       : "
        f"{stats['losses']}"
    )

    print(
        f"Win Rate     : "
        f"{stats['win_rate']:.2f}%"
    )

    print(
        f"Profit Factor: "
        f"{stats['profit_factor']}"
    )

    print(
        f"Net R        : "
        f"{stats['net_r']:.2f}"
    )

    print(
        f"Average R    : "
        f"{stats['average_r']:.4f}"
    )

    print(
        f"Max Drawdown : "
        f"{stats['max_drawdown']:.2f}%"
    )

    print(
        f"Balance      : "
        f"${stats['final_balance']:.2f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 64
    )

    print(
        "SCORE HUNTER v11"
    )

    print(
        "LBANK REAL DATA BACKTEST"
    )

    print(
        "1H TREND + 15M BREAKOUT / RETEST"
    )

    print(
        "REGIME FILTER + ADX + RELATIVE VOLUME"
    )

    print(
        "ATR STOP + DYNAMIC RISK"
    )

    print(
        "FEES + SLIPPAGE"
    )

    print(
        "CLOSED CANDLE ONLY"
    )

    print(
        "ENTRY = NEXT 15M OPEN"
    )

    print(
        "NO PANDAS / NO NUMPY"
    )

    print(
        "=" * 64
    )

    all_trades = []
    all_is = []
    all_oos = []

    per_symbol = {}

    for symbol in SYMBOLS:

        candles_15m = (
            get_lbank_klines(
                symbol,
                "minute15"
            )
        )

        candles_1h = (
            get_lbank_klines(
                symbol,
                "hour1"
            )
        )

        print(
            f"{symbol}: "
            f"{len(candles_15m)} "
            f"15M candles"
        )

        print(
            f"{symbol}: "
            f"{len(candles_1h)} "
            f"1H candles"
        )

        if (
            len(candles_15m)
            < 1000
            or len(candles_1h)
            < 250
        ):

            print(
                f"{symbol}: "
                f"NOT ENOUGH DATA"
            )

            continue

        (
            trades,
            is_trades,
            oos_trades
        ) = run_backtest(
            symbol,
            candles_15m,
            candles_1h
        )

        all_trades.extend(
            trades
        )

        all_is.extend(
            is_trades
        )

        all_oos.extend(
            oos_trades
        )

        stats = calculate_stats(
            trades,
            INITIAL_CAPITAL
        )

        per_symbol[
            symbol
        ] = stats

        print_stats(
            symbol,
            stats
        )

    # ========================================================
    # ALL
    # ========================================================

    all_stats = calculate_stats(
        all_trades,
        INITIAL_CAPITAL
    )

    is_stats = calculate_stats(
        all_is,
        INITIAL_CAPITAL
    )

    oos_stats = calculate_stats(
        all_oos,
        INITIAL_CAPITAL
    )

    print_stats(
        "ALL COINS",
        all_stats
    )

    print_stats(
        "IN SAMPLE",
        is_stats
    )

    print_stats(
        "OUT OF SAMPLE",
        oos_stats
    )

    # ========================================================
    # SIGNAL FREQUENCY
    # ========================================================

    if all_trades:

        first_time = min(
            t["entry_time"]
            for t in all_trades
        )

        last_time = max(
            t["entry_time"]
            for t in all_trades
        )

        days = max(
            1.0,
            (
                last_time
                - first_time
            )
            / 86400.0
        )

        signals_per_day = (
            len(all_trades)
            / days
        )

    else:

        days = 0.0
        signals_per_day = 0.0

    print(
        "\n"
        + "=" * 64
    )

    print(
        "SIGNAL FREQUENCY"
    )

    print(
        "=" * 64
    )

    print(
        f"Trading days : "
        f"{days:.1f}"
    )

    print(
        f"Total signals: "
        f"{len(all_trades)}"
    )

    print(
        f"Signals/day  : "
        f"{signals_per_day:.2f}"
    )

    # ========================================================
    # TARGET CHECK
    # ========================================================

    win_rate = all_stats[
        "win_rate"
    ]

    oos_win_rate = oos_stats[
        "win_rate"
    ]

    pf = all_stats[
        "profit_factor"
    ]

    oos_pf = oos_stats[
        "profit_factor"
    ]

    print(
        "\n"
        + "=" * 64
    )

    print(
        "TARGET / QUALITY CHECK"
    )

    print(
        "=" * 64
    )

    print(
        "Win Rate 60-70% : "
        + (
            "PASS"
            if 60 <= win_rate <= 70
            else "FAIL"
        )
    )

    print(
        "Signals >= 2/day: "
        + (
            "PASS"
            if signals_per_day >= 2
            else "FAIL"
        )
    )

    print(
        "Profit Factor >1.20: "
        + (
            "PASS"
            if pf is not None
            and pf > 1.20
            else "FAIL"
        )
    )

    print(
        "OOS Win Rate 55%+: "
        + (
            "PASS"
            if oos_win_rate >= 55
            else "FAIL"
        )
    )

    print(
        "OOS Profit Factor >1.10: "
        + (
            "PASS"
            if oos_pf is not None
            and oos_pf > 1.10
            else "FAIL"
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    results = {
        "config": {
            "strategy":
                "1H Trend + 15M Breakout/Retest",
            "ema_fast":
                EMA_FAST,
            "ema_slow":
                EMA_SLOW,
            "adx_min":
                ADX_MIN,
            "rr":
                RR,
            "atr_sl_mult":
                ATR_SL_MULT,
            "fee_rate":
                FEE_RATE,
            "slippage_rate":
                SLIPPAGE_RATE,
            "risk_per_trade":
                RISK_PER_TRADE,
            "relative_volume_min":
                RELATIVE_VOLUME_MIN,
            "cooldown_bars":
                COOLDOWN_BARS,
            "oos_percent":
                OOS_PERCENT
        },

        "all": all_stats,

        "in_sample": is_stats,

        "out_of_sample":
            oos_stats,

        "signals_per_day":
            signals_per_day,

        "per_symbol":
            per_symbol,

        "trades":
            all_trades
    }

    with open(
        "lbank_v11_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print(
        "\nResults saved to:"
    )

    print(
        "lbank_v11_results.json"
    )

    print(
        "\nBACKTEST FINISHED."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
