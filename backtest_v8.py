import json
import time
import math
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timezone


# ============================================================
# SCORE HUNTER v12
# LBANK REAL DATA
#
# 1H TREND
# 15M ENTRY
# BREAKOUT + RETEST
# EMA + ADX + RSI + ATR + RELATIVE VOLUME
#
# CLOSED CANDLE ONLY
# ENTRY = NEXT 15M OPEN
#
# TP = 2R
# SL = ATR / STRUCTURE
#
# FEES + SLIPPAGE INCLUDED
#
# NO PANDAS
# NO NUMPY
# ============================================================


# -----------------------------
# SETTINGS
# -----------------------------

BASE_URLS = [
    "https://api.lbank.info",
    "https://api.lbkex.com",
    "https://www.lbkex.net",
]

SYMBOLS = [
    "btc_usdt",
    "eth_usdt",
    "sol_usdt",
    "xrp_usdt",
    "ada_usdt",
    "link_usdt",
]

TIMEFRAME_ENTRY = "minute15"
TIMEFRAME_TREND = "hour1"

# Number of candles requested.
# 15m: 20,000 ~= 208 days
# 1h : 20,000 ~= 833 days
ENTRY_CANDLES = 20000
TREND_CANDLES = 20000

PAGE_SIZE = 2000

INITIAL_CAPITAL = 1000.0

# Risk per trade
RISK_PERCENT = 1.0

# Reward / risk
RR = 2.0

# ATR
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.35

# EMA
EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

# RSI
RSI_PERIOD = 14

# ADX
ADX_PERIOD = 14
ADX_MIN = 20.0

# Relative volume
VOLUME_PERIOD = 20
RELATIVE_VOLUME_MIN = 1.15

# Breakout lookback
BREAKOUT_LOOKBACK = 20

# Retest tolerance
RETEST_ATR_MULTIPLIER = 0.35

# Minimum candle body
MIN_BODY_PERCENT = 0.45

# Fees
# Change this if your actual LBank fee differs.
FEE_RATE = 0.0010

# Slippage
SLIPPAGE_RATE = 0.0003

# Maximum simultaneous positions
MAX_OPEN_POSITIONS = 1

# Minimum bars between signals
MIN_BARS_BETWEEN_TRADES = 4

# OOS
OOS_DAYS = 60

# Initial warmup
MIN_REQUIRED_BARS = 300

OUTPUT_FILE = "lbank_v12_results.json"


# ============================================================
# HTTP
# ============================================================

def http_get(url, params, timeout=20):
    query = urllib.parse.urlencode(params)
    full_url = url + "/v2/kline.do?" + query

    request = urllib.request.Request(
        full_url,
        headers={
            "User-Agent": "Mozilla/5.0 SCORE-HUNTER-BACKTEST"
        }
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")

    return json.loads(raw)


# ============================================================
# GET KLINES
# ============================================================

def extract_data(payload):
    if isinstance(payload, dict):
        data = payload.get("data")

        if data is None:
            return []

        return data

    if isinstance(payload, list):
        return payload

    return []


def download_klines(symbol, timeframe, target_count):
    """
    Download historical candles page-by-page.

    LBank documents:
    size = 1..2000
    time = timestamp in seconds

    We walk backwards using the oldest candle returned
    from each page.
    """

    all_rows = {}
    end_time = int(time.time())

    pages = 0
    last_oldest = None

    while len(all_rows) < target_count:
        pages += 1

        if pages > 100:
            break

        params = {
            "symbol": symbol,
            "size": PAGE_SIZE,
            "type": timeframe,
            "time": end_time,
        }

        payload = None

        for base in BASE_URLS:
            try:
                payload = http_get(base, params)
                break
            except Exception as exc:
                last_error = exc
                continue

        if payload is None:
            print(
                f"{symbol} {timeframe}: request failed: "
                f"{last_error}"
            )
            break

        rows = extract_data(payload)

        if not rows:
            break

        added = 0

        for row in rows:
            if not isinstance(row, list):
                continue

            if len(row) < 6:
                continue

            try:
                ts = int(float(row[0]))
                op = float(row[1])
                hi = float(row[2])
                lo = float(row[3])
                cl = float(row[4])
                vol = float(row[5])

                if hi <= 0 or lo <= 0 or op <= 0 or cl <= 0:
                    continue

                all_rows[ts] = {
                    "timestamp": ts,
                    "open": op,
                    "high": hi,
                    "low": lo,
                    "close": cl,
                    "volume": vol,
                }

                added += 1

            except Exception:
                continue

        if not all_rows:
            break

        oldest = min(all_rows.keys())

        if last_oldest == oldest:
            break

        last_oldest = oldest

        end_time = oldest - 1

        if pages % 1 == 0:
            print(
                f"{symbol} {timeframe}: "
                f"{len(all_rows)} candles"
            )

        if added == 0:
            break

        time.sleep(0.08)

    candles = list(all_rows.values())
    candles.sort(key=lambda x: x["timestamp"])

    # Keep only requested amount
    if len(candles) > target_count:
        candles = candles[-target_count:]

    # Remove currently forming candle.
    now = int(time.time())

    if timeframe == "minute15":
        interval = 15 * 60
    elif timeframe == "hour1":
        interval = 60 * 60
    else:
        interval = 60

    closed = []

    for candle in candles:
        if candle["timestamp"] + interval <= now:
            closed.append(candle)

    return closed


# ============================================================
# INDICATORS
# ============================================================

def sma(values, period):
    if len(values) < period:
        return None

    return sum(values[-period:]) / period


def ema_series(values, period):
    result = [None] * len(values)

    if len(values) < period:
        return result

    seed = sum(values[:period]) / period
    result[period - 1] = seed

    multiplier = 2.0 / (period + 1.0)

    prev = seed

    for i in range(period, len(values)):
        prev = (
            (values[i] - prev) * multiplier
            + prev
        )

        result[i] = prev

    return result


def true_range(candles):
    tr = [0.0] * len(candles)

    for i in range(len(candles)):
        if i == 0:
            tr[i] = (
                candles[i]["high"]
                - candles[i]["low"]
            )
        else:
            high = candles[i]["high"]
            low = candles[i]["low"]
            prev_close = candles[i - 1]["close"]

            tr[i] = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )

    return tr


def atr_series(candles, period):
    tr = true_range(candles)

    atr = [None] * len(candles)

    if len(tr) < period:
        return atr

    initial = sum(tr[:period]) / period
    atr[period - 1] = initial

    prev = initial

    for i in range(period, len(tr)):
        prev = (
            ((prev * (period - 1)) + tr[i])
            / period
        )

        atr[i] = prev

    return atr


def rsi_series(candles, period):
    result = [None] * len(candles)

    if len(candles) <= period:
        return result

    gains = []
    losses = []

    for i in range(1, period + 1):
        change = (
            candles[i]["close"]
            - candles[i - 1]["close"]
        )

        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (
            100.0 / (1.0 + rs)
        )

    for i in range(period + 1, len(candles)):
        change = (
            candles[i]["close"]
            - candles[i - 1]["close"]
        )

        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        avg_gain = (
            (avg_gain * (period - 1) + gain)
            / period
        )

        avg_loss = (
            (avg_loss * (period - 1) + loss)
            / period
        )

        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (
                100.0 / (1.0 + rs)
            )

    return result


def adx_series(candles, period):
    n = len(candles)

    adx = [None] * n

    if n < period * 2 + 2:
        return adx

    tr = [0.0] * n
    plus_dm = [0.0] * n
    minus_dm = [0.0] * n

    for i in range(1, n):
        high = candles[i]["high"]
        low = candles[i]["low"]

        prev_high = candles[i - 1]["high"]
        prev_low = candles[i - 1]["low"]
        prev_close = candles[i - 1]["close"]

        tr[i] = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )

        up_move = high - prev_high
        down_move = prev_low - low

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move

        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    atr = sum(tr[1:period + 1]) / period
    plus = sum(plus_dm[1:period + 1]) / period
    minus = sum(minus_dm[1:period + 1]) / period

    dx_values = []

    for i in range(period + 1, n):
        atr = (
            (atr * (period - 1) + tr[i])
            / period
        )

        plus = (
            (plus * (period - 1) + plus_dm[i])
            / period
        )

        minus = (
            (minus * (period - 1) + minus_dm[i])
            / period
        )

        if atr <= 0:
            continue

        plus_di = 100.0 * plus / atr
        minus_di = 100.0 * minus / atr

        denominator = plus_di + minus_di

        if denominator <= 0:
            dx = 0.0
        else:
            dx = (
                100.0
                * abs(plus_di - minus_di)
                / denominator
            )

        dx_values.append(dx)

        if len(dx_values) >= period:
            if adx[i] is None:
                adx[i] = (
                    sum(dx_values[-period:])
                    / period
                )
            else:
                adx[i] = (
                    (
                        adx[i - 1] * (period - 1)
                        + dx
                    )
                    / period
                )

    return adx


# ============================================================
# MERGE 1H TREND INTO 15M
# ============================================================

def get_trend_index(hourly, timestamp):
    """
    Return the latest CLOSED 1H candle that existed
    before the 15M candle.

    This avoids look-ahead.
    """

    lo = 0
    hi = len(hourly) - 1
    answer = None

    while lo <= hi:
        mid = (lo + hi) // 2

        if hourly[mid]["timestamp"] <= timestamp:
            answer = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return answer


# ============================================================
# TRADE ENGINE
# ============================================================

def apply_entry_slippage(price, side):
    if side == "LONG":
        return price * (1.0 + SLIPPAGE_RATE)

    return price * (1.0 - SLIPPAGE_RATE)


def apply_exit_slippage(price, side):
    if side == "LONG":
        return price * (1.0 - SLIPPAGE_RATE)

    return price * (1.0 + SLIPPAGE_RATE)


def calculate_trade_r(entry, exit_price, sl, side):
    if side == "LONG":
        risk = entry - sl

        if risk <= 0:
            return 0.0

        return (exit_price - entry) / risk

    risk = sl - entry

    if risk <= 0:
        return 0.0

    return (entry - exit_price) / risk


def run_backtest(
    entry_candles,
    trend_candles,
    initial_capital=1000.0,
):
    closes = [
        x["close"]
        for x in entry_candles
    ]

    volumes = [
        x["volume"]
        for x in entry_candles
    ]

    ema20 = ema_series(closes, EMA_FAST)
    ema50 = ema_series(closes, EMA_MID)
    ema200 = ema_series(closes, EMA_SLOW)

    atr = atr_series(
        entry_candles,
        ATR_PERIOD
    )

    rsi = rsi_series(
        entry_candles,
        RSI_PERIOD
    )

    adx = adx_series(
        entry_candles,
        ADX_PERIOD
    )

    # Trend timeframe indicators
    trend_closes = [
        x["close"]
        for x in trend_candles
    ]

    trend_ema20 = ema_series(
        trend_closes,
        EMA_FAST
    )

    trend_ema50 = ema_series(
        trend_closes,
        EMA_MID
    )

    trend_ema200 = ema_series(
        trend_closes,
        EMA_SLOW
    )

    capital = initial_capital
    peak = capital
    max_dd = 0.0

    trades = []

    active_trade = None

    last_trade_index = -999999

    # We need enough data for all indicators
    start = max(
        EMA_SLOW + 10,
        ATR_PERIOD + 10,
        RSI_PERIOD + 10,
        ADX_PERIOD * 2 + 10,
        BREAKOUT_LOOKBACK + 5,
    )

    for i in range(start, len(entry_candles) - 1):

        candle = entry_candles[i]

        # ----------------------------------------------------
        # Manage open position first
        # ----------------------------------------------------

        if active_trade is not None:

            side = active_trade["side"]
            sl = active_trade["sl"]
            tp = active_trade["tp"]

            high = candle["high"]
            low = candle["low"]

            result = None
            exit_price = None

            # Conservative assumption:
            # if SL and TP are both touched in same candle,
            # assume SL first.
            if side == "LONG":

                if low <= sl:
                    result = "SL"
                    exit_price = sl

                elif high >= tp:
                    result = "TP"
                    exit_price = tp

            else:

                if high >= sl:
                    result = "SL"
                    exit_price = sl

                elif low <= tp:
                    result = "TP"
                    exit_price = tp

            if result is not None:

                exit_price = apply_exit_slippage(
                    exit_price,
                    side
                )

                gross_r = calculate_trade_r(
                    active_trade["entry"],
                    exit_price,
                    active_trade["sl"],
                    side
                )

                # Round trip fee
                fee_r = (
                    2.0
                    * FEE_RATE
                    * active_trade["entry"]
                    / max(
                        abs(
                            active_trade["entry"]
                            - active_trade["sl"]
                        ),
                        1e-12
                    )
                )

                net_r = gross_r - fee_r

                risk_amount = active_trade[
                    "risk_amount"
                ]

                pnl = risk_amount * net_r

                capital += pnl

                trades.append({
                    "timestamp": candle["timestamp"],
                    "date": datetime.fromtimestamp(
                        candle["timestamp"],
                        tz=timezone.utc
                    ).isoformat(),
                    "side": side,
                    "entry": active_trade["entry"],
                    "exit": exit_price,
                    "sl": sl,
                    "tp": tp,
                    "result": result,
                    "gross_r": gross_r,
                    "fee_r": fee_r,
                    "net_r": net_r,
                    "pnl": pnl,
                    "capital": capital,
                })

                active_trade = None

                if capital > peak:
                    peak = capital

                dd = (
                    (peak - capital)
                    / peak
                    * 100.0
                )

                if dd > max_dd:
                    max_dd = dd

                # No new trade on same candle
                continue

        # ----------------------------------------------------
        # Indicator values
        # ----------------------------------------------------

        if (
            ema20[i] is None
            or ema50[i] is None
            or ema200[i] is None
            or atr[i] is None
            or rsi[i] is None
            or adx[i] is None
        ):
            continue

        if (
            capital <= 0
            or capital < 1
        ):
            break

        if (
            i - last_trade_index
            < MIN_BARS_BETWEEN_TRADES
        ):
            continue

        current_atr = atr[i]

        if current_atr <= 0:
            continue

        # ----------------------------------------------------
        # 1H TREND
        # ----------------------------------------------------

        trend_i = get_trend_index(
            trend_candles,
            candle["timestamp"]
        )

        if trend_i is None:
            continue

        if trend_i < EMA_SLOW:
            continue

        if (
            trend_ema20[trend_i] is None
            or trend_ema50[trend_i] is None
            or trend_ema200[trend_i] is None
        ):
            continue

        trend_close = (
            trend_candles[
                trend_i
            ]["close"]
        )

        bullish_1h = (
            trend_close
            > trend_ema20[trend_i]
            > trend_ema50[trend_i]
            > trend_ema200[trend_i]
        )

        bearish_1h = (
            trend_close
            < trend_ema20[trend_i]
            < trend_ema50[trend_i]
            < trend_ema200[trend_i]
        )

        if not bullish_1h and not bearish_1h:
            continue

        # ----------------------------------------------------
        # 15M TREND
        # ----------------------------------------------------

        bullish_15m = (
            ema20[i]
            > ema50[i]
            > ema200[i]
        )

        bearish_15m = (
            ema20[i]
            < ema50[i]
            < ema200[i]
        )

        # ----------------------------------------------------
        # ADX
        # ----------------------------------------------------

        if adx[i] < ADX_MIN:
            continue

        # ----------------------------------------------------
        # RELATIVE VOLUME
        # ----------------------------------------------------

        if i < VOLUME_PERIOD + 1:
            continue

        avg_volume = sum(
            volumes[
                i - VOLUME_PERIOD:i
            ]
        ) / VOLUME_PERIOD

        if avg_volume <= 0:
            continue

        relative_volume = (
            volumes[i]
            / avg_volume
        )

        if (
            relative_volume
            < RELATIVE_VOLUME_MIN
        ):
            continue

        # ----------------------------------------------------
        # CANDLE QUALITY
        # ----------------------------------------------------

        candle_range = (
            candle["high"]
            - candle["low"]
        )

        if candle_range <= 0:
            continue

        body = abs(
            candle["close"]
            - candle["open"]
        )

        body_ratio = (
            body / candle_range
        )

        if body_ratio < MIN_BODY_PERCENT:
            continue

        # ----------------------------------------------------
        # BREAKOUT LEVEL
        # ----------------------------------------------------

        lookback_start = (
            i - BREAKOUT_LOOKBACK
        )

        previous_high = max(
            x["high"]
            for x in entry_candles[
                lookback_start:i
            ]
        )

        previous_low = min(
            x["low"]
            for x in entry_candles[
                lookback_start:i
            ]
        )

        # ----------------------------------------------------
        # LONG BREAKOUT / RETEST
        # ----------------------------------------------------

        long_breakout = (
            candle["high"]
            > previous_high
        )

        long_retest = (
            candle["low"]
            <= previous_high
            + current_atr
            * RETEST_ATR_MULTIPLIER
        )

        long_close_strength = (
            candle["close"]
            > candle["open"]
            and candle["close"]
            > previous_high
        )

        long_rsi = (
            52.0
            <= rsi[i]
            <= 72.0
        )

        long_signal = (
            bullish_1h
            and bullish_15m
            and long_breakout
            and long_retest
            and long_close_strength
            and long_rsi
        )

        # ----------------------------------------------------
        # SHORT BREAKOUT / RETEST
        # ----------------------------------------------------

        short_breakout = (
            candle["low"]
            < previous_low
        )

        short_retest = (
            candle["high"]
            >= previous_low
            - current_atr
            * RETEST_ATR_MULTIPLIER
        )

        short_close_strength = (
            candle["close"]
            < candle["open"]
            and candle["close"]
            < previous_low
        )

        short_rsi = (
            28.0
            <= rsi[i]
            <= 48.0
        )

        short_signal = (
            bearish_1h
            and bearish_15m
            and short_breakout
            and short_retest
            and short_close_strength
            and short_rsi
        )

        # ----------------------------------------------------
        # NEXT CANDLE OPEN ENTRY
        # ----------------------------------------------------

        if long_signal:

            next_candle = entry_candles[i + 1]

            entry = apply_entry_slippage(
                next_candle["open"],
                "LONG"
            )

            swing_low = min(
                x["low"]
                for x in entry_candles[
                    max(0, i - 10):i + 1
                ]
            )

            atr_sl = (
                entry
                - current_atr
                * ATR_MULTIPLIER
            )

            sl = min(
                swing_low,
                atr_sl
            )

            risk = entry - sl

            if risk <= 0:
                continue

            # Reject absurdly wide stops
            if risk > current_atr * 3.5:
                continue

            tp = (
                entry
                + risk * RR
            )

            risk_amount = (
                capital
                * RISK_PERCENT
                / 100.0
            )

            active_trade = {
                "side": "LONG",
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "risk_amount": risk_amount,
                "signal_index": i,
            }

            last_trade_index = i

        elif short_signal:

            next_candle = entry_candles[i + 1]

            entry = apply_entry_slippage(
                next_candle["open"],
                "SHORT"
            )

            swing_high = max(
                x["high"]
                for x in entry_candles[
                    max(0, i - 10):i + 1
                ]
            )

            atr_sl = (
                entry
                + current_atr
                * ATR_MULTIPLIER
            )

            sl = max(
                swing_high,
                atr_sl
            )

            risk = sl - entry

            if risk <= 0:
                continue

            if risk > current_atr * 3.5:
                continue

            tp = (
                entry
                - risk * RR
            )

            risk_amount = (
                capital
                * RISK_PERCENT
                / 100.0
            )

            active_trade = {
                "side": "SHORT",
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "risk_amount": risk_amount,
                "signal_index": i,
            }

            last_trade_index = i

        if capital > peak:
            peak = capital

        dd = (
            (peak - capital)
            / peak
            * 100.0
        )

        if dd > max_dd:
            max_dd = dd

    # --------------------------------------------------------
    # Force-close open trade at last close
    # --------------------------------------------------------

    if active_trade is not None:

        last = entry_candles[-1]

        exit_price = apply_exit_slippage(
            last["close"],
            active_trade["side"]
        )

        gross_r = calculate_trade_r(
            active_trade["entry"],
            exit_price,
            active_trade["sl"],
            active_trade["side"]
        )

        fee_r = (
            2.0
            * FEE_RATE
            * active_trade["entry"]
            / max(
                abs(
                    active_trade["entry"]
                    - active_trade["sl"]
                ),
                1e-12
            )
        )

        net_r = gross_r - fee_r

        pnl = (
            active_trade["risk_amount"]
            * net_r
        )

        capital += pnl

        trades.append({
            "timestamp": last["timestamp"],
            "date": datetime.fromtimestamp(
                last["timestamp"],
                tz=timezone.utc
            ).isoformat(),
            "side": active_trade["side"],
            "entry": active_trade["entry"],
            "exit": exit_price,
            "sl": active_trade["sl"],
            "tp": active_trade["tp"],
            "result": "FORCED_EXIT",
            "gross_r": gross_r,
            "fee_r": fee_r,
            "net_r": net_r,
            "pnl": pnl,
            "capital": capital,
        })

    return {
        "final_balance": capital,
        "trades": trades,
        "max_drawdown": max_dd,
    }


# ============================================================
# STATISTICS
# ============================================================

def calculate_stats(
    trades,
    initial_capital=1000.0
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
            "final_balance": initial_capital,
            "signals_per_day": 0.0,
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
        profit_factor = (
            gross_profit
            / gross_loss
        )
    else:
        profit_factor = None

    net_r = sum(
        t["net_r"]
        for t in trades
    )

    average_r = (
        net_r / len(trades)
    )

    capital_curve = [
        initial_capital
    ]

    for t in trades:
        capital_curve.append(
            t["capital"]
        )

    peak = capital_curve[0]
    max_dd = 0.0

    for value in capital_curve:

        if value > peak:
            peak = value

        if peak > 0:
            dd = (
                (peak - value)
                / peak
                * 100.0
            )

            max_dd = max(
                max_dd,
                dd
            )

    first_ts = trades[0]["timestamp"]
    last_ts = trades[-1]["timestamp"]

    days = max(
        1.0,
        (last_ts - first_ts)
        / 86400.0
    )

    signals_per_day = (
        len(trades)
        / days
    )

    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (
            len(wins)
            / len(trades)
            * 100.0
        ),
        "profit_factor": profit_factor,
        "net_r": net_r,
        "average_r": average_r,
        "max_drawdown": max_dd,
        "final_balance": trades[-1]["capital"],
        "signals_per_day": signals_per_day,
    }


# ============================================================
# SPLIT IS / OOS
# ============================================================

def split_trades(
    trades,
    oos_days=60
):
    if not trades:
        return [], []

    last_timestamp = trades[-1]["timestamp"]

    cutoff = (
        last_timestamp
        - oos_days * 86400
    )

    is_trades = []
    oos_trades = []

    for trade in trades:

        if trade["timestamp"] < cutoff:
            is_trades.append(trade)
        else:
            oos_trades.append(trade)

    return is_trades, oos_trades


# ============================================================
# PRINT
# ============================================================

def print_stats(title, stats):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

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

    print(
        f"Signals/day  : "
        f"{stats['signals_per_day']:.2f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SCORE HUNTER v12")
    print("LBANK REAL DATA BACKTEST")
    print("1H TREND + 15M BREAKOUT / RETEST")
    print("EMA + ADX + RSI + ATR + RELATIVE VOLUME")
    print("CLOSED CANDLE ONLY")
    print("ENTRY = NEXT 15M OPEN")
    print("TP = 2R")
    print("FEES + SLIPPAGE")
    print("PAGINATED HISTORICAL DATA")
    print("NO PANDAS / NO NUMPY")
    print("=" * 70)

    all_trades = []
    coin_results = {}

    for symbol in SYMBOLS:

        print()
        print(
            f"Downloading {symbol} "
            f"{TIMEFRAME_ENTRY}..."
        )

        entry_candles = download_klines(
            symbol,
            TIMEFRAME_ENTRY,
            ENTRY_CANDLES
        )

        print(
            f"{symbol} {TIMEFRAME_ENTRY}: "
            f"{len(entry_candles)} closed candles"
        )

        print(
            f"Downloading {symbol} "
            f"{TIMEFRAME_TREND}..."
        )

        trend_candles = download_klines(
            symbol,
            TIMEFRAME_TREND,
            TREND_CANDLES
        )

        print(
            f"{symbol} {TIMEFRAME_TREND}: "
            f"{len(trend_candles)} closed candles"
        )

        if (
            len(entry_candles)
            < MIN_REQUIRED_BARS
            or
            len(trend_candles)
            < MIN_REQUIRED_BARS
        ):
            print(
                f"{symbol}: NOT ENOUGH DATA"
            )
            continue

        result = run_backtest(
            entry_candles,
            trend_candles,
            INITIAL_CAPITAL
        )

        stats = calculate_stats(
            result["trades"],
            INITIAL_CAPITAL
        )

        coin_results[symbol] = {
            "candles_15m": len(
                entry_candles
            ),
            "candles_1h": len(
                trend_candles
            ),
            "stats": stats,
        }

        for trade in result["trades"]:
            trade["symbol"] = symbol

        all_trades.extend(
            result["trades"]
        )

        print_stats(
            symbol,
            stats
        )

    # --------------------------------------------------------
    # Sort all trades
    # --------------------------------------------------------

    all_trades.sort(
        key=lambda x: x["timestamp"]
    )

    # --------------------------------------------------------
    # ALL COINS
    # --------------------------------------------------------

    all_stats = calculate_stats(
        all_trades,
        INITIAL_CAPITAL
    )

    print_stats(
        "ALL COINS",
        all_stats
    )

    # --------------------------------------------------------
    # IS / OOS
    # --------------------------------------------------------

    is_trades, oos_trades = split_trades(
        all_trades,
        OOS_DAYS
    )

    is_stats = calculate_stats(
        is_trades,
        INITIAL_CAPITAL
    )

    oos_stats = calculate_stats(
        oos_trades,
        INITIAL_CAPITAL
    )

    print_stats(
        f"IN SAMPLE - BEFORE LAST {OOS_DAYS} DAYS",
        is_stats
    )

    print_stats(
        f"OUT OF SAMPLE - LAST {OOS_DAYS} DAYS",
        oos_stats
    )

    # --------------------------------------------------------
    # TARGET CHECK
    # --------------------------------------------------------

    win_rate_ok = (
        60.0
        <= all_stats["win_rate"]
        <= 70.0
    )

    frequency_ok = (
        all_stats["signals_per_day"]
        >= 2.0
    )

    pf_ok = (
        all_stats["profit_factor"]
        is not None
        and
        all_stats["profit_factor"]
        > 1.20
    )

    oos_win_ok = (
        oos_stats["trades"] >= 20
        and
        oos_stats["win_rate"] >= 55.0
    )

    oos_pf_ok = (
        oos_stats["trades"] >= 20
        and
        oos_stats["profit_factor"]
        is not None
        and
        oos_stats["profit_factor"]
        > 1.10
    )

    print()
    print("=" * 70)
    print("TARGET / QUALITY CHECK")
    print("=" * 70)

    print(
        "Win Rate 60-70%       : "
        + ("PASS" if win_rate_ok else "FAIL")
    )

    print(
        "Signals >= 2/day     : "
        + ("PASS" if frequency_ok else "FAIL")
    )

    print(
        "Profit Factor > 1.20 : "
        + ("PASS" if pf_ok else "FAIL")
    )

    print(
        "OOS Win Rate >= 55%  : "
        + ("PASS" if oos_win_ok else "FAIL")
    )

    print(
        "OOS PF > 1.10        : "
        + ("PASS" if oos_pf_ok else "FAIL")
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output = {
        "strategy": "SCORE HUNTER v12",
        "exchange": "LBank",
        "data": {
            "entry_timeframe": TIMEFRAME_ENTRY,
            "trend_timeframe": TIMEFRAME_TREND,
            "entry_candles": ENTRY_CANDLES,
            "trend_candles": TREND_CANDLES,
            "closed_candles_only": True,
            "entry_next_candle_open": True,
        },
        "settings": {
            "initial_capital": INITIAL_CAPITAL,
            "risk_percent": RISK_PERCENT,
            "rr": RR,
            "atr_period": ATR_PERIOD,
            "atr_multiplier": ATR_MULTIPLIER,
            "ema_fast": EMA_FAST,
            "ema_mid": EMA_MID,
            "ema_slow": EMA_SLOW,
            "rsi_period": RSI_PERIOD,
            "adx_period": ADX_PERIOD,
            "adx_min": ADX_MIN,
            "relative_volume_min": RELATIVE_VOLUME_MIN,
            "breakout_lookback": BREAKOUT_LOOKBACK,
            "retest_atr_multiplier": RETEST_ATR_MULTIPLIER,
            "fee_rate": FEE_RATE,
            "slippage_rate": SLIPPAGE_RATE,
        },
        "all_coins": all_stats,
        "in_sample": is_stats,
        "out_of_sample": oos_stats,
        "coins": coin_results,
        "trades": all_trades,
        "target_check": {
            "win_rate_60_70": win_rate_ok,
            "signals_2_per_day": frequency_ok,
            "profit_factor_gt_1_20": pf_ok,
            "oos_win_rate_55_plus": oos_win_ok,
            "oos_profit_factor_gt_1_10": oos_pf_ok,
        },
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print(
        "Results saved to:"
    )
    print(
        OUTPUT_FILE
    )

    print()
    print(
        "BACKTEST FINISHED."
    )


if __name__ == "__main__":
    main()
