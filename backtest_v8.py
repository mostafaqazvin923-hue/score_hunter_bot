import os
import sys
import time
import argparse
import warnings
from datetime import datetime, timedelta, timezone

try:
    import ccxt
except ImportError:
    print("📦 در حال نصب کتابخانه ccxt...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

DAYS = 365
WARMUP_DAYS = 60

TIMEFRAME = "1h"
FETCH_LIMIT = 1000

RR = 2.0

# هزینه تقریبی رفت و برگشت
FEE_RATE = 0.0006

# اسلیپیج هر طرف
SLIPPAGE = 0.0003

# حداکثر ریسک هر معامله
MAX_RISK_PCT = 0.035

# حداکثر زمان نگهداری معامله
MAX_HOLD_BARS = 40

# حداکثر فاصله retest
MAX_RETEST_BARS = 8

# تلورانس retest
RETEST_ATR = 0.35

MIN_DATA_RATIO = 0.90

# برای اینکه به زور 3-4 معامله نسازیم:
# ابتدا تعداد واقعی معاملات استراتژی گزارش می‌شود.
MAX_DAILY_TRADES = None


# ============================================================
# SYMBOLS
# ============================================================

SYMBOLS = {
    "BTC": "BTC",
    "ETH": "ETH",
    "SOL": "SOL",
    "XRP": "XRP",
    "ADA": "ADA",
    "AVAX": "AVAX",
    "LINK": "LINK",
    "NEAR": "NEAR",
    "SUI": "SUI",
    "DOT": "DOT",
}


# ============================================================
# EXCHANGE
# ============================================================

exchange = ccxt.lbank({
    "enableRateLimit": True,
    "options": {
        "defaultType": "swap",
    }
})


# ============================================================
# HELPERS
# ============================================================

def now_utc():
    return datetime.now(timezone.utc)


def resolve_symbol(base):
    """
    اول بازار Perpetual را پیدا می‌کند.
    """

    candidates = [
        f"{base}/USDT:USDT",
        f"{base}/USDT",
    ]

    for symbol in candidates:

        if symbol not in exchange.markets:
            continue

        market = exchange.markets[symbol]

        if market.get("swap") or market.get("contract"):
            return symbol

    raise ValueError(
        f"❌ بازار Futures/Swap برای {base} در LBank پیدا نشد."
    )


def load_markets():

    print("🔄 در حال دریافت لیست بازارهای LBank...")

    exchange.load_markets()

    resolved = {}

    for name, base in SYMBOLS.items():

        try:
            symbol = resolve_symbol(base)
            resolved[name] = symbol

            print(f"  ✔️ {name} -> {symbol}")

        except Exception as e:
            print(f"  ⚠️ {name}: {e}")

    if not resolved:
        raise RuntimeError(
            "هیچ بازار Futures قابل استفاده‌ای در LBank پیدا نشد."
        )

    return resolved


# ============================================================
# DOWNLOAD LBank DATA WITH CCXT
# ============================================================

def download_lbank(symbol, days):

    print(f"\n📥 دانلود دیتای {symbol}")

    end_ms = exchange.milliseconds()

    start_ms = (
        end_ms
        - int((days + WARMUP_DAYS) * 24 * 60 * 60 * 1000)
    )

    all_ohlcv = []

    current_since = start_ms

    while current_since < end_ms:

        try:

            ohlcv = exchange.fetch_ohlcv(
                symbol,
                timeframe=TIMEFRAME,
                since=current_since,
                limit=FETCH_LIMIT
            )

        except Exception as e:

            print(f"  ❌ خطا: {e}")

            time.sleep(5)
            continue

        if not ohlcv:
            break

        all_ohlcv.extend(ohlcv)

        last_timestamp = ohlcv[-1][0]

        next_since = last_timestamp + 1

        if next_since <= current_since:
            break

        current_since = next_since

        print(
            f"  📊 دریافت شد: {len(all_ohlcv)} کندل",
            end="\r"
        )

        if len(ohlcv) < FETCH_LIMIT:
            break

        time.sleep(exchange.rateLimit / 1000)

    print()

    if not all_ohlcv:
        raise RuntimeError(
            f"❌ هیچ دیتایی برای {symbol} دریافت نشد."
        )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        all_ohlcv,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True
    )

    df = df.drop_duplicates(
        subset=["timestamp"]
    )

    df = df.sort_values("timestamp")

    # فقط تا زمان فعلی
    current_time = pd.Timestamp.now(tz="UTC")

    df = df[
        df["timestamp"] <= current_time
    ]

    df = df.set_index("timestamp")

    df = df[
        ["open", "high", "low", "close", "volume"]
    ]

    df = df.dropna()

    # --------------------------------------------------------
    # Remove incomplete current candle
    # --------------------------------------------------------

    current_hour = pd.Timestamp.now(
        tz="UTC"
    ).floor("h")

    df = df[df.index < current_hour]

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    expected = int(
        (days + WARMUP_DAYS) * 24
    )

    minimum = int(
        expected * MIN_DATA_RATIO
    )

    if len(df) < minimum:

        raise RuntimeError(
            f"""
❌ دیتای {symbol} ناقص است.

دریافت شده: {len(df)}
حداقل موردنیاز: {minimum}

بک‌تست متوقف شد تا نتیجه جعلی تولید نشود.
"""
        )

    print(
        f"  ✔️ {symbol}: {len(df)} کندل 1H"
    )

    print(
        f"  🕐 {df.index.min()} -> {df.index.max()}"
    )

    return df


# ============================================================
# INDICATORS
# ============================================================

def ema(series, period):

    return series.ewm(
        span=period,
        adjust=False
    ).mean()


def rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    return 100 - (
        100 / (1 + rs)
    )


def atr(df, period=14):

    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]

    tr2 = (
        df["high"] - prev_close
    ).abs()

    tr3 = (
        df["low"] - prev_close
    ).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


def adx(df, period=14):

    high = df["high"]
    low = df["low"]

    up_move = high.diff()

    down_move = -low.diff()

    plus_dm = np.where(
        (up_move > down_move) &
        (up_move > 0),
        up_move,
        0
    )

    minus_dm = np.where(
        (down_move > up_move) &
        (down_move > 0),
        down_move,
        0
    )

    tr = atr(df, period)

    plus_di = (
        100 *
        pd.Series(
            plus_dm,
            index=df.index
        ).ewm(
            alpha=1 / period,
            adjust=False
        ).mean()
        / tr
    )

    minus_di = (
        100 *
        pd.Series(
            minus_dm,
            index=df.index
        ).ewm(
            alpha=1 / period,
            adjust=False
        ).mean()
        / tr
    )

    dx = (
        100 *
        (plus_di - minus_di).abs()
        /
        (plus_di + minus_di)
        .replace(0, np.nan)
    )

    return dx.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


# ============================================================
# 1H INDICATORS
# ============================================================

def prepare_1h(df):

    x = df.copy()

    x["EMA20"] = ema(x["close"], 20)
    x["EMA50"] = ema(x["close"], 50)
    x["EMA200"] = ema(x["close"], 200)

    x["RSI"] = rsi(x["close"], 14)

    x["ATR"] = atr(x, 14)

    x["ADX"] = adx(x, 14)

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    x["VolumeMedian20"] = (
        x["volume"]
        .rolling(20)
        .median()
    )

    x["RVOL"] = (
        x["volume"]
        /
        x["VolumeMedian20"]
        .replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Candle structure
    # --------------------------------------------------------

    candle_range = (
        x["high"] - x["low"]
    )

    x["Body"] = (
        x["close"] - x["open"]
    ).abs()

    x["BodyATR"] = (
        x["Body"]
        /
        x["ATR"].replace(0, np.nan)
    )

    x["CloseLocation"] = (
        x["close"] - x["low"]
    ) / candle_range.replace(
        0,
        np.nan
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    x["ATR_PCT"] = (
        x["ATR"] /
        x["close"]
    )

    # --------------------------------------------------------
    # Structure
    # --------------------------------------------------------

    x["Prior20High"] = (
        x["high"]
        .rolling(20)
        .max()
        .shift(1)
    )

    x["Prior20Low"] = (
        x["low"]
        .rolling(20)
        .min()
        .shift(1)
    )

    # --------------------------------------------------------
    # EMA slopes
    # --------------------------------------------------------

    x["EMA20Slope"] = (
        x["EMA20"] -
        x["EMA20"].shift(5)
    )

    x["EMA50Slope"] = (
        x["EMA50"] -
        x["EMA50"].shift(5)
    )

    x["EMA200Slope"] = (
        x["EMA200"] -
        x["EMA200"].shift(10)
    )

    return x


# ============================================================
# 4H DATA
# ============================================================

def prepare_4h(df1h):

    # فقط کندل‌های کامل 4H
    df4h = (
        df1h
        .resample(
            "4h",
            label="right",
            closed="right"
        )
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })
        .dropna()
    )

    df4h["EMA20_4H"] = ema(
        df4h["close"],
        20
    )

    df4h["EMA50_4H"] = ema(
        df4h["close"],
        50
    )

    df4h["EMA200_4H"] = ema(
        df4h["close"],
        200
    )

    df4h["RSI_4H"] = rsi(
        df4h["close"],
        14
    )

    df4h["ATR_4H"] = atr(
        df4h,
        14
    )

    df4h["ADX_4H"] = adx(
        df4h,
        14
    )

    df4h["EMA50Slope_4H"] = (
        df4h["EMA50_4H"]
        -
        df4h["EMA50_4H"].shift(3)
    )

    df4h["EMA200Slope_4H"] = (
        df4h["EMA200_4H"]
        -
        df4h["EMA200_4H"].shift(5)
    )

    # --------------------------------------------------------
    # VERY IMPORTANT:
    # Shift one full 4H candle.
    #
    # Therefore 1H candles can NEVER use the still-forming
    # 4H candle.
    # --------------------------------------------------------

    context = df4h.shift(1)

    return context


def merge_4h_context(df1h, context4h):

    x = df1h.copy()

    x = pd.merge_asof(
        x.sort_index(),
        context4h.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward"
    )

    return x


# ============================================================
# 4H REGIME
# ============================================================

def get_4h_regime(row):

    required = [
        "close",
        "EMA20_4H",
        "EMA50_4H",
        "EMA200_4H",
        "RSI_4H",
        "ADX_4H",
        "EMA50Slope_4H",
        "EMA200Slope_4H"
    ]

    if any(
        pd.isna(row[x])
        for x in required
    ):
        return None

    # --------------------------------------------------------
    # LONG REGIME
    # --------------------------------------------------------

    long_regime = (

        row["close"] >
        row["EMA200_4H"]

        and

        row["EMA20_4H"] >
        row["EMA50_4H"]

        and

        row["EMA50_4H"] >
        row["EMA200_4H"]

        and

        row["EMA50Slope_4H"] > 0

        and

        row["EMA200Slope_4H"] >= 0

        and

        row["ADX_4H"] >= 18

        and

        row["RSI_4H"] >= 52
    )

    if long_regime:
        return "LONG"

    # --------------------------------------------------------
    # SHORT REGIME
    # --------------------------------------------------------

    short_regime = (

        row["close"] <
        row["EMA200_4H"]

        and

        row["EMA20_4H"] <
        row["EMA50_4H"]

        and

        row["EMA50_4H"] <
        row["EMA200_4H"]

        and

        row["EMA50Slope_4H"] < 0

        and

        row["EMA200Slope_4H"] <= 0

        and

        row["ADX_4H"] >= 18

        and

        row["RSI_4H"] <= 48
    )

    if short_regime:
        return "SHORT"

    return None


# ============================================================
# SCORE
# ============================================================

def calculate_score(row, direction):

    score = 0

    # ========================================================
    # 4H
    # ========================================================

    regime = get_4h_regime(row)

    if regime != direction:
        return 0

    score += 2

    if row["ADX_4H"] >= 22:
        score += 1

    # ========================================================
    # 1H TREND
    # ========================================================

    if direction == "LONG":

        if (
            row["close"] >
            row["EMA50"]

            and

            row["EMA20"] >
            row["EMA50"]

            and

            row["EMA50"] >
            row["EMA200"]

        ):
            score += 2

    else:

        if (
            row["close"] <
            row["EMA50"]

            and

            row["EMA20"] <
            row["EMA50"]

            and

            row["EMA50"] <
            row["EMA200"]

        ):
            score += 2

    # ========================================================
    # BREAKOUT
    # ========================================================

    if direction == "LONG":

        breakout = (
            row["close"] >
            row["Prior20High"]
        )

        strong_close = (
            row["CloseLocation"] >= 0.70
        )

    else:

        breakout = (
            row["close"] <
            row["Prior20Low"]
        )

        strong_close = (
            row["CloseLocation"] <= 0.30
        )

    if breakout:
        score += 2

    if strong_close:
        score += 1

    # ========================================================
    # VOLUME
    # ========================================================

    if row["RVOL"] >= 1.15:
        score += 1

    # ========================================================
    # MOMENTUM
    # ========================================================

    if direction == "LONG":

        if row["RSI"] >= 55:
            score += 1

    else:

        if row["RSI"] <= 45:
            score += 1

    # ========================================================
    # CANDLE BODY
    # ========================================================

    if row["BodyATR"] >= 0.45:
        score += 1

    # ========================================================
    # EMA SLOPE
    # ========================================================

    if direction == "LONG":

        if row["EMA20Slope"] > 0:
            score += 1

    else:

        if row["EMA20Slope"] < 0:
            score += 1

    return score


# ============================================================
# BREAKOUT DETECTION
# ============================================================

def is_breakout(row, direction):

    if direction == "LONG":

        return (
            row["close"] >
            row["Prior20High"]

            and

            row["BodyATR"] >= 0.45

            and

            row["CloseLocation"] >= 0.70

            and

            row["RVOL"] >= 1.15
        )

    else:

        return (
            row["close"] <
            row["Prior20Low"]

            and

            row["BodyATR"] >= 0.45

            and

            row["CloseLocation"] <= 0.30

            and

            row["RVOL"] >= 1.15
        )


# ============================================================
# RETEST
# ============================================================

def find_retest(
    df,
    breakout_i,
    direction,
    breakout_level
):

    last_i = min(
        len(df) - 2,
        breakout_i + MAX_RETEST_BARS
    )

    for j in range(
        breakout_i + 1,
        last_i + 1
    ):

        row = df.iloc[j]

        if pd.isna(row["ATR"]):
            continue

        tolerance = (
            row["ATR"] *
            RETEST_ATR
        )

        # ====================================================
        # LONG
        # ====================================================

        if direction == "LONG":

            touched = (
                row["low"] <=
                breakout_level + tolerance
            )

            recovered = (
                row["close"] >=
                breakout_level
            )

            bullish = (
                row["close"] >
                row["open"]
            )

            momentum = (
                row["RSI"] >= 50
            )

            volume_ok = (
                row["RVOL"] >= 0.85
            )

            if (
                touched
                and recovered
                and bullish
                and momentum
                and volume_ok
            ):
                return j

        # ====================================================
        # SHORT
        # ====================================================

        else:

            touched = (
                row["high"] >=
                breakout_level - tolerance
            )

            rejected = (
                row["close"] <=
                breakout_level
            )

            bearish = (
                row["close"] <
                row["open"]
            )

            momentum = (
                row["RSI"] <= 50
            )

            volume_ok = (
                row["RVOL"] >= 0.85
            )

            if (
                touched
                and rejected
                and bearish
                and momentum
                and volume_ok
            ):
                return j

    return None


# ============================================================
# STOP LOSS
# ============================================================

def calculate_stop(
    df,
    entry_i,
    direction
):

    row = df.iloc[entry_i]

    atr_value = row["ATR"]

    if pd.isna(atr_value):
        return None

    lookback_start = max(
        0,
        entry_i - 5
    )

    lookback = df.iloc[
        lookback_start:
        entry_i + 1
    ]

    if direction == "LONG":

        swing_low = lookback["low"].min()

        stop = (
            swing_low
            -
            0.20 * atr_value
        )

        return stop

    else:

        swing_high = lookback["high"].max()

        stop = (
            swing_high
            +
            0.20 * atr_value
        )

        return stop


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(
    df,
    signal_i,
    direction,
    stop
):

    # --------------------------------------------------------
    # Signal is confirmed at candle CLOSE.
    #
    # Therefore entry is NEXT candle OPEN.
    #
    # This prevents lookahead within the signal candle.
    # --------------------------------------------------------

    entry_i = signal_i + 1

    if entry_i >= len(df):
        return None

    entry_row = df.iloc[entry_i]

    raw_entry = entry_row["open"]

    # --------------------------------------------------------
    # Slippage
    # --------------------------------------------------------

    if direction == "LONG":

        entry = raw_entry * (
            1 + SLIPPAGE
        )

    else:

        entry = raw_entry * (
            1 - SLIPPAGE
        )

    # --------------------------------------------------------
    # Validate SL
    # --------------------------------------------------------

    if direction == "LONG":

        if stop >= entry:
            return None

    else:

        if stop <= entry:
            return None

    risk = abs(
        entry - stop
    )

    if risk <= 0:
        return None

    # --------------------------------------------------------
    # Risk as %
    # --------------------------------------------------------

    risk_pct = (
        risk / entry
    )

    if risk_pct > MAX_RISK_PCT:
        return None

    # --------------------------------------------------------
    # TP
    # --------------------------------------------------------

    if direction == "LONG":

        target = (
            entry +
            RR * risk
        )

    else:

        target = (
            entry -
            RR * risk
        )

    # --------------------------------------------------------
    # Future candles are used ONLY AFTER entry.
    #
    # This is legitimate trade simulation.
    # --------------------------------------------------------

    last_i = min(
        len(df) - 1,
        entry_i + MAX_HOLD_BARS
    )

    exit_i = None
    exit_price = None
    result = None

    for j in range(
        entry_i,
        last_i + 1
    ):

        row = df.iloc[j]

        high = row["high"]
        low = row["low"]

        if direction == "LONG":

            hit_sl = (
                low <= stop
            )

            hit_tp = (
                high >= target
            )

            # ------------------------------------------------
            # Same candle:
            # conservative assumption = SL first
            # ------------------------------------------------

            if hit_sl and hit_tp:

                exit_i = j
                exit_price = stop
                result = "LOSS"
                break

            elif hit_sl:

                exit_i = j
                exit_price = stop
                result = "LOSS"
                break

            elif hit_tp:

                exit_i = j
                exit_price = target
                result = "WIN"
                break

        else:

            hit_sl = (
                high >= stop
            )

            hit_tp = (
                low <= target
            )

            if hit_sl and hit_tp:

                exit_i = j
                exit_price = stop
                result = "LOSS"
                break

            elif hit_sl:

                exit_i = j
                exit_price = stop
                result = "LOSS"
                break

            elif hit_tp:

                exit_i = j
                exit_price = target
                result = "WIN"
                break

    # --------------------------------------------------------
    # Timeout
    # --------------------------------------------------------

    if result is None:

        exit_i = last_i

        exit_price = df.iloc[
            exit_i
        ]["close"]

        result = "TIMEOUT"

    # --------------------------------------------------------
    # Slippage on exit
    # --------------------------------------------------------

    if direction == "LONG":

        if result == "WIN":

            exit_price *= (
                1 - SLIPPAGE
            )

        else:

            exit_price *= (
                1 - SLIPPAGE
            )

    else:

        exit_price *= (
            1 + SLIPPAGE
        )

    # --------------------------------------------------------
    # Gross R
    # --------------------------------------------------------

    if direction == "LONG":

        pnl_price = (
            exit_price -
            entry
        )

    else:

        pnl_price = (
            entry -
            exit_price
        )

    gross_r = (
        pnl_price /
        risk
    )

    # --------------------------------------------------------
    # Fees
    #
    # Fee converted approximately to R.
    # --------------------------------------------------------

    total_notional_fee = (
        entry +
        exit_price
    ) * FEE_RATE

    fee_r = (
        total_notional_fee /
        risk
    )

    net_r = (
        gross_r -
        fee_r
    )

    return {
        "signal_i": signal_i,
        "entry_i": entry_i,
        "exit_i": exit_i,

        "entry_time": df.index[
            entry_i
        ],

        "exit_time": df.index[
            exit_i
        ],

        "direction": direction,

        "entry": entry,
        "stop": stop,
        "target": target,
        "exit": exit_price,

        "risk_pct": risk_pct,

        "gross_r": gross_r,
        "fee_r": fee_r,
        "net_r": net_r,

        "result": result
    }


# ============================================================
# BACKTEST ONE SYMBOL
# ============================================================

def backtest_symbol(
    symbol_name,
    df,
    days
):

    print(
        f"\n{'=' * 70}"
    )

    print(
        f"🚀 BACKTEST {symbol_name}"
    )

    print(
        f"{'=' * 70}"
    )

    trades = []

    i = 250

    # --------------------------------------------------------
    # Keep signal processing chronological.
    # --------------------------------------------------------

    while i < len(df) - MAX_HOLD_BARS - 2:

        row = df.iloc[i]

        if row.isna().any():
            i += 1
            continue

        regime = get_4h_regime(row)

        if regime is None:
            i += 1
            continue

        # ----------------------------------------------------
        # Determine direction
        # ----------------------------------------------------

        direction = regime

        # ----------------------------------------------------
        # Score
        # ----------------------------------------------------

        score = calculate_score(
            row,
            direction
        )

        # Need strong setup
        if score < 9:
            i += 1
            continue

        # ----------------------------------------------------
        # Breakout
        # ----------------------------------------------------

        if not is_breakout(
            row,
            direction
        ):

            i += 1
            continue

        # ----------------------------------------------------
        # Breakout level
        # ----------------------------------------------------

        if direction == "LONG":

            breakout_level = (
                row["Prior20High"]
            )

        else:

            breakout_level = (
                row["Prior20Low"]
            )

        if pd.isna(breakout_level):
            i += 1
            continue

        # ----------------------------------------------------
        # Retest
        # ----------------------------------------------------

        retest_i = find_retest(
            df,
            i,
            direction,
            breakout_level
        )

        if retest_i is None:

            i += 1
            continue

        # ----------------------------------------------------
        # Retest candle is the actual signal candle.
        # ----------------------------------------------------

        signal_row = df.iloc[
            retest_i
        ]

        # ----------------------------------------------------
        # Re-check 4H regime on retest candle.
        # ----------------------------------------------------

        retest_regime = (
            get_4h_regime(
                signal_row
            )
        )

        if retest_regime != direction:

            i = retest_i + 1
            continue

        # ----------------------------------------------------
        # Stop
        # ----------------------------------------------------

        stop = calculate_stop(
            df,
            retest_i,
            direction
        )

        if stop is None:

            i = retest_i + 1
            continue

        # ----------------------------------------------------
        # Simulate
        # ----------------------------------------------------

        trade = simulate_trade(
            df,
            retest_i,
            direction,
            stop
        )

        if trade is None:

            i = retest_i + 1
            continue

        # ----------------------------------------------------
        # Add metadata
        # ----------------------------------------------------

        trade["symbol"] = symbol_name
        trade["score"] = score
        trade["breakout_time"] = df.index[i]
        trade["breakout_level"] = breakout_level

        trades.append(
            trade
        )

        print(
            f"  {trade['entry_time']} | "
            f"{direction:<5} | "
            f"Score {score:02d} | "
            f"{trade['result']:<8} | "
            f"{trade['net_r']:+.2f}R"
        )

        # ----------------------------------------------------
        # Move forward after completed trade.
        #
        # Prevents overlapping trades on same symbol.
        # ----------------------------------------------------

        i = trade["exit_i"] + 1

    print(
        f"  ✅ {symbol_name}: "
        f"{len(trades)} trades"
    )

    return trades


# ============================================================
# DAILY LIMIT - CAUSAL VERSION
# ============================================================

def apply_daily_limit(trades):

    if MAX_DAILY_TRADES is None:
        return trades

    if not trades:
        return trades

    trades = sorted(
        trades,
        key=lambda x: x["entry_time"]
    )

    counts = {}

    output = []

    for trade in trades:

        day = (
            trade["entry_time"]
            .date()
        )

        current = counts.get(
            day,
            0
        )

        if current >= MAX_DAILY_TRADES:
            continue

        output.append(
            trade
        )

        counts[day] = (
            current + 1
        )

    return output


# ============================================================
# PORTFOLIO REPORT
# ============================================================

def print_report(trades):

    print()
    print("=" * 70)
    print("🏆 HUNTER-X 70/2 CLEAN BACKTEST")
    print("=" * 70)

    if not trades:

        print(
            "❌ No trades generated."
        )

        return

    trades = sorted(
        trades,
        key=lambda x: x["entry_time"]
    )

    total = len(trades)

    wins = sum(
        t["result"] == "WIN"
        for t in trades
    )

    losses = sum(
        t["result"] == "LOSS"
        for t in trades
    )

    timeouts = sum(
        t["result"] == "TIMEOUT"
        for t in trades
    )

    net_r = sum(
        t["net_r"]
        for t in trades
    )

    win_r = [
        t["net_r"]
        for t in trades
        if t["result"] == "WIN"
    ]

    loss_r = [
        t["net_r"]
        for t in trades
        if t["result"] == "LOSS"
    ]

    if wins + losses > 0:

        win_rate_wl = (
            wins /
            (wins + losses)
            * 100
        )

    else:

        win_rate_wl = 0

    win_rate_all = (
        wins /
        total *
        100
    )

    gross_profit = sum(
        max(
            0,
            t["net_r"]
        )
        for t in trades
    )

    gross_loss = abs(
        sum(
            min(
                0,
                t["net_r"]
            )
            for t in trades
        )
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit /
            gross_loss
        )

    else:

        profit_factor = float("inf")

    expectancy = (
        net_r /
        total
    )

    avg_win = (
        np.mean(win_r)
        if win_r
        else 0
    )

    avg_loss = (
        np.mean(loss_r)
        if loss_r
        else 0
    )

    # --------------------------------------------------------
    # Equity / Drawdown
    # --------------------------------------------------------

    equity = 0
    peak = 0
    max_dd = 0

    for trade in trades:

        equity += trade["net_r"]

        peak = max(
            peak,
            equity
        )

        dd = (
            peak -
            equity
        )

        max_dd = max(
            max_dd,
            dd
        )

    # --------------------------------------------------------
    # Trades/day
    # --------------------------------------------------------

    first_day = (
        trades[0]["entry_time"]
        .date()
    )

    last_day = (
        trades[-1]["entry_time"]
        .date()
    )

    calendar_days = (
        last_day -
        first_day
    ).days + 1

    trades_per_day = (
        total /
        calendar_days
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"Total Trades       : {total}"
    )

    print(
        f"Wins               : {wins}"
    )

    print(
        f"Losses             : {losses}"
    )

    print(
        f"Timeouts           : {timeouts}"
    )

    print(
        f"Win Rate (W/L)     : {win_rate_wl:.2f}%"
    )

    print(
        f"Win Rate (All)      : {win_rate_all:.2f}%"
    )

    print(
        f"Net Profit          : {net_r:+.2f}R"
    )

    print(
        f"Profit Factor       : {profit_factor:.2f}"
    )

    print(
        f"Expectancy          : {expectancy:+.4f}R"
    )

    print(
        f"Average Win         : {avg_win:+.2f}R"
    )

    print(
        f"Average Loss        : {avg_loss:+.2f}R"
    )

    print(
        f"Max Drawdown        : {max_dd:.2f}R"
    )

    print(
        f"RR                  : 1:{RR}"
    )

    print(
        f"Trades / Day        : {trades_per_day:.2f}"
    )

    # ========================================================
    # SYMBOL REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("📊 SYMBOL REPORT")
    print("=" * 70)

    symbol_rows = []

    for symbol in sorted(
        set(
            t["symbol"]
            for t in trades
        )
    ):

        st = [
            t for t in trades
            if t["symbol"] == symbol
        ]

        sw = sum(
            t["result"] == "WIN"
            for t in st
        )

        sl = sum(
            t["result"] == "LOSS"
            for t in st
        )

        sr = sum(
            t["net_r"]
            for t in st
        )

        wr = (
            sw /
            (sw + sl) *
            100
            if sw + sl
            else 0
        )

        symbol_rows.append({
            "Symbol": symbol,
            "Trades": len(st),
            "Wins": sw,
            "Losses": sl,
            "WinRate": wr,
            "NetR": sr
        })

        print(
            f"{symbol:<8} "
            f"{len(st):>4} trades | "
            f"W {sw:>3} | "
            f"L {sl:>3} | "
            f"WR {wr:>6.2f}% | "
            f"{sr:+.2f}R"
        )

    # ========================================================
    # SCORE REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("🎯 SCORE REPORT")
    print("=" * 70)

    score_values = sorted(
        set(
            t["score"]
            for t in trades
        )
    )

    for score in score_values:

        st = [
            t for t in trades
            if t["score"] == score
        ]

        sw = sum(
            t["result"] == "WIN"
            for t in st
        )

        sl = sum(
            t["result"] == "LOSS"
            for t in st
        )

        sr = sum(
            t["net_r"]
            for t in st
        )

        wr = (
            sw /
            (sw + sl) *
            100
            if sw + sl
            else 0
        )

        print(
            f"Score {score:02d}: "
            f"{len(st)} trades | "
            f"WR {wr:.2f}% | "
            f"Net {sr:+.2f}R"
        )

    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "win_rate_wl": win_rate_wl,
        "win_rate_all": win_rate_all,
        "net_r": net_r,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_drawdown": max_dd,
        "trades_per_day": trades_per_day,
        "symbol_rows": symbol_rows
    }


# ============================================================
# SAVE TRADES
# ============================================================

def save_trades(trades):

    if not trades:
        return

    rows = []

    for t in trades:

        rows.append({
            "symbol": t["symbol"],
            "direction": t["direction"],
            "score": t["score"],

            "breakout_time": t[
                "breakout_time"
            ],

            "entry_time": t[
                "entry_time"
            ],

            "exit_time": t[
                "exit_time"
            ],

            "entry": t["entry"],
            "stop": t["stop"],
            "target": t["target"],
            "exit": t["exit"],

            "risk_pct": t[
                "risk_pct"
            ],

            "gross_r": t[
                "gross_r"
            ],

            "fee_r": t[
                "fee_r"
            ],

            "net_r": t[
                "net_r"
            ],

            "result": t[
                "result"
            ]
        })

    out = pd.DataFrame(rows)

    out.to_csv(
        "HUNTER_X_70_2_TRADES.csv",
        index=False
    )

    print()
    print(
        "💾 معاملات ذخیره شدند:"
    )

    print(
        "HUNTER_X_70_2_TRADES.csv"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    global DAYS

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--days",
        type=int,
        default=DAYS
    )

    args = parser.parse_args()

    DAYS = args.days

    print()
    print("=" * 70)
    print("🔥 HUNTER-X 70/2 CLEAN")
    print("=" * 70)

    print(
        f"📅 Backtest: {DAYS} days"
    )

    print(
        f"⏱ Timeframe: 1H + 4H"
    )

    print(
        f"🎯 RR: 1:{RR}"
    )

    print(
        "🛡 Lookahead: DISABLED"
    )

    print(
        "🕯 Closed 4H candles only"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Markets
    # --------------------------------------------------------

    resolved_symbols = load_markets()

    all_trades = []

    # --------------------------------------------------------
    # Process symbols
    # --------------------------------------------------------

    for name, lbank_symbol in resolved_symbols.items():

        try:

            df1h = download_lbank(
                lbank_symbol,
                DAYS
            )

            # ------------------------------------------------
            # 1H indicators
            # ------------------------------------------------

            df1h = prepare_1h(
                df1h
            )

            # ------------------------------------------------
            # 4H indicators
            # ------------------------------------------------

            context4h = prepare_4h(
                df1h
            )

            # ------------------------------------------------
            # Merge ONLY CLOSED 4H
            # ------------------------------------------------

            df = merge_4h_context(
                df1h,
                context4h
            )

            # ------------------------------------------------
            # Backtest
            # ------------------------------------------------

            trades = backtest_symbol(
                name,
                df,
                DAYS
            )

            all_trades.extend(
                trades
            )

        except Exception as e:

            print()
            print(
                f"❌ {name}: {e}"
            )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    all_trades = sorted(
        all_trades,
        key=lambda x:
        x["entry_time"]
    )

    # --------------------------------------------------------
    # Optional daily limit
    #
    # None = no artificial limitation
    # --------------------------------------------------------

    all_trades = apply_daily_limit(
        all_trades
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = print_report(
        all_trades
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_trades(
        all_trades
    )

    print()
    print("=" * 70)
    print("🏁 BACKTEST FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()
