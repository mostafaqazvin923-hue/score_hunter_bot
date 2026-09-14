# CLTS v1 — Causal Liquidity/Structure Trend System
# LBank USDT-M Futures / CCXT
#
# IMPORTANT:
# - Uses only closed 1H and 4H candles.
# - Entry is at the NEXT 1H candle open after a confirmed signal.
# - Confirmed pivots are timestamped at the candle on which confirmation becomes available.
# - No timeout / max-bars exit.
# - No overlapping position per symbol.
# - Portfolio max positions + correlation-group lock.
# - If SL and TP are both touched in the same candle, SL wins (conservative).
# - A candle that closes a trade is NOT allowed to create a new signal for that symbol.
# - The script prints portfolio + per-symbol results and every consecutive-loss streak.
#
# Install:
#   pip install ccxt pandas numpy
#
# Run:
#   python clts_lbank_backtest.py
#
# NOTE:
# The script fetches public market data from LBank. No API key is needed for
# historical market data. It is deliberately NOT an order-execution bot.

import sys
import subprocess
from datetime import datetime, timedelta, timezone

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DAYS = 365

# Exact requested universe. These are LBank USDT perpetual contracts.
SYMBOLS = {
    "BTC": "BTC/USDT:USDT",
    "ETH": "ETH/USDT:USDT",
    "SOL": "SOL/USDT:USDT",
    "XRP": "XRP/USDT:USDT",
    "SUI": "SUI/USDT:USDT",
    "NEAR": "NEAR/USDT:USDT",
    "ADA": "ADA/USDT:USDT",
    "LINK": "LINK/USDT:USDT",
    "AVAX": "AVAX/USDT:USDT",
    "DOT": "DOT/USDT:USDT",
}

# Correlation/exposure groups.
# Only one position from each group can be open at once.
CORRELATION_GROUP = {
    "BTC": "MAJOR",
    "ETH": "MAJOR",
    "SOL": "L1_HIGH_BETA",
    "SUI": "L1_HIGH_BETA",
    "AVAX": "L1_HIGH_BETA",
    "XRP": "PAYMENTS",
    "ADA": "PAYMENTS",
    "DOT": "L1_OTHER",
    "LINK": "INFRA",
    "NEAR": "INFRA",
}

MAX_OPEN_POSITIONS = 3

# Strategy parameters — baseline, intentionally not aggressively optimized.
PIVOT_LEFT = 2
PIVOT_RIGHT = 2

EMA_FAST = 50
EMA_SLOW = 200
ADX_LEN = 14
ATR_LEN = 14
RSI_LEN = 14
RVOL_LEN = 20

ADX_MIN = 20.0
RVOL_MIN = 1.20
RSI_LONG_MIN = 50.0
RSI_SHORT_MAX = 50.0

BOS_ATR_BUFFER = 0.10
RETEST_ATR_BUFFER = 0.15
SL_ATR_BUFFER = 0.20

MAX_RISK_PCT_OF_ENTRY = 0.025  # reject trades whose structural stop is >2.5%

TP_R = 2.0

# Backtest execution assumptions.
# LBank currently publishes a 0.06% taker fee in its futures fee documentation.
# We model both entry and exit as taker executions by default.
TAKER_FEE = 0.0006

# Conservative modeled slippage. This is an assumption, not historical LBank tick data.
SLIPPAGE = 0.0002

# No timeout. This constant exists only as an explicit audit marker.
TIMEOUT_ENABLED = False

# If both SL and TP are inside the same candle, always take the conservative SL first.
CONSERVATIVE_INTRABAR = True


# ============================================================
# EXCHANGE
# ============================================================

exchange = ccxt.lbank({
    "enableRateLimit": True,
    "options": {
        "defaultType": "swap",
    },
})


# ============================================================
# DATA HELPERS
# ============================================================

def utc_now():
    return datetime.now(timezone.utc)


def timeframe_ms(timeframe: str) -> int:
    return {
        "1h": 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
    }[timeframe]


def floor_timestamp_ms(ts_ms: int, tf_ms: int) -> int:
    return (ts_ms // tf_ms) * tf_ms


def fetch_ohlcv_paginated(symbol, timeframe, since_ms, until_ms):
    """Fetch historical OHLCV in chronological pages."""
    tf_ms = timeframe_ms(timeframe)
    rows = []
    cursor = since_ms

    while cursor < until_ms:
        try:
            batch = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=cursor,
                limit=1000,
            )
        except Exception as exc:
            print(f"  ERROR {symbol} {timeframe}: {exc}")
            break

        if not batch:
            break

        rows.extend(batch)

        last_ts = int(batch[-1][0])
        next_cursor = last_ts + tf_ms

        if next_cursor <= cursor:
            break

        cursor = next_cursor

        if len(batch) < 1000:
            break

    if not rows:
        return pd.DataFrame(
            columns=["Date", "Open", "High", "Low", "Close", "Volume"]
        )

    df = pd.DataFrame(
        rows,
        columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"],
    )

    df["Timestamp"] = df["Timestamp"].astype(np.int64)

    # Only candles whose OPEN time is inside requested range.
    df = df[
        (df["Timestamp"] >= since_ms) &
        (df["Timestamp"] < until_ms)
    ].copy()

    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)

    df = df[
        ["Date", "Open", "High", "Low", "Close", "Volume"]
    ].drop_duplicates("Date").sort_values("Date").reset_index(drop=True)

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    # Sanity validation.
    bad = (
        (df["High"] < df[["Open", "Close"]].max(axis=1)) |
        (df["Low"] > df[["Open", "Close"]].min(axis=1)) |
        (df["High"] < df["Low"]) |
        (df["Volume"] < 0)
    )
    if bad.any():
        print(f"  WARNING: removing {int(bad.sum())} invalid candles for {symbol} {timeframe}")
        df = df.loc[~bad].reset_index(drop=True)

    # Remove the currently-open candle.
    now_ms = exchange.milliseconds()
    last_complete_open = floor_timestamp_ms(now_ms, tf_ms) - tf_ms
    df = df[df["Timestamp"] <= last_complete_open].copy()

    return df.reset_index(drop=True)


# ============================================================
# INDICATORS — ALL CAUSAL
# ============================================================

def true_range(df):
    prev_close = df["Close"].shift(1)
    return pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr_wilder(df, length=14):
    tr = true_range(df)
    # ewm(..., adjust=False) is causal.
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def rsi_wilder(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def adx_wilder(df, length=14):
    high = df["High"]
    low = df["Low"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0,
        ),
        index=df.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0,
        ),
        index=df.index,
    )

    tr = true_range(df)

    atr = tr.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length,
    ).mean()

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / length,
            adjust=False,
            min_periods=length,
        ).mean()
        / atr
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / length,
            adjust=False,
            min_periods=length,
        ).mean()
        / atr
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    return dx.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length,
    ).mean()


def add_indicators(df):
    df = df.copy()

    df["EMA50"] = df["Close"].ewm(
        span=EMA_FAST,
        adjust=False,
        min_periods=EMA_FAST,
    ).mean()

    df["EMA200"] = df["Close"].ewm(
        span=EMA_SLOW,
        adjust=False,
        min_periods=EMA_SLOW,
    ).mean()

    df["ATR"] = atr_wilder(df, ATR_LEN)
    df["ADX"] = adx_wilder(df, ADX_LEN)
    df["RSI"] = rsi_wilder(df["Close"], RSI_LEN)

    volume_mean = df["Volume"].rolling(
        RVOL_LEN,
        min_periods=RVOL_LEN,
    ).mean()

    df["RVOL"] = df["Volume"] / volume_mean.replace(0, np.nan)

    return df


# ============================================================
# CONFIRMED PIVOTS
# ============================================================

def build_confirmed_pivots(df):
    """
    A pivot at i is NOT usable at i.

    It becomes usable only at i + PIVOT_RIGHT.

    This avoids assigning future-confirmed structure to the past.
    """
    n = len(df)

    swing_high_confirmed = [None] * n
    swing_low_confirmed = [None] * n

    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()

    for i in range(PIVOT_LEFT, n - PIVOT_RIGHT):
        left_highs = highs[i - PIVOT_LEFT:i]
        right_highs = highs[i + 1:i + 1 + PIVOT_RIGHT]

        left_lows = lows[i - PIVOT_LEFT:i]
        right_lows = lows[i + 1:i + 1 + PIVOT_RIGHT]

        is_high = (
            highs[i] > left_highs.max()
            and highs[i] >= right_highs.max()
        )

        is_low = (
            lows[i] < left_lows.min()
            and lows[i] <= right_lows.min()
        )

        confirmation_index = i + PIVOT_RIGHT

        if is_high:
            swing_high_confirmed[confirmation_index] = {
                "pivot_index": i,
                "confirmed_index": confirmation_index,
                "price": float(highs[i]),
            }

        if is_low:
            swing_low_confirmed[confirmation_index] = {
                "pivot_index": i,
                "confirmed_index": confirmation_index,
                "price": float(lows[i]),
            }

    df["NewSwingHigh"] = swing_high_confirmed
    df["NewSwingLow"] = swing_low_confirmed

    return df


# ============================================================
# 4H REGIME
# ============================================================

def regime_from_4h(row, previous_row):
    vals = [
        row["EMA50"],
        row["EMA200"],
        row["ADX"],
        row["Close"],
    ]

    if any(pd.isna(x) for x in vals):
        return "NEUTRAL"

    if pd.isna(previous_row["EMA50"]):
        return "NEUTRAL"

    bullish = (
        row["EMA50"] > row["EMA200"]
        and row["Close"] > row["EMA50"]
        and row["EMA50"] > previous_row["EMA50"]
        and row["ADX"] >= ADX_MIN
    )

    bearish = (
        row["EMA50"] < row["EMA200"]
        and row["Close"] < row["EMA50"]
        and row["EMA50"] < previous_row["EMA50"]
        and row["ADX"] >= ADX_MIN
    )

    if bullish:
        return "BULL"
    if bearish:
        return "BEAR"
    return "NEUTRAL"


def build_4h_regime(df4h):
    df = df4h.copy()
    regimes = ["NEUTRAL"] * len(df)

    for i in range(1, len(df)):
        regimes[i] = regime_from_4h(df.iloc[i], df.iloc[i - 1])

    df["Regime"] = regimes
    return df


# ============================================================
# 4H -> 1H CAUSAL ALIGNMENT
# ============================================================

def attach_4h_regime(df1h, df4h):
    """
    Each 1H candle receives only the LAST COMPLETED 4H regime.

    4H candle timestamp is its OPEN time. Its regime is available
    only after that 4H candle closes, i.e. at open_time + 4h.

    Therefore the 1H candle at the exact 4H close can use it.
    """
    h4 = df4h.copy()

    h4["RegimeAvailableAt"] = h4["Date"] + pd.Timedelta(hours=4)

    regime_map = h4[["RegimeAvailableAt", "Regime"]].sort_values(
        "RegimeAvailableAt"
    )

    out = pd.merge_asof(
        df1h.sort_values("Date"),
        regime_map,
        left_on="Date",
        right_on="RegimeAvailableAt",
        direction="backward",
        allow_exact_matches=True,
    )

    out["Regime"] = out["Regime"].fillna("NEUTRAL")
    return out


# ============================================================
# SIGNAL STATE
# ============================================================

def find_recent_confirmed_pivot(pivot_column, upto_index):
    for j in range(upto_index, -1, -1):
        x = pivot_column.iloc[j]
        if isinstance(x, dict):
            return x
    return None


def generate_signal_at(df, i):
    """
    Signal is generated ONLY from information available at candle close i.

    Entry is always at candle i+1 OPEN.
    """
    if i < 1:
        return None

    row = df.iloc[i]

    if row["Regime"] not in ("BULL", "BEAR"):
        return None

    if any(pd.isna(row[c]) for c in ["ATR", "RSI", "RVOL"]):
        return None

    atr = float(row["ATR"])

    if atr <= 0:
        return None

    # We only use pivots already confirmed by time i.
    prior_high = find_recent_confirmed_pivot(df["NewSwingHigh"], i)
    prior_low = find_recent_confirmed_pivot(df["NewSwingLow"], i)

    if prior_high is None or prior_low is None:
        return None

    # Need a pivot before current candle, not one created at current candle.
    if prior_high["confirmed_index"] >= i:
        return None
    if prior_low["confirmed_index"] >= i:
        return None

    # ========================================================
    # LONG: liquidity sweep below confirmed swing low
    # followed by bullish BOS, then retest
    # ========================================================
    if row["Regime"] == "BULL":
        sweep_level = prior_low["price"]

        swept = (
            float(row["Low"]) < sweep_level
            and float(row["Close"]) > sweep_level
        )

        if not swept:
            return None

        # The swing high used for BOS must be a confirmed swing
        # that existed before this sweep candle.
        pre_sweep_high = None

        for j in range(i - 1, -1, -1):
            x = df["NewSwingHigh"].iloc[j]
            if isinstance(x, dict):
                pre_sweep_high = x
                break

        if pre_sweep_high is None:
            return None

        bos_level = float(pre_sweep_high["price"])

        # BOS must occur AFTER the sweep, not on the same candle.
        # Therefore this function alone does not create the trade.
        # The state machine below tracks sweep -> BOS -> retest.
        return {
            "event": "SWEEP_LONG",
            "index": i,
            "sweep_low": sweep_level,
            "bos_reference": bos_level,
            "atr": atr,
        }

    # ========================================================
    # SHORT: liquidity sweep above confirmed swing high
    # ========================================================
    if row["Regime"] == "BEAR":
        sweep_level = prior_high["price"]

        swept = (
            float(row["High"]) > sweep_level
            and float(row["Close"]) < sweep_level
        )

        if not swept:
            return None

        pre_sweep_low = None

        for j in range(i - 1, -1, -1):
            x = df["NewSwingLow"].iloc[j]
            if isinstance(x, dict):
                pre_sweep_low = x
                break

        if pre_sweep_low is None:
            return None

        bos_level = float(pre_sweep_low["price"])

        return {
            "event": "SWEEP_SHORT",
            "index": i,
            "sweep_high": sweep_level,
            "bos_reference": bos_level,
            "atr": atr,
        }

    return None


# ============================================================
# PER-SYMBOL SETUP STATE MACHINE
# ============================================================

def build_signals(df):
    """
    Stateful, chronological signal generator.

    States:
        IDLE
        AFTER_SWEEP_LONG / AFTER_BOS_LONG
        AFTER_SWEEP_SHORT / AFTER_BOS_SHORT

    No future candle is referenced for signal timing.
    """
    df = df.copy()

    pending = None
    signals = []

    for i in range(len(df)):
        row = df.iloc[i]

        # We need enough history.
        if i < max(EMA_SLOW, RVOL_LEN, ATR_LEN, ADX_LEN) + 5:
            continue

        # ----------------------------------------------------
        # Existing pending setup
        # ----------------------------------------------------
        if pending is not None:

            # Regime must still support the direction.
            if (
                (pending["side"] == "LONG" and row["Regime"] != "BULL")
                or
                (pending["side"] == "SHORT" and row["Regime"] != "BEAR")
            ):
                pending = None
                continue

            atr = float(row["ATR"]) if not pd.isna(row["ATR"]) else np.nan
            if not np.isfinite(atr) or atr <= 0:
                continue

            # ---------------- LONG ----------------
            if pending["side"] == "LONG":

                # BOS must occur after the sweep.
                if pending["stage"] == "SWEEP":
                    if (
                        float(row["Close"])
                        > pending["bos_level"] + BOS_ATR_BUFFER * atr
                    ):
                        pending["stage"] = "BOS"
                        pending["bos_index"] = i
                        pending["bos_level"] = pending["bos_level"]

                        # Do NOT enter on BOS candle.
                        continue

                elif pending["stage"] == "BOS":
                    bos = pending["bos_level"]

                    retest = (
                        float(row["Low"]) <= bos + RETEST_ATR_BUFFER * atr
                        and float(row["Close"]) > bos
                    )

                    if retest:
                        # Confirmation is complete at close i.
                        # Entry is at OPEN i+1.
                        if i + 1 < len(df):
                            structure_low = min(
                                float(df["Low"].iloc[pending["sweep_index"]:i + 1].min()),
                                float(df["Low"].iloc[i]),
                            )

                            # Entry uses next candle open, which is not
                            # known until next candle arrives. The signal
                            # stores only causal information.
                            signals.append({
                                "signal_index": i,
                                "entry_index": i + 1,
                                "side": "LONG",
                                "bos_level": bos,
                                "structure_low": structure_low,
                            })

                        pending = None
                        continue

            # ---------------- SHORT ----------------
            else:
                if pending["stage"] == "SWEEP":
                    if (
                        float(row["Close"])
                        < pending["bos_level"] - BOS_ATR_BUFFER * atr
                    ):
                        pending["stage"] = "BOS"
                        pending["bos_index"] = i
                        continue

                elif pending["stage"] == "BOS":
                    bos = pending["bos_level"]

                    retest = (
                        float(row["High"]) >= bos - RETEST_ATR_BUFFER * atr
                        and float(row["Close"]) < bos
                    )

                    if retest:
                        if i + 1 < len(df):
                            structure_high = max(
                                float(df["High"].iloc[pending["sweep_index"]:i + 1].max()),
                                float(df["High"].iloc[i]),
                            )

                            signals.append({
                                "signal_index": i,
                                "entry_index": i + 1,
                                "side": "SHORT",
                                "bos_level": bos,
                                "structure_high": structure_high,
                            })

                        pending = None
                        continue

        # ----------------------------------------------------
        # No pending setup: detect a NEW sweep
        # ----------------------------------------------------
        if pending is None:
            event = generate_signal_at(df, i)

            if event is None:
                continue

            if event["event"] == "SWEEP_LONG":
                pending = {
                    "side": "LONG",
                    "stage": "SWEEP",
                    "sweep_index": i,
                    "bos_level": event["bos_reference"],
                }

            elif event["event"] == "SWEEP_SHORT":
                pending = {
                    "side": "SHORT",
                    "stage": "SWEEP",
                    "sweep_index": i,
                    "bos_level": event["bos_reference"],
                }

    return signals


# ============================================================
# PREPARE ALL SYMBOL DATA
# ============================================================

def prepare_symbol(symbol_name, unified_symbol, since_ms, until_ms):
    print(f"\n📥 {symbol_name}: downloading LBank Futures 1H + 4H")

    df1h = fetch_ohlcv_paginated(
        unified_symbol,
        "1h",
        since_ms,
        until_ms,
    )

    df4h = fetch_ohlcv_paginated(
        unified_symbol,
        "4h",
        since_ms - 400 * 4 * 60 * 60 * 1000,
        until_ms,
    )

    if len(df1h) < 500 or len(df4h) < 250:
        print(
            f"  ⚠️ insufficient data: 1H={len(df1h)}, 4H={len(df4h)}"
        )
        return None

    df4h = add_indicators(df4h)
    df4h = build_4h_regime(df4h)

    df1h = add_indicators(df1h)
    df1h = build_confirmed_pivots(df1h)
    df1h = attach_4h_regime(df1h, df4h)

    signals = build_signals(df1h)

    # Keep only signals whose entry candle is inside requested test range.
    signals = [
        s for s in signals
        if s["entry_index"] < len(df1h)
    ]

    print(
        f"  ✓ 1H candles={len(df1h)}, 4H candles={len(df4h)}, "
        f"candidate signals={len(signals)}"
    )

    return {
        "df": df1h,
        "signals": signals,
    }


# ============================================================
# EXECUTION HELPERS
# ============================================================

def apply_entry_slippage(side, price):
    if side == "LONG":
        return price * (1.0 + SLIPPAGE)
    return price * (1.0 - SLIPPAGE)


def apply_exit_slippage(side, price):
    # Long closes by selling; short closes by buying.
    if side == "LONG":
        return price * (1.0 - SLIPPAGE)
    return price * (1.0 + SLIPPAGE)


def calculate_net_r(pos, exit_price, fee_rate=TAKER_FEE):
    entry = pos["entry_price"]
    sl = pos["stop_loss"]
    risk = abs(entry - sl)

    if risk <= 0:
        return 0.0

    if pos["side"] == "LONG":
        gross_r = (exit_price - entry) / risk
    else:
        gross_r = (entry - exit_price) / risk

    # Fee is charged on entry + exit notional.
    # Approximate R deduction using the entry risk denominator.
    entry_fee = entry * fee_rate
    exit_fee = exit_price * fee_rate
    fee_r = (entry_fee + exit_fee) / risk

    return gross_r - fee_r


# ============================================================
# SIGNAL SCORE
# ============================================================

def score_signal(df, signal):
    i = signal["signal_index"]
    row = df.iloc[i]

    score = 0.0

    adx = float(row["ADX"])
    rvol = float(row["RVOL"])
    rsi = float(row["RSI"])

    # Trend strength
    if adx >= 30:
        score += 2
    elif adx >= 20:
        score += 1

    # Volume
    if rvol >= 1.50:
        score += 2
    elif rvol >= 1.20:
        score += 1

    # Momentum
    if signal["side"] == "LONG":
        if rsi >= 60:
            score += 2
        elif rsi >= 50:
            score += 1
    else:
        if rsi <= 40:
            score += 2
        elif rsi <= 50:
            score += 1

    # Structure displacement
    atr = float(row["ATR"])

    if signal["side"] == "LONG":
        displacement = float(row["Close"]) - signal["bos_level"]
    else:
        displacement = signal["bos_level"] - float(row["Close"])

    if displacement >= 0.25 * atr:
        score += 2
    elif displacement >= 0.10 * atr:
        score += 1

    return score


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(processed):
    """
    Chronological event engine.

    Critical ordering:
        1. Existing positions are checked for exits using current candle.
        2. If a position closes on this candle, that symbol is LOCKED
           against new signals on the same candle.
        3. New candidate signals are considered only from signals whose
           signal candle already closed before this entry candle.
        4. Entries happen at the next candle OPEN.
    """
    active_positions = {}
    all_trades = []

    # Build global event timestamps from all 1H candles.
    timestamps = sorted({
        ts
        for item in processed.values()
        for ts in item["df"]["Date"]
    })

    # Pre-index signals by their ENTRY timestamp.
    signals_by_entry_ts = {}

    for symbol, item in processed.items():
        df = item["df"]

        for sig in item["signals"]:
            entry_i = sig["entry_index"]

            if entry_i >= len(df):
                continue

            entry_ts = df["Date"].iloc[entry_i]

            sig2 = dict(sig)
            sig2["symbol"] = symbol
            sig2["entry_ts"] = entry_ts

            # Signal score was computed from the CLOSED signal candle.
            sig2["score"] = score_signal(df, sig)

            signals_by_entry_ts.setdefault(entry_ts, []).append(sig2)

    # Prevent re-entry on the candle that closes a trade.
    blocked_until_next_candle = set()

    for ts in timestamps:

        # ====================================================
        # 1) EXIT OPEN POSITIONS
        # ====================================================
        closed_symbols = set()

        for symbol in list(active_positions.keys()):
            pos = active_positions[symbol]
            df = processed[symbol]["df"]

            matches = df.index[df["Date"] == ts]
            if len(matches) == 0:
                continue

            i = int(matches[0])
            candle = df.iloc[i]

            hit_sl = False
            hit_tp = False

            if pos["side"] == "LONG":
                hit_sl = float(candle["Low"]) <= pos["stop_loss"]
                hit_tp = float(candle["High"]) >= pos["take_profit"]

            else:
                hit_sl = float(candle["High"]) >= pos["stop_loss"]
                hit_tp = float(candle["Low"]) <= pos["take_profit"]

            if not (hit_sl or hit_tp):
                continue

            # Conservative intrabar rule:
            # If both are touched, SL wins.
            if hit_sl and hit_tp and CONSERVATIVE_INTRABAR:
                outcome = "LOSS"
                exit_reason = "SL_AMBIGUOUS"
                raw_exit = pos["stop_loss"]
            elif hit_sl:
                outcome = "LOSS"
                exit_reason = "SL"
                raw_exit = pos["stop_loss"]
            else:
                outcome = "WIN"
                exit_reason = "TP"
                raw_exit = pos["take_profit"]

            exit_price = apply_exit_slippage(pos["side"], raw_exit)

            gross_r = (
                -1.0 if outcome == "LOSS" else TP_R
            )

            net_r = calculate_net_r(pos, exit_price)

            # Record both theoretical R and after-cost R.
            trade = {
                "Timestamp": ts,
                "Symbol": symbol,
                "Side": pos["side"],
                "SignalTimestamp": pos["signal_timestamp"],
                "EntryTimestamp": pos["entry_timestamp"],
                "EntryPrice": pos["entry_price"],
                "StopLoss": pos["stop_loss"],
                "TakeProfit": pos["take_profit"],
                "ExitPrice": exit_price,
                "ExitReason": exit_reason,
                "Outcome": outcome,
                "Gross_R": gross_r,
                "Net_R": net_r,
                "BarsHeld": i - pos["entry_index"],
                "HoldingHours": (i - pos["entry_index"]) * 1.0,
                "Score": pos["score"],
            }

            all_trades.append(trade)
            closed_symbols.add(symbol)

        # Remove positions only after recording exits.
        for symbol in closed_symbols:
            del active_positions[symbol]

        # ====================================================
        # 2) SAME-CANDLE RE-ENTRY LOCK
        # ====================================================
        # A symbol that just closed is forbidden from re-entering
        # on this exact candle, even if a signal is otherwise queued.
        blocked_until_next_candle = closed_symbols

        # ====================================================
        # 3) NEW ENTRIES
        # ====================================================
        candidates = signals_by_entry_ts.get(ts, [])

        if not candidates:
            continue

        # Sort all candidates before allocation.
        candidates = sorted(
            candidates,
            key=lambda x: (
                -x["score"],
                -float(processed[x["symbol"]]["df"].loc[
                    processed[x["symbol"]]["df"]["Date"] == x["entry_ts"],
                    "ADX"
                ].iloc[0]),
                x["symbol"],
            ),
        )

        used_groups = {
            CORRELATION_GROUP[symbol]
            for symbol in active_positions.keys()
        }

        for sig in candidates:
            symbol = sig["symbol"]

            if symbol in blocked_until_next_candle:
                continue

            if symbol in active_positions:
                continue

            if len(active_positions) >= MAX_OPEN_POSITIONS:
                break

            group = CORRELATION_GROUP[symbol]

            if group in used_groups:
                continue

            df = processed[symbol]["df"]
            entry_i = sig["entry_index"]

            if entry_i >= len(df):
                continue

            row = df.iloc[entry_i]

            # The signal was formed on entry_i - 1.
            # Entry uses ONLY this candle's OPEN.
            entry_raw = float(row["Open"])

            side = sig["side"]

            entry_price = apply_entry_slippage(side, entry_raw)

            signal_i = sig["signal_index"]
            signal_row = df.iloc[signal_i]

            atr = float(signal_row["ATR"])

            if side == "LONG":
                structure_low = sig["structure_low"]
                stop_loss = structure_low - SL_ATR_BUFFER * atr

                risk = entry_price - stop_loss

                if risk <= 0:
                    continue

                risk_pct = risk / entry_price

                if risk_pct > MAX_RISK_PCT_OF_ENTRY:
                    continue

                take_profit = entry_price + TP_R * risk

            else:
                structure_high = sig["structure_high"]
                stop_loss = structure_high + SL_ATR_BUFFER * atr

                risk = stop_loss - entry_price

                if risk <= 0:
                    continue

                risk_pct = risk / entry_price

                if risk_pct > MAX_RISK_PCT_OF_ENTRY:
                    continue

                take_profit = entry_price - TP_R * risk

            active_positions[symbol] = {
                "side": side,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "entry_index": entry_i,
                "entry_timestamp": ts,
                "signal_timestamp": df["Date"].iloc[signal_i],
                "score": sig["score"],
            }

            used_groups.add(group)

    # No artificial closing of open positions at the end.
    # This is intentional: NO TIMEOUT.
    return pd.DataFrame(all_trades), active_positions


# ============================================================
# REPORTING
# ============================================================

def loss_streaks(outcomes):
    streaks = []
    current = 0

    for outcome in outcomes:
        if outcome == "LOSS":
            current += 1
        else:
            if current > 0:
                streaks.append(current)
                current = 0

    if current > 0:
        streaks.append(current)

    return streaks


def max_streak(outcomes, target):
    best = 0
    cur = 0

    for x in outcomes:
        if x == target:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0

    return best


def max_drawdown_from_r(trades):
    if trades.empty:
        return 0.0

    equity = trades["Net_R"].cumsum()
    peak = equity.cummax()
    dd = equity - peak

    return float(dd.min())


def report(trades, processed):
    print("\n" + "=" * 72)
    print("📊 CLTS v1 — LBank Futures Causal Backtest")
    print("=" * 72)

    if trades.empty:
        print("⚠️ No completed trades.")
        print("این نتیجه به معنی صفر معامله کامل‌شده است؛ پوزیشن‌های باز انتهای دیتاست")
        print("عمداً بسته نشده‌اند چون Time-Out ممنوع است.")
        return

    trades = trades.sort_values(
        ["Timestamp", "Symbol"]
    ).reset_index(drop=True)

    total = len(trades)
    wins = int((trades["Outcome"] == "WIN").sum())
    losses = int((trades["Outcome"] == "LOSS").sum())

    win_rate = 100 * wins / total if total else 0.0

    gross_r = trades["Gross_R"].sum()
    net_r = trades["Net_R"].sum()

    gross_profit = trades.loc[trades["Net_R"] > 0, "Net_R"].sum()
    gross_loss = -trades.loc[trades["Net_R"] < 0, "Net_R"].sum()

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0 else np.inf
    )

    avg_win = (
        trades.loc[trades["Outcome"] == "WIN", "Net_R"].mean()
        if wins else 0.0
    )

    avg_loss = (
        trades.loc[trades["Outcome"] == "LOSS", "Net_R"].mean()
        if losses else 0.0
    )

    expectancy = trades["Net_R"].mean()

    avg_trades_day = total / DAYS

    max_dd = max_drawdown_from_r(trades)

    outcomes = trades["Outcome"].tolist()
    streaks = loss_streaks(outcomes)

    print(f"🔸 Total Trades:             {total}")
    print(f"🔸 Wins:                     {wins}")
    print(f"🔸 Losses:                   {losses}")
    print(f"🎯 Win Rate:                 {win_rate:.2f}%")
    print(f"📈 Gross R (before costs):   {gross_r:.4f}R")
    print(f"💰 Net R (after costs):      {net_r:.4f}R")
    print(f"📊 Profit Factor:            {profit_factor:.4f}")
    print(f"➕ Average Win:              {avg_win:.4f}R")
    print(f"➖ Average Loss:             {avg_loss:.4f}R")
    print(f"📐 Expectancy:              {expectancy:.4f}R/trade")
    print(f"📉 Max Drawdown:            {max_dd:.4f}R")
    print(f"🔥 Max Consecutive Wins:    {max_streak(outcomes, 'WIN')}")
    print(f"❄️ Max Consecutive Losses:  {max_streak(outcomes, 'LOSS')}")
    print(f"📅 Avg Trades / Day:        {avg_trades_day:.4f}")
    print(
        f"⏱️ Avg Holding Time:        "
        f"{trades['HoldingHours'].mean():.2f} hours"
    )
    print(
        f"🟢 Long Trades:             "
        f"{int((trades['Side'] == 'LONG').sum())}"
    )
    print(
        f"🔴 Short Trades:            "
        f"{int((trades['Side'] == 'SHORT').sum())}"
    )

    print("\n" + "-" * 72)
    print("📉 ALL CONSECUTIVE LOSS STREAKS")
    print("-" * 72)

    if streaks:
        print(", ".join(map(str, streaks)))
        print(f"تعداد زنجیره‌های ضرر: {len(streaks)}")
    else:
        print("هیچ زنجیره ضرری ثبت نشد.")

    print("\n" + "-" * 72)
    print("📈 SYMBOL-BY-SYMBOL")
    print("-" * 72)

    rows = []

    for sym in SYMBOLS.keys():
        x = trades[trades["Symbol"] == sym]

        if x.empty:
            rows.append({
                "Symbol": sym,
                "Trades": 0,
                "Wins": 0,
                "Losses": 0,
                "WinRate%": 0.0,
                "GrossR": 0.0,
                "NetR": 0.0,
                "PF": 0.0,
                "AvgWinR": 0.0,
                "AvgLossR": 0.0,
                "MaxLossStreak": 0,
                "AvgHoldH": 0.0,
            })
            continue

        w = int((x["Outcome"] == "WIN").sum())
        l = int((x["Outcome"] == "LOSS").sum())

        gp = x.loc[x["Net_R"] > 0, "Net_R"].sum()
        gl = -x.loc[x["Net_R"] < 0, "Net_R"].sum()

        pf = gp / gl if gl > 0 else np.inf

        rows.append({
            "Symbol": sym,
            "Trades": len(x),
            "Wins": w,
            "Losses": l,
            "WinRate%": round(100 * w / len(x), 2),
            "GrossR": round(x["Gross_R"].sum(), 3),
            "NetR": round(x["Net_R"].sum(), 3),
            "PF": round(pf, 3) if np.isfinite(pf) else np.inf,
            "AvgWinR": round(
                x.loc[x["Outcome"] == "WIN", "Net_R"].mean()
                if w else 0.0, 3
            ),
            "AvgLossR": round(
                x.loc[x["Outcome"] == "LOSS", "Net_R"].mean()
                if l else 0.0, 3
            ),
            "MaxLossStreak": max_streak(
                x["Outcome"].tolist(),
                "LOSS",
            ),
            "AvgHoldH": round(x["HoldingHours"].mean(), 2),
        })

    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "-" * 72)
    print("🧾 COMPLETED TRADE LEDGER")
    print("-" * 72)

    cols = [
        "Timestamp",
        "Symbol",
        "Side",
        "Outcome",
        "EntryPrice",
        "StopLoss",
        "TakeProfit",
        "ExitPrice",
        "ExitReason",
        "Gross_R",
        "Net_R",
        "BarsHeld",
    ]

    print(trades[cols].to_string(index=False))

    print("\n" + "-" * 72)
    print("🔐 AUDIT")
    print("-" * 72)

    print("✓ No timeout:", not TIMEOUT_ENABLED)
    print("✓ No max-bars exit:", not TIMEOUT_ENABLED)
    print("✓ Entry = next 1H candle open after closed signal candle")
    print("✓ Confirmed pivots become usable only at confirmation time")
    print("✓ No negative shift / centered rolling / backfill")
    print("✓ No same-symbol overlapping positions")
    print("✓ Max portfolio positions:", MAX_OPEN_POSITIONS)
    print("✓ One position per correlation group")
    print("✓ Same-candle re-entry after exit is blocked")
    print("✓ Same-candle SL+TP ambiguity -> SL first")
    print("✓ Open positions at dataset end are NOT forcibly closed")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("🚀 CLTS v1 — LBank USDT Futures")
    print("   4H Regime → 1H Sweep → BOS → Retest → Entry → SL/2R")
    print("=" * 72)

    print("\nLoading LBank markets...")
    exchange.load_markets()

    available = set(exchange.symbols)

    resolved = {}

    for name, symbol in SYMBOLS.items():
        if symbol in available:
            resolved[name] = symbol
        else:
            # Fallback diagnostic for CCXT/LBank symbol naming changes.
            matches = [
                s for s in exchange.symbols
                if s.startswith(f"{name}/USDT")
            ]
            print(f"⚠️ {name}: {symbol} not found. Candidates: {matches[:10]}")

    if not resolved:
        raise RuntimeError(
            "None of the requested LBank USDT perpetual contracts were found."
        )

    end_dt = utc_now()

    # Add warm-up before the actual one-year test.
    # This is NOT part of the reported test period.
    start_dt = end_dt - timedelta(days=DAYS)
    warmup_dt = start_dt - timedelta(days=100)

    since_ms = int(warmup_dt.timestamp() * 1000)
    until_ms = int(end_dt.timestamp() * 1000)

    print(
        f"\nTest period target: {start_dt.isoformat()} → {end_dt.isoformat()}"
    )
    print(
        "Warm-up period is used only to initialize indicators/structure."
    )

    processed = {}

    for name, symbol in resolved.items():
        try:
            item = prepare_symbol(
                name,
                symbol,
                since_ms,
                until_ms,
            )
            if item is not None:
                # Trim actual backtest rows to the requested one-year window.
                df = item["df"].copy()
                df = df[
                    df["Date"] >= pd.Timestamp(start_dt)
                ].reset_index(drop=True)

                # Rebuild signals after trim is NOT allowed because that could
                # accidentally alter state. Signals were already generated
                # chronologically from the warmup + test period.
                #
                # Instead, retain only signals whose signal/entry timestamps
                # belong to the final test window.
                original_df = item["df"]

                kept_signals = []
                for s in item["signals"]:
                    sig_ts = original_df["Date"].iloc[s["signal_index"]]
                    entry_ts = original_df["Date"].iloc[s["entry_index"]]

                    if (
                        pd.Timestamp(start_dt)
                        <= sig_ts
                        < pd.Timestamp(end_dt)
                        and
                        pd.Timestamp(start_dt)
                        <= entry_ts
                        < pd.Timestamp(end_dt)
                    ):
                        # Convert indices to trimmed dataframe indices.
                        entry_match = df.index[
                            df["Date"] == entry_ts
                        ]
                        sig_match = df.index[
                            df["Date"] == sig_ts
                        ]

                        if len(entry_match) and len(sig_match):
                            s2 = dict(s)

                            s2["entry_index"] = int(entry_match[0])
                            s2["signal_index"] = int(sig_match[0])

                            kept_signals.append(s2)

                item["df"] = df
                item["signals"] = kept_signals

                processed[name] = item

        except Exception as exc:
            print(f"❌ {name} failed: {type(exc).__name__}: {exc}")

    if not processed:
        raise RuntimeError("No symbol data was successfully prepared.")

    print("\n⚙️ Starting chronological backtest...")
    trades, open_positions = run_backtest(processed)

    report(trades, processed)

    if open_positions:
        print("\n" + "-" * 72)
        print("🔓 POSITIONS STILL OPEN AT END OF DATA")
        print("-" * 72)

        for symbol, pos in open_positions.items():
            print(
                symbol,
                pos["side"],
                "entry=", pos["entry_price"],
                "SL=", pos["stop_loss"],
                "TP=", pos["take_profit"],
                "entry_time=", pos["entry_timestamp"],
            )

        print(
            "\nاین پوزیشن‌ها عمداً در آمار معاملات بسته‌شده وارد نشده‌اند؛ "
            "بستن اجباری انتهای دیتاست خلاف قانون NO TIMEOUT است."
        )


if __name__ == "__main__":
    main()
