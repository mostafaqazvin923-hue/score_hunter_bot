import subprocess
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt


# ============================================================
# HUNTER-V137
# CAUSAL LIQUIDITY -> BOS -> RETEST ENGINE
#
# Purpose:
# Build a genuinely executable version after V136 showed that
# the old "same-candle confirmation / same-candle entry" logic
# does not survive causal execution.
#
# Core rules:
#   1) Only completed candles are used for decisions.
#   2) 4H regime uses only COMPLETED 4H candles.
#   3) Liquidity sweep must happen BEFORE displacement/BOS.
#   4) Retest happens AFTER BOS.
#   5) Entry is NEXT 15m OPEN after a completed retest candle.
#   6) Fixed nominal RR = 1:2.
#   7) No artificial trade timeout.
#   8) No overlapping position per symbol.
#   9) No new entry on the same candle where a position closes.
#  10) Conservative SL priority if SL and TP are both touched.
#
# This is a research backtest, not a promise of profitability.
# ============================================================


# ============================================================
# EXACT LBank DATA INFRASTRUCTURE USED BY V123/V129
# ============================================================

exchange = ccxt.lbank({
    "enableRateLimit": True,
    "timeout": 20000
})

SYMBOLS = {
    "CRV": "CRV/USDT",
    "DOGE": "DOGE/USDT",
    "ICP": "ICP/USDT",
    "APT": "APT/USDT",
    "PENDLE": "PENDLE/USDT",
    "WIF": "WIF/USDT",
    "ONDO": "ONDO/USDT",
    "NEAR": "NEAR/USDT",
    "SEI": "SEI/USDT",
    "XLM": "XLM/USDT",
    "ADA": "ADA/USDT",
    "BNB": "BNB/USDT",
    "SOL": "SOL/USDT",
    "ETH": "ETH/USDT",
}

TIMEFRAME_BASE = "15m"

SLIPPAGE = 0.0003
FEE_RATE = 0.0007

TRADE_MARGIN = 100.0
LEVERAGE = 50.0

DAYS = 365


# ============================================================
# STRATEGY PARAMETERS
# ============================================================

# 4H trend regime
EMA_FAST = 50
EMA_SLOW = 200

# 1H structure
STRUCTURE_LOOKBACK = 12
ATR_1H_PERIOD = 14

# 15m execution
ATR_15M_PERIOD = 14
BODY_AVG_PERIOD = 20
DISPLACEMENT_MULT = 1.50
MIN_CLOSE_LOCATION = 0.65
VOLUME_PERIOD = 20
MIN_RVOL = 1.05

# Sweep / BOS
SWEEP_LOOKBACK = 16
BOS_LOOKBACK = 8
BOS_BUFFER_ATR = 0.05

# Retest:
# The displacement candle defines the impulse zone.
# We require a later candle to trade back into the zone and
# close back in the direction of the impulse.
RETEST_MIN_FRACTION = 0.35
RETEST_MAX_FRACTION = 0.70

# Structural stop protection.
# The stop is below/above the sweep extreme with a volatility
# buffer. This prevents microscopic stops while keeping the
# stop structural rather than a fixed percentage.
SL_BUFFER_ATR = 0.20
MAX_STOP_ATR = 3.0
MIN_STOP_ATR = 0.70

# No setup timeout. A setup is invalidated only by structure.
# To avoid an old setup surviving through an opposite BOS, a new
# valid setup replaces the old one for that symbol.
# This is structural invalidation, NOT a time timeout.


# ============================================================
# DATA FETCH
# ============================================================

def fetch_chunk_data(lbank_symbol, start_dt, end_dt):
    since_ts = int((start_dt - timedelta(days=15)).timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)

    all_ohlcv = []
    current_since = since_ts

    try:
        while current_since < end_ts:
            batch = exchange.fetch_ohlcv(
                lbank_symbol,
                timeframe="15m",
                since=current_since,
                limit=1000,
            )

            if not batch:
                break

            all_ohlcv.extend(batch)

            last_ts = batch[-1][0]

            if last_ts <= current_since:
                break

            current_since = last_ts + 1

            if len(batch) < 1000:
                break

            if last_ts >= end_ts:
                break

            time.sleep(0.2)

    except Exception as e:
        print(f"  ERROR fetching {lbank_symbol}: {e}")
        return None

    if not all_ohlcv:
        return None

    df = pd.DataFrame(
        all_ohlcv,
        columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"],
    )

    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")

    df = df[
        ["Date", "Open", "High", "Low", "Close", "Volume"]
    ]

    df.dropna(inplace=True)
    df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.set_index("Date", inplace=True)

    df = df[
        (df.index >= start_dt)
        & (df.index <= end_dt)
    ]

    return df


# ============================================================
# INDICATORS
# ============================================================

def add_atr(df, period):
    high_low = df["High"] - df["Low"]
    prev_close = df["Close"].shift(1)

    tr = pd.concat(
        [
            high_low,
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(period).mean()


def prepare_data(df_15m):
    df = df_15m.copy()

    # --------------------------------------------------------
    # 4H
    # --------------------------------------------------------
    df_4h = (
        df.resample("4h")
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        })
        .dropna()
    )

    df_4h["EMA_50"] = df_4h["Close"].ewm(
        span=EMA_FAST,
        adjust=False,
    ).mean()

    df_4h["EMA_200"] = df_4h["Close"].ewm(
        span=EMA_SLOW,
        adjust=False,
    ).mean()

    df_4h["Bull"] = (
        (df_4h["Close"] > df_4h["EMA_200"])
        & (df_4h["EMA_50"] > df_4h["EMA_200"])
    )

    df_4h["Bear"] = (
        (df_4h["Close"] < df_4h["EMA_200"])
        & (df_4h["EMA_50"] < df_4h["EMA_200"])
    )

    # --------------------------------------------------------
    # 1H
    # --------------------------------------------------------
    df_1h = (
        df.resample("1h")
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        })
        .dropna()
    )

    df_1h["ATR"] = add_atr(df_1h, ATR_1H_PERIOD)

    # Confirmed swing levels are deliberately causal.
    # A swing at t is confirmed only after two bars to its right.
    df_1h["PivotHigh"] = (
        (df_1h["High"] > df_1h["High"].shift(1))
        & (df_1h["High"] > df_1h["High"].shift(2))
        & (df_1h["High"] >= df_1h["High"].shift(-1))
        & (df_1h["High"] >= df_1h["High"].shift(-2))
    )

    df_1h["PivotLow"] = (
        (df_1h["Low"] < df_1h["Low"].shift(1))
        & (df_1h["Low"] < df_1h["Low"].shift(2))
        & (df_1h["Low"] <= df_1h["Low"].shift(-1))
        & (df_1h["Low"] <= df_1h["Low"].shift(-2))
    )

    # These columns are only read from pivots that are already
    # confirmed by the time the 15m decision is made.
    # --------------------------------------------------------
    # 15M
    # --------------------------------------------------------
    df["ATR"] = add_atr(df, ATR_15M_PERIOD)

    df["Body"] = (df["Close"] - df["Open"]).abs()

    df["Avg_Body"] = (
        df["Body"]
        .rolling(BODY_AVG_PERIOD)
        .mean()
        .shift(1)
    )

    df["RVOL"] = (
        df["Volume"]
        / df["Volume"].rolling(VOLUME_PERIOD).mean().shift(1)
    )

    rng = (df["High"] - df["Low"]).replace(0, np.nan)

    df["CloseLocation"] = (
        (df["Close"] - df["Low"]) / rng
    )

    return df, df_1h, df_4h


# ============================================================
# COMPLETED HTF LOOKUP
# ============================================================

def completed_htf_row(df_htf, t_curr, timeframe_hours):
    """
    Return only an HTF candle whose entire interval ended before
    the 15m candle t_curr began.

    Example:
      10:15 decision cannot use the 10:00-11:00 1H candle.
    """
    if df_htf.empty:
        return None

    completed = df_htf[
        (df_htf.index + pd.Timedelta(hours=timeframe_hours))
        <= t_curr
    ]

    if completed.empty:
        return None

    return completed.iloc[-1]


# ============================================================
# CONFIRMED 1H LIQUIDITY LEVELS
# ============================================================

def confirmed_1h_levels(df_1h, t_curr):
    """
    Return the most recent confirmed 1H pivot high/low before t_curr.

    A pivot at timestamp p is usable only after p + 2 hours,
    because the two right-hand 1H candles are then complete.
    """
    if df_1h.empty:
        return None, None

    usable = df_1h[
        (df_1h.index + pd.Timedelta(hours=2))
        <= t_curr
    ]

    if usable.empty:
        return None, None

    highs = usable[usable["PivotHigh"]]
    lows = usable[usable["PivotLow"]]

    last_high = (
        float(highs["High"].iloc[-1])
        if not highs.empty
        else None
    )

    last_low = (
        float(lows["Low"].iloc[-1])
        if not lows.empty
        else None
    )

    return last_high, last_low


# ============================================================
# CAUSAL STATE MACHINE
# ============================================================

def run_v137_engine(
    symbol,
    df_15,
    df_1h,
    df_4h,
    start_dt,
    end_dt,
):
    trades = []

    position = None
    setup = None
    last_exit_index = -1

    # Start far enough into history for all indicators.
    for i in range(250, len(df_15)):
        t = df_15.index[i]

        if t < start_dt:
            continue

        if t > end_dt:
            break

        row = df_15.iloc[i]

        # ----------------------------------------------------
        # 1) Manage an existing position FIRST.
        # ----------------------------------------------------
        if position is not None:
            high = float(row["High"])
            low = float(row["Low"])

            side = position["Side"]
            entry = position["Entry_Price"]
            sl = position["SL"]
            tp = position["TP"]
            be_trigger = position["BE_Trigger"]

            if side == "LONG":
                if not position["BE_Active"] and high >= be_trigger:
                    position["SL"] = entry
                    position["BE_Active"] = True
                    sl = entry

                hit_sl = low <= sl
                hit_tp = high >= tp

                if hit_sl:
                    exit_price = entry if position["BE_Active"] else sl
                    outcome = "BE" if position["BE_Active"] else "LOSS"

                    exit_idx = i
                    trades.append({
                        "Timestamp": position["SignalTimestamp"],
                        "EntryTimestamp": position["EntryTimestamp"],
                        "ExitTimestamp": df_15.index[exit_idx],
                        "Symbol": symbol,
                        "Side": side,
                        "Outcome": outcome,
                        "Dollar_PnL": pnl_for_trade(
                            side, entry, exit_price
                        ),
                        "Entry_Price": entry,
                        "Exit_Price": exit_price,
                        "SL": position["Initial_SL"],
                        "TP": tp,
                        "Entry_Index": position["Entry_Index"],
                        "Exit_Index": exit_idx,
                    })

                    position = None
                    setup = None
                    last_exit_index = i

                    # IMPORTANT:
                    # No new signal is evaluated on this same candle.
                    continue

                if hit_tp:
                    exit_price = tp
                    trades.append({
                        "Timestamp": position["SignalTimestamp"],
                        "EntryTimestamp": position["EntryTimestamp"],
                        "ExitTimestamp": df_15.index[i],
                        "Symbol": symbol,
                        "Side": side,
                        "Outcome": "WIN",
                        "Dollar_PnL": pnl_for_trade(
                            side, entry, exit_price
                        ),
                        "Entry_Price": entry,
                        "Exit_Price": exit_price,
                        "SL": position["Initial_SL"],
                        "TP": tp,
                        "Entry_Index": position["Entry_Index"],
                        "Exit_Index": i,
                    })

                    position = None
                    setup = None
                    last_exit_index = i
                    continue

            else:
                if not position["BE_Active"] and low <= be_trigger:
                    position["SL"] = entry
                    position["BE_Active"] = True
                    sl = entry

                hit_sl = high >= sl
                hit_tp = low <= tp

                if hit_sl:
                    exit_price = entry if position["BE_Active"] else sl
                    outcome = "BE" if position["BE_Active"] else "LOSS"

                    trades.append({
                        "Timestamp": position["SignalTimestamp"],
                        "EntryTimestamp": position["EntryTimestamp"],
                        "ExitTimestamp": df_15.index[i],
                        "Symbol": symbol,
                        "Side": side,
                        "Outcome": outcome,
                        "Dollar_PnL": pnl_for_trade(
                            side, entry, exit_price
                        ),
                        "Entry_Price": entry,
                        "Exit_Price": exit_price,
                        "SL": position["Initial_SL"],
                        "TP": tp,
                        "Entry_Index": position["Entry_Index"],
                        "Exit_Index": i,
                    })

                    position = None
                    setup = None
                    last_exit_index = i
                    continue

                if hit_tp:
                    exit_price = tp
                    trades.append({
                        "Timestamp": position["SignalTimestamp"],
                        "EntryTimestamp": position["EntryTimestamp"],
                        "ExitTimestamp": df_15.index[i],
                        "Symbol": symbol,
                        "Side": side,
                        "Outcome": "WIN",
                        "Dollar_PnL": pnl_for_trade(
                            side, entry, exit_price
                        ),
                        "Entry_Price": entry,
                        "Exit_Price": exit_price,
                        "SL": position["Initial_SL"],
                        "TP": tp,
                        "Entry_Index": position["Entry_Index"],
                        "Exit_Index": i,
                    })

                    position = None
                    setup = None
                    last_exit_index = i
                    continue

        # No entry or setup processing on a candle that just closed
        # an existing position.
        if i == last_exit_index:
            continue

        # ----------------------------------------------------
        # 2) HTF regime — COMPLETED candles only.
        # ----------------------------------------------------
        h4 = completed_htf_row(df_4h, t, 4)

        if h4 is None:
            continue

        bull_regime = bool(h4["Bull"])
        bear_regime = bool(h4["Bear"])

        if not bull_regime and not bear_regime:
            # No directional regime. A previous setup is not
            # allowed to survive a neutral HTF regime.
            setup = None
            continue

        # ----------------------------------------------------
        # 3) Detect a NEW liquidity sweep on the completed
        #    candle i-1.
        #
        #    The sweep level is made only from candles BEFORE
        #    the sweep candle.
        # ----------------------------------------------------
        if i < max(SWEEP_LOOKBACK + 5, 30):
            continue

        sweep_idx = i - 1
        sweep = df_15.iloc[sweep_idx]

        prior = df_15.iloc[
            max(0, sweep_idx - SWEEP_LOOKBACK):sweep_idx
        ]

        if len(prior) < SWEEP_LOOKBACK:
            continue

        prior_low = float(prior["Low"].min())
        prior_high = float(prior["High"].max())

        sweep_low = (
            float(sweep["Low"]) < prior_low
            and float(sweep["Close"]) > prior_low
        )

        sweep_high = (
            float(sweep["High"]) > prior_high
            and float(sweep["Close"]) < prior_high
        )

        # ----------------------------------------------------
        # 4) Displacement/BOS on the CURRENT completed candle.
        #
        #    This is known only AFTER candle i closes.
        #    Therefore any entry based on it can only happen
        #    on a later candle.
        # ----------------------------------------------------
        atr = float(row["ATR"]) if np.isfinite(row["ATR"]) else np.nan

        if not np.isfinite(atr) or atr <= 0:
            continue

        body = float(row["Body"])
        avg_body = float(row["Avg_Body"]) if np.isfinite(row["Avg_Body"]) else np.nan
        rvol = float(row["RVOL"]) if np.isfinite(row["RVOL"]) else np.nan
        close_loc = float(row["CloseLocation"]) if np.isfinite(row["CloseLocation"]) else np.nan

        if not np.isfinite(avg_body) or avg_body <= 0:
            continue

        if not np.isfinite(rvol) or rvol < MIN_RVOL:
            continue

        displacement_up = (
            row["Close"] > row["Open"]
            and body >= DISPLACEMENT_MULT * avg_body
            and close_loc >= MIN_CLOSE_LOCATION
        )

        displacement_down = (
            row["Close"] < row["Open"]
            and body >= DISPLACEMENT_MULT * avg_body
            and close_loc <= (1.0 - MIN_CLOSE_LOCATION)
        )

        # BOS level comes from candles BEFORE the displacement candle.
        bos_prior = df_15.iloc[
            max(0, i - BOS_LOOKBACK):i
        ]

        if len(bos_prior) < BOS_LOOKBACK:
            continue

        bos_high = float(bos_prior["High"].max())
        bos_low = float(bos_prior["Low"].min())

        bos_up = float(row["Close"]) > bos_high + BOS_BUFFER_ATR * atr
        bos_down = float(row["Close"]) < bos_low - BOS_BUFFER_ATR * atr

        new_long_setup = (
            bull_regime
            and sweep_low
            and displacement_up
            and bos_up
        )

        new_short_setup = (
            bear_regime
            and sweep_high
            and displacement_down
            and bos_down
        )

        # ----------------------------------------------------
        # 5) Create a setup.
        #
        #    The displacement candle becomes the impulse candle.
        #    The zone is its 35%-70% retracement.
        #
        #    For LONG:
        #      zone is from 35% to 70% down from high.
        #
        #    For SHORT:
        #      zone is from 35% to 70% up from low.
        # ----------------------------------------------------
        if new_long_setup:
            impulse_high = float(row["High"])
            impulse_low = float(row["Low"])
            impulse_range = impulse_high - impulse_low

            if impulse_range > 0:
                zone_top = impulse_high - RETEST_MIN_FRACTION * impulse_range
                zone_bottom = impulse_high - RETEST_MAX_FRACTION * impulse_range

                sweep_extreme = float(sweep["Low"])
                structural_sl = (
                    sweep_extreme - SL_BUFFER_ATR * atr
                )

                setup = {
                    "Side": "LONG",
                    "Created_Index": i,
                    "SignalTimestamp": t,
                    "ZoneTop": zone_top,
                    "ZoneBottom": zone_bottom,
                    "SweepExtreme": sweep_extreme,
                    "SL": structural_sl,
                    "ATR": atr,
                    "ImpulseHigh": impulse_high,
                    "ImpulseLow": impulse_low,
                }

        elif new_short_setup:
            impulse_high = float(row["High"])
            impulse_low = float(row["Low"])
            impulse_range = impulse_high - impulse_low

            if impulse_range > 0:
                zone_bottom = impulse_low + RETEST_MIN_FRACTION * impulse_range
                zone_top = impulse_low + RETEST_MAX_FRACTION * impulse_range

                sweep_extreme = float(sweep["High"])
                structural_sl = (
                    sweep_extreme + SL_BUFFER_ATR * atr
                )

                setup = {
                    "Side": "SHORT",
                    "Created_Index": i,
                    "SignalTimestamp": t,
                    "ZoneTop": zone_top,
                    "ZoneBottom": zone_bottom,
                    "SweepExtreme": sweep_extreme,
                    "SL": structural_sl,
                    "ATR": atr,
                    "ImpulseHigh": impulse_high,
                    "ImpulseLow": impulse_low,
                }

        # ----------------------------------------------------
        # 6) Retest confirmation.
        #
        #    The setup cannot be entered on its creation candle.
        #    We require a LATER completed candle to:
        #      - trade into the impulse zone
        #      - close back in the intended direction
        #
        #    Entry is the NEXT 15m OPEN.
        # ----------------------------------------------------
        if setup is None:
            continue

        # A setup can be invalidated structurally, but NOT by age.
        if setup["Side"] == "LONG":
            if float(row["Low"]) <= setup["SL"]:
                setup = None
                continue

            touched_zone = (
                float(row["Low"]) <= setup["ZoneTop"]
                and float(row["High"]) >= setup["ZoneBottom"]
            )

            bullish_reclaim = (
                float(row["Close"]) > float(row["Open"])
                and float(row["Close"]) >= setup["ZoneTop"]
            )

            # Retest must occur AFTER creation candle.
            if (
                i > setup["Created_Index"]
                and touched_zone
                and bullish_reclaim
            ):
                entry_idx = i + 1

                if entry_idx >= len(df_15):
                    continue

                entry_row = df_15.iloc[entry_idx]
                entry_price = float(entry_row["Open"]) * (1.0 + SLIPPAGE)

                sl = float(setup["SL"])
                risk = entry_price - sl

                if risk <= 0:
                    setup = None
                    continue

                if risk < MIN_STOP_ATR * atr or risk > MAX_STOP_ATR * atr:
                    setup = None
                    continue

                tp = entry_price + 2.0 * risk
                be_trigger = entry_price + risk

                position = {
                    "Side": "LONG",
                    "SignalTimestamp": setup["SignalTimestamp"],
                    "EntryTimestamp": df_15.index[entry_idx],
                    "Entry_Index": entry_idx,
                    "Entry_Price": entry_price,
                    "Initial_SL": sl,
                    "SL": sl,
                    "TP": tp,
                    "BE_Trigger": be_trigger,
                    "BE_Active": False,
                }

                setup = None

                # The entry candle is handled normally on its own
                # future loop iteration; no same-candle exit is
                # evaluated here.
                continue

        else:
            if float(row["High"]) >= setup["SL"]:
                setup = None
                continue

            touched_zone = (
                float(row["High"]) >= setup["ZoneBottom"]
                and float(row["Low"]) <= setup["ZoneTop"]
            )

            bearish_reclaim = (
                float(row["Close"]) < float(row["Open"])
                and float(row["Close"]) <= setup["ZoneBottom"]
            )

            if (
                i > setup["Created_Index"]
                and touched_zone
                and bearish_reclaim
            ):
                entry_idx = i + 1

                if entry_idx >= len(df_15):
                    continue

                entry_row = df_15.iloc[entry_idx]
                entry_price = float(entry_row["Open"]) * (1.0 - SLIPPAGE)

                sl = float(setup["SL"])
                risk = sl - entry_price

                if risk <= 0:
                    setup = None
                    continue

                if risk < MIN_STOP_ATR * atr or risk > MAX_STOP_ATR * atr:
                    setup = None
                    continue

                tp = entry_price - 2.0 * risk
                be_trigger = entry_price - risk

                position = {
                    "Side": "SHORT",
                    "SignalTimestamp": setup["SignalTimestamp"],
                    "EntryTimestamp": df_15.index[entry_idx],
                    "Entry_Index": entry_idx,
                    "Entry_Price": entry_price,
                    "Initial_SL": sl,
                    "SL": sl,
                    "TP": tp,
                    "BE_Trigger": be_trigger,
                    "BE_Active": False,
                }

                setup = None
                continue

    # --------------------------------------------------------
    # No artificial timeout:
    # If a position remains open at dataset end, record it as
    # OPEN and exclude it from realized PnL.
    # --------------------------------------------------------
    if position is not None:
        trades.append({
            "Timestamp": position["SignalTimestamp"],
            "EntryTimestamp": position["EntryTimestamp"],
            "ExitTimestamp": None,
            "Symbol": symbol,
            "Side": position["Side"],
            "Outcome": "OPEN",
            "Dollar_PnL": 0.0,
            "Entry_Price": position["Entry_Price"],
            "Exit_Price": None,
            "SL": position["Initial_SL"],
            "TP": position["TP"],
            "Entry_Index": position["Entry_Index"],
            "Exit_Index": None,
        })

    return trades


# ============================================================
# PNL
# ============================================================

def pnl_for_trade(side, entry_price, exit_price):
    notional = TRADE_MARGIN * LEVERAGE

    if side == "LONG":
        price_ret = (
            exit_price - entry_price
        ) / entry_price
    else:
        price_ret = (
            entry_price - exit_price
        ) / entry_price

    return (
        notional * price_ret
        - notional * FEE_RATE * 2.0
    )


# ============================================================
# REPORTING
# ============================================================

def max_loss_streak(outcomes):
    streaks = []
    current = 0

    for outcome in outcomes:
        if outcome == "LOSS":
            current += 1
        else:
            if current:
                streaks.append(current)
            current = 0

    if current:
        streaks.append(current)

    return max(streaks) if streaks else 0, streaks


def print_report(trades_df):
    if trades_df.empty:
        print("\nNO TRADES GENERATED")
        return

    trades_df.sort_values(
        ["EntryTimestamp", "Symbol"],
        inplace=True,
    )
    trades_df.reset_index(drop=True, inplace=True)

    realized = trades_df[
        trades_df["Outcome"].isin(["WIN", "LOSS", "BE"])
    ].copy()

    wins = realized[realized["Outcome"] == "WIN"]
    losses = realized[realized["Outcome"] == "LOSS"]
    bes = realized[realized["Outcome"] == "BE"]
    opens = trades_df[trades_df["Outcome"] == "OPEN"]

    total = len(trades_df)
    realized_count = len(realized)

    wr = len(wins) / total * 100.0 if total else 0.0
    lr = len(losses) / total * 100.0 if total else 0.0
    ber = len(bes) / total * 100.0 if total else 0.0

    net_pnl = float(realized["Dollar_PnL"].sum())

    gross_profit = float(wins["Dollar_PnL"].sum())
    gross_loss = abs(float(losses["Dollar_PnL"].sum()))

    pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else float("inf")
    )

    avg_win = (
        float(wins["Dollar_PnL"].mean())
        if not wins.empty else 0.0
    )

    avg_loss = (
        float(losses["Dollar_PnL"].mean())
        if not losses.empty else 0.0
    )

    if not realized.empty:
        equity = realized["Dollar_PnL"].cumsum()
        peak = equity.cummax()
        dd = equity - peak
        max_dd = float(dd.min())
    else:
        max_dd = 0.0

    max_streak, streaks = max_loss_streak(
        realized["Outcome"].tolist()
    )

    print()
    print("=" * 80)
    print("===== HUNTER-V137 — CAUSAL LIQUIDITY / BOS / RETEST =====")
    print("=" * 80)
    print(f"Total Trades:          {total}")
    print(f"Realized Trades:       {realized_count}")
    print(f"Open at Dataset End:   {len(opens)}")
    print(f"Trades / Month:        {total / 12.0:.1f}")
    print(f"Win Rate:              {wr:.2f}%")
    print(f"Loss Rate:             {lr:.2f}%")
    print(f"Break Even Rate:       {ber:.2f}%")
    print(f"Net PnL:               ${net_pnl:,.2f}")
    print(f"Profit Factor:         {pf:.2f}")
    print(f"Average Win:           ${avg_win:,.2f}")
    print(f"Average Loss:          ${avg_loss:,.2f}")
    print(f"Max Drawdown:          ${max_dd:,.2f}")
    print(f"Maximum Consecutive Losses: {max_streak}")
    print(f"Loss Streaks:          {streaks}")
    print("-" * 80)

    print("INTEGRITY:")
    print("  Entry:                NEXT 15m OPEN after confirmed retest")
    print("  HTF:                  COMPLETED 1H/4H candles only")
    print("  Sweep:                BEFORE BOS")
    print("  BOS:                  CLOSED 15m candle")
    print("  Retest:               LATER CLOSED 15m candle")
    print("  RR:                   1:2 FIXED")
    print("  Timeout:              DISABLED")
    print("  Lookahead:            NONE BY DESIGN")
    print("  Overlap per symbol:  LOCKED")
    print("  Same-candle reentry: BLOCKED")
    print("-" * 80)

    print("BY SYMBOL:")
    for symbol in SYMBOLS:
        sub = trades_df[trades_df["Symbol"] == symbol]
        if sub.empty:
            continue

        wr_s = (
            sub["Outcome"].eq("WIN").mean() * 100.0
        )
        pnl_s = float(sub["Dollar_PnL"].sum())

        print(
            f"  {symbol:7} -> "
            f"Trades: {len(sub):4}, "
            f"Win Rate: {wr_s:6.2f}%, "
            f"PnL: ${pnl_s:10,.2f}"
        )

    print("-" * 80)
    print("BY DIRECTION:")

    for side in ("LONG", "SHORT"):
        sub = trades_df[trades_df["Side"] == side]
        if sub.empty:
            continue

        wr_s = sub["Outcome"].eq("WIN").mean() * 100.0
        pnl_s = float(sub["Dollar_PnL"].sum())

        print(
            f"  {side:7} -> "
            f"Trades: {len(sub):4}, "
            f"Win Rate: {wr_s:6.2f}%, "
            f"PnL: ${pnl_s:10,.2f}"
        )

    print("-" * 80)
    print("BY MONTH:")

    trades_df["Month"] = pd.to_datetime(
        trades_df["EntryTimestamp"]
    ).dt.strftime("%Y-%m")

    for month, sub in trades_df.groupby("Month"):
        wr_s = sub["Outcome"].eq("WIN").mean() * 100.0
        pnl_s = float(sub["Dollar_PnL"].sum())

        print(
            f"  {month} -> "
            f"Trades: {len(sub):4}, "
            f"Win Rate: {wr_s:6.2f}%, "
            f"PnL: ${pnl_s:10,.2f}"
        )

    print("=" * 80)

    try:
        trades_df.to_csv(
            "v137_causal_trades.csv",
            index=False,
        )
        print("Trade log saved: v137_causal_trades.csv")
    except Exception as e:
        print(f"Could not save trade log: {e}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("HUNTER-V137 — CAUSAL LIQUIDITY / BOS / RETEST ENGINE")
    print("=" * 80)

    now = datetime.now()
    start_dt = now - timedelta(days=DAYS)
    end_dt = now

    print(f"Period: {start_dt} -> {end_dt}")
    print()
    print("Data connection: EXACT V123/V129 LBank infrastructure")
    print("Universe:        EXACT V123/V129 14-symbol basket")
    print("Base timeframe:  15m")
    print("Strategy:        Liquidity Sweep -> BOS -> Retest")
    print("Entry:           Next 15m open")
    print("RR:              1:2 fixed")
    print("Timeout:         Disabled")
    print("=" * 80)

    all_trades = []

    for symbol, lbank_symbol in SYMBOLS.items():
        print()
        print("-" * 80)
        print(f"{symbol} -> {lbank_symbol}")
        print("-" * 80)

        df = fetch_chunk_data(
            lbank_symbol,
            start_dt,
            end_dt,
        )

        if df is None:
            print("  NO DATA")
            continue

        print(f"  Candles: {len(df)}")

        if len(df) < 1000:
            print("  INSUFFICIENT DATA")
            continue

        df_15, df_1h, df_4h = prepare_data(df)

        symbol_trades = run_v137_engine(
            symbol,
            df_15,
            df_1h,
            df_4h,
            start_dt,
            end_dt,
        )

        all_trades.extend(symbol_trades)

        print(f"  Trades: {len(symbol_trades)}")

    if not all_trades:
        print()
        print("=" * 80)
        print("NO TRADES GENERATED")
        print("=" * 80)
        return

    trades_df = pd.DataFrame(all_trades)

    print_report(trades_df)


if __name__ == "__main__":
    main()
