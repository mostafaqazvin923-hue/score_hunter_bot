import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import numpy as np
import pandas as pd

# ============================================================
# HUNTER-V131-A
# Portfolio-Level / Strict No-Lookahead / Pure 1:2
#
# Architecture:
#   4H  = regime
#   1H  = structure + liquidity reference
#   15M = sweep + displacement + FVG/value confirmation
#
# Rules:
#   - Signal is confirmed only on a CLOSED 15M candle.
#   - 1H/4H context uses only COMPLETED higher-timeframe candles.
#   - Entry is at the NEXT 15M candle open.
#   - Fixed RR = 1:2.
#   - No BE, no trailing stop, no timeout.
#   - Portfolio-level max 3 open positions.
#   - Max 1 open position per correlation cluster.
#   - Portfolio-level loss-streak pause.
#   - Existing positions are always managed during pause.
#   - If SL and TP occur on the same candle, SL wins (conservative).
#   - Final open positions are reported separately and not counted as
#     completed trades.
#
# NOTE:
# This is a research/backtest engine, NOT an order-execution bot.
# ============================================================

EXCHANGE = ccxt.lbank({
    "enableRateLimit": True,
    "timeout": 20000,
})

SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "SUI": "SUI/USDT",
    "AVAX": "AVAX/USDT",
    "NEAR": "NEAR/USDT",
    "ADA": "ADA/USDT",
    "BNB": "BNB/USDT",
    "APT": "APT/USDT",
    "CRV": "CRV/USDT",
    "ONDO": "ONDO/USDT",
    "PENDLE": "PENDLE/USDT",
    "ICP": "ICP/USDT",
    "WIF": "WIF/USDT",
}

CORRELATION_CLUSTERS = {
    "BTC": "MAJOR",
    "ETH": "MAJOR",
    "SOL": "L1",
    "SUI": "L1",
    "AVAX": "L1",
    "NEAR": "L1",
    "ADA": "L1",
    "BNB": "L1",
    "APT": "L1",
    "CRV": "DEFI",
    "ONDO": "DEFI",
    "PENDLE": "DEFI",
    "ICP": "OTHER",
    "WIF": "MEME",
}

# -----------------------------
# Backtest configuration
# -----------------------------
DAYS = 365
TIMEFRAME = "15m"

INITIAL_CAPITAL = 1000.0
TRADE_MARGIN = 100.0
LEVERAGE = 50.0

SLIPPAGE = 0.0003
FEE_RATE = 0.0007

MAX_OPEN_POSITIONS = 3
MAX_ONE_PER_CLUSTER = True

MAX_LOSS_STREAK = 4
LOSS_PAUSE_BARS = 16  # 16 x 15m = 4 hours

ATR_PERIOD = 14
BODY_AVG_PERIOD = 20

# Structure / liquidity
STRUCTURE_LOOKBACK_1H = 15
SWEEP_BUFFER_ATR = 0.15

# Displacement
DISPLACEMENT_ATR = 0.80
DISPLACEMENT_BODY_RATIO = 0.60

# FVG
USE_FVG_CONFIRMATION = True
FVG_MIN_ATR = 0.05

# Confirmation score:
# Core conditions are mandatory:
#   regime + 1H structure + liquidity sweep
#
# Additional confirmations contribute score.
MIN_CONFIRMATION_SCORE = 2

# A hard cap prevents absurdly wide structural stops.
MAX_STOP_ATR = 2.5

# Higher-timeframe alignment:
# 4H bar timestamp represents its opening time after resample.
# To use ONLY completed 4H bars at a 15M timestamp t,
# we require the 4H bar to end <= t.
# Same for 1H.
FOUR_HOURS = pd.Timedelta(hours=4)
ONE_HOUR = pd.Timedelta(hours=1)
FIFTEEN_MIN = pd.Timedelta(minutes=15)


# ============================================================
# DATA
# ============================================================

def fetch_symbol_data(lbank_symbol, start_dt, end_dt):
    """
    Fetch 15M OHLCV with warmup.
    Warmup is deliberately kept before start_dt so indicators have
    historical context, but trades are only allowed inside [start_dt, end_dt].
    """
    warmup_start = start_dt - timedelta(days=30)
    since_ts = int(warmup_start.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)

    all_ohlcv = []
    current_since = since_ts
    last_seen = None

    try:
        while current_since < end_ts:
            batch = EXCHANGE.fetch_ohlcv(
                lbank_symbol,
                timeframe=TIMEFRAME,
                since=current_since,
                limit=1000,
            )

            if not batch:
                break

            if last_seen is not None and batch[-1][0] <= last_seen:
                break

            all_ohlcv.extend(batch)
            last_seen = batch[-1][0]
            current_since = last_seen + 1

            if len(batch) < 1000:
                break

            time.sleep(0.15)

    except Exception as e:
        print(f"  خطا در دریافت {lbank_symbol}: {e}")
        return None

    if not all_ohlcv:
        return None

    df = pd.DataFrame(
        all_ohlcv,
        columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"],
    )

    # UTC makes cross-symbol alignment deterministic.
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)
    df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)

    # Remove future/incomplete candle conservatively:
    # a 15M candle whose close time is after now is not usable.
    now_utc = pd.Timestamp.now(tz="UTC")
    df = df[df.index + FIFTEEN_MIN <= now_utc]

    if len(df) < 500:
        return None

    return df


def resample_closed(df_15m, rule):
    """
    Build higher timeframe bars with timestamp at the BAR OPEN.
    The caller must only use bars whose END <= current 15M candle close.
    """
    out = df_15m.resample(
        rule,
        label="left",
        closed="left",
        origin="epoch",
    ).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()

    return out


def prepare_data(df_15m):
    df_15 = df_15m.copy()

    # -----------------------------
    # 4H
    # -----------------------------
    df_4h = resample_closed(df_15, "4h")

    df_4h["EMA_50"] = df_4h["Close"].ewm(
        span=50, adjust=False
    ).mean()

    df_4h["EMA_200"] = df_4h["Close"].ewm(
        span=200, adjust=False
    ).mean()

    df_4h["EMA_Slope"] = (
        df_4h["EMA_200"] - df_4h["EMA_200"].shift(5)
    )

    df_4h["TR"] = pd.concat([
        df_4h["High"] - df_4h["Low"],
        (df_4h["High"] - df_4h["Close"].shift(1)).abs(),
        (df_4h["Low"] - df_4h["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)

    df_4h["ATR"] = df_4h["TR"].rolling(ATR_PERIOD).mean()
    df_4h["ATR_PCT"] = df_4h["ATR"] / df_4h["Close"]

    # ADX, calculated causally on 4H.
    up_move = df_4h["High"].diff()
    down_move = -df_4h["Low"].diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0,
        ),
        index=df_4h.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0,
        ),
        index=df_4h.index,
    )

    atr14 = df_4h["TR"].rolling(ATR_PERIOD).mean()

    plus_di = 100.0 * plus_dm.rolling(ATR_PERIOD).mean() / atr14
    minus_di = 100.0 * minus_dm.rolling(ATR_PERIOD).mean() / atr14

    dx = (
        100.0
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    df_4h["ADX"] = dx.rolling(ATR_PERIOD).mean()
    df_4h["PLUS_DI"] = plus_di
    df_4h["MINUS_DI"] = minus_di

    df_4h["Regime_Bullish"] = (
        (df_4h["Close"] > df_4h["EMA_200"])
        & (df_4h["EMA_50"] > df_4h["EMA_200"])
        & (df_4h["EMA_Slope"] > 0)
        & (df_4h["PLUS_DI"] > df_4h["MINUS_DI"])
    )

    df_4h["Regime_Bearish"] = (
        (df_4h["Close"] < df_4h["EMA_200"])
        & (df_4h["EMA_50"] < df_4h["EMA_200"])
        & (df_4h["EMA_Slope"] < 0)
        & (df_4h["MINUS_DI"] > df_4h["PLUS_DI"])
    )

    # -----------------------------
    # 1H
    # -----------------------------
    df_1h = resample_closed(df_15, "1h")

    # Confirmed local pivots.
    # A pivot at t requires two bars on both sides.
    # We then shift the pivot series so the engine only sees confirmed pivots.
    left = 2
    right = 2

    roll_high = df_1h["High"].rolling(
        window=left + right + 1,
        center=True,
    ).max()

    roll_low = df_1h["Low"].rolling(
        window=left + right + 1,
        center=True,
    ).min()

    pivot_high_raw = df_1h["High"].where(df_1h["High"] == roll_high)
    pivot_low_raw = df_1h["Low"].where(df_1h["Low"] == roll_low)

    # Confirmation becomes known right bars later.
    df_1h["Confirmed_Pivot_High"] = pivot_high_raw.shift(right)
    df_1h["Confirmed_Pivot_Low"] = pivot_low_raw.shift(right)

    # Causal forward-fill of already confirmed pivots.
    df_1h["Last_Swing_High"] = (
        df_1h["Confirmed_Pivot_High"].ffill()
    )
    df_1h["Last_Swing_Low"] = (
        df_1h["Confirmed_Pivot_Low"].ffill()
    )

    # Previous structure levels, always from prior completed 1H bars.
    df_1h["Prior_15H_High"] = (
        df_1h["High"].shift(1).rolling(15).max()
    )
    df_1h["Prior_15H_Low"] = (
        df_1h["Low"].shift(1).rolling(15).min()
    )

    # -----------------------------
    # 15M
    # -----------------------------
    tr15 = pd.concat([
        df_15["High"] - df_15["Low"],
        (df_15["High"] - df_15["Close"].shift(1)).abs(),
        (df_15["Low"] - df_15["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)

    df_15["TR"] = tr15
    df_15["ATR"] = tr15.rolling(ATR_PERIOD).mean()
    df_15["Body"] = (df_15["Close"] - df_15["Open"]).abs()
    df_15["Range"] = df_15["High"] - df_15["Low"]
    df_15["Avg_Body"] = df_15["Body"].rolling(BODY_AVG_PERIOD).mean()
    df_15["Volume_MA"] = df_15["Volume"].rolling(20).mean()

    # Simple rolling VWAP using only past/current completed 15M candles.
    pv = df_15["Close"] * df_15["Volume"]
    df_15["VWAP_48"] = (
        pv.rolling(48).sum()
        / df_15["Volume"].rolling(48).sum().replace(0, np.nan)
    )

    return df_15, df_1h, df_4h


# ============================================================
# HELPERS
# ============================================================

def completed_htf_row(df_htf, current_15m_close, timeframe_delta):
    """
    Return the latest HIGHER-TF bar that is fully completed by
    current_15m candle close.
    """
    eligible = df_htf.index + timeframe_delta <= current_15m_close
    rows = df_htf.loc[eligible]
    if rows.empty:
        return None
    return rows.iloc[-1]


def find_1h_context(df_1h, current_15m_close):
    eligible = df_1h.index + ONE_HOUR <= current_15m_close
    rows = df_1h.loc[eligible]
    if rows.empty:
        return None
    return rows


def get_regime(df_4h, current_15m_close):
    row = completed_htf_row(
        df_4h,
        current_15m_close,
        FOUR_HOURS,
    )

    if row is None:
        return None

    if not np.isfinite(row.get("EMA_200", np.nan)):
        return None

    if bool(row["Regime_Bullish"]):
        return "LONG"

    if bool(row["Regime_Bearish"]):
        return "SHORT"

    return None


def get_structure_context(h_rows):
    if len(h_rows) < STRUCTURE_LOOKBACK_1H:
        return None

    recent = h_rows.iloc[-STRUCTURE_LOOKBACK_1H:]

    prior_high = recent["High"].shift(1).max()
    prior_low = recent["Low"].shift(1).min()

    last_close = recent.iloc[-1]["Close"]

    if not np.isfinite(prior_high) or not np.isfinite(prior_low):
        return None

    return {
        "prior_high": float(prior_high),
        "prior_low": float(prior_low),
        "last_close": float(last_close),
        "last_swing_high": float(recent["Last_Swing_High"].iloc[-1])
        if np.isfinite(recent["Last_Swing_High"].iloc[-1])
        else np.nan,
        "last_swing_low": float(recent["Last_Swing_Low"].iloc[-1])
        if np.isfinite(recent["Last_Swing_Low"].iloc[-1])
        else np.nan,
    }


def calculate_fvg(df_15, i, side):
    """
    Detect a 3-candle FVG ending on the current CLOSED candle i.
    Long bullish FVG:
        candle i-2 high < candle i low
    Short bearish FVG:
        candle i-2 low > candle i high
    """
    if i < 2:
        return None

    a = df_15.iloc[i - 2]
    c = df_15.iloc[i]

    atr = c["ATR"]
    if not np.isfinite(atr) or atr <= 0:
        return None

    if side == "LONG":
        gap_low = float(a["High"])
        gap_high = float(c["Low"])

        if gap_high > gap_low and (gap_high - gap_low) >= FVG_MIN_ATR * atr:
            return (gap_low, gap_high)

    else:
        gap_low = float(c["High"])
        gap_high = float(a["Low"])

        if gap_high > gap_low and (gap_high - gap_low) >= FVG_MIN_ATR * atr:
            return (gap_low, gap_high)

    return None


def signal_on_closed_candle(df_15, df_1h, df_4h, i):
    """
    Generate a signal ONLY from information known at the CLOSE of
    15M candle i. Entry occurs on candle i+1 open.
    """
    if i < max(60, ATR_PERIOD + BODY_AVG_PERIOD + 5):
        return None

    row = df_15.iloc[i]
    prev = df_15.iloc[i - 1]

    current_close_time = df_15.index[i] + FIFTEEN_MIN

    regime = get_regime(df_4h, current_close_time)
    if regime is None:
        return None

    h_rows = find_1h_context(df_1h, current_close_time)
    if h_rows is None:
        return None

    structure = get_structure_context(h_rows)
    if structure is None:
        return None

    atr = row["ATR"]
    candle_range = row["Range"]

    if not np.isfinite(atr) or atr <= 0:
        return None

    if not np.isfinite(candle_range) or candle_range <= 0:
        return None

    # Liquidity reference uses PRIOR completed 1H candles.
    support = structure["prior_low"]
    resistance = structure["prior_high"]

    sweep_buffer = SWEEP_BUFFER_ATR * atr

    sweep_low = (
        prev["Low"] < (support - sweep_buffer)
        and prev["Close"] > support
    )

    sweep_high = (
        prev["High"] > (resistance + sweep_buffer)
        and prev["Close"] < resistance
    )

    # Displacement on current CLOSED 15M candle.
    body = row["Body"]
    body_ratio = body / candle_range

    displacement_up = (
        regime == "LONG"
        and sweep_low
        and row["Close"] > row["Open"]
        and body >= DISPLACEMENT_ATR * atr
        and body_ratio >= DISPLACEMENT_BODY_RATIO
    )

    displacement_down = (
        regime == "SHORT"
        and sweep_high
        and row["Close"] < row["Open"]
        and body >= DISPLACEMENT_ATR * atr
        and body_ratio >= DISPLACEMENT_BODY_RATIO
    )

    if not displacement_up and not displacement_down:
        return None

    side = "LONG" if displacement_up else "SHORT"

    score = 0
    confirmations = []

    # 1) Displacement
    score += 1
    confirmations.append("DISPLACEMENT")

    # 2) FVG
    fvg = calculate_fvg(df_15, i, side)
    if USE_FVG_CONFIRMATION and fvg is not None:
        score += 1
        confirmations.append("FVG")

    # 3) VWAP alignment
    vwap = row["VWAP_48"]
    if np.isfinite(vwap):
        if side == "LONG" and row["Close"] > vwap:
            score += 1
            confirmations.append("VWAP")
        elif side == "SHORT" and row["Close"] < vwap:
            score += 1
            confirmations.append("VWAP")

    # 4) Volume expansion
    vol_ma = row["Volume_MA"]
    if np.isfinite(vol_ma) and vol_ma > 0:
        if row["Volume"] > 1.10 * vol_ma:
            score += 1
            confirmations.append("VOLUME")

    if score < MIN_CONFIRMATION_SCORE:
        return None

    # Structural SL:
    # Long below the liquidity sweep low.
    # Short above the liquidity sweep high.
    if side == "LONG":
        structural_extreme = min(float(prev["Low"]), float(row["Low"]))
        sl_reference = structural_extreme - 0.15 * atr
    else:
        structural_extreme = max(float(prev["High"]), float(row["High"]))
        sl_reference = structural_extreme + 0.15 * atr

    # Entry will be next candle OPEN, so risk is calculated from
    # that actual future execution price. This is NOT used to
    # decide whether the signal exists.
    return {
        "side": side,
        "score": score,
        "confirmations": ",".join(confirmations),
        "sl_reference": float(sl_reference),
        "atr_signal": float(atr),
        "signal_time": df_15.index[i],
        "signal_close_time": current_close_time,
    }


# ============================================================
# PORTFOLIO ENGINE
# ============================================================

def run_portfolio_backtest(processed_data, start_dt, end_dt):
    """
    Event-driven portfolio engine.

    Every 15M timestamp:
      1) Manage all existing positions.
      2) Apply completed outcomes to portfolio loss streak.
      3) If not paused and capacity exists, scan signals.
      4) New positions enter only at the next candle OPEN.

    Because signal execution is next-bar open, candidate signals
    are collected on the current bar and opened on the next bar.
    """
    all_times = sorted({
        ts for df, _, _ in processed_data.values()
        for ts in df.index
        if start_dt <= ts <= end_dt
    })

    if not all_times:
        return [], []

    active = {}
    trades = []
    pending_entries = []

    equity = INITIAL_CAPITAL
    peak_equity = INITIAL_CAPITAL
    max_dd = 0.0

    portfolio_loss_streak = 0
    pause_counter = 0

    def cluster_opened(cluster):
        return any(
            p["cluster"] == cluster
            for p in active.values()
        )

    def record_trade(pos, outcome, exit_price, exit_time):
        nonlocal equity, peak_equity, max_dd, portfolio_loss_streak, pause_counter

        notional = TRADE_MARGIN * LEVERAGE

        if pos["side"] == "LONG":
            gross = notional * (
                (exit_price - pos["entry_price"])
                / pos["entry_price"]
            )
        else:
            gross = notional * (
                (pos["entry_price"] - exit_price)
                / pos["entry_price"]
            )

        fees = notional * FEE_RATE * 2.0
        net = gross - fees

        equity += net
        peak_equity = max(peak_equity, equity)
        dd = equity - peak_equity
        max_dd = min(max_dd, dd)

        trades.append({
            "Timestamp": pos["entry_time"],
            "ExitTimestamp": exit_time,
            "Symbol": pos["symbol"],
            "Cluster": pos["cluster"],
            "Side": pos["side"],
            "Outcome": outcome,
            "Dollar_PnL": net,
            "Entry_Price": pos["entry_price"],
            "Exit_Price": exit_price,
            "SL": pos["sl"],
            "TP": pos["tp"],
            "SignalTime": pos["signal_time"],
            "Score": pos["score"],
            "Confirmations": pos["confirmations"],
            "Equity_After": equity,
        })

        if outcome == "LOSS":
            portfolio_loss_streak += 1

            if portfolio_loss_streak >= MAX_LOSS_STREAK:
                pause_counter = LOSS_PAUSE_BARS

        elif outcome == "WIN":
            portfolio_loss_streak = 0

        return net

    for k, ts in enumerate(all_times):
        # --------------------------------------------------------
        # 1) Manage existing positions FIRST.
        # --------------------------------------------------------
        for symbol, pos in list(active.items()):
            df_15, _, _ = processed_data[symbol]

            if ts not in df_15.index:
                continue

            candle = df_15.loc[ts]

            if pos["side"] == "LONG":
                hit_sl = candle["Low"] <= pos["sl"]
                hit_tp = candle["High"] >= pos["tp"]

                # Conservative intrabar rule:
                # if both are hit in one 15M candle, SL wins.
                if hit_sl and hit_tp:
                    record_trade(
                        pos, "LOSS", pos["sl"], ts
                    )
                    del active[symbol]

                elif hit_sl:
                    record_trade(
                        pos, "LOSS", pos["sl"], ts
                    )
                    del active[symbol]

                elif hit_tp:
                    record_trade(
                        pos, "WIN", pos["tp"], ts
                    )
                    del active[symbol]

            else:
                hit_sl = candle["High"] >= pos["sl"]
                hit_tp = candle["Low"] <= pos["tp"]

                if hit_sl and hit_tp:
                    record_trade(
                        pos, "LOSS", pos["sl"], ts
                    )
                    del active[symbol]

                elif hit_sl:
                    record_trade(
                        pos, "LOSS", pos["sl"], ts
                    )
                    del active[symbol]

                elif hit_tp:
                    record_trade(
                        pos, "WIN", pos["tp"], ts
                    )
                    del active[symbol]

        # --------------------------------------------------------
        # 2) Apply pause countdown AFTER position management.
        # --------------------------------------------------------
        if pause_counter > 0:
            pause_counter -= 1
            if pause_counter == 0:
                portfolio_loss_streak = 0

        # --------------------------------------------------------
        # 3) Execute pending entries at THIS candle's OPEN.
        #    Signals were generated on the previous CLOSED candle.
        # --------------------------------------------------------
        if pending_entries:
            # Deterministic order: highest score first, then symbol.
            pending_entries.sort(
                key=lambda x: (-x["score"], x["symbol"])
            )

            for sig in pending_entries:
                if sig["symbol"] in active:
                    continue

                if len(active) >= MAX_OPEN_POSITIONS:
                    break

                cluster = sig["cluster"]

                if (
                    MAX_ONE_PER_CLUSTER
                    and cluster_opened(cluster)
                ):
                    continue

                df_15, _, _ = processed_data[sig["symbol"]]

                if ts not in df_15.index:
                    continue

                candle = df_15.loc[ts]
                raw_open = float(candle["Open"])

                if sig["side"] == "LONG":
                    entry = raw_open * (1.0 + SLIPPAGE)
                    sl = sig["sl_reference"]

                    risk = entry - sl

                    if risk <= 0:
                        continue

                    if risk > sig["atr_signal"] * MAX_STOP_ATR:
                        continue

                    tp = entry + 2.0 * risk

                else:
                    entry = raw_open * (1.0 - SLIPPAGE)
                    sl = sig["sl_reference"]

                    risk = sl - entry

                    if risk <= 0:
                        continue

                    if risk > sig["atr_signal"] * MAX_STOP_ATR:
                        continue

                    tp = entry - 2.0 * risk

                active[sig["symbol"]] = {
                    "symbol": sig["symbol"],
                    "cluster": cluster,
                    "side": sig["side"],
                    "entry_price": entry,
                    "sl": sl,
                    "tp": tp,
                    "entry_time": ts,
                    "signal_time": sig["signal_time"],
                    "score": sig["score"],
                    "confirmations": sig["confirmations"],
                }

            pending_entries = []

        # --------------------------------------------------------
        # 4) Generate signals on THIS CLOSED 15M candle.
        #    They can only execute on the NEXT 15M candle.
        # --------------------------------------------------------
        if (
            pause_counter == 0
            and len(active) < MAX_OPEN_POSITIONS
            and k < len(all_times) - 1
        ):
            next_ts = all_times[k + 1]

            # Entry is next chronological 15M candle.
            for symbol, (df_15, df_1h, df_4h) in processed_data.items():

                if symbol in active:
                    continue

                if ts not in df_15.index:
                    continue

                # Need next bar to exist for execution.
                if next_ts not in df_15.index:
                    continue

                signal = signal_on_closed_candle(
                    df_15,
                    df_1h,
                    df_4h,
                    df_15.index.get_loc(ts),
                )

                if signal is None:
                    continue

                cluster = CORRELATION_CLUSTERS.get(
                    symbol, "OTHER"
                )

                # Capacity and cluster checks are repeated here
                # so pending signals do not explode.
                if (
                    MAX_ONE_PER_CLUSTER
                    and cluster_opened(cluster)
                ):
                    continue

                pending_entries.append({
                    **signal,
                    "symbol": symbol,
                    "cluster": cluster,
                })

                # Don't queue more than remaining capacity + a small
                # deterministic buffer. Final execution sorts by score.
                if len(pending_entries) >= MAX_OPEN_POSITIONS * 2:
                    break

    # ------------------------------------------------------------
    # Final open positions: report separately, don't fabricate exits.
    # ------------------------------------------------------------
    open_positions = []

    for symbol, pos in active.items():
        df_15, _, _ = processed_data[symbol]
        last_ts = df_15.index[-1]
        last_close = float(df_15.iloc[-1]["Close"])

        if pos["side"] == "LONG":
            unrealized = (
                TRADE_MARGIN
                * LEVERAGE
                * ((last_close - pos["entry_price"]) / pos["entry_price"])
            )
        else:
            unrealized = (
                TRADE_MARGIN
                * LEVERAGE
                * ((pos["entry_price"] - last_close) / pos["entry_price"])
            )

        open_positions.append({
            "Symbol": symbol,
            "Cluster": pos["cluster"],
            "Side": pos["side"],
            "EntryTime": pos["entry_time"],
            "EntryPrice": pos["entry_price"],
            "LastPrice": last_close,
            "SL": pos["sl"],
            "TP": pos["tp"],
            "Unrealized_Gross_PnL": unrealized,
        })

    return trades, open_positions


# ============================================================
# REPORTING
# ============================================================

def calculate_loss_streaks(trades_df):
    if trades_df.empty:
        return [], 0

    ordered = trades_df.sort_values(
        ["ExitTimestamp", "Timestamp"],
        kind="stable",
    )

    streaks = []
    current = 0
    max_streak = 0

    for outcome in ordered["Outcome"]:
        if outcome == "LOSS":
            current += 1
            max_streak = max(max_streak, current)
        elif outcome == "WIN":
            if current > 0:
                streaks.append(current)
            current = 0
        # No BE in V131-A.

    if current > 0:
        streaks.append(current)

    return streaks, max_streak


def print_symbol_report(trades_df):
    print("\n" + "-" * 90)
    print("PER-SYMBOL")
    print("-" * 90)

    if trades_df.empty:
        return

    rows = []

    for symbol, g in trades_df.groupby("Symbol"):
        wins = int((g["Outcome"] == "WIN").sum())
        losses = int((g["Outcome"] == "LOSS").sum())
        total = len(g)

        wr = 100.0 * wins / total if total else 0.0

        gp = float(
            g.loc[g["Outcome"] == "WIN", "Dollar_PnL"].sum()
        )
        gl = abs(float(
            g.loc[g["Outcome"] == "LOSS", "Dollar_PnL"].sum()
        ))

        pf = gp / gl if gl > 0 else np.inf

        rows.append({
            "Symbol": symbol,
            "Trades": total,
            "Wins": wins,
            "Losses": losses,
            "WinRate%": wr,
            "NetPnL": float(g["Dollar_PnL"].sum()),
            "PF": pf,
        })

    report = pd.DataFrame(rows).sort_values(
        "NetPnL",
        ascending=False,
    )

    for _, r in report.iterrows():
        pf_text = (
            "inf" if np.isinf(r["PF"])
            else f"{r['PF']:.2f}"
        )

        print(
            f"{r['Symbol']:>7} | "
            f"{int(r['Trades']):>4} trades | "
            f"WR {r['WinRate%']:>6.2f}% | "
            f"PnL ${r['NetPnL']:>10.2f} | "
            f"PF {pf_text}"
        )


def summarize(trades, open_positions):
    print("\n" + "=" * 90)
    print("HUNTER-V131-A — FINAL AUDITED REPORT")
    print("=" * 90)

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        print("هیچ معامله بسته‌شده‌ای ثبت نشد.")
        if open_positions:
            print(f"Open positions: {len(open_positions)}")
        return

    trades_df.sort_values(
        ["ExitTimestamp", "Timestamp"],
        kind="stable",
        inplace=True,
    )
    trades_df.reset_index(drop=True, inplace=True)

    total = len(trades_df)
    wins = int((trades_df["Outcome"] == "WIN").sum())
    losses = int((trades_df["Outcome"] == "LOSS").sum())

    win_rate = 100.0 * wins / total if total else 0.0

    net_pnl = float(trades_df["Dollar_PnL"].sum())

    gross_profit = float(
        trades_df.loc[
            trades_df["Outcome"] == "WIN",
            "Dollar_PnL"
        ].sum()
    )

    gross_loss = abs(float(
        trades_df.loc[
            trades_df["Outcome"] == "LOSS",
            "Dollar_PnL"
        ].sum()
    ))

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    equity = INITIAL_CAPITAL + trades_df["Dollar_PnL"].cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    max_dd = float(drawdown.min())

    streaks, max_streak = calculate_loss_streaks(trades_df)

    avg_win = (
        float(
            trades_df.loc[
                trades_df["Outcome"] == "WIN",
                "Dollar_PnL"
            ].mean()
        )
        if wins > 0 else 0.0
    )

    avg_loss = (
        float(
            trades_df.loc[
                trades_df["Outcome"] == "LOSS",
                "Dollar_PnL"
            ].mean()
        )
        if losses > 0 else 0.0
    )

    expectancy = float(
        trades_df["Dollar_PnL"].mean()
    )

    start = trades_df["Timestamp"].min()
    end = trades_df["ExitTimestamp"].max()

    days = max(
        1.0,
        (end - start).total_seconds() / 86400.0,
    )

    trades_per_day = total / days

    long_df = trades_df[
        trades_df["Side"] == "LONG"
    ]
    short_df = trades_df[
        trades_df["Side"] == "SHORT"
    ]

    long_wr = (
        100.0 * (long_df["Outcome"] == "WIN").sum()
        / len(long_df)
        if len(long_df) else 0.0
    )

    short_wr = (
        100.0 * (short_df["Outcome"] == "WIN").sum()
        / len(short_df)
        if len(short_df) else 0.0
    )

    final_equity = INITIAL_CAPITAL + net_pnl

    print(f"Initial Capital:          ${INITIAL_CAPITAL:,.2f}")
    print(f"Trade Margin:             ${TRADE_MARGIN:,.2f}")
    print(f"Leverage:                 {LEVERAGE:.0f}x")
    print(f"RR:                       1:2")
    print("-" * 90)
    print(f"Closed Trades:            {total}")
    print(f"Wins:                     {wins}")
    print(f"Losses:                   {losses}")
    print(f"Win Rate:                 {win_rate:.2f}%")
    print(f"Long Win Rate:            {long_wr:.2f}%")
    print(f"Short Win Rate:           {short_wr:.2f}%")
    print("-" * 90)
    print(f"Net PnL:                  ${net_pnl:,.2f}")
    print(f"Final Closed Equity:      ${final_equity:,.2f}")
    print(f"Profit Factor:            {profit_factor:.2f}")
    print(f"Average Win:              ${avg_win:,.2f}")
    print(f"Average Loss:             ${avg_loss:,.2f}")
    print(f"Expectancy / Trade:       ${expectancy:,.2f}")
    print(f"Max Drawdown:             ${max_dd:,.2f}")
    print(f"Max Portfolio Loss Streak:{max_streak}")
    print(f"Trades / Day:             {trades_per_day:.2f}")
    print("-" * 90)

    print("Loss streak sequence:")
    print(
        ", ".join(map(str, streaks))
        if streaks else "None"
    )

    if open_positions:
        print("\n" + "-" * 90)
        print("OPEN POSITIONS AT END OF DATA — NOT COUNTED AS CLOSED TRADES")
        print("-" * 90)

        for p in open_positions:
            print(
                f"{p['Symbol']:>7} | "
                f"{p['Side']:>5} | "
                f"Entry {p['EntryPrice']:.8g} | "
                f"Last {p['LastPrice']:.8g} | "
                f"Gross Unrealized ${p['Unrealized_Gross_PnL']:.2f}"
            )

    print_symbol_report(trades_df)

    print("\n" + "=" * 90)
    print("AUDIT NOTES")
    print("=" * 90)
    print("✓ No same-candle signal/entry")
    print("✓ Entry = next 15M candle OPEN + slippage")
    print("✓ 1H/4H context uses completed higher-timeframe candles")
    print("✓ SL/TP = conservative 1:2 structural setup")
    print("✓ No Break-Even / trailing / timeout")
    print("✓ Existing positions remain managed during loss pause")
    print("✓ Portfolio-level max 3 positions")
    print("✓ Max 1 position per correlation cluster")
    print("✓ Same-candle SL+TP conflict resolves to LOSS")
    print("✓ Final open positions are reported separately")
    print("=" * 90)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 90)
    print("HUNTER-V131-A — LIQUIDITY / STRUCTURE / REPRICING")
    print("STRICT PORTFOLIO-LEVEL NO-LOOKAHEAD BACKTEST")
    print("=" * 90)

    now = pd.Timestamp.now(tz="UTC")
    start_dt = now - pd.Timedelta(days=DAYS)
    end_dt = now

    processed_data = {}

    for symbol, lbank_symbol in SYMBOLS.items():
        print(f"\nدریافت و آماده‌سازی: {symbol} ...")

        df = fetch_symbol_data(
            lbank_symbol,
            start_dt.to_pydatetime(),
            end_dt.to_pydatetime(),
        )

        if df is None:
            print("  -> داده معتبر کافی نبود.")
            continue

        try:
            df_15, df_1h, df_4h = prepare_data(df)

            processed_data[symbol] = (
                df_15,
                df_1h,
                df_4h,
            )

            print(
                f"  -> 15M={len(df_15):,} | "
                f"1H={len(df_1h):,} | "
                f"4H={len(df_4h):,}"
            )

        except Exception as e:
            print(f"  -> خطا در prepare_data: {e}")

    print(f"\nنمادهای معتبر: {len(processed_data)} / {len(SYMBOLS)}")

    if not processed_data:
        print("هیچ داده‌ای برای بک‌تست موجود نیست.")
        return

    print("\nشروع موتور Portfolio-Level ...")

    trades, open_positions = run_portfolio_backtest(
        processed_data,
        start_dt,
        end_dt,
    )

    summarize(
        trades,
        open_positions,
    )


if __name__ == "__main__":
    main()
