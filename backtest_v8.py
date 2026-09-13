
"""
HUNTER-X / LSCS — LBank USDT-M Perpetual Portfolio Backtest
============================================================

هدف:
- داده واقعی LBank Futures / Perpetual
- 1H execution + 4H regime
- Long / Short
- Sweep -> BOS -> Retest -> Score -> NEXT-CANDLE Entry
- RR ثابت 1:2
- No Lookahead / No Repainting
- No signal on the same candle whose close determines the signal
- Per-symbol overlap lock: while a position on a symbol is unresolved,
  no second position/signal is accepted for that symbol.
- Portfolio-level max open positions + correlated-altcoin filter
- Intrabar ambiguity (SL and TP both touched in one candle) => LOSS
- Result is resolved only by forward scanning after entry.
- 1-year LBank data, paginated through CCXT.

IMPORTANT:
1) Signals are evaluated ONLY at CLOSED 1H candles.
2) Entry occurs at the OPEN of the NEXT 1H candle.
3) The entry candle itself cannot be used to create the signal.
4) A 4H candle becomes available only AFTER its close.
5) Pivots/structure use only candles strictly BEFORE the decision candle.
6) No "future_window < TP" or any other future-based filtering is used.
7) A trade is never removed because its future outcome is known.
"""

import os
import sys
import subprocess
from datetime import datetime, timedelta, timezone

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ccxt"])
    import ccxt

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    # Data
    "days_back": 365,
    "timeframe": "1h",
    "min_bars": 2500,
    "fetch_limit": 1000,

    # 1H structure
    "structure_lookback": 20,
    "sweep_confirm_window": 8,
    "retest_window": 6,
    "retest_atr_mult": 0.25,

    # Trend / volatility
    "adx_min_4h": 20.0,
    "min_atr_pct_1h": 0.0025,      # 0.25%
    "max_atr_pct_1h": 0.04,        # reject abnormal volatility > 4%
    "min_bos_atr": 0.20,
    "bos_volume_mult": 1.20,

    # Retest / momentum
    "long_rsi_min": 50.0,
    "long_rsi_max": 68.0,
    "short_rsi_min": 32.0,
    "short_rsi_max": 50.0,

    # SL
    "sl_atr_buffer": 0.30,
    "min_sl_atr": 0.80,
    "max_sl_atr": 2.50,

    # TP
    "rr": 2.0,

    # Score
    "min_score": 70.0,

    # Forward resolution
    "max_holding_bars": 96,        # 4 days maximum
    "ambiguous_bar_is_loss": True,

    # Costs
    "taker_fee": 0.0005,           # 0.05% each side
    "slippage_pct": 0.0005,        # 0.05% each side
    "funding_cost_R": 0.0,         # 0 by default; replace with historical funding model if available

    # Portfolio
    "max_open_positions": 3,
    "max_new_entries_per_timestamp": 1,
    "same_symbol_cooldown_bars": 0,

    # Correlation clusters. At most one position per cluster.
    "correlation_clusters": {
        "MAJOR": {"BTC", "ETH"},
        "L1": {"SOL", "SUI", "NEAR", "AVAX", "ADA", "DOT"},
        "PAYMENTS": {"XRP"},
        "DEFI": {"LINK"},
    },

    # Ranking weights
    "rank_score_weight": 1.0,
    "rank_rr_quality_weight": 10.0,
}


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


exchange = ccxt.lbank({
    "enableRateLimit": True,
    "options": {
        "defaultType": "swap",
    },
})


# =============================================================================
# DATA
# =============================================================================

def normalize_ohlcv(rows):
    if not rows:
        return None

    df = pd.DataFrame(
        rows,
        columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
    )

    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (
        df.dropna()
          .drop_duplicates("Date")
          .sort_values("Date")
          .reset_index(drop=True)
    )

    return df


def fetch_1h_data(symbol_key, symbol, days_back):
    """
    Fetch only the requested 1H period.

    We deliberately discard the last candle if it is still forming.
    A live/unfinished candle must NEVER participate in the backtest.
    """
    print(f"\n📥 {symbol_key}: downloading LBank 1H perpetual data...")

    try:
        exchange.load_markets()
        if symbol not in exchange.markets:
            # Try the unified market name without settlement suffix.
            candidates = [
                s for s in exchange.markets
                if s.replace(":USDT", "") == symbol.replace(":USDT", "")
            ]
            if candidates:
                symbol = candidates[0]
            else:
                print(f"❌ {symbol_key}: market not found on LBank.")
                return None

        now_ms = exchange.milliseconds()
        start_ms = int(
            (datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp()
            * 1000
        )

        all_rows = []
        since = start_ms
        last_ts = None

        while since < now_ms:
            rows = exchange.fetch_ohlcv(
                symbol,
                timeframe="1h",
                since=since,
                limit=CONFIG["fetch_limit"],
            )

            if not rows:
                break

            all_rows.extend(rows)

            newest = rows[-1][0]

            # Hard protection against a broken exchange pagination response.
            if last_ts is not None and newest <= last_ts:
                break

            last_ts = newest
            since = newest + 1

            if len(rows) < CONFIG["fetch_limit"]:
                break

        df = normalize_ohlcv(all_rows)

        if df is None or df.empty:
            print(f"❌ {symbol_key}: no data.")
            return None

        # Never use a currently-forming candle.
        current_hour = pd.Timestamp.now(tz="UTC").floor("h")
        df = df[df["Date"] < current_hour].copy()

        # Keep only the requested lookback.
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back)
        df = df[df["Date"] >= cutoff].reset_index(drop=True)

        if len(df) < CONFIG["min_bars"]:
            print(
                f"⚠️ {symbol_key}: only {len(df)} closed candles; "
                f"minimum requested = {CONFIG['min_bars']}"
            )

        df.to_csv(f"{symbol_key}_LBank_1H.csv", index=False)

        print(f"✅ {symbol_key}: {len(df)} closed 1H candles.")
        return df

    except Exception as e:
        print(f"❌ {symbol_key}: {type(e).__name__}: {e}")
        return None


# =============================================================================
# INDICATORS — PAST ONLY
# =============================================================================

def wilder_rma(series, period):
    # pandas ewm(alpha=1/period, adjust=False) is causal.
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def calculate_indicators(df):
    df = df.copy()

    # EMA
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    # RSI (Wilder-style, causal)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = wilder_rma(gain, 14)
    avg_loss = wilder_rma(loss, 14)

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # ATR (Wilder)
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)

    df["TR"] = tr
    df["ATR"] = wilder_rma(tr, 14)
    df["ATR_pct"] = df["ATR"] / df["Close"]

    # ADX
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0
        ),
        index=df.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0
        ),
        index=df.index,
    )

    atr14 = df["ATR"]

    plus_di = 100 * wilder_rma(plus_dm, 14) / atr14.replace(0, np.nan)
    minus_di = 100 * wilder_rma(minus_dm, 14) / atr14.replace(0, np.nan)

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    df["PLUS_DI"] = plus_di
    df["MINUS_DI"] = minus_di
    df["ADX"] = wilder_rma(dx, 14)

    # Volume baseline.
    # shift(1) is important: current candle is NOT included in its own baseline.
    df["VOL_MEAN_20_PREV"] = df["Volume"].rolling(20).mean().shift(1)
    df["RVOL"] = df["Volume"] / df["VOL_MEAN_20_PREV"].replace(0, np.nan)

    # Candle momentum
    df["BODY"] = (df["Close"] - df["Open"]).abs()
    df["BODY_ATR"] = df["BODY"] / df["ATR"]

    return df


# =============================================================================
# 4H CONTEXT — ONLY CLOSED 4H CANDLES ARE AVAILABLE
# =============================================================================

def build_4h_context(df1h):
    """
    Important timestamp convention:

    A 1H candle at 10:00 represents [10:00, 11:00).
    Its close is available at 11:00.

    A 4H candle starting at 08:00 represents [08:00, 12:00).
    Its values are available at 12:00.

    Therefore the 4H context is attached using available_at = 4H start + 4h.
    A 1H candle can access only a 4H candle whose close time <= that 1H candle's
    OPEN time. This is deliberately conservative and prevents leakage.
    """
    base = df1h.set_index("Date")

    df4h = (
        base.resample("4h", label="left", closed="left")
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        })
        .dropna()
        .reset_index()
    )

    df4h = calculate_indicators(df4h)

    # Previous closed 4H values.
    df4h["EMA200_PREV"] = df4h["EMA200"].shift(1)
    df4h["EMA20_PREV"] = df4h["EMA20"].shift(1)
    df4h["EMA50_PREV"] = df4h["EMA50"].shift(1)

    df4h["available_at"] = df4h["Date"] + pd.Timedelta(hours=4)

    ctx = df4h[[
        "available_at",
        "EMA20",
        "EMA50",
        "EMA200",
        "EMA20_PREV",
        "EMA50_PREV",
        "EMA200_PREV",
        "RSI",
        "ADX",
        "ATR",
        "ATR_pct",
        "PLUS_DI",
        "MINUS_DI",
    ]].rename(columns={
        "EMA20": "EMA20_4H",
        "EMA50": "EMA50_4H",
        "EMA200": "EMA200_4H",
        "EMA20_PREV": "EMA20_PREV_4H",
        "EMA50_PREV": "EMA50_PREV_4H",
        "EMA200_PREV": "EMA200_PREV_4H",
        "RSI": "RSI_4H",
        "ADX": "ADX_4H",
        "ATR": "ATR_4H",
        "ATR_pct": "ATR_PCT_4H",
        "PLUS_DI": "PLUS_DI_4H",
        "MINUS_DI": "MINUS_DI_4H",
    })

    out = pd.merge_asof(
        df1h.sort_values("Date"),
        ctx.sort_values("available_at"),
        left_on="Date",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )

    return out


# =============================================================================
# MARKET REGIME
# =============================================================================

def add_regime(df):
    df = df.copy()

    bullish = (
        (df["EMA20_4H"] > df["EMA50_4H"])
        & (df["EMA50_4H"] > df["EMA200_4H"])
        & (df["EMA200_4H"] >= df["EMA200_PREV_4H"])
        & (df["ADX_4H"] >= CONFIG["adx_min_4h"])
        & (df["RSI_4H"] >= 52)
        & (df["PLUS_DI_4H"] > df["MINUS_DI_4H"])
    )

    bearish = (
        (df["EMA20_4H"] < df["EMA50_4H"])
        & (df["EMA50_4H"] < df["EMA200_4H"])
        & (df["EMA200_4H"] <= df["EMA200_PREV_4H"])
        & (df["ADX_4H"] >= CONFIG["adx_min_4h"])
        & (df["RSI_4H"] <= 48)
        & (df["MINUS_DI_4H"] > df["PLUS_DI_4H"])
    )

    # Ranging / neutral is deliberately not traded.
    df["REGIME"] = np.where(
        bullish,
        "BULL",
        np.where(bearish, "BEAR", "NO_TRADE")
    )

    return df


# =============================================================================
# SETUP / SCORE
# =============================================================================

def get_structure(df, i, lookback):
    """
    ONLY candles [i-lookback, i-1].
    The decision candle i is excluded.
    """
    w = df.iloc[i - lookback:i]

    return {
        "high": float(w["High"].max()),
        "low": float(w["Low"].min()),
        "vol_mean": float(w["Volume"].mean()),
    }


def score_long(df, sweep_i, bos_i, retest_i, structure_high, structure_low):
    r = df.iloc[retest_i]
    b = df.iloc[bos_i]
    s = df.iloc[sweep_i]

    atr = float(r["ATR"])
    if not np.isfinite(atr) or atr <= 0:
        return 0.0

    score = 0.0

    # 20: 4H trend strength
    adx = float(r["ADX_4H"])
    score += min(20.0, max(0.0, (adx - 20.0) / 15.0 * 20.0))

    # 20: sweep quality
    sweep_depth = max(0.0, (structure_low - float(s["Low"])) / atr)
    score += min(20.0, sweep_depth / 0.75 * 20.0)

    # 20: BOS quality
    bos_size = max(0.0, (float(b["Close"]) - structure_high) / atr)
    score += min(20.0, bos_size / 0.75 * 20.0)

    # 15: retest quality
    dist = abs(float(r["Low"]) - structure_high) / atr
    if dist <= CONFIG["retest_atr_mult"]:
        score += 15.0

    # 15: volume / momentum
    rvol = float(b["RVOL"]) if np.isfinite(b["RVOL"]) else 0.0
    if rvol >= CONFIG["bos_volume_mult"]:
        score += min(10.0, (rvol - 1.0) / 0.75 * 10.0)

    if float(r["RSI"]) > float(df.iloc[retest_i - 1]["RSI"]):
        score += 5.0

    return float(min(100.0, score))


def score_short(df, sweep_i, bos_i, retest_i, structure_high, structure_low):
    r = df.iloc[retest_i]
    b = df.iloc[bos_i]
    s = df.iloc[sweep_i]

    atr = float(r["ATR"])
    if not np.isfinite(atr) or atr <= 0:
        return 0.0

    score = 0.0

    adx = float(r["ADX_4H"])
    score += min(20.0, max(0.0, (adx - 20.0) / 15.0 * 20.0))

    sweep_depth = max(0.0, (float(s["High"]) - structure_high) / atr)
    score += min(20.0, sweep_depth / 0.75 * 20.0)

    bos_size = max(0.0, (structure_low - float(b["Close"])) / atr)
    score += min(20.0, bos_size / 0.75 * 20.0)

    dist = abs(float(r["High"]) - structure_low) / atr
    if dist <= CONFIG["retest_atr_mult"]:
        score += 15.0

    rvol = float(b["RVOL"]) if np.isfinite(b["RVOL"]) else 0.0
    if rvol >= CONFIG["bos_volume_mult"]:
        score += min(10.0, (rvol - 1.0) / 0.75 * 10.0)

    if float(r["RSI"]) < float(df.iloc[retest_i - 1]["RSI"]):
        score += 5.0

    return float(min(100.0, score))


# =============================================================================
# CANDIDATE GENERATION
# =============================================================================

def find_candidate(df, i):
    """
    Returns a candidate only when candle i is CLOSED.

    Crucially:
      - sweep may have happened on an earlier candle
      - BOS and retest are searched forward from that sweep
      - once retest candle p closes, that candle becomes the SIGNAL candle
      - ENTRY = OPEN OF p+1
      - therefore the signal candle's future is never used for entry.
    """
    if i < CONFIG["structure_lookback"]:
        return None

    row = df.iloc[i]

    # We only generate a signal on the retest candle itself.
    if row["REGIME"] not in ("BULL", "BEAR"):
        return None

    if not np.isfinite(row["ATR"]) or row["ATR"] <= 0:
        return None

    atr = float(row["ATR"])
    atr_pct = float(row["ATR_pct"])

    if atr_pct < CONFIG["min_atr_pct_1h"]:
        return None

    if atr_pct > CONFIG["max_atr_pct_1h"]:
        return None

    lb = CONFIG["structure_lookback"]

    # Search only backwards for a valid sweep.
    # The sweep/BOS/retest sequence is causal because all referenced candles
    # are already CLOSED by candle i.
    search_start = max(lb, i - CONFIG["retest_window"] - CONFIG["sweep_confirm_window"] - 2)

    for sweep_i in range(search_start, i):
        sweep = df.iloc[sweep_i]

        if sweep["REGIME"] != row["REGIME"]:
            continue

        structure = get_structure(df, sweep_i, lb)
        sh = structure["high"]
        sl = structure["low"]

        # LONG sweep: take sell-side liquidity and close back above it.
        if row["REGIME"] == "BULL":
            if not (
                float(sweep["Low"]) < sl
                and float(sweep["Close"]) > sl
            ):
                continue

            # BOS must happen after sweep and before current retest.
            bos_found = None

            bos_end = min(
                sweep_i + 1 + CONFIG["sweep_confirm_window"],
                i
            )

            for bos_i in range(sweep_i + 1, bos_end):
                b = df.iloc[bos_i]

                # If price decisively destroys the setup before BOS, abandon it.
                if float(b["Close"]) < sl - atr:
                    break

                if float(b["Close"]) > sh:
                    vol_ok = (
                        np.isfinite(b["RVOL"])
                        and float(b["RVOL"]) >= CONFIG["bos_volume_mult"]
                    )

                    bos_size_ok = (
                        float(b["Close"]) - sh
                    ) / atr >= CONFIG["min_bos_atr"]

                    if vol_ok and bos_size_ok:
                        bos_found = bos_i
                        break

            if bos_found is None:
                continue

            # Current candle i must be inside the retest window after BOS.
            if not (
                bos_found < i <= bos_found + CONFIG["retest_window"]
            ):
                continue

            # Retest zone around broken resistance.
            in_zone = (
                float(row["Low"])
                <= sh + CONFIG["retest_atr_mult"] * atr
                and float(row["Low"])
                >= sh - CONFIG["retest_atr_mult"] * atr
            )

            bullish_close = float(row["Close"]) > float(row["Open"])
            rsi_ok = CONFIG["long_rsi_min"] <= float(row["RSI"]) <= CONFIG["long_rsi_max"]

            if not (in_zone and bullish_close and rsi_ok):
                continue

            score = score_long(
                df, sweep_i, bos_found, i, sh, sl
            )

            if score < CONFIG["min_score"]:
                continue

            # SL: lowest low from BOS through retest + ATR buffer.
            pullback_low = float(
                df.iloc[bos_found:i + 1]["Low"].min()
            )

            entry = float(df.iloc[i + 1]["Open"]) if i + 1 < len(df) else np.nan
            if not np.isfinite(entry):
                continue

            stop = pullback_low - CONFIG["sl_atr_buffer"] * atr
            risk = entry - stop

            if risk <= 0:
                continue

            risk_atr = risk / atr

            if not (
                CONFIG["min_sl_atr"]
                <= risk_atr
                <= CONFIG["max_sl_atr"]
            ):
                continue

            tp = entry + CONFIG["rr"] * risk

            return {
                "SignalIdx": i,
                "EntryIdx": i + 1,
                "Side": "LONG",
                "SignalTime": row["Date"],
                "EntryTime": df.iloc[i + 1]["Date"],
                "Entry": entry,
                "SL": stop,
                "TP": tp,
                "Risk": risk,
                "RiskATR": risk_atr,
                "Score": score,
                "SweepIdx": sweep_i,
                "BOSIdx": bos_found,
            }

        # SHORT
        else:
            if not (
                float(sweep["High"]) > sh
                and float(sweep["Close"]) < sh
            ):
                continue

            bos_found = None

            bos_end = min(
                sweep_i + 1 + CONFIG["sweep_confirm_window"],
                i
            )

            for bos_i in range(sweep_i + 1, bos_end):
                b = df.iloc[bos_i]

                if float(b["Close"]) > sh + atr:
                    break

                if float(b["Close"]) < sl:
                    vol_ok = (
                        np.isfinite(b["RVOL"])
                        and float(b["RVOL"]) >= CONFIG["bos_volume_mult"]
                    )

                    bos_size_ok = (
                        sl - float(b["Close"])
                    ) / atr >= CONFIG["min_bos_atr"]

                    if vol_ok and bos_size_ok:
                        bos_found = bos_i
                        break

            if bos_found is None:
                continue

            if not (
                bos_found < i <= bos_found + CONFIG["retest_window"]
            ):
                continue

            in_zone = (
                float(row["High"])
                <= sl + CONFIG["retest_atr_mult"] * atr
                and float(row["High"])
                >= sl - CONFIG["retest_atr_mult"] * atr
            )

            bearish_close = float(row["Close"]) < float(row["Open"])
            rsi_ok = CONFIG["short_rsi_min"] <= float(row["RSI"]) <= CONFIG["short_rsi_max"]

            if not (in_zone and bearish_close and rsi_ok):
                continue

            score = score_short(
                df, sweep_i, bos_found, i, sh, sl
            )

            if score < CONFIG["min_score"]:
                continue

            pullback_high = float(
                df.iloc[bos_found:i + 1]["High"].max()
            )

            entry = float(df.iloc[i + 1]["Open"]) if i + 1 < len(df) else np.nan
            if not np.isfinite(entry):
                continue

            stop = pullback_high + CONFIG["sl_atr_buffer"] * atr
            risk = stop - entry

            if risk <= 0:
                continue

            risk_atr = risk / atr

            if not (
                CONFIG["min_sl_atr"]
                <= risk_atr
                <= CONFIG["max_sl_atr"]
            ):
                continue

            tp = entry - CONFIG["rr"] * risk

            return {
                "SignalIdx": i,
                "EntryIdx": i + 1,
                "Side": "SHORT",
                "SignalTime": row["Date"],
                "EntryTime": df.iloc[i + 1]["Date"],
                "Entry": entry,
                "SL": stop,
                "TP": tp,
                "Risk": risk,
                "RiskATR": risk_atr,
                "Score": score,
                "SweepIdx": sweep_i,
                "BOSIdx": bos_found,
            }

    return None


# =============================================================================
# FORWARD TRADE RESOLUTION
# =============================================================================

def resolve_trade(df, candidate):
    """
    IMPORTANT:
    Starts at EntryIdx + 1.

    The entry occurs at the OPEN of EntryIdx.
    Therefore the candle containing the entry price is NOT allowed to decide
    the outcome. This removes a common same-candle ambiguity/lookahead issue.

    Outcome is determined only by future closed candles after the entry candle.

    If SL and TP are both touched in the same candle:
        LOSS (conservative worst-case)
    """
    entry_idx = int(candidate["EntryIdx"])
    side = candidate["Side"]
    entry = float(candidate["Entry"])
    sl = float(candidate["SL"])
    tp = float(candidate["TP"])

    last = min(
        len(df),
        entry_idx + 1 + CONFIG["max_holding_bars"]
    )

    for j in range(entry_idx + 1, last):
        c = df.iloc[j]

        if side == "LONG":
            hit_sl = float(c["Low"]) <= sl
            hit_tp = float(c["High"]) >= tp
        else:
            hit_sl = float(c["High"]) >= sl
            hit_tp = float(c["Low"]) <= tp

        if hit_sl and hit_tp:
            return {
                "Outcome": "LOSS",
                "ExitIdx": j,
                "ExitPrice": sl,
                "BarsHeld": j - entry_idx,
            }

        if hit_sl:
            return {
                "Outcome": "LOSS",
                "ExitIdx": j,
                "ExitPrice": sl,
                "BarsHeld": j - entry_idx,
            }

        if hit_tp:
            return {
                "Outcome": "WIN",
                "ExitIdx": j,
                "ExitPrice": tp,
                "BarsHeld": j - entry_idx,
            }

    # Time exit: neither TP nor SL was reached.
    # We do NOT classify it as a WIN.
    # For fixed 1:2 system, it is treated as LOSS-equivalent in R accounting
    # only if explicitly enabled. Here it is marked TIMEOUT and realized using
    # the close of the final available candle.
    j = last - 1
    return {
        "Outcome": "TIMEOUT",
        "ExitIdx": j,
        "ExitPrice": float(df.iloc[j]["Close"]),
        "BarsHeld": j - entry_idx,
    }


# =============================================================================
# TRADE ACCOUNTING
# =============================================================================

def apply_costs(candidate, resolution):
    entry = float(candidate["Entry"])
    sl = float(candidate["SL"])
    tp = float(candidate["TP"])
    side = candidate["Side"]

    risk = abs(entry - sl)

    if risk <= 0:
        return None

    gross_R = 0.0

    if resolution["Outcome"] == "WIN":
        gross_R = CONFIG["rr"]

    elif resolution["Outcome"] == "LOSS":
        gross_R = -1.0

    elif resolution["Outcome"] == "TIMEOUT":
        exit_price = float(resolution["ExitPrice"])

        if side == "LONG":
            pnl_price = exit_price - entry
        else:
            pnl_price = entry - exit_price

        gross_R = pnl_price / risk

    # Entry + exit notional approximation.
    # This is a conservative transaction-cost model in R.
    cost_pct_roundtrip = (
        2.0 * CONFIG["taker_fee"]
        + 2.0 * CONFIG["slippage_pct"]
    )

    cost_R = cost_pct_roundtrip / (risk / entry)
    net_R = gross_R - cost_R - CONFIG["funding_cost_R"]

    return gross_R, cost_R, net_R


def make_trade_record(symbol, candidate, resolution):
    accounting = apply_costs(candidate, resolution)
    if accounting is None:
        return None

    gross_R, cost_R, net_R = accounting

    return {
        "Symbol": symbol,
        "Side": candidate["Side"],
        "Outcome": resolution["Outcome"],

        "SignalTime": candidate["SignalTime"],
        "EntryTime": candidate["EntryTime"],

        "ExitTime": None,  # filled by caller
        "Entry": candidate["Entry"],
        "SL": candidate["SL"],
        "TP": candidate["TP"],
        "ExitPrice": resolution["ExitPrice"],

        "Score": candidate["Score"],
        "RiskATR": candidate["RiskATR"],

        "BarsHeld": resolution["BarsHeld"],

        "Gross_R": gross_R,
        "Cost_R": cost_R,
        "Net_R": net_R,

        "SweepIdx": candidate["SweepIdx"],
        "BOSIdx": candidate["BOSIdx"],
        "SignalIdx": candidate["SignalIdx"],
        "EntryIdx": candidate["EntryIdx"],
        "ExitIdx": resolution["ExitIdx"],
    }


# =============================================================================
# PORTFOLIO HELPERS
# =============================================================================

def cluster_of(symbol):
    for name, members in CONFIG["correlation_clusters"].items():
        if symbol in members:
            return name
    return symbol


def can_open_candidate(candidate, symbol, open_positions, last_exit_by_symbol):
    """
    Portfolio constraints are evaluated using ONLY positions that are already
    known at this timestamp. No future outcome is consulted.
    """
    if len(open_positions) >= CONFIG["max_open_positions"]:
        return False

    # One position per symbol.
    if symbol in open_positions:
        return False

    # One position per correlated cluster.
    cluster = cluster_of(symbol)
    for pos in open_positions.values():
        if cluster_of(pos["Symbol"]) == cluster:
            return False

    # Optional same-symbol cooldown.
    if symbol in last_exit_by_symbol:
        last_exit = last_exit_by_symbol[symbol]
        entry_time = candidate["EntryTime"]
        hours = (entry_time - last_exit).total_seconds() / 3600.0

        if hours < CONFIG["same_symbol_cooldown_bars"]:
            return False

    return True


# =============================================================================
# SINGLE-SYMBOL CANDIDATE PRECOMPUTATION
# =============================================================================

def prepare_symbol(symbol_key, df):
    """
    Computes all candidate signals causally.

    No future outcome is involved here.
    Candidate generation stops at the current closed candle.
    """
    candidates = []

    for i in range(
        CONFIG["structure_lookback"],
        len(df) - 1
    ):
        candidate = find_candidate(df, i)

        if candidate is not None:
            candidate["Symbol"] = symbol_key
            candidates.append(candidate)

    return candidates


# =============================================================================
# PORTFOLIO ENGINE
# =============================================================================

def run_portfolio_backtest(data_by_symbol):
    """
    Event-driven portfolio engine.

    Key rule:
        We do NOT backtest each symbol independently and then merge results.

    All candidate entries are synchronized by actual timestamp, so portfolio
    limits and correlation filters are applied in chronological order.
    """
    candidate_by_time = {}

    for symbol, df in data_by_symbol.items():
        candidates = prepare_symbol(symbol, df)

        for c in candidates:
            candidate_by_time.setdefault(c["EntryTime"], []).append(c)

    timestamps = sorted(candidate_by_time.keys())

    # Active positions.
    # Because each candidate is resolved before its next candidate on the same
    # symbol can be considered, overlap is locked per symbol.
    open_positions = {}

    trades = []
    last_exit_by_symbol = {}

    diagnostics = {
        "candidates": 0,
        "accepted": 0,
        "rejected_max_positions": 0,
        "rejected_same_symbol": 0,
        "rejected_correlation": 0,
    }

    for ts in timestamps:
        candidates = candidate_by_time[ts]
        diagnostics["candidates"] += len(candidates)

        # ---------------------------------------------------------------------
        # First, close positions whose exit candle has arrived.
        #
        # Resolution was already performed when the position was accepted.
        # We use exit time only to know whether the position is still active at
        # this entry timestamp.
        # ---------------------------------------------------------------------
        to_remove = []

        for symbol, pos in open_positions.items():
            if pos["ExitTime"] < ts:
                to_remove.append(symbol)

        for symbol in to_remove:
            last_exit_by_symbol[symbol] = open_positions[symbol]["ExitTime"]
            del open_positions[symbol]

        # ---------------------------------------------------------------------
        # Rank simultaneous candidates.
        #
        # Ranking uses ONLY information available at signal/entry time:
        # score + SL quality.
        # No outcome or future movement is used.
        # ---------------------------------------------------------------------
        ranked = []

        for c in candidates:
            entry = float(c["Entry"])
            risk = float(c["Risk"])
            risk_pct = risk / entry

            rr_quality = 1.0 / max(risk_pct, 1e-9)

            rank = (
                CONFIG["rank_score_weight"] * float(c["Score"])
                + CONFIG["rank_rr_quality_weight"] * rr_quality
            )

            ranked.append((rank, c))

        ranked.sort(key=lambda x: x[0], reverse=True)

        entries_this_timestamp = 0

        for _, candidate in ranked:
            symbol = candidate["Symbol"]

            # A candidate cannot be accepted after the portfolio entry quota
            # for the same timestamp has been reached.
            if entries_this_timestamp >= CONFIG["max_new_entries_per_timestamp"]:
                break

            if len(open_positions) >= CONFIG["max_open_positions"]:
                diagnostics["rejected_max_positions"] += 1
                continue

            if symbol in open_positions:
                diagnostics["rejected_same_symbol"] += 1
                continue

            candidate_cluster = cluster_of(symbol)

            if any(
                cluster_of(p["Symbol"]) == candidate_cluster
                for p in open_positions.values()
            ):
                diagnostics["rejected_correlation"] += 1
                continue

            # -----------------------------------------------------------------
            # ACCEPT TRADE
            # -----------------------------------------------------------------
            df = data_by_symbol[symbol]
            resolution = resolve_trade(df, candidate)

            # ExitTime is known only from the forward resolution of this
            # already-accepted trade. It is NOT used to decide whether the
            # candidate itself was a winner/loser.
            exit_idx = resolution["ExitIdx"]
            exit_time = df.iloc[exit_idx]["Date"]

            trade = make_trade_record(
                symbol,
                candidate,
                resolution,
            )

            if trade is None:
                continue

            trade["ExitTime"] = exit_time

            trades.append(trade)

            open_positions[symbol] = {
                "Symbol": symbol,
                "EntryTime": candidate["EntryTime"],
                "ExitTime": exit_time,
                "Side": candidate["Side"],
            }

            diagnostics["accepted"] += 1
            entries_this_timestamp += 1

    return pd.DataFrame(trades), diagnostics


# =============================================================================
# METRICS
# =============================================================================

def max_consecutive(values, target):
    best = 0
    cur = 0

    for v in values:
        if v == target:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0

    return best


def compute_drawdown(equity):
    running_max = equity.cummax()
    dd = equity - running_max
    return dd.min()


def compute_metrics(df, r_col="Net_R"):
    if df.empty:
        return {}

    x = df.sort_values("EntryTime").copy()

    total = len(x)

    wins = x[x["Outcome"] == "WIN"]
    losses = x[x["Outcome"] == "LOSS"]
    timeouts = x[x["Outcome"] == "TIMEOUT"]

    win_rate = len(wins) / total * 100.0
    loss_rate = len(losses) / total * 100.0

    gross_profit = wins[r_col].sum()
    gross_loss = losses[r_col].sum()

    net = x[r_col].sum()

    pf = (
        gross_profit / abs(gross_loss)
        if gross_loss != 0
        else np.inf
    )

    expectancy = x[r_col].mean()

    avg_win = wins[r_col].mean() if not wins.empty else np.nan
    avg_loss = losses[r_col].mean() if not losses.empty else np.nan

    equity = x[r_col].cumsum()
    max_dd = compute_drawdown(equity)

    outcomes = x["Outcome"].tolist()

    max_wins = max_consecutive(outcomes, "WIN")
    max_losses = max_consecutive(outcomes, "LOSS")

    span_days = max(
        (x["EntryTime"].max() - x["EntryTime"].min()).total_seconds()
        / 86400.0,
        1.0,
    )

    avg_trades_day = total / span_days
    avg_hold_hours = x["BarsHeld"].mean()

    # Sharpe on per-trade R is only an approximation.
    std = x[r_col].std(ddof=1)
    sharpe_trade = (
        x[r_col].mean() / std * np.sqrt(252.0)
        if std and np.isfinite(std)
        else np.nan
    )

    # Fixed-risk equity interpretation:
    # 1R = 1% account risk is NOT assumed here.
    # Total R is strategy-level normalized return.
    total_R = x[r_col].sum()

    return {
        "Total Trades": total,
        "Wins": len(wins),
        "Losses": len(losses),
        "Timeouts": len(timeouts),
        "Win Rate %": win_rate,
        "Loss Rate %": loss_rate,
        "Gross Profit R": gross_profit,
        "Gross Loss R": gross_loss,
        "Net R": net,
        "Profit Factor": pf,
        "Average Win R": avg_win,
        "Average Loss R": avg_loss,
        "Expectancy R/trade": expectancy,
        "Max Drawdown R": max_dd,
        "Max Consecutive Wins": max_wins,
        "Max Consecutive Losses": max_losses,
        "Average Trades/Day": avg_trades_day,
        "Average Holding Hours": avg_hold_hours,
        "Total R": total_R,
        "Sharpe approx": sharpe_trade,
    }


def print_metrics(title, df):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if df.empty:
        print("⚠️ No trades.")
        return

    m = compute_metrics(df)

    for k, v in m.items():
        if isinstance(v, (float, np.floating)):
            print(f"{k:30s}: {v:.4f}")
        else:
            print(f"{k:30s}: {v}")

    print("\n--- Per Symbol ---")

    rows = []

    for symbol, g in df.groupby("Symbol"):
        mm = compute_metrics(g)

        rows.append({
            "Symbol": symbol,
            "Trades": mm["Total Trades"],
            "Wins": mm["Wins"],
            "Losses": mm["Losses"],
            "Timeouts": mm["Timeouts"],
            "WinRate%": mm["Win Rate %"],
            "NetR": mm["Net R"],
            "PF": mm["Profit Factor"],
            "AvgHoldH": mm["Average Holding Hours"],
        })

    print(
        pd.DataFrame(rows)
        .sort_values("NetR", ascending=False)
        .to_string(index=False)
    )


# =============================================================================
# DATA QUALITY / ANTI-LOOKAHEAD AUDIT
# =============================================================================

def anti_lookahead_audit(data_by_symbol, trades):
    """
    Mechanical sanity checks.

    This does NOT prove a strategy is mathematically bug-free, but catches
    several common accidental leaks.
    """
    print("\n" + "=" * 80)
    print("🔒 ANTI-LOOKAHEAD AUDIT")
    print("=" * 80)

    errors = []

    for symbol, df in data_by_symbol.items():
        if not df["Date"].is_monotonic_increasing:
            errors.append(f"{symbol}: Date is not sorted.")

        if df["Date"].duplicated().any():
            errors.append(f"{symbol}: duplicate timestamps.")

        # 4H context must have been available no later than the 1H candle.
        bad_ctx = (
            df["available_at"].notna()
            & (df["available_at"] > df["Date"])
        )

        if bad_ctx.any():
            errors.append(
                f"{symbol}: future 4H context detected."
            )

    if not trades.empty:
        bad_entry = trades["EntryTime"] <= trades["SignalTime"]

        if bad_entry.any():
            errors.append(
                "At least one trade has EntryTime <= SignalTime."
            )

        bad_exit = trades["ExitTime"] < trades["EntryTime"]

        if bad_exit.any():
            errors.append(
                "At least one trade exits before entry."
            )

        # Same-symbol overlap audit.
        for symbol, g in trades.groupby("Symbol"):
            g = g.sort_values("EntryTime")

            prev_exit = None

            for _, r in g.iterrows():
                if prev_exit is not None and r["EntryTime"] <= prev_exit:
                    errors.append(
                        f"{symbol}: overlapping positions detected."
                    )
                    break

                prev_exit = r["ExitTime"]

    if errors:
        print("❌ AUDIT FAILED")
        for e in errors:
            print(" -", e)
    else:
        print("✅ No structural lookahead/overlap violation detected.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("HUNTER-X / LSCS — LBank 1Y Futures Portfolio Backtest")
    print("=" * 80)
    print("Execution TF : 1H")
    print("Regime TF    : 4H")
    print("RR           : 1:2")
    print("Entry        : NEXT 1H candle OPEN after closed signal candle")
    print("Overlap lock : ON per symbol")
    print("Correlation  : ON")
    print("Lookahead    : OFF")
    print("=" * 80)

    data_by_symbol = {}

    # -------------------------------------------------------------------------
    # 1) Download + indicators + 4H context + regime
    # -------------------------------------------------------------------------
    for key, symbol in SYMBOLS.items():
        raw = fetch_1h_data(
            key,
            symbol,
            CONFIG["days_back"]
        )

        if raw is None:
            continue

        df = calculate_indicators(raw)
        df = build_4h_context(df)
        df = add_regime(df)

        required = [
            "ATR",
            "RSI",
            "ADX",
            "EMA20_4H",
            "EMA50_4H",
            "EMA200_4H",
            "available_at",
        ]

        df = df.dropna(subset=required).reset_index(drop=True)

        if len(df) < CONFIG["min_bars"]:
            print(
                f"⏭️ {key}: skipped because "
                f"{len(df)} usable bars < {CONFIG['min_bars']}"
            )
            continue

        data_by_symbol[key] = df

    if not data_by_symbol:
        print("❌ No symbols with sufficient data.")
        return

    print(
        f"\n✅ Loaded {len(data_by_symbol)}/{len(SYMBOLS)} symbols."
    )

    # -------------------------------------------------------------------------
    # 2) Event-driven portfolio backtest
    # -------------------------------------------------------------------------
    trades, diagnostics = run_portfolio_backtest(
        data_by_symbol
    )

    if trades.empty:
        print("⚠️ No accepted trades.")
        print("Diagnostics:", diagnostics)
        return

    trades = trades.sort_values(
        ["EntryTime", "Symbol"]
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 3) Save complete trade log
    # -------------------------------------------------------------------------
    trades.to_csv(
        "lbank_hunter_x_1y_all_trades.csv",
        index=False
    )

    # -------------------------------------------------------------------------
    # 4) Reports
    # -------------------------------------------------------------------------
    print_metrics(
        "📊 PORTFOLIO — NET RESULT AFTER FEES + SLIPPAGE",
        trades
    )

    # Reconstruct gross result for comparison.
    gross_view = trades.copy()
    print_metrics(
        "📊 PORTFOLIO — GROSS RESULT BEFORE FEES + SLIPPAGE",
        gross_view.assign(Net_R=gross_view["Gross_R"])
    )

    # -------------------------------------------------------------------------
    # 5) Diagnostics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🔬 PORTFOLIO DIAGNOSTICS")
    print("=" * 80)

    for k, v in diagnostics.items():
        print(f"{k:35s}: {v}")

    # -------------------------------------------------------------------------
    # 6) Anti-lookahead audit
    # -------------------------------------------------------------------------
    anti_lookahead_audit(
        data_by_symbol,
        trades
    )

    # -------------------------------------------------------------------------
    # 7) Basic target check — NOT optimization
    # -------------------------------------------------------------------------
    m = compute_metrics(trades)

    print("\n" + "=" * 80)
    print("🎯 TARGET CHECK — NO PARAMETER OPTIMIZATION")
    print("=" * 80)

    print(
        f"Win Rate: {m['Win Rate %']:.2f}% "
        f"(target ≈ 70%)"
    )

    print(
        f"Avg Trades/Day: {m['Average Trades/Day']:.2f} "
        f"(target ≈ 3–4)"
    )

    print(
        "R:R is structurally fixed at 1:2 for WIN trades."
    )

    print(
        "\n⚠️ Important: this section does NOT modify parameters "
        "to force the target."
    )

    print("\nFiles:")
    print("  - lbank_hunter_x_1y_all_trades.csv")
    for symbol in data_by_symbol:
        print(f"  - {symbol}_LBank_1H.csv")

    print("\n✅ Backtest completed.")


if __name__ == "__main__":
    main()
