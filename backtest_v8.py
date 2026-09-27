#!/usr/bin/env python3
"""
HUNTER-V5.1 — CROSS-ASSET FLOW-STATE & LIQUIDITY TRANSITION
STRICT CAUSAL XT USDT-M FUTURES BACKTEST ENGINE

Fixes:
- Robust XT Futures REST pagination with compact/verbose payload parsing.
- Exact 15m candle progression without pagination stalls.
- 365-day test + 60-day warmup.
- Closed-candle-only signal generation.
- Causal daily regime features.
- Deterministic cross-sectional candidate ranking.
- Single global non-overlapping position.
- No timeout, no forced end-of-test loss.
- Same-candle SL+TP resolves as LOSS.
- Entry on next available 1H candle open.
- Full portfolio/per-symbol statistics.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# =========================
# CONFIG
# =========================

SYMBOLS = [
    "btc_usdt", "eth_usdt", "sol_usdt", "sui_usdt", "avax_usdt",
    "near_usdt", "ada_usdt", "bnb_usdt", "apt_usdt", "crv_usdt",
    "ondo_usdt", "pendle_usdt", "icp_usdt", "wif_usdt",
]

DATA_DIR = Path("data/xt_futures_v5")

TOTAL_DAYS = 365
WARMUP_DAYS = 60

INITIAL_CAPITAL = 1000.0
MARGIN_PER_TRADE = 100.0
LEVERAGE = 50.0
RR = 2.0

FEE_RATE = 0.0007
SLIPPAGE = 0.0003

INTERVAL = "15m"
INTERVAL_MS = 15 * 60 * 1000
PAGE_LIMIT = 1500

MIN_ROWS = 38000
MIN_SPAN_DAYS = 350.0
MAX_GAPS = 100

REQUEST_TIMEOUT = 20
MAX_RETRIES = 4


# =========================
# ARGUMENTS
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(DATA_DIR))
    return p.parse_args()


# =========================
# XT RESPONSE PARSING
# =========================

def unwrap_rows(payload):
    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        return None

    for key in ("result", "data", "rows", "list"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            nested = unwrap_rows(value)
            if nested is not None:
                return nested

    # Some endpoints may return a single compact kline object.
    if "t" in payload and ("o" in payload or "open" in payload):
        return [payload]

    return None


def parse_kline_row(row):
    if isinstance(row, (list, tuple)):
        if len(row) < 6:
            return None
        raw = {
            "timestamp": row[0],
            "open": row[1],
            "high": row[2],
            "low": row[3],
            "close": row[4],
            "volume": row[5],
        }
    elif isinstance(row, dict):
        raw = {
            "timestamp": row.get(
                "t",
                row.get("time", row.get("timestamp", row.get("ts")))
            ),
            "open": row.get("o", row.get("open")),
            "high": row.get("h", row.get("high")),
            "low": row.get("l", row.get("low")),
            "close": row.get("c", row.get("close")),
            # XT compact response commonly exposes "a" as amount/volume.
            "volume": row.get(
                "a",
                row.get("v", row.get("volume", row.get("amount")))
            ),
        }
    else:
        return None

    try:
        ts = int(float(raw["timestamp"]))
        o = float(raw["open"])
        h = float(raw["high"])
        l = float(raw["low"])
        c = float(raw["close"])
        v = float(raw["volume"])

        # Reject impossible rows.
        if ts <= 0:
            return None
        if not all(np.isfinite([o, h, l, c, v])):
            return None
        if o <= 0 or h <= 0 or l <= 0 or c <= 0:
            return None
        if h < max(o, c, l):
            return None
        if l > min(o, c, h):
            return None
        if v < 0:
            return None

        return [ts, o, h, l, c, v]
    except (TypeError, ValueError, OverflowError):
        return None


def parse_batch(payload, start_ms, end_ms):
    rows = unwrap_rows(payload)
    if rows is None:
        return []

    parsed = []
    for row in rows:
        item = parse_kline_row(row)
        if item is None:
            continue

        ts = item[0]
        if start_ms <= ts <= end_ms:
            parsed.append(item)

    return parsed


# =========================
# CACHE VALIDATION
# =========================

def validate_dataframe(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{symbol}: missing columns {missing}")

    out = df[required].copy()

    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna()
    out = out[out["timestamp"] > 0]
    out = out.sort_values("timestamp")
    out = out.drop_duplicates("timestamp", keep="last")
    out = out.reset_index(drop=True)

    if out.empty:
        raise RuntimeError(f"{symbol}: empty dataframe")

    dt = pd.to_datetime(out["timestamp"], unit="ms", utc=True)
    span_days = (dt.iloc[-1] - dt.iloc[0]).total_seconds() / 86400.0
    diffs = dt.diff().dt.total_seconds().dropna()

    # A gap is greater than one normal 15m interval.
    gaps = int((diffs > 900.0).sum())

    if len(out) < MIN_ROWS:
        raise RuntimeError(
            f"{symbol}: only {len(out)} rows; required >= {MIN_ROWS}"
        )

    if span_days < MIN_SPAN_DAYS:
        raise RuntimeError(
            f"{symbol}: span {span_days:.1f}d; required >= {MIN_SPAN_DAYS:.1f}d"
        )

    if gaps > MAX_GAPS:
        raise RuntimeError(
            f"{symbol}: {gaps} gaps; maximum allowed {MAX_GAPS}"
        )

    return out


def cache_is_valid(path: Path, symbol: str) -> bool:
    if not path.exists() or path.stat().st_size < 1000:
        return False

    try:
        df = pd.read_csv(path)
        validate_dataframe(df, symbol)
        return True
    except Exception:
        return False


# =========================
# XT FUTURES DATA
# =========================

def fetch_xt_futures_data(symbol: str, data_dir: Path) -> pd.DataFrame:
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / f"{symbol.upper()}_15m.csv"

    if cache_is_valid(file_path, symbol):
        df = pd.read_csv(file_path)
        print(f"[CACHE] {symbol.upper()} rows={len(df)}")
        return df

    url = "https://fapi.xt.com/future/market/v1/public/q/kline"

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - int((TOTAL_DAYS + WARMUP_DAYS) * 86400 * 1000)

    # Last fully completed 15m candle.
    current_floor = (now_ms // INTERVAL_MS) * INTERVAL_MS
    final_end_ms = current_floor - 1

    current_start = start_ms
    all_rows = []
    page = 0

    while current_start <= final_end_ms:
        page += 1

        # Request exactly PAGE_LIMIT candle slots.
        requested_end = current_start + PAGE_LIMIT * INTERVAL_MS - 1
        window_end = min(final_end_ms, requested_end)

        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "startTime": int(current_start),
            "endTime": int(window_end),
            "limit": PAGE_LIMIT,
        }

        payload = None
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = requests.get(
                    url,
                    params=params,
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=REQUEST_TIMEOUT,
                )
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}")

                payload = r.json()
                break
            except Exception as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    time.sleep(min(2.0 * attempt, 5.0))

        if payload is None:
            raise RuntimeError(
                f"{symbol}: API request failed page={page}: {last_error}"
            )

        parsed = parse_batch(payload, current_start, window_end)

        # XT can occasionally return the boundary candle from just before
        # startTime. If filtering removes the entire page, do not blindly
        # abort: inspect the raw response and advance by one candle only when
        # the API clearly returned data before the requested interval.
        raw_rows = unwrap_rows(payload) or []

        if not parsed:
            raw_timestamps = []
            for row in raw_rows:
                try:
                    if isinstance(row, (list, tuple)):
                        raw_timestamps.append(int(float(row[0])))
                    elif isinstance(row, dict):
                        raw_timestamps.append(
                            int(float(row.get(
                                "t",
                                row.get(
                                    "time",
                                    row.get("timestamp", row.get("ts"))
                                )
                            )))
                        )
                except Exception:
                    continue

            if raw_timestamps and max(raw_timestamps) < current_start:
                current_start += INTERVAL_MS
                continue

            # If this is the final window and all returned rows are newer than
            # the requested window, we are done. Otherwise this is a real
            # pagination/data-integrity failure.
            if raw_timestamps and min(raw_timestamps) > window_end:
                break

            sample = raw_rows[0] if raw_rows else None
            raise RuntimeError(
                f"{symbol}: parsed batch empty page={page}; "
                f"requested=[{current_start},{window_end}], sample={sample}"
            )

        all_rows.extend(parsed)

        max_ts = max(row[0] for row in parsed)

        # Normal progression is one interval after the newest parsed candle.
        next_start = max_ts + INTERVAL_MS

        # Critical guard: if API response does not move forward, retry the
        # exact window once via a smaller boundary, then fail loudly.
        if next_start <= current_start:
            raise RuntimeError(
                f"{symbol}: pagination stalled page={page}; "
                f"next_start={next_start}, current_start={current_start}"
            )

        print(
            f"{symbol.upper()} page={page} rows={len(parsed)} "
            f"{pd.to_datetime(min(x[0] for x in parsed), unit='ms', utc=True)} "
            f"-> {pd.to_datetime(max_ts, unit='ms', utc=True)}"
        )

        current_start = next_start

        # If the API returned fewer candles than requested, the endpoint may
        # have ignored the exact upper boundary. Continue from the actual
        # newest timestamp instead of stopping prematurely.
        time.sleep(0.05)

        # Safety against pathological APIs.
        if page > 100:
            raise RuntimeError(f"{symbol}: excessive pagination pages")

    if not all_rows:
        raise RuntimeError(f"{symbol}: no klines fetched")

    df = pd.DataFrame(
        all_rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )

    df = validate_dataframe(df, symbol)

    # Remove any currently incomplete candle.
    now_ms = int(time.time() * 1000)
    current_floor = (now_ms // INTERVAL_MS) * INTERVAL_MS
    df = df[df["timestamp"] < current_floor].copy()

    df = validate_dataframe(df, symbol)

    dt = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    print(
        f"[VALIDATION] {symbol.upper()} rows={len(df)} "
        f"first={dt.iloc[0]} last={dt.iloc[-1]} "
        f"span={(dt.iloc[-1] - dt.iloc[0]).total_seconds()/86400.0:.1f}d"
    )

    df.to_csv(file_path, index=False)
    return df


# =========================
# RESAMPLING
# =========================

def load_and_resample(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    x = df.copy()

    x["timestamp_dt"] = pd.to_datetime(
        x["timestamp"], unit="ms", utc=True
    )

    x = (
        x.set_index("timestamp_dt")
        .sort_index()
        .loc[:, ["open", "high", "low", "close", "volume"]]
    )

    x = x[~x.index.duplicated(keep="last")]

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    # UTC boundaries are deterministic. 1H/4H/1D bars are composed only from
    # completed 15m candles already present in the source dataset.
    h1 = x.resample("1h", label="left", closed="left").agg(agg).dropna()
    h4 = x.resample("4h", label="left", closed="left").agg(agg).dropna()
    d1 = x.resample("1d", label="left", closed="left").agg(agg).dropna()

    return {
        "15m": x.copy(),
        "1h": h1,
        "4h": h4,
        "1d": d1,
    }


# =========================
# FEATURES
# =========================

def calculate_features(dfs: dict[str, pd.DataFrame]) -> None:
    h1 = dfs["1h"].copy()

    prev_close = h1["close"].shift(1)

    h1["return_1h"] = h1["close"].pct_change(1)
    h1["return_4h"] = h1["close"].pct_change(4)
    h1["return_12h"] = h1["close"].pct_change(12)
    h1["return_24h"] = h1["close"].pct_change(24)
    h1["return_72h"] = h1["close"].pct_change(72)

    hl_range = (h1["high"] - h1["low"]).replace(0, np.nan)
    clv = (
        (h1["close"] - h1["low"])
        - (h1["high"] - h1["close"])
    ) / hl_range

    h1["svp"] = (clv.fillna(0.0) * h1["volume"]).astype(float)

    # All flow values are strictly from completed bars before the signal bar.
    h1["svp_1h"] = h1["svp"].shift(1)
    h1["svp_4h"] = h1["svp"].shift(1).rolling(4, min_periods=4).sum()
    h1["svp_12h"] = h1["svp"].shift(1).rolling(12, min_periods=12).sum()

    tr = pd.concat(
        [
            h1["high"] - h1["low"],
            (h1["high"] - prev_close).abs(),
            (h1["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    h1["tr"] = tr

    h1["atr14"] = tr.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    dollar_vol = (h1["close"] * h1["volume"]).replace(0, np.nan)

    h1["amihud"] = (
        h1["return_1h"].abs() / dollar_vol
    ).shift(1)

    vol_median = (
        h1["volume"]
        .shift(1)
        .rolling(24, min_periods=24)
        .median()
        .replace(0, np.nan)
    )

    h1["vol_shock"] = h1["volume"].shift(1) / vol_median

    range_median = (
        tr.shift(1)
        .rolling(24, min_periods=24)
        .median()
        .replace(0, np.nan)
    )

    h1["range_shock"] = tr.shift(1) / range_median

    h1["liquidity_stress_score"] = (
        h1["vol_shock"] * h1["range_shock"]
    )

    rv_1h = (
        h1["return_1h"]
        .shift(1)
        .rolling(24, min_periods=24)
        .std()
    )

    rv_4h = (
        h1["return_1h"]
        .shift(1)
        .rolling(96, min_periods=96)
        .std()
    )

    h1["rv_ratio"] = rv_1h / rv_4h.replace(0, np.nan)

    # Percentile is based only on observations before the current signal bar.
    h1["vol_percentile"] = (
        rv_1h
        .rolling(168, min_periods=168)
        .rank(pct=True)
    )

    # Daily regime: use only the last COMPLETED daily candle.
    d1 = dfs["1d"].copy()

    d1["ema50"] = d1["close"].shift(1).ewm(
        span=50,
        adjust=False,
        min_periods=50,
    ).mean()

    d1["ema200"] = d1["close"].shift(1).ewm(
        span=200,
        adjust=False,
        min_periods=200,
    ).mean()

    d1["ema50_slope"] = d1["ema50"].diff()

    # Use previous completed daily close for regime decisions.
    d1["regime_close"] = d1["close"].shift(1)

    dfs["1h"] = h1
    dfs["1d"] = d1


# =========================
# BACKTEST
# =========================

def run_backtest(
    all_symbol_data: dict[str, dict[str, pd.DataFrame]]
) -> tuple[list[dict], dict | None]:

    all_times = sorted(
        set().union(
            *(dfs["1h"].index for dfs in all_symbol_data.values())
        )
    )

    active_position = None
    trades: list[dict] = []

    cooldowns = {sym: 0 for sym in all_symbol_data}

    for ts in all_times:

        # Cooldown is measured in portfolio clock hours.
        for sym in cooldowns:
            if cooldowns[sym] > 0:
                cooldowns[sym] -= 1

        closed_this_bar = False

        # ---------------------------------
        # 1) Manage existing position
        # ---------------------------------
        if active_position is not None:
            sym = active_position["symbol"]
            h1 = all_symbol_data[sym]["1h"]

            if ts in h1.index and ts >= active_position["entry_ts"]:
                bar = h1.loc[ts]

                high = float(bar["high"])
                low = float(bar["low"])

                side = active_position["side"]
                sl = active_position["sl"]
                tp = active_position["tp"]
                entry = active_position["entry"]

                if side == "LONG":
                    hit_sl = low <= sl
                    hit_tp = high >= tp
                else:
                    hit_sl = high >= sl
                    hit_tp = low <= tp

                if hit_sl or hit_tp:

                    # Conservative same-candle rule:
                    # if both levels are touched, LOSS.
                    if hit_sl and hit_tp:
                        outcome = "LOSS"
                        exit_price = sl
                    elif hit_sl:
                        outcome = "LOSS"
                        exit_price = sl
                    else:
                        outcome = "WIN"
                        exit_price = tp

                    notional = MARGIN_PER_TRADE * LEVERAGE

                    if side == "LONG":
                        gross = (
                            (exit_price - entry) / entry
                        ) * notional
                    else:
                        gross = (
                            (entry - exit_price) / entry
                        ) * notional

                    fees = notional * FEE_RATE * 2.0
                    pnl = gross - fees

                    trades.append(
                        {
                            "symbol": sym,
                            "side": side,
                            "entry_ts": active_position["entry_ts"],
                            "exit_ts": ts,
                            "outcome": outcome,
                            "pnl": float(pnl),
                            "entry": float(entry),
                            "exit": float(exit_price),
                            "sl": float(sl),
                            "tp": float(tp),
                        }
                    )

                    cooldowns[sym] = 3
                    active_position = None
                    closed_this_bar = True

        # Never open another position on the same candle in which the
        # previous position's outcome became known.
        if closed_this_bar:
            continue

        # ---------------------------------
        # 2) Generate new signal
        # ---------------------------------
        if active_position is not None:
            continue

        cross_returns = {}

        for sym, dfs in all_symbol_data.items():
            h1 = dfs["1h"]

            if ts not in h1.index:
                continue

            val = h1.loc[ts, "return_24h"]

            if pd.notna(val) and np.isfinite(float(val)):
                cross_returns[sym] = float(val)

        if len(cross_returns) < 5:
            continue

        ranks = pd.Series(cross_returns).rank(
            method="average",
            pct=True,
        )

        candidates = []

        for sym, dfs in all_symbol_data.items():

            if cooldowns[sym] > 0:
                continue

            h1 = dfs["1h"]
            d1 = dfs["1d"]

            if ts not in h1.index:
                continue

            h1_idx = h1.index.get_loc(ts)

            if h1_idx < 100:
                continue

            # Most recent COMPLETED daily bar strictly before the current
            # 1H signal candle.
            d1_idx = d1.index.searchsorted(ts, side="right") - 1

            if d1_idx < 1:
                continue

            drow = d1.iloc[d1_idx]
            hrow = h1.iloc[h1_idx]

            regime_close = drow.get("regime_close", np.nan)
            ema50 = drow.get("ema50", np.nan)
            ema200 = drow.get("ema200", np.nan)
            slope = drow.get("ema50_slope", np.nan)

            atr = hrow.get("atr14", np.nan)
            svp4 = hrow.get("svp_4h", np.nan)
            vol_pct = hrow.get("vol_percentile", np.nan)

            if not all(
                np.isfinite(float(x))
                for x in [regime_close, ema50, ema200, slope, atr, svp4, vol_pct]
            ):
                continue

            if float(atr) <= 0:
                continue

            daily_long = (
                regime_close > ema200
                and ema50 > ema200
                and slope > 0
            )

            daily_short = (
                regime_close < ema200
                and ema50 < ema200
                and slope < 0
            )

            rank_val = float(ranks.get(sym, np.nan))

            if not np.isfinite(rank_val):
                continue

            # Flow-continuation setup.
            if (
                daily_long
                and rank_val >= 0.70
                and svp4 > 0
                and vol_pct < 0.85
            ):
                sl = float(hrow["low"] - 0.5 * atr)

                candidates.append(
                    {
                        "symbol": sym,
                        "side": "LONG",
                        "rank": rank_val,
                        "flow": float(svp4),
                        "sl": sl,
                        "atr": float(atr),
                    }
                )

            elif (
                daily_short
                and rank_val <= 0.30
                and svp4 < 0
                and vol_pct < 0.85
            ):
                sl = float(hrow["high"] + 0.5 * atr)

                candidates.append(
                    {
                        "symbol": sym,
                        "side": "SHORT",
                        "rank": rank_val,
                        "flow": float(svp4),
                        "sl": sl,
                        "atr": float(atr),
                    }
                )

        if not candidates:
            continue

        # Deterministic strongest-momentum selection.
        # This replaces arbitrary candidates[0].
        candidates.sort(
            key=lambda x: (
                x["rank"] if x["side"] == "LONG"
                else 1.0 - x["rank"],
                abs(x["flow"]),
                -x["atr"],
            ),
            reverse=True,
        )

        selected = candidates[0]
        sym = selected["symbol"]
        side = selected["side"]
        sl = selected["sl"]

        h1 = all_symbol_data[sym]["1h"]

        future_indices = h1.index[h1.index > ts]

        if len(future_indices) == 0:
            continue

        entry_ts = future_indices[0]

        # The entry candle itself must be complete enough to provide its open;
        # no high/low/close from that candle is used to create the signal.
        raw_open = float(h1.loc[entry_ts, "open"])

        if side == "LONG":
            entry = raw_open * (1.0 + SLIPPAGE)
            risk = entry - sl
        else:
            entry = raw_open * (1.0 - SLIPPAGE)
            risk = sl - entry

        if not np.isfinite(risk) or risk <= 0:
            continue

        tp = (
            entry + RR * risk
            if side == "LONG"
            else entry - RR * risk
        )

        active_position = {
            "symbol": sym,
            "side": side,
            "entry_ts": entry_ts,
            "entry": float(entry),
            "sl": float(sl),
            "tp": float(tp),
        }

    return trades, active_position


# =========================
# REPORTING
# =========================

def calculate_metrics(trades: list[dict]):
    ordered = sorted(
        trades,
        key=lambda t: pd.Timestamp(t["exit_ts"]),
    )

    total = len(ordered)
    wins = sum(t["outcome"] == "WIN" for t in ordered)
    losses = total - wins

    wr = wins / total * 100.0 if total else 0.0

    gross_profit = sum(
        t["pnl"] for t in ordered if t["pnl"] > 0
    )

    gross_loss = abs(
        sum(t["pnl"] for t in ordered if t["pnl"] < 0)
    )

    if gross_loss > 0:
        pf = gross_profit / gross_loss
    elif gross_profit > 0:
        pf = float("inf")
    else:
        pf = 0.0

    net = sum(t["pnl"] for t in ordered)

    equity = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    max_dd_dollar = 0.0
    max_dd_pct = 0.0

    streak = 0
    max_streak = 0

    for t in ordered:
        equity += t["pnl"]

        if equity > peak:
            peak = equity

        dd_dollar = peak - equity
        dd_pct = (
            dd_dollar / peak * 100.0
            if peak > 0 else 0.0
        )

        max_dd_dollar = max(max_dd_dollar, dd_dollar)
        max_dd_pct = max(max_dd_pct, dd_pct)

        if t["outcome"] == "LOSS":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "wr": wr,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "pf": pf,
        "net": net,
        "max_dd_dollar": max_dd_dollar,
        "max_dd_pct": max_dd_pct,
        "max_streak": max_streak,
    }


def print_report(
    trades: list[dict],
    open_pos: dict | None,
    all_symbol_data: dict[str, dict[str, pd.DataFrame]],
) -> None:

    metrics = calculate_metrics(trades)

    all_times = sorted(
        set().union(
            *(dfs["1h"].index for dfs in all_symbol_data.values())
        )
    )

    if len(all_times) >= 2:
        span_days = (
            all_times[-1] - all_times[0]
        ).total_seconds() / 86400.0
    else:
        span_days = 1.0

    tpd = (
        metrics["total"] / span_days
        if span_days > 0 else 0.0
    )

    print("\n" + "=" * 30 + " PER-SYMBOL BREAKDOWN " + "=" * 30)
    print(
        f"{'Symbol':<12} | {'Trades':<7} | {'Wins':<6} | "
        f"{'Losses':<7} | {'WR':<9} | {'PnL':<12}"
    )
    print("-" * 70)

    for sym in SYMBOLS:
        st = [t for t in trades if t["symbol"] == sym]
        total = len(st)
        wins = sum(t["outcome"] == "WIN" for t in st)
        losses = total - wins
        wr = wins / total * 100.0 if total else 0.0
        pnl = sum(t["pnl"] for t in st)

        print(
            f"{sym.upper():<12} | {total:<7} | {wins:<6} | "
            f"{losses:<7} | {wr:>6.2f}% | ${pnl:>10.2f}"
        )

    print("\n" + "=" * 30 + " PORTFOLIO BACKTEST RESULTS " + "=" * 30)

    print(f"Total Trades         : {metrics['total']}")
    print(f"Wins                 : {metrics['wins']}")
    print(f"Losses               : {metrics['losses']}")
    print(f"Win Rate             : {metrics['wr']:.2f}%")
    print(f"Gross Profit         : ${metrics['gross_profit']:,.2f}")
    print(f"Gross Loss           : ${metrics['gross_loss']:,.2f}")
    print(f"Profit Factor        : {metrics['pf']:.2f}")
    print(f"Net PnL              : ${metrics['net']:,.2f}")
    print(
        f"Max Drawdown ($)     : "
        f"${metrics['max_dd_dollar']:,.2f}"
    )
    print(
        f"Max Drawdown (%)     : "
        f"{metrics['max_dd_pct']:.2f}%"
    )
    print(
        f"Max Consecutive Loss : "
        f"{metrics['max_streak']}"
    )
    print(f"Trades Per Day       : {tpd:.4f}")
    print(
        f"Open Positions At End: "
        f"{1 if open_pos is not None else 0}"
    )

    print("=" * 68)

    eligible = (
        metrics["wr"] > 50.0
        and metrics["pf"] > 1.20
        and metrics["net"] > 0
        and metrics["max_streak"] <= 4
        and metrics["total"] >= 150
    )

    print("\n" + "=" * 30 + " OOS ELIGIBILITY " + "=" * 30)
    print(
        f"Win Rate        : {metrics['wr']:.2f}% "
        f"| requirement > 50% "
        f"| {'PASS' if metrics['wr'] > 50 else 'FAIL'}"
    )
    print(
        f"Profit Factor   : {metrics['pf']:.2f} "
        f"| requirement > 1.20 "
        f"| {'PASS' if metrics['pf'] > 1.20 else 'FAIL'}"
    )
    print(
        f"Net PnL         : ${metrics['net']:,.2f} "
        f"| requirement > $0 "
        f"| {'PASS' if metrics['net'] > 0 else 'FAIL'}"
    )
    print(
        f"Max Loss Streak : {metrics['max_streak']} "
        f"| requirement <= 4 "
        f"| {'PASS' if metrics['max_streak'] <= 4 else 'FAIL'}"
    )
    print(
        f"Sample Size     : {metrics['total']} "
        f"| requirement >= 150 "
        f"| {'PASS' if metrics['total'] >= 150 else 'FAIL'}"
    )
    print(
        f"OOS ELIGIBLE    : "
        f"{'TRUE (ELIGIBLE)' if eligible else 'FALSE (REJECTED)'}"
    )
    print("=" * 68)


# =========================
# MAIN
# =========================

def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HUNTER-V5.1 — STRICT XT FUTURES FLOW-STATE BACKTEST")
    print("=" * 70)
    print(
        f"Universe={len(SYMBOLS)} | Test={TOTAL_DAYS}d | "
        f"Warmup={WARMUP_DAYS}d | RR=1:{RR}"
    )

    all_symbol_data = {}

    for symbol in SYMBOLS:
        try:
            raw = fetch_xt_futures_data(symbol, data_dir)
            dfs = load_and_resample(raw)
            calculate_features(dfs)
            all_symbol_data[symbol] = dfs
        except Exception as exc:
            print(
                f"\n[ABORT] Critical error for mandatory symbol "
                f"{symbol.upper()}: {exc}"
            )
            print(
                "[ABORT] Complete universe required. "
                "Backtest NOT executed."
            )
            sys.exit(1)

    print(
        f"\n[DATA] Validated {len(all_symbol_data)}/"
        f"{len(SYMBOLS)} symbols."
    )

    trades, open_pos = run_backtest(all_symbol_data)

    print_report(
        trades,
        open_pos,
        all_symbol_data,
    )


if __name__ == "__main__":
    main()
