#!/usr/bin/env python3
"""
HUNTER-V9.1 — AUDITED DERIVATIVES REGIME / MICROSTRUCTURE ENGINE

Purpose:
- XT USDT-M Futures 15m data
- strict causal feature construction
- futures/spot basis + volume-ratio diagnostics
- factor discovery on TRAIN only
- strategy selection on TRAIN/VALIDATION only
- untouched OOS evaluation
- fixed RR 1:2
- no timeout / no BE / no trailing
- no overlapping positions
- same-candle SL+TP = LOSS
- unresolved position at dataset end = OPEN

Important:
This version does NOT pretend that a simple technical setup is a
"microstructure edge". It first measures the available causal factors.
If the discovered factors do not support a robust rule, the final
acceptance gate remains FALSE.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests


SYMBOLS = [
    "btc_usdt", "eth_usdt", "sol_usdt", "sui_usdt", "avax_usdt",
    "near_usdt", "ada_usdt", "bnb_usdt", "apt_usdt", "crv_usdt",
    "ondo_usdt", "pendle_usdt", "icp_usdt", "wif_usdt",
]

FUTURES_URL = "https://fapi.xt.com/future/market/v1/public/q/kline"

DATA_DIR = Path("data/xt_futures_v9_1")
TOTAL_DAYS = 365
WARMUP_DAYS = 60
INTERVAL_MS = 15 * 60 * 1000
KLINE_LIMIT = 1500

INITIAL_CAPITAL = 1000.0
MARGIN_PER_TRADE = 100.0
LEVERAGE = 50.0
RR = 2.0

FEE_RATE = 0.0007
SLIPPAGE = 0.0003

MIN_DATA_ROWS = 38000
MAX_GAP_MINUTES = 15.0

# Factor-discovery horizons on 1H data.
HORIZONS = (1, 4, 8, 24)

# OOS is never used to choose parameters.
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20

# The strategy family is deliberately small and predeclared.
# No post-OOS threshold hunting is permitted.
TOP_RS = 3
MIN_COMMON_SYMBOLS_FRAC = 0.70
BB_LOOKBACK = 20
BB_PERCENTILE_LOOKBACK = 96
ATR_PERIOD = 14

# Acceptance requirements.
MIN_OOS_TRADES = 150
MIN_OOS_WR = 50.0
MIN_OOS_PF = 1.20
MAX_OOS_LOSS_STREAK = 4


@dataclass(frozen=True)
class Split:
    start: pd.Timestamp
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    end: pd.Timestamp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--refresh", action="store_true")
    return p.parse_args()


def _extract_payload(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("result", "data", "rows", "items", "list"):
            if key in obj:
                found = _extract_payload(obj[key])
                if found is not None:
                    return found
    return None


def _parse_kline_row(row):
    if isinstance(row, (list, tuple)):
        if len(row) < 6:
            return None
        ts, o, h, l, c, v = row[:6]
    elif isinstance(row, dict):
        def pick(*keys):
            for k in keys:
                if k in row and row[k] is not None:
                    return row[k]
            raise KeyError(keys)

        ts = pick("t", "time", "timestamp")
        o = pick("o", "open")
        h = pick("h", "high")
        l = pick("l", "low")
        c = pick("c", "close")
        # XT responses seen in prior validated runs used compact volume keys.
        # Prefer v, then volume, then a. Never silently invent volume.
        v = pick("v", "volume", "a")
    else:
        return None

    ts = int(float(ts))
    vals = [float(o), float(h), float(l), float(c), float(v)]
    if not all(np.isfinite(x) for x in vals):
        return None
    if vals[1] < vals[2] or vals[1] < max(vals[0], vals[3]) or vals[2] > min(vals[0], vals[3]):
        return None
    if ts <= 0:
        return None
    return [ts, *vals]


def _validate_15m(df: pd.DataFrame, target_start_ms: int, final_end_ms: int, symbol: str):
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    if any(c not in df.columns for c in required):
        raise RuntimeError(f"{symbol}: missing required columns")

    df = df[required].copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    for c in required[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().copy()

    df["timestamp"] = df["timestamp"].astype(np.int64)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    # Only complete candles inside the requested historical range.
    df = df[(df["timestamp"] >= target_start_ms) & (df["timestamp"] <= final_end_ms)].copy()
    if df.empty:
        raise RuntimeError(f"{symbol}: no rows in requested range")

    current_floor = (int(datetime.now(timezone.utc).timestamp() * 1000) // INTERVAL_MS) * INTERVAL_MS
    df = df[df["timestamp"] < current_floor].copy()

    if df.empty:
        raise RuntimeError(f"{symbol}: all rows are incomplete/current")

    diffs = df["timestamp"].diff().dropna()
    gap_count = int((diffs > INTERVAL_MS).sum())
    duplicate_count = int(df["timestamp"].duplicated().sum())
    monotonic = bool(df["timestamp"].is_monotonic_increasing)

    first_ts = int(df["timestamp"].iloc[0])
    last_ts = int(df["timestamp"].iloc[-1])
    span_days = (last_ts - first_ts) / 86400000.0

    # Allow only a small edge truncation caused by request alignment.
    start_tolerance = 2 * 3600000
    end_tolerance = 2 * 3600000

    print(
        f"[VALIDATE] {symbol.upper()} rows={len(df)} "
        f"first={pd.to_datetime(first_ts, unit='ms', utc=True)} "
        f"last={pd.to_datetime(last_ts, unit='ms', utc=True)} "
        f"span={span_days:.1f}d gaps={gap_count} "
        f"duplicates={duplicate_count} monotonic={monotonic}"
    )

    if len(df) < MIN_DATA_ROWS:
        raise RuntimeError(f"{symbol}: too few rows: {len(df)}")
    if span_days < TOTAL_DAYS + WARMUP_DAYS - 5:
        raise RuntimeError(f"{symbol}: insufficient span: {span_days:.1f}d")
    if gap_count > 0:
        raise RuntimeError(f"{symbol}: data gaps detected: {gap_count}")
    if duplicate_count != 0 or not monotonic:
        raise RuntimeError(f"{symbol}: duplicate/non-monotonic timestamps")
    if first_ts > target_start_ms + start_tolerance:
        raise RuntimeError(f"{symbol}: starts too late")
    if last_ts < final_end_ms - end_tolerance:
        raise RuntimeError(f"{symbol}: ends too early")

    return df.reset_index(drop=True)


def fetch_xt_futures_data(symbol: str, data_dir: Path, refresh: bool = False) -> pd.DataFrame:
    file_path = data_dir / f"{symbol.upper()}_15m.csv"

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    target_start_ms = now_ms - int((TOTAL_DAYS + WARMUP_DAYS) * 86400 * 1000)
    final_end_ms = (now_ms // INTERVAL_MS) * INTERVAL_MS - 1

    if file_path.exists() and file_path.stat().st_size > 1000 and not refresh:
        try:
            cached = pd.read_csv(file_path)
            validated = _validate_15m(cached, target_start_ms, final_end_ms, symbol)
            print(f"[CACHE] {symbol.upper()} accepted")
            return validated
        except Exception as exc:
            print(f"[CACHE] {symbol.upper()} invalid; refetching: {exc}")

    data_dir.mkdir(parents=True, exist_ok=True)
    cursor = target_start_ms
    all_rows = []
    page = 0
    session = requests.Session()
    session.headers.update({"User-Agent": "HUNTER-V9.1/1.0"})

    while cursor <= final_end_ms:
        page += 1
        window_end = min(final_end_ms, cursor + KLINE_LIMIT * INTERVAL_MS - 1)
        params = {
            "symbol": symbol,
            "interval": "15m",
            "startTime": cursor,
            "endTime": window_end,
            "limit": KLINE_LIMIT,
        }

        payload = None
        last_error = None
        for attempt in range(3):
            try:
                r = session.get(FUTURES_URL, params=params, timeout=20)
                r.raise_for_status()
                payload = _extract_payload(r.json())
                if payload is None:
                    raise RuntimeError("API payload has no list")
                break
            except Exception as exc:
                last_error = exc
                time.sleep(1.0 + attempt)

        if payload is None:
            raise RuntimeError(f"{symbol}: page {page} failed: {last_error}")

        parsed = []
        for row in payload:
            try:
                parsed_row = _parse_kline_row(row)
                if parsed_row is not None:
                    ts = parsed_row[0]
                    if cursor <= ts <= window_end:
                        parsed.append(parsed_row)
            except Exception:
                continue

        if parsed:
            all_rows.extend(parsed)
            max_ts = max(x[0] for x in parsed)
            next_cursor = max_ts + INTERVAL_MS
        else:
            # A page can contain stale/out-of-window rows near the API's
            # historical boundary. Do not treat that as a valid page.
            returned_ts = []
            for row in payload:
                try:
                    pr = _parse_kline_row(row)
                    if pr is not None:
                        returned_ts.append(pr[0])
                except Exception:
                    pass

            if returned_ts and max(returned_ts) < cursor:
                break
            raise RuntimeError(
                f"{symbol}: page {page} produced no rows in requested window "
                f"[{cursor}, {window_end}]"
            )

        if next_cursor <= cursor:
            raise RuntimeError(f"{symbol}: pagination stalled at page {page}")

        cursor = next_cursor
        if page % 10 == 0:
            print(f"[FETCH] {symbol.upper()} page={page} rows={len(all_rows)}")
        time.sleep(0.05)

        if cursor > final_end_ms:
            break

    if not all_rows:
        raise RuntimeError(f"{symbol}: no klines fetched")

    raw = pd.DataFrame(
        all_rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    validated = _validate_15m(raw, target_start_ms, final_end_ms, symbol)
    validated.to_csv(file_path, index=False)
    return validated


def resample_complete(df: pd.DataFrame, rule: str, expected_bars: int) -> pd.DataFrame:
    x = df.copy()
    x["timestamp_dt"] = pd.to_datetime(x["timestamp"], unit="ms", utc=True)
    x = x.set_index("timestamp_dt").sort_index()
    x = x[~x.index.duplicated(keep="last")]

    # Left-closed, right-labelled intervals:
    # [10:00,11:00) is labelled 11:00 and is fully known at 11:00.
    out = x.resample(rule, closed="left", label="right").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        count=("close", "count"),
    )

    out = out[out["count"] == expected_bars].drop(columns="count")
    return out


def load_and_resample(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    x = df.copy()
    x["timestamp_dt"] = pd.to_datetime(x["timestamp"], unit="ms", utc=True)
    x = x.set_index("timestamp_dt").sort_index()
    x = x[~x.index.duplicated(keep="last")]

    h1 = resample_complete(df, "1h", 4)
    h4 = resample_complete(df, "4h", 16)
    d1 = resample_complete(df, "1d", 96)

    return {"15m": x, "1h": h1, "4h": h4, "1d": d1}


def add_features(dfs: dict[str, pd.DataFrame]):
    h1 = dfs["1h"].copy()
    h4 = dfs["4h"].copy()
    d1 = dfs["1d"].copy()

    # 1H causal features. At timestamp t, all values refer to the
    # completed bar ending at t. Entry, however, occurs at t+1H.
    prev_close = h1["close"].shift(1)
    tr = pd.concat(
        [
            h1["high"] - h1["low"],
            (h1["high"] - prev_close).abs(),
            (h1["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    h1["tr"] = tr
    h1["atr14"] = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean()

    h1["ret_1h"] = h1["close"].pct_change(1)
    h1["ret_4h"] = h1["close"].pct_change(4)
    h1["ret_8h"] = h1["close"].pct_change(8)
    h1["ret_24h"] = h1["close"].pct_change(24)

    atr_mean = h1["atr14"].rolling(24, min_periods=24).mean()
    h1["norm_ret_24h"] = (
        (h1["close"] - h1["close"].shift(24))
        / atr_mean.replace(0, np.nan)
    )

    # Previous completed closes avoid using the current bar in the
    # compression statistic itself.
    pc = h1["close"].shift(1)
    std20 = pc.rolling(BB_LOOKBACK, min_periods=BB_LOOKBACK).std()
    sma20 = pc.rolling(BB_LOOKBACK, min_periods=BB_LOOKBACK).mean()
    bb_width = 4.0 * std20 / sma20.replace(0, np.nan)

    def last_rank_pct(x):
        if len(x) == 0:
            return np.nan
        return pd.Series(x).rank(pct=True).iloc[-1]

    h1["bb_width_pct"] = bb_width.rolling(
        BB_PERCENTILE_LOOKBACK,
        min_periods=BB_PERCENTILE_LOOKBACK,
    ).apply(last_rank_pct, raw=False)

    h1["vol_expansion"] = tr / h1["atr14"].replace(0, np.nan)

    # Volume shock is a causal proxy; no historical OI is fabricated.
    vol_med = h1["volume"].rolling(24, min_periods=24).median().shift(1)
    h1["volume_shock"] = h1["volume"] / vol_med.replace(0, np.nan)

    # Confirmed 4H pivots. Candidate pivot is t-2 and requires bars
    # t-4,t-3,t-1,t to confirm. Thus the value first appears at t.
    hh = h4["high"]
    ll = h4["low"]
    h4["pivot_high"] = h4["high"].shift(2).where(
        (hh.shift(2) > hh.shift(4))
        & (hh.shift(2) > hh.shift(3))
        & (hh.shift(2) > hh.shift(1))
        & (hh.shift(2) > hh)
    )
    h4["pivot_low"] = h4["low"].shift(2).where(
        (ll.shift(2) < ll.shift(4))
        & (ll.shift(2) < ll.shift(3))
        & (ll.shift(2) < ll.shift(1))
        & (ll.shift(2) < ll)
    )
    h4["swing_high"] = h4["pivot_high"].ffill()
    h4["swing_low"] = h4["pivot_low"].ffill()

    h4["atr4"] = (
        pd.concat(
            [
                h4["high"] - h4["low"],
                (h4["high"] - h4["close"].shift(1)).abs(),
                (h4["low"] - h4["close"].shift(1)).abs(),
            ],
            axis=1,
        )
        .max(axis=1)
        .ewm(alpha=1 / 14, adjust=False, min_periods=14)
        .mean()
    )

    # Daily regime is explicitly based on prior completed daily close.
    # At a daily timestamp t, the EMA describes the history through t-1.
    prior_dclose = d1["close"].shift(1)
    d1["ema50"] = prior_dclose.ewm(span=50, adjust=False, min_periods=50).mean()
    d1["ema200"] = prior_dclose.ewm(span=200, adjust=False, min_periods=200).mean()
    d1["ema50_slope"] = d1["ema50"].diff()

    # Store back.
    dfs["1h"] = h1
    dfs["4h"] = h4
    dfs["1d"] = d1


def align_asof_row(df: pd.DataFrame, ts: pd.Timestamp):
    # DataFrame indices are completion timestamps. right search means
    # the bar ending exactly at ts is eligible.
    if df.empty:
        return None
    idx = df.index.searchsorted(ts, side="right") - 1
    if idx < 0:
        return None
    return df.iloc[idx]


def build_basis_features(
    futures: dict[str, dict[str, pd.DataFrame]],
    data_dir: Path,
) -> None:
    """
    Optional futures/spot basis research.

    XT spot history is not used as the primary signal. If a valid spot
    collector is unavailable, basis is explicitly marked unavailable.
    This function intentionally does not manufacture spot/OI/funding data.
    """
    # The current XT public futures collector is mandatory. Historical
    # spot endpoint availability is exchange/API-version dependent.
    # We therefore leave basis unavailable unless a validated spot file
    # already exists in data_dir/spot.
    spot_dir = data_dir / "spot"
    if not spot_dir.exists():
        print("[FACTOR] Spot history directory not present; basis factors unavailable.")
        return

    for sym in SYMBOLS:
        spot_file = spot_dir / f"{sym.upper()}_15m.csv"
        if not spot_file.exists():
            continue

        try:
            spot = pd.read_csv(spot_file)
            required = {"timestamp", "close", "volume"}
            if not required.issubset(spot.columns):
                print(f"[FACTOR] Spot file incomplete: {spot_file}")
                continue

            spot["timestamp_dt"] = pd.to_datetime(spot["timestamp"], unit="ms", utc=True)
            spot = spot.set_index("timestamp_dt").sort_index()
            fut = futures[sym]["15m"].copy()

            # Align exact 15m timestamps only.
            f = fut[["close", "volume"]].rename(
                columns={"close": "fut_close", "volume": "fut_volume"}
            )
            s = spot[["close", "volume"]].rename(
                columns={"close": "spot_close", "volume": "spot_volume"}
            )
            z = f.join(s, how="inner")
            z["basis_pct"] = z["fut_close"] / z["spot_close"] - 1.0
            z["volume_ratio"] = z["fut_volume"] / z["spot_volume"].replace(0, np.nan)

            h1 = z.resample("1h", closed="left", label="right").agg(
                fut_close=("fut_close", "last"),
                spot_close=("spot_close", "last"),
                basis_pct=("basis_pct", "last"),
                volume_ratio=("volume_ratio", "mean"),
            )
            h1.to_csv(spot_dir / f"{sym.upper()}_basis_1h.csv")
            print(f"[FACTOR] Basis prepared: {sym.upper()} rows={len(h1)}")
        except Exception as exc:
            print(f"[FACTOR] Basis failed for {sym.upper()}: {exc}")


def future_return_series(h1: pd.DataFrame, horizon: int) -> pd.Series:
    # At completion timestamp t, this is the close-to-close return
    # from t to t+horizon. It is NEVER used as an input feature.
    return h1["close"].shift(-horizon) / h1["close"] - 1.0


def spearman_ic(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    z = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(z) < 30:
        return np.nan, len(z)
    return float(z.iloc[:, 0].corr(z.iloc[:, 1], method="spearman")), len(z)


def factor_discovery(
    all_symbol_data: dict[str, dict[str, pd.DataFrame]],
    start: pd.Timestamp,
    end: pd.Timestamp,
):
    """
    Factor audit only.

    It computes forward returns after the factor is known. It does not
    feed future returns back into the strategy.
    """
    records = []

    feature_names = [
        "ret_1h",
        "ret_4h",
        "ret_8h",
        "ret_24h",
        "norm_ret_24h",
        "bb_width_pct",
        "vol_expansion",
        "volume_shock",
    ]

    for sym, dfs in all_symbol_data.items():
        h1 = dfs["1h"].loc[start:end].copy()
        for h in HORIZONS:
            y = future_return_series(dfs["1h"], h).loc[start:end]
            for feature in feature_names:
                ic, n = spearman_ic(h1[feature], y)
                if np.isfinite(ic):
                    records.append(
                        {
                            "symbol": sym,
                            "horizon_h": h,
                            "feature": feature,
                            "spearman_ic": ic,
                            "abs_ic": abs(ic),
                            "n": n,
                        }
                    )

    audit = pd.DataFrame(records)
    if audit.empty:
        print("[FACTOR] No valid factor observations.")
        return audit

    summary = (
        audit.groupby(["feature", "horizon_h"], as_index=False)
        .agg(
            median_ic=("spearman_ic", "median"),
            median_abs_ic=("abs_ic", "median"),
            mean_ic=("spearman_ic", "mean"),
            symbols=("symbol", "nunique"),
            observations=("n", "sum"),
        )
        .sort_values(["median_abs_ic", "observations"], ascending=[False, False])
    )

    print("\n================ FACTOR DISCOVERY (TRAIN) ================")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("==========================================================")
    return audit


def _regime_and_signal(
    symbol: str,
    ts: pd.Timestamp,
    data: dict[str, pd.DataFrame],
    rs_map: dict[str, float],
):
    h1 = data["1h"]
    h4 = data["4h"]
    d1 = data["1d"]

    if ts not in h1.index:
        return None

    h1_row = h1.loc[ts]
    h4_row = align_asof_row(h4, ts)
    d1_row = align_asof_row(d1, ts)

    if h4_row is None or d1_row is None:
        return None

    vals = [
        h1_row.get("atr14", np.nan),
        h1_row.get("bb_width_pct", np.nan),
        h1_row.get("vol_expansion", np.nan),
        h1_row.get("norm_ret_24h", np.nan),
        d1_row.get("ema50", np.nan),
        d1_row.get("ema200", np.nan),
        d1_row.get("ema50_slope", np.nan),
        h4_row.get("swing_high", np.nan),
        h4_row.get("swing_low", np.nan),
    ]
    if not all(np.isfinite(v) for v in vals):
        return None

    daily_long = (
        d1_row["close"] > d1_row["ema200"]
        and d1_row["ema50"] > d1_row["ema200"]
        and d1_row["ema50_slope"] > 0
    )
    daily_short = (
        d1_row["close"] < d1_row["ema200"]
        and d1_row["ema50"] < d1_row["ema200"]
        and d1_row["ema50_slope"] < 0
    )

    # This is a predeclared compression/expansion state, not an
    # optimized-after-OOS threshold.
    compression = h1_row["bb_width_pct"] <= 0.30
    expansion = h1_row["vol_expansion"] >= 1.05

    if not (compression and expansion):
        return None

    rs = rs_map.get(symbol, np.nan)
    if not np.isfinite(rs):
        return None

    close = float(h1_row["close"])
    low = float(h1_row["low"])
    high = float(h1_row["high"])
    atr = float(h1_row["atr14"])

    sh = float(h4_row["swing_high"])
    slw = float(h4_row["swing_low"])

    candidates = []

    if daily_long and symbol in rs_map["top"]:
        breakout = close > sh + 0.10 * atr
        retest = low <= sh + 0.20 * atr and close >= sh - 0.20 * atr
        if breakout or retest:
            stop = slw - 0.20 * atr if np.isfinite(slw) else low - 1.5 * atr
            candidates.append(("LONG", stop, abs(rs) + float(h1_row["vol_expansion"])))

    if daily_short and symbol in rs_map["bottom"]:
        breakout = close < slw - 0.10 * atr
        retest = high >= slw - 0.20 * atr and close <= slw + 0.20 * atr
        if breakout or retest:
            stop = sh + 0.20 * atr if np.isfinite(sh) else high + 1.5 * atr
            candidates.append(("SHORT", stop, abs(rs) + float(h1_row["vol_expansion"])))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (-x[2], x[0]))
    return candidates[0]


def _next_timestamp(index: pd.DatetimeIndex, ts: pd.Timestamp):
    pos = index.searchsorted(ts, side="right")
    if pos >= len(index):
        return None
    return index[pos]


def run_backtest(
    all_symbol_data: dict[str, dict[str, pd.DataFrame]],
    start: pd.Timestamp,
    end: pd.Timestamp,
):
    # Global 1H event clock.
    common = sorted(
        set().union(*(set(d["1h"].index) for d in all_symbol_data.values()))
    )
    times = [t for t in common if start <= t <= end]

    active = None
    trades = []
    cooldown_until = {s: pd.Timestamp.min.tz_localize("UTC") for s in all_symbol_data}

    for ts in times:
        # Manage the active trade only after its actual entry timestamp.
        if active is not None and ts >= active["entry_ts"]:
            sym = active["symbol"]
            h1 = all_symbol_data[sym]["1h"]
            if ts in h1.index:
                bar = h1.loc[ts]
                hi = float(bar["high"])
                lo = float(bar["low"])

                if active["side"] == "LONG":
                    hit_sl = lo <= active["sl"]
                    hit_tp = hi >= active["tp"]
                else:
                    hit_sl = hi >= active["sl"]
                    hit_tp = lo <= active["tp"]

                if hit_sl or hit_tp:
                    # Conservative rule: if both are touched in one bar,
                    # the stop is assumed to have occurred first.
                    outcome = "LOSS" if hit_sl else "WIN"
                    exit_price = active["sl"] if hit_sl else active["tp"]

                    notional = MARGIN_PER_TRADE * LEVERAGE
                    if active["side"] == "LONG":
                        gross = (exit_price - active["entry"]) / active["entry"] * notional
                    else:
                        gross = (active["entry"] - exit_price) / active["entry"] * notional

                    fees = notional * FEE_RATE * 2.0
                    pnl = gross - fees

                    trades.append(
                        {
                            "symbol": sym,
                            "side": active["side"],
                            "entry_ts": active["entry_ts"],
                            "exit_ts": ts,
                            "outcome": outcome,
                            "pnl": float(pnl),
                            "entry": float(active["entry"]),
                            "exit": float(exit_price),
                        }
                    )

                    cooldown_until[sym] = ts + pd.Timedelta(hours=3)
                    active = None
                    # No same-candle re-entry.
                    continue

        if active is not None:
            continue

        # Cross-sectional rank at the SAME completed 1H timestamp.
        rs_values = {}
        for sym, data in all_symbol_data.items():
            h1 = data["1h"]
            if ts in h1.index:
                v = h1.loc[ts].get("norm_ret_24h", np.nan)
                if np.isfinite(v):
                    rs_values[sym] = float(v)

        if len(rs_values) < max(2, int(math.ceil(len(SYMBOLS) * MIN_COMMON_SYMBOLS_FRAC))):
            continue

        ranked = sorted(rs_values.items(), key=lambda kv: (kv[1], kv[0]))
        top = {s for s, _ in ranked[-TOP_RS:]}
        bottom = {s for s, _ in ranked[:TOP_RS]}

        rs_context = dict(rs_values)
        rs_context["top"] = top
        rs_context["bottom"] = bottom

        candidates = []
        for sym, data in all_symbol_data.items():
            if ts < cooldown_until[sym]:
                continue

            signal = _regime_and_signal(sym, ts, data, rs_context)
            if signal is None:
                continue

            side, stop, score = signal
            entry_ts = _next_timestamp(data["1h"].index, ts)
            if entry_ts is None or entry_ts > end:
                continue

            raw_open = float(data["1h"].loc[entry_ts, "open"])
            entry = raw_open * (1.0 + SLIPPAGE if side == "LONG" else 1.0 - SLIPPAGE)

            risk = entry - stop if side == "LONG" else stop - entry
            if not np.isfinite(risk) or risk <= 0:
                continue

            tp = entry + RR * risk if side == "LONG" else entry - RR * risk

            candidates.append(
                {
                    "symbol": sym,
                    "side": side,
                    "score": float(score),
                    "trigger_ts": ts,
                    "entry_ts": entry_ts,
                    "entry": entry,
                    "sl": float(stop),
                    "tp": float(tp),
                }
            )

        if candidates:
            candidates.sort(
                key=lambda c: (
                    -c["score"],
                    c["symbol"],
                    c["side"],
                )
            )
            active = candidates[0]

    return trades, active


def calculate_stats(trades: list[dict]):
    if not trades:
        return {
            "trades": 0, "wins": 0, "losses": 0, "wr": 0.0,
            "gross_profit": 0.0, "gross_loss": 0.0, "pf": 0.0,
            "net": 0.0, "max_dd": 0.0, "max_dd_pct": 0.0,
            "max_streak": 0, "trades_day": 0.0,
        }

    ordered = sorted(trades, key=lambda t: pd.Timestamp(t["exit_ts"]))
    pnls = np.array([t["pnl"] for t in ordered], dtype=float)
    wins = int(sum(t["outcome"] == "WIN" for t in ordered))
    losses = int(sum(t["outcome"] == "LOSS" for t in ordered))
    gross_profit = float(pnls[pnls > 0].sum())
    gross_loss = float(-pnls[pnls < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    equity = INITIAL_CAPITAL + np.cumsum(pnls)
    peaks = np.maximum.accumulate(np.r_[INITIAL_CAPITAL, equity])[1:]
    dd = peaks - equity
    max_dd = float(dd.max()) if len(dd) else 0.0
    max_dd_pct = float((dd / np.maximum(peaks, 1e-9)).max() * 100.0) if len(dd) else 0.0

    streak = max_streak = 0
    for t in ordered:
        if t["outcome"] == "LOSS":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    first = pd.Timestamp(ordered[0]["entry_ts"])
    last = pd.Timestamp(ordered[-1]["exit_ts"])
    days = max((last - first).total_seconds() / 86400.0, 1e-9)

    return {
        "trades": len(ordered),
        "wins": wins,
        "losses": losses,
        "wr": wins / len(ordered) * 100.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "pf": pf,
        "net": float(pnls.sum()),
        "max_dd": max_dd,
        "max_dd_pct": max_dd_pct,
        "max_streak": max_streak,
        "trades_day": len(ordered) / days,
        "avg_win": gross_profit / wins if wins else 0.0,
        "avg_loss": -gross_loss / losses if losses else 0.0,
        "expectancy": float(pnls.mean()),
        "median": float(np.median(pnls)),
    }


def print_report(name: str, stats: dict):
    print(f"\n================ {name} ================")
    print(f"Trades              : {stats['trades']}")
    print(f"Wins                : {stats['wins']}")
    print(f"Losses              : {stats['losses']}")
    print(f"Win Rate            : {stats['wr']:.2f}%")
    print(f"Gross Profit        : ${stats['gross_profit']:,.2f}")
    print(f"Gross Loss          : ${stats['gross_loss']:,.2f}")
    print(f"Profit Factor       : {stats['pf']:.3f}")
    print(f"Net PnL             : ${stats['net']:,.2f}")
    print(f"Max Drawdown        : ${stats['max_dd']:,.2f}")
    print(f"Max Drawdown %      : {stats['max_dd_pct']:.2f}%")
    print(f"Max Loss Streak     : {stats['max_streak']}")
    print(f"Trades / Day        : {stats['trades_day']:.4f}")
    print(f"Average Win         : ${stats.get('avg_win', 0):,.2f}")
    print(f"Average Loss        : ${stats.get('avg_loss', 0):,.2f}")
    print(f"Expectancy / Trade  : ${stats.get('expectancy', 0):,.2f}")
    print(f"Median Trade PnL    : ${stats.get('median', 0):,.2f}")
    print("=" * 50)


def split_timeline(all_symbol_data) -> Split:
    first = max(d["1h"].index[0] for d in all_symbol_data.values())
    last = min(d["1h"].index[-1] for d in all_symbol_data.values())
    span = last - first

    train_end = first + span * TRAIN_FRAC
    val_end = first + span * (TRAIN_FRAC + VAL_FRAC)

    return Split(first, train_end, val_end, last)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("HUNTER-V9.1 — AUDITED DERIVATIVES REGIME / MICROSTRUCTURE ENGINE")
    print("=" * 78)

    all_symbol_data = {}
    for sym in SYMBOLS:
        try:
            raw = fetch_xt_futures_data(sym, data_dir, refresh=args.refresh)
            dfs = load_and_resample(raw)
            add_features(dfs)
            all_symbol_data[sym] = dfs
        except Exception as exc:
            print(f"[ABORT] {sym.upper()}: {exc}")
            sys.exit(1)

    split = split_timeline(all_symbol_data)
    print(f"[TIMELINE] Train: {split.start} -> {split.train_end}")
    print(f"[TIMELINE] Validation: {split.train_end} -> {split.val_end}")
    print(f"[TIMELINE] OOS: {split.val_end} -> {split.end}")

    # Factor discovery is diagnostic and TRAIN-only.
    factor_discovery(all_symbol_data, split.start, split.train_end)

    # Optional basis research from already validated spot files.
    build_basis_features(all_symbol_data, data_dir)

    # The strategy parameters are frozen before OOS.
    # We intentionally do NOT tune them on OOS.
    print("\n[BACKTEST] Validation run")
    val_trades, val_open = run_backtest(
        all_symbol_data, split.train_end, split.val_end
    )
    val_stats = calculate_stats(val_trades)
    print_report("VALIDATION", val_stats)
    print(f"Validation open position: {'YES' if val_open else 'NO'}")

    print("\n[BACKTEST] Untouched OOS run")
    oos_trades, oos_open = run_backtest(
        all_symbol_data, split.val_end, split.end
    )
    oos_stats = calculate_stats(oos_trades)
    print_report("OOS PERFORMANCE", oos_stats)
    print(f"OOS open position at end: {'YES' if oos_open else 'NO'}")

    accepted = (
        oos_stats["trades"] >= MIN_OOS_TRADES
        and oos_stats["wr"] > MIN_OOS_WR
        and oos_stats["pf"] > MIN_OOS_PF
        and oos_stats["net"] > 0
        and oos_stats["max_streak"] <= MAX_OOS_LOSS_STREAK
        and oos_stats["max_dd_pct"] < 50.0
    )

    print("\n================ ACCEPTANCE GATE ================")
    print(f"OOS trades >= {MIN_OOS_TRADES}       : {'PASS' if oos_stats['trades'] >= MIN_OOS_TRADES else 'FAIL'}")
    print(f"OOS WR > {MIN_OOS_WR:.1f}%              : {'PASS' if oos_stats['wr'] > MIN_OOS_WR else 'FAIL'}")
    print(f"OOS PF > {MIN_OOS_PF:.2f}              : {'PASS' if oos_stats['pf'] > MIN_OOS_PF else 'FAIL'}")
    print(f"OOS Net PnL > $0                  : {'PASS' if oos_stats['net'] > 0 else 'FAIL'}")
    print(f"OOS Max loss streak <= {MAX_OOS_LOSS_STREAK}: {'PASS' if oos_stats['max_streak'] <= MAX_OOS_LOSS_STREAK else 'FAIL'}")
    print(f"OOS Max DD < 50%                   : {'PASS' if oos_stats['max_dd_pct'] < 50.0 else 'FAIL'}")
    print(f"ACCEPTED                           : {accepted}")
    print("==================================================")

    if not accepted:
        print("\n[DECISION] REJECTED — no claim of robust edge.")


if __name__ == "__main__":
    main()
