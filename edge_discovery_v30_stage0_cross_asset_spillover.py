
import os, time, json, math
from pathlib import Path
import requests
import numpy as np
import pandas as pd

# ============================================================
# V30-STAGE0 — CROSS-ASSET SPILLOVER / POSITIVE LEAD-LAG
# ============================================================
# Research-only event study. No parameter optimization.
#
# Hypothesis:
# Lagged returns of other crypto assets contain information about
# the next return of a focal asset. This is a continuation /
# delayed-information hypothesis, NOT the negative seesaw tested
# in V28.
#
# Locked research protocol:
# - XT USDT-M perpetual futures, 1H OHLCV
# - 14-symbol fixed universe
# - 455 days + 30-day warmup
# - all features causal: signal at t uses data <= t-1
# - no synthetic OI/CVD/liquidation/funding
# - no future symbol selection
# - next-bar-open execution for RR path labels
# - 1 ATR stop / 2 ATR target
# - same-candle SL+TP = LOSS
# - censored trades at sample end excluded
# - Discovery / Development / Validation = 50/25/25
#
# The event study deliberately avoids selecting a winning threshold.
# It reports target relative-position buckets and forward horizons.
# A strategy is only built if the effect is economically meaningful
# and directionally stable across Development and Validation.
# ============================================================

BASE_URL = "https://fapi.xt.com/future/market/v1/public/q/kline"
OUT_DIR = Path("reports/xt_v30_stage0_cross_asset_spillover")

SYMBOLS = [
    "btc_usdt", "eth_usdt", "sol_usdt", "sui_usdt", "avax_usdt",
    "near_usdt", "ada_usdt", "bnb_usdt", "apt_usdt", "crv_usdt",
    "ondo_usdt", "pendle_usdt", "icp_usdt", "wif_usdt"
]

DAYS = 455
WARMUP_DAYS = 30
BAR_MS = 60 * 60 * 1000
EXPECTED_BARS = DAYS * 24
TOTAL_BARS = (DAYS + WARMUP_DAYS) * 24
LIMIT = 1000
MIN_COVERAGE = 0.97

# Fixed, pre-registered horizons.
PEER_LOOKBACK = 4          # 4H
PEER_Z_WINDOW = 168        # 7 days
FORWARD_HORIZONS = [1, 4, 8, 24]  # 1H, 4H, 8H, 24H
ATR_PERIOD = 14
RR_STOP_ATR = 1.0
RR_TARGET_ATR = 2.0
RR_HORIZON = 24

REQUEST_TIMEOUT = 20
RETRIES = 4
SLEEP = 0.20

session = requests.Session()
session.headers.update({"User-Agent": "v30-stage0-research/1.0"})


def parse_rows(payload):
    """Accept common XT response wrappers and return raw kline rows."""
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    candidates = [
        payload.get("result"),
        payload.get("data"),
        payload.get("rows"),
        payload.get("list"),
    ]
    for x in candidates:
        if isinstance(x, list):
            return x
        if isinstance(x, dict):
            for k in ("data", "list", "rows", "result"):
                if isinstance(x.get(k), list):
                    return x[k]
    return []


def normalize_row(r):
    if isinstance(r, dict):
        t = r.get("t", r.get("time", r.get("timestamp")))
        o = r.get("o", r.get("open"))
        h = r.get("h", r.get("high"))
        l = r.get("l", r.get("low"))
        c = r.get("c", r.get("close"))
        v = r.get("v", r.get("volume", r.get("vol")))
        return [t, o, h, l, c, v]
    if isinstance(r, (list, tuple)) and len(r) >= 6:
        return list(r[:6])
    return None


def fetch_symbol(symbol, start_ms, end_ms):
    """
    XT's kline endpoint can return only the first LIMIT rows for a
    large requested interval. A single 485-day request therefore
    returns about 1000/11640 rows and looks like a coverage failure.

    Use deterministic bounded windows instead of cursor pagination.
    Each request covers at most 900 one-hour candles. This mirrors
    the bounded-window approach that previously produced complete
    XT historical coverage.
    """
    rows = []
    window_bars = 900
    window_ms = window_bars * BAR_MS
    window_start = start_ms

    while window_start < end_ms:
        window_end = min(window_start + window_ms, end_ms)

        params = {
            "symbol": symbol,
            "interval": "1h",
            "startTime": window_start,
            "endTime": window_end,
            "limit": LIMIT,
        }

        payload = None
        for attempt in range(RETRIES):
            try:
                r = session.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                payload = r.json()
                break
            except Exception:
                if attempt == RETRIES - 1:
                    raise
                time.sleep(1.0 + attempt)

        raw = parse_rows(payload)
        batch = []

        for x in raw:
            y = normalize_row(x)
            if y is None:
                continue
            try:
                ts = int(float(y[0]))
                if ts < 10_000_000_000:
                    ts *= 1000

                # Keep only the exact requested half-open window.
                if window_start <= ts < window_end:
                    batch.append([
                        ts,
                        float(y[1]), float(y[2]), float(y[3]),
                        float(y[4]), float(y[5])
                    ])
            except Exception:
                continue

        rows.extend(batch)

        # Advance by the requested window, not by the response's last
        # timestamp. This prevents XT's response ordering/pagination
        # behavior from truncating the historical range.
        window_start = window_end
        time.sleep(SLEEP)

    df = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = (
        df.drop_duplicates("timestamp")
          .sort_values("timestamp")
          .reset_index(drop=True)
    )

    # Never use the currently incomplete candle.
    now_ms = int(time.time() * 1000)
    cutoff_ms = now_ms - (now_ms % BAR_MS)
    cutoff = pd.Timestamp(cutoff_ms, unit="ms", tz="UTC")
    df = df[df["timestamp"] < cutoff].copy()

    return df


def fetch_all():
    end = int(time.time() * 1000)
    # Round to the start of the current hour, then exclude that hour.
    end = end - (end % BAR_MS)
    start = end - (DAYS + WARMUP_DAYS) * 24 * BAR_MS

    coverage = []
    frames = {}

    for symbol in SYMBOLS:
        df = fetch_symbol(symbol, start, end)
        expected = TOTAL_BARS
        actual = len(df)
        cov = actual / expected if expected else 0.0
        gaps = int(df["timestamp"].diff().dropna().ne(pd.Timedelta(hours=1)).sum()) if len(df) > 1 else 0

        coverage.append({
            "symbol": symbol,
            "expected_rows": expected,
            "actual_rows": actual,
            "coverage": cov,
            "gaps": gaps,
            "first_timestamp": df["timestamp"].min(),
            "last_timestamp": df["timestamp"].max(),
            "status": "PASS" if cov >= MIN_COVERAGE and gaps == 0 else "FAIL",
        })

        if cov < MIN_COVERAGE or gaps != 0:
            raise RuntimeError(
                f"Coverage failure {symbol}: coverage={cov:.4f}, "
                f"rows={actual}/{expected}, gaps={gaps}, "
                f"first={df['timestamp'].min()}, last={df['timestamp'].max()}"
            )

        frames[symbol] = df

    cov_df = pd.DataFrame(coverage)
    return frames, cov_df


def atr(df):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()


def build_panel(frames):
    # Common timestamps only: no forward filling across missing assets.
    common = None
    for df in frames.values():
        s = pd.DatetimeIndex(df["timestamp"])
        common = s if common is None else common.intersection(s)

    common = common.sort_values()
    if len(common) < EXPECTED_BARS * 0.97:
        raise RuntimeError(f"Common panel coverage too low: {len(common)} rows")

    closes = pd.DataFrame(index=common)
    highs = pd.DataFrame(index=common)
    lows = pd.DataFrame(index=common)
    atrs = pd.DataFrame(index=common)

    for symbol, df in frames.items():
        x = df.set_index("timestamp").reindex(common)
        closes[symbol] = x["close"]
        highs[symbol] = x["high"]
        lows[symbol] = x["low"]
        atrs[symbol] = atr(x)

    # Use log returns. The return at t is the move from t-1 close to t close.
    rets = np.log(closes / closes.shift(1))

    # The signal for bar t is generated from data available at t-1.
    # Therefore all peer statistics below are explicitly shifted by one.
    peer_ret = pd.DataFrame(index=common, columns=SYMBOLS, dtype=float)
    for target in SYMBOLS:
        others = [s for s in SYMBOLS if s != target]
        peer_ret[target] = rets[others].mean(axis=1)

    peer_mom = peer_ret.rolling(PEER_LOOKBACK).sum()
    peer_mu = peer_mom.rolling(PEER_Z_WINDOW).mean().shift(1)
    peer_sd = peer_mom.rolling(PEER_Z_WINDOW).std(ddof=0).shift(1)
    peer_z = (peer_mom.shift(1) - peer_mu) / peer_sd.replace(0, np.nan)

    target_mom = rets.rolling(PEER_LOOKBACK).sum().shift(1)
    relative = target_mom - peer_mom.shift(1)

    # Cross-sectional relative rank is computed only from t-1 information.
    rel_rank = relative.rank(axis=1, pct=True, method="average")

    rows = []
    for ts in common:
        for symbol in SYMBOLS:
            rows.append({
                "timestamp": ts,
                "symbol": symbol,
                "close": closes.loc[ts, symbol],
                "atr": atrs.loc[ts, symbol],
                "peer_momentum_4h": peer_mom.loc[ts, symbol],
                "peer_z_7d": peer_z.loc[ts, symbol],
                "target_momentum_4h": target_mom.loc[ts, symbol],
                "relative_momentum": relative.loc[ts, symbol],
                "relative_rank": rel_rank.loc[ts, symbol],
            })

    panel = pd.DataFrame(rows)
    return panel, closes, highs, lows, atrs


def rr_label(entry, future_high, future_low, direction, stop_dist, target_dist):
    if direction == "LONG":
        sl = entry - stop_dist
        tp = entry + target_dist
        for hi, lo in zip(future_high, future_low):
            hit_sl = lo <= sl
            hit_tp = hi >= tp
            if hit_sl and hit_tp:
                return -1.0, "SL_TP_SAME_BAR"
            if hit_sl:
                return -1.0, "SL"
            if hit_tp:
                return 2.0, "TP"
    else:
        sl = entry + stop_dist
        tp = entry - target_dist
        for hi, lo in zip(future_high, future_low):
            hit_sl = hi >= sl
            hit_tp = lo <= tp
            if hit_sl and hit_tp:
                return -1.0, "SL_TP_SAME_BAR"
            if hit_sl:
                return -1.0, "SL"
            if hit_tp:
                return 2.0, "TP"
    return np.nan, "CENSORED"


def add_splits(panel):
    ts = pd.DatetimeIndex(sorted(panel["timestamp"].unique()))
    n = len(ts)
    a = int(n * 0.50)
    b = int(n * 0.75)
    mp = {t: ("DISCOVERY" if i < a else "DEVELOPMENT" if i < b else "VALIDATION")
          for i, t in enumerate(ts)}
    panel["split"] = panel["timestamp"].map(mp)
    return panel


def make_event_report(panel, closes, highs, lows, atrs):
    # Event classes are pre-defined and are NOT chosen after seeing results.
    # For positive peer shocks: under-reacting targets are the lower relative ranks.
    # For negative peer shocks: over-reacting targets are the higher relative ranks.
    # Buckets are 5 fixed quantiles to show monotonicity, not to select a threshold.
    work = panel.copy()
    work["peer_dir"] = np.where(work["peer_z_7d"] >= 1.0, "UP",
                         np.where(work["peer_z_7d"] <= -1.0, "DOWN", "NONE"))

    work["rel_bucket"] = pd.cut(
        work["relative_rank"],
        bins=[0, .2, .4, .6, .8, 1.0],
        labels=["Q1_LOW","Q2","Q3","Q4","Q5_HIGH"],
        include_lowest=True
    )

    # Only peer shocks are tested.
    work = work[work["peer_dir"] != "NONE"].copy()

    close_w = closes
    high_w = highs
    low_w = lows
    atr_w = atrs

    results = []
    for split in ["DISCOVERY","DEVELOPMENT","VALIDATION"]:
        sub = work[work["split"] == split]
        for direction in ["LONG","SHORT"]:
            dsub = sub[(sub["peer_dir"] == ("UP" if direction == "LONG" else "DOWN"))].copy()
            for bucket in ["Q1_LOW","Q2","Q3","Q4","Q5_HIGH"]:
                bsub = dsub[dsub["rel_bucket"] == bucket]
                for horizon in FORWARD_HORIZONS:
                    vals = []
                    for _, r in bsub.iterrows():
                        ts = r["timestamp"]; sym = r["symbol"]
                        idx = close_w.index.get_loc(ts)
                        if idx + horizon >= len(close_w.index):
                            continue
                        c0 = close_w.iloc[idx][sym]
                        c1 = close_w.iloc[idx + horizon][sym]
                        ret = (c1 / c0 - 1.0)
                        vals.append(ret if direction == "LONG" else -ret)

                    if vals:
                        arr = np.asarray(vals, dtype=float)
                        results.append({
                            "split": split,
                            "peer_direction": direction,
                            "relative_bucket": bucket,
                            "horizon_bars": horizon,
                            "n": len(arr),
                            "mean_directional_return": float(arr.mean()),
                            "median_directional_return": float(np.median(arr)),
                            "win_rate": float((arr > 0).mean()),
                        })

    # RR 1:2 path test for the economically motivated extreme buckets:
    # positive peer shock -> Q1 (underreaction) long
    # negative peer shock -> Q5 (overreaction) short
    rr_rows = []
    for split in ["DISCOVERY","DEVELOPMENT","VALIDATION"]:
        for direction, pdir, bucket in [
            ("LONG","UP","Q1_LOW"),
            ("SHORT","DOWN","Q5_HIGH"),
        ]:
            sub = work[
                (work["split"] == split) &
                (work["peer_dir"] == pdir) &
                (work["rel_bucket"] == bucket)
            ]
            vals = []
            exits = []
            for _, r in sub.iterrows():
                ts = r["timestamp"]; sym = r["symbol"]
                idx = close_w.index.get_loc(ts)
                # Entry at NEXT bar open. Need OHLC of t+1 ... t+RR_HORIZON.
                if idx + RR_HORIZON + 1 >= len(close_w.index):
                    continue
                entry = close_w.iloc[idx + 1][sym]
                a = atr_w.iloc[idx][sym]  # ATR known at signal time.
                if not np.isfinite(entry) or not np.isfinite(a) or a <= 0:
                    continue
                fut_hi = high_w.iloc[idx + 1: idx + 1 + RR_HORIZON][sym].to_numpy()
                fut_lo = low_w.iloc[idx + 1: idx + 1 + RR_HORIZON][sym].to_numpy()
                rr, ex = rr_label(
                    float(entry), fut_hi, fut_lo, direction,
                    float(a) * RR_STOP_ATR, float(a) * RR_TARGET_ATR
                )
                if np.isfinite(rr):
                    vals.append(rr); exits.append(ex)

            if vals:
                arr = np.asarray(vals, dtype=float)
                wins = int((arr == 2.0).sum())
                losses = int((arr == -1.0).sum())
                rr_rows.append({
                    "split": split,
                    "direction": direction,
                    "relative_bucket": bucket,
                    "trades": len(arr),
                    "wins": wins,
                    "losses": losses,
                    "win_rate": wins / len(arr),
                    "pf_gross": (2.0 * wins / losses) if losses else np.inf,
                    "mean_R": float(arr.mean()),
                    "same_candle_sl_tp": int(exits.count("SL_TP_SAME_BAR")),
                })

    event_df = pd.DataFrame(results)
    rr_df = pd.DataFrame(rr_rows)
    return event_df, rr_df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames, coverage = fetch_all()
    coverage.to_csv(OUT_DIR / "kline_coverage_audit.csv", index=False)

    panel, closes, highs, lows, atrs = build_panel(frames)
    panel = add_splits(panel)
    panel.to_csv(OUT_DIR / "research_panel.csv", index=False)

    event_df, rr_df = make_event_report(panel, closes, highs, lows, atrs)
    event_df.to_csv(OUT_DIR / "event_study_report.csv", index=False)
    rr_df.to_csv(OUT_DIR / "rr12_event_report.csv", index=False)

    meta = {
        "strategy": "V30-STAGE0 Cross-Asset Spillover / Positive Lead-Lag",
        "hypothesis": "Positive cross-crypto lead-lag / delayed information diffusion",
        "exchange": "XT USDT-M perpetual futures",
        "timeframe": "1h",
        "symbols": SYMBOLS,
        "days": DAYS,
        "warmup_days": WARMUP_DAYS,
        "peer_lookback_bars": PEER_LOOKBACK,
        "peer_z_window_bars": PEER_Z_WINDOW,
        "forward_horizons": FORWARD_HORIZONS,
        "rr_stop_atr": RR_STOP_ATR,
        "rr_target_atr": RR_TARGET_ATR,
        "rr_horizon_bars": RR_HORIZON,
        "splits": "50/25/25",
        "lookahead_policy": "All signal features use information through t-1; entry is t+1 open.",
        "same_candle_sl_tp": "LOSS",
        "censored_at_end": "EXCLUDED",
        "optimization": "NONE",
        "status": "COMPLETE"
    }
    with open(OUT_DIR / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"V30 STAGE-0 COMPLETE: panel_rows={len(panel):,} events={len(event_df):,} rr_rows={len(rr_df):,}")
    print(f"Artifacts: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
