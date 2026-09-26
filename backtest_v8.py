# HUNTER-FACTOR-AUDIT-1
# Perpetual dislocation research audit: FUTURES vs SPOT basis + price/volume factors.
# No strategy, no parameter sweep, no OOS tuning. This file only measures whether factors
# have forward information before a trading setup is designed.

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

BASE_FUT = "https://fapi.xt.com"
BASE_SPOT = "https://sapi.xt.com"
DAYS = 365
WARMUP_DAYS = 30
INTERVAL_MS = 15 * 60 * 1000
LIMIT_FUT = 1500
LIMIT_SPOT = 1000
REQUEST_TIMEOUT = 20
RETRIES = 4

SYMBOLS = [
    "BTC", "ETH", "SOL", "SUI", "AVAX", "NEAR", "ADA", "BNB",
    "APT", "CRV", "ONDO", "PENDLE", "ICP", "WIF",
]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "HUNTER-FACTOR-AUDIT/1.0"})


def utc_now_ms() -> int:
    return int(time.time() * 1000)


def request_json(url: str, params: dict) -> dict:
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            obj = r.json()
            if not isinstance(obj, dict):
                raise RuntimeError("non-dict JSON response")
            if obj.get("rc") not in (None, 0, "0"):
                raise RuntimeError(f"XT API error rc={obj.get('rc')} mc={obj.get('mc')} ma={obj.get('ma')}")
            return obj
        except Exception as exc:
            last = exc
            if attempt < RETRIES:
                time.sleep(0.6 * attempt)
    raise RuntimeError(f"request failed after {RETRIES} attempts: {url} params={params} err={last}")


def extract_list(obj: dict) -> list:
    result = obj.get("result")
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for key in ("items", "data", "list", "rows"):
            val = result.get(key)
            if isinstance(val, list):
                return val
    for key in ("data", "list", "rows"):
        val = obj.get(key)
        if isinstance(val, list):
            return val
    return []


def normalize_kline(rows: list) -> pd.DataFrame:
    out = []
    for x in rows:
        if isinstance(x, dict):
            t = x.get("t")
            o = x.get("o")
            h = x.get("h")
            l = x.get("l")
            c = x.get("c")
            v = x.get("a", x.get("q", x.get("volume")))
        elif isinstance(x, (list, tuple)) and len(x) >= 6:
            t, o, h, l, c, v = x[:6]
        else:
            continue
        try:
            out.append((int(t), float(o), float(h), float(l), float(c), float(v)))
        except Exception:
            continue
    df = pd.DataFrame(out, columns=["ts", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def fetch_futures(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Forward-paginate XT futures using the known-good V143/V144 pattern.

    XT can occasionally return records outside the requested page window.  Never use
    those records to advance the cursor; filter to the current window first.
    """
    rows: List = []
    cursor = start_ms
    guard = 0
    while cursor <= end_ms:
        guard += 1
        if guard > 1000:
            raise RuntimeError(f"futures pagination guard tripped for {symbol}")
        window_end = min(end_ms, cursor + LIMIT_FUT * INTERVAL_MS - 1)
        params = {
            "symbol": f"{symbol.lower()}_usdt",
            "interval": "15m",
            "startTime": cursor,
            "endTime": window_end,
            "limit": LIMIT_FUT,
        }
        batch = extract_list(request_json(f"{BASE_FUT}/future/market/v1/public/q/kline", params))
        if not batch:
            cursor = window_end + 1
            continue

        # XT may return an occasional record outside the requested window.
        # Only in-window timestamps are allowed to contribute to the cursor.
        in_window = []
        for x in batch:
            try:
                ts = int(x.get("t")) if isinstance(x, dict) else int(x[0])
            except Exception:
                continue
            if cursor <= ts <= window_end:
                in_window.append(x)

        if in_window:
            rows.extend(in_window)
            mx = max(int(x.get("t")) if isinstance(x, dict) else int(x[0]) for x in in_window)
            cursor = mx + 1
        else:
            # No usable record in this requested window: move to the next page
            # boundary rather than treating an out-of-window response as progress.
            cursor = window_end + 1

        time.sleep(0.03)

    df = normalize_kline(rows)
    return df[(df.ts >= start_ms) & (df.ts <= end_ms)].copy()

def fetch_spot(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Forward-paginate XT spot with the same guarded window semantics."""
    rows: List = []
    cursor = start_ms
    guard = 0
    while cursor <= end_ms:
        guard += 1
        if guard > 1000:
            raise RuntimeError(f"spot pagination guard tripped for {symbol}")
        window_end = min(end_ms, cursor + LIMIT_SPOT * INTERVAL_MS - 1)
        params = {
            "symbol": f"{symbol.lower()}_usdt",
            "interval": "15m",
            "startTime": cursor,
            "endTime": window_end,
            "limit": LIMIT_SPOT,
        }
        batch = extract_list(request_json(f"{BASE_SPOT}/v4/public/kline", params))
        if not batch:
            cursor = window_end + 1
            continue

        in_window = []
        for x in batch:
            try:
                ts = int(x.get("t")) if isinstance(x, dict) else int(x[0])
            except Exception:
                continue
            if cursor <= ts <= window_end:
                in_window.append(x)

        if in_window:
            rows.extend(in_window)
            mx = max(int(x.get("t")) if isinstance(x, dict) else int(x[0]) for x in in_window)
            cursor = mx + 1
        else:
            cursor = window_end + 1

        time.sleep(0.03)

    df = normalize_kline(rows)
    return df[(df.ts >= start_ms) & (df.ts <= end_ms)].copy()

def aggregate_1h(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    x = df.set_index("dt")
    out = x.resample("1h", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna().reset_index()
    out["ts"] = (out["dt"].astype("int64") // 10**6).astype("int64")
    return out


def zscore(s: pd.Series, n: int) -> pd.Series:
    mu = s.rolling(n, min_periods=n).mean().shift(1)
    sd = s.rolling(n, min_periods=n).std(ddof=0).shift(1)
    return (s - mu) / sd.replace(0, np.nan)


def add_features(fut: pd.DataFrame, spot: pd.DataFrame, btc_fut: pd.DataFrame) -> pd.DataFrame:
    f = aggregate_1h(fut).rename(columns={c: f"f_{c}" for c in ["open","high","low","close","volume"]})
    s = aggregate_1h(spot).rename(columns={c: f"s_{c}" for c in ["open","high","low","close","volume"]})
    x = f.merge(s[["dt","s_close","s_volume"]], on="dt", how="inner").sort_values("dt").reset_index(drop=True)
    x["basis"] = np.log(x.f_close / x.s_close)
    x["basis_pct"] = x.f_close / x.s_close - 1.0
    x["basis_z96"] = zscore(x["basis"], 96)
    x["basis_z192"] = zscore(x["basis"], 192)
    x["basis_ch4"] = x["basis"].diff(4)
    x["basis_ch24"] = x["basis"].diff(24)

    x["ret1"] = x.f_close.pct_change(1)
    x["ret4"] = x.f_close.pct_change(4)
    x["ret8"] = x.f_close.pct_change(8)
    x["ret24"] = x.f_close.pct_change(24)
    x["vol_z24"] = zscore(np.log1p(x.f_volume), 24)
    x["spot_vol_z24"] = zscore(np.log1p(x.s_volume), 24)
    x["volume_ratio"] = x.f_volume / x.s_volume.replace(0, np.nan)
    x["vol_ratio_z96"] = zscore(np.log(x["volume_ratio"].replace(0, np.nan).abs()), 96)

    # Price-volume factor: contemporaneous signed return times standardized volume,
    # then lagged by one hour for the predictive test so the feature is fully closed.
    x["pv_factor"] = x["ret4"] * x["vol_z24"]
    x["pv_factor_z96"] = zscore(x["pv_factor"], 96)

    b = aggregate_1h(btc_fut)[["dt", "close"]].rename(columns={"close": "btc_close"})
    x = x.merge(b, on="dt", how="left")
    x["rs24"] = x.f_close.pct_change(24) - x.btc_close.pct_change(24)
    x["rs72"] = x.f_close.pct_change(72) - x.btc_close.pct_change(72)

    # Forward outcomes; every factor row uses information available at the row's close.
    for h in (1, 4, 8, 24):
        x[f"fwd_ret_{h}h"] = x.f_close.shift(-h) / x.f_close - 1.0
        x[f"fwd_absret_{h}h"] = x[f"fwd_ret_{h}h"].abs()

    return x


def quantile_report(df: pd.DataFrame, factor: str, outcome: str, q: int = 5) -> pd.DataFrame:
    z = df[[factor, outcome]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(z) < 200:
        return pd.DataFrame()
    try:
        z["q"] = pd.qcut(z[factor], q=q, labels=False, duplicates="drop") + 1
    except Exception:
        return pd.DataFrame()
    g = z.groupby("q", observed=True)
    r = g[outcome].agg(["count", "mean", "median"])
    r["hit_rate"] = g[outcome].apply(lambda s: (s > 0).mean())
    r = r.reset_index().rename(columns={"q":"quantile"})
    r.insert(0, "factor", factor)
    r.insert(1, "outcome", outcome)
    return r


def summarize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    factors = [
        "basis", "basis_pct", "basis_z96", "basis_z192", "basis_ch4", "basis_ch24",
        "vol_z24", "spot_vol_z24", "volume_ratio", "vol_ratio_z96",
        "pv_factor", "pv_factor_z96", "rs24", "rs72",
    ]
    outcomes = ["fwd_ret_1h", "fwd_ret_4h", "fwd_ret_8h", "fwd_ret_24h", "fwd_absret_8h", "fwd_absret_24h"]
    qrows = []
    corr_rows = []
    for f in factors:
        for o in outcomes:
            sub = df[[f,o]].replace([np.inf,-np.inf],np.nan).dropna()
            if len(sub) < 200:
                continue
            pearson = sub[f].corr(sub[o], method="pearson")
            spearman = sub[f].corr(sub[o], method="spearman")
            corr_rows.append({"factor":f,"outcome":o,"n":len(sub),"pearson":pearson,"spearman":spearman})
            qr = quantile_report(df, f, o)
            if not qr.empty:
                qrows.append(qr)
    return pd.DataFrame(corr_rows), pd.concat(qrows, ignore_index=True) if qrows else pd.DataFrame()


def main() -> None:
    print("=" * 96)
    print("HUNTER-FACTOR-AUDIT-1 — PERPETUAL BASIS + PRICE/VOLUME FACTOR AUDIT")
    print("15m XT FUTURES + XT SPOT -> 1H | NO TRADING STRATEGY / NO PARAMETER OPTIMIZATION")
    print("=" * 96)

    now = utc_now_ms()
    start_ms = now - int((DAYS + WARMUP_DAYS) * 86400 * 1000)
    interval_end = (now // INTERVAL_MS) * INTERVAL_MS - 1

    futures: Dict[str, pd.DataFrame] = {}
    spots: Dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(SYMBOLS, 1):
        f = fetch_futures(sym, start_ms, interval_end)
        s = fetch_spot(sym, start_ms, interval_end)
        if len(f) < 30000:
            raise RuntimeError(f"Futures coverage too short for {sym}: {len(f)} rows")
        if len(s) < 25000:
            raise RuntimeError(f"Spot coverage too short for {sym}: {len(s)} rows")
        futures[sym] = f
        spots[sym] = s
        print(f"{i:02d}/{len(SYMBOLS)} {sym:<7} futures={len(f):>6} spot={len(s):>6} "
              f"f_start={f.dt.min()} f_end={f.dt.max()} s_start={s.dt.min()} s_end={s.dt.max()}")

    btc = futures["BTC"]
    all_corr = []
    all_quant = []
    for sym in SYMBOLS:
        feat = add_features(futures[sym], spots[sym], btc)
        # Drop the warmup tail only after features are calculated; retain the complete 365d audit window.
        cutoff = pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=DAYS))
        feat = feat[feat.dt >= cutoff].copy()
        corr, quant = summarize(feat)
        if not corr.empty:
            corr.insert(0, "symbol", sym)
            all_corr.append(corr)
        if not quant.empty:
            quant.insert(0, "symbol", sym)
            all_quant.append(quant)
        print(f"AUDIT {sym:<7} 1H rows={len(feat):>5} factors={len(corr):>3} correlation tests")

    corr_all = pd.concat(all_corr, ignore_index=True) if all_corr else pd.DataFrame()
    quant_all = pd.concat(all_quant, ignore_index=True) if all_quant else pd.DataFrame()

    if corr_all.empty:
        raise RuntimeError("No factor audit rows produced")

    # Aggregate only descriptive evidence: median cross-sectional rank of |Spearman| and sign consistency.
    score_rows = []
    for factor, g in corr_all.groupby("factor"):
        pred = g[g.outcome.str.startswith("fwd_ret_")].copy()
        vol = g[g.outcome.str.startswith("fwd_absret_")].copy()
        for frame, kind in ((pred, "direction"), (vol, "volatility")):
            if frame.empty:
                continue
            score_rows.append({
                "factor": factor,
                "test_type": kind,
                "median_abs_spearman": float(frame.spearman.abs().median()),
                "median_spearman": float(frame.spearman.median()),
                "positive_sign_fraction": float((frame.spearman > 0).mean()),
                "tests": int(len(frame)),
            })
    score = pd.DataFrame(score_rows).sort_values(["test_type","median_abs_spearman"], ascending=[True,False])

    corr_all.to_csv("factor_audit_correlations.csv", index=False)
    quant_all.to_csv("factor_audit_quantiles.csv", index=False)
    score.to_csv("factor_audit_summary.csv", index=False)

    print("\n" + "=" * 96)
    print("FACTOR AUDIT SUMMARY — DESCRIPTIVE ONLY")
    print("=" * 96)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 180)
    print(score.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nFiles written: factor_audit_correlations.csv, factor_audit_quantiles.csv, factor_audit_summary.csv")
    print("No trading setup was selected from these results; no OOS tuning was performed.")
    print("Research basis: recent perpetual-futures literature identifies basis and price-volume as important systematic drivers; this run tests whether the relationship appears in XT data before strategy construction.")


if __name__ == "__main__":
    main()
