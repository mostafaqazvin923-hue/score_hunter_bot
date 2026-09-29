#!/usr/bin/env python3
"""HUNTER-V23.10-STAGE0 — FUTURES BASIS DISLOCATION / REPRICING

LOCKED RESEARCH HYPOTHESIS
--------------------------
Perpetual-vs-spot basis dislocation/repricing contains short-horizon
information in crypto futures. Signal logic is fixed before Stage-0 results:
  LONG : high basis tail + basis falling + spot 24h return >= 0
  SHORT: low basis tail  + basis rising  + spot 24h return <= 0
with causal cross-sectional historical basis z-score.

IMPORTANT
---------
This version is rebuilt as a clean data/backtest engine. Funding is NOT a
signal input and is not fetched. No missing candle is fabricated. Spot gaps
up to 6h are tolerated only as real missing observations; timestamp-dependent
features become NaN at affected timestamps. Futures gaps are fatal.

Backtest rules
--------------
- 1H completed candles only
- 455 days research window + 7 days warmup
- initial equity $1,000; $100 margin; $5,000 notional (50x)
- RR 1:2
- fee 0.07% per side; slippage 0.03% per side
- one position per symbol; max 10 simultaneous positions
- next-bar-open entry
- no timeout / BE / trailing / pyramiding
- same candle SL+TP => LOSS
- no re-entry on the exit candle
- no carrying an open trade past the available data
- fixed universe; no future symbol selection
- Stage-0 tests all three pre-registered ATR stops; no selection here
"""

import hashlib
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SYMBOLS = [
    "btc_usdt", "eth_usdt", "sol_usdt", "sui_usdt", "avax_usdt",
    "near_usdt", "ada_usdt", "bnb_usdt", "apt_usdt", "crv_usdt",
    "ondo_usdt", "pendle_usdt", "icp_usdt", "wif_usdt",
]

FUT_URL = "https://fapi.xt.com/future/market/v1/public/q/kline"
SPOT_URL = "https://sapi.xt.com/v4/public/kline"

LOOKBACK_DAYS = 455
WARMUP_DAYS = 7
REQUIRED_DAYS = LOOKBACK_DAYS + WARMUP_DAYS
H = 3_600_000
LIMIT = 1000

INITIAL = 1000.0
MARGIN = 100.0
NOTIONAL = 5000.0
RR = 2.0
FEE = 0.0007
SLIP = 0.0003
MAX_POS = 10

ATR_N = 14
BASIS_Z_N = 120
BASIS_CHANGE_N = 24
QUANT = 0.20
MIN_Z = 1.0
MIN_CS = 8
STOPS = (1.0, 1.25, 1.5)

CACHE = Path("data/xt_v23_10")
REPORT = Path("reports/xt_v23_10")

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "hunter-v23.9-stage0/1.0"})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now_hour_ms() -> int:
    return (int(time.time() * 1000) // H) * H


def rows(payload):
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return None
    for key in ("result", "data", "rows", "list"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    result = payload.get("result")
    if isinstance(result, dict):
        for key in ("items", "list", "rows", "data"):
            value = result.get(key)
            if isinstance(value, list):
                return value
    return None


def parse_kline_row(row):
    try:
        if isinstance(row, dict):
            values = [
                row.get("t", row.get("timestamp")),
                row.get("o", row.get("open")),
                row.get("h", row.get("high")),
                row.get("l", row.get("low")),
                row.get("c", row.get("close")),
                row.get("a", row.get("v", row.get("volume", row.get("amount")))),
            ]
        elif isinstance(row, (list, tuple)) and len(row) >= 6:
            values = list(row[:6])
        else:
            return None
        if any(v is None for v in values):
            return None
        ts = int(float(values[0]))
        if ts < 10_000_000_000:
            ts *= 1000
        return (ts,) + tuple(float(v) for v in values[1:])
    except Exception:
        return None


def validate_ohlcv(df, name, start_ms, end_ms, allow_spot_gaps):
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name}: missing columns {missing}")

    x = df[required].copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    if x.isna().any().any():
        bad = int(x.isna().sum().sum())
        raise RuntimeError(f"{name}: invalid/NaN OHLCV values={bad}")

    x = x.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    start = pd.to_datetime(start_ms, unit="ms", utc=True)
    end = pd.to_datetime(end_ms, unit="ms", utc=True)
    x = x[(x.timestamp >= start) & (x.timestamp <= end)].copy()

    if x.empty:
        raise RuntimeError(f"{name}: zero candles after date filtering")
    if (x[["open", "high", "low", "close"]] <= 0).any().any():
        raise RuntimeError(f"{name}: non-positive OHLC")
    if (x.volume < 0).any():
        raise RuntimeError(f"{name}: negative volume")

    gaps = x.timestamp.diff().dropna().dt.total_seconds() / 3600.0
    bad = gaps > 1.0 + 1e-9
    if bad.any():
        max_gap = float(gaps.max())
        count = int(bad.sum())
        if not allow_spot_gaps:
            raise RuntimeError(
                f"{name}: FUTURES data gap detected; max_gap={max_gap:.2f}h; refusing fabrication"
            )
        if max_gap > 6.0:
            raise RuntimeError(
                f"{name}: SPOT gap too large; max_gap={max_gap:.2f}h; refusing fabrication"
            )
        print(f"[DATA] {name}: {count} spot gap(s), max_gap={max_gap:.2f}h; no filling")

    expected = REQUIRED_DAYS * 24
    min_required = int(expected * 0.95)
    if len(x) < min_required:
        span_days = (x.timestamp.iloc[-1] - x.timestamp.iloc[0]).total_seconds() / 86400.0
        raise RuntimeError(
            f"{name}: insufficient history: rows={len(x)}, need>={min_required}, span={span_days:.1f}d"
        )

    return x.reset_index(drop=True)


def request_json(url, params, label, attempts=4):
    last = None
    for attempt in range(1, attempts + 1):
        try:
            response = SESSION.get(url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            return payload
        except Exception as exc:
            last = exc
            if attempt < attempts:
                time.sleep(attempt)
    raise RuntimeError(f"{label}: request failed after {attempts} attempts: {last}")


def fetch_kline(url, symbol, kind):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{kind}_{symbol}.csv"

    end_ms = now_hour_ms() - H
    expected_rows = REQUIRED_DAYS * 24
    start_ms = end_ms - (expected_rows - 1) * H
    allow_spot = kind == "spot"
    WINDOW_CANDLES = 900

    if path.exists():
        try:
            cached = pd.read_csv(path)
            return validate_ohlcv(
                cached, f"{kind}_{symbol}[cache]", start_ms, end_ms, allow_spot
            )
        except Exception as exc:
            print(f"[CACHE] {kind}_{symbol}: invalid/stale cache ({exc}); refetching")

    # XT can return one fewer candle than the requested bounded window.  The
    # safe solution is not to assume page-size == response-size and not to
    # make a tiny one-hour request at the very end.  We advance by the actual
    # newest timestamp received, then use one final full-width overlapping
    # window for the tail.  All rows are deduplicated by timestamp and the
    # validator remains the authority on real missing candles.
    cur = start_ms
    all_rows = []
    page = 0
    previous_max = None

    while cur <= end_ms:
        page += 1
        if page > 1000:
            raise RuntimeError(f"{kind}_{symbol}: pagination guard exceeded")

        remaining = ((end_ms - cur) // H) + 1
        final_window = remaining <= WINDOW_CANDLES

        if final_window:
            window_start = max(start_ms, end_ms - (WINDOW_CANDLES - 1) * H)
            window_end = end_ms
        else:
            window_start = cur
            window_end = cur + (WINDOW_CANDLES - 1) * H

        params = {
            "symbol": symbol,
            "interval": "1h",
            "startTime": int(window_start),
            # XT behaves as if endTime is exclusive on this endpoint.
            # Request one extra hour and filter it back to window_end.
            "endTime": int(window_end + H),
            "limit": LIMIT,
        }
        payload = request_json(url, params, f"{kind}_{symbol} page={page}")
        raw = rows(payload)
        if raw is None:
            raise RuntimeError(
                f"{kind}_{symbol} page={page}: API returned no candle list"
            )

        parsed = [parse_kline_row(r) for r in raw]
        inside = sorted({
            r for r in parsed
            if r is not None and window_start <= r[0] <= window_end
        })

        # A final one-hour request is an XT boundary edge case.  Do not treat
        # an empty tail request as proof of a valid dataset; instead validate
        # the rows already collected.  Any actual missing candle/gap will be
        # rejected by validate_ohlcv below.  Non-final empty pages are fatal.
        if not inside:
            if final_window:
                break
            raise RuntimeError(
                f"{kind}_{symbol}: empty bounded page={page} at "
                f"{pd.to_datetime(window_start, unit='ms', utc=True)} -> "
                f"{pd.to_datetime(window_end, unit='ms', utc=True)}; "
                f"refusing skip/fabricate"
            )

        all_rows.extend(inside)
        unique_ts = len({r[0] for r in all_rows})
        max_ts = max(r[0] for r in inside)
        min_ts = min(r[0] for r in inside)

        if previous_max is not None and not final_window and max_ts <= previous_max:
            raise RuntimeError(
                f"{kind}_{symbol}: pagination stalled at page={page}; "
                f"max timestamp did not move forward"
            )
        previous_max = max(previous_max or max_ts, max_ts)

        if page % 4 == 0 or unique_ts >= expected_rows or final_window:
            print(
                f"[FETCH] {kind}_{symbol} page={page} rows={unique_ts} "
                f"oldest={pd.to_datetime(min(r[0] for r in all_rows), unit='ms', utc=True)} "
                f"newest={pd.to_datetime(max(r[0] for r in all_rows), unit='ms', utc=True)}"
            )

        if unique_ts >= expected_rows or final_window:
            break

        next_cur = max_ts + H
        if next_cur <= cur:
            raise RuntimeError(f"{kind}_{symbol}: pagination stalled at page={page}")
        cur = next_cur
        time.sleep(0.05)

    frame = pd.DataFrame(
        all_rows, columns=["ts", "open", "high", "low", "close", "volume"]
    )
    frame["timestamp"] = pd.to_datetime(frame.pop("ts"), unit="ms", utc=True)
    frame = frame[
        (frame.timestamp >= pd.to_datetime(start_ms, unit="ms", utc=True))
        & (frame.timestamp <= pd.to_datetime(end_ms, unit="ms", utc=True))
    ]
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last")
    frame = validate_ohlcv(frame, f"{kind}_{symbol}", start_ms, end_ms, allow_spot)
    frame.to_csv(path, index=False)
    return frame

def true_range(x):
    prev_close = x["close"].shift(1)
    return pd.concat(
        [
            x["high"] - x["low"],
            (x["high"] - prev_close).abs(),
            (x["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def build_stream(futures, spot):
    f = futures.set_index("timestamp").sort_index()
    s = spot.set_index("timestamp").sort_index()

    x = f.join(
        s[["close", "open", "high", "low", "volume"]].add_suffix("_spot"),
        how="inner",
    )
    if x.empty:
        raise RuntimeError("merged futures/spot series is empty")

    x["basis"] = (x["close"] - x["close_spot"]) / x["close_spot"]
    x["atr"] = true_range(x).rolling(ATR_N, min_periods=ATR_N).mean()

    # Exact elapsed-time lookup. A missing spot candle can never be converted
    # into a fake 24-row/24-hour observation.
    x["spot_ret24"] = np.nan
    x["basis_chg24"] = np.nan
    prior_ts = x.index - pd.Timedelta(hours=BASIS_CHANGE_N)
    exact_prior = prior_ts.isin(x.index)

    spot_lookup = pd.Series(x["close_spot"].to_numpy(), index=x.index)
    basis_lookup = pd.Series(x["basis"].to_numpy(), index=x.index)
    spot_prev = spot_lookup.reindex(prior_ts)
    basis_prev = basis_lookup.reindex(prior_ts)
    spot_prev.index = x.index
    basis_prev.index = x.index

    x.loc[exact_prior, "spot_ret24"] = (
        x.loc[exact_prior, "close_spot"] / spot_prev.loc[exact_prior] - 1.0
    )
    x.loc[exact_prior, "basis_chg24"] = (
        x.loc[exact_prior, "basis"] - basis_prev.loc[exact_prior]
    )

    x = x.replace([np.inf, -np.inf], np.nan)
    return x


def build_panel(streams):
    # OUTER join is intentional. Missing data for one symbol must not delete
    # valid observations from every other symbol. No values are filled.
    basis = pd.concat({s: x["basis"] for s, x in streams.items()}, axis=1, join="outer")
    change = pd.concat(
        {s: x["basis_chg24"] for s, x in streams.items()}, axis=1, join="outer"
    )

    median = basis.median(axis=1, skipna=True)
    deviation = basis.sub(median, axis=0)

    # Historical cross-sectional factor statistics are shifted one bar so the
    # current signal cannot influence its own z-score.
    mu = deviation.rolling(BASIS_Z_N, min_periods=BASIS_Z_N).mean().shift(1)
    sd = deviation.rolling(BASIS_Z_N, min_periods=BASIS_Z_N).std(ddof=0).shift(1)
    z = (deviation - mu) / sd.replace(0.0, np.nan)
    return basis, change, z




def _self_test():
    """Offline integrity tests; no network access and no strategy tuning."""
    # 1) Validate exact hourly sequence for a synthetic futures dataset.
    start = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    periods = REQUIRED_DAYS * 24
    idx = pd.date_range(start, periods=periods, freq="1h", tz="UTC")
    base = np.linspace(100.0, 120.0, periods)
    df = pd.DataFrame({
        "timestamp": idx,
        "open": base,
        "high": base + 1.0,
        "low": base - 1.0,
        "close": base + 0.2,
        "volume": np.ones(periods),
    })
    checked = validate_ohlcv(
        df, "SELFTEST", int(idx[0].timestamp() * 1000),
        int(idx[-1].timestamp() * 1000), False
    )
    assert len(checked) == periods

    # 2) A futures gap must be rejected, never filled.
    gap_df = df.drop(index=123).reset_index(drop=True)
    try:
        validate_ohlcv(
            gap_df, "SELFTEST_GAP", int(idx[0].timestamp() * 1000),
            int(idx[-1].timestamp() * 1000), False
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("futures gap was not rejected")

    # 3) Causal feature construction must produce a finite basis/ATR stream.
    spot = df.copy()
    spot[["open", "high", "low", "close"]] *= 0.999
    stream = build_stream(df, spot)
    assert stream["basis"].notna().sum() == periods
    assert stream["atr"].notna().sum() == periods - ATR_N + 1

    # 4) Panel/z-score construction works with two symbols and no fill.
    streams = {"a_usdt": stream, "b_usdt": stream.copy()}
    basis, change, z = build_panel(streams)
    assert basis.shape[0] == periods
    assert change.shape == basis.shape
    assert z.shape == basis.shape

    # 5) Pagination must stop after the required candles and must not make
    # an unnecessary boundary request. Synthetic API pages are bounded exactly
    # like the XT endpoint contract.
    old_request = globals()["request_json"]
    calls = []
    fetch_end = pd.to_datetime(now_hour_ms() - H, unit="ms", utc=True)
    fetch_start = fetch_end - pd.Timedelta(hours=periods - 1)
    fetch_idx = pd.date_range(fetch_start, periods=periods, freq="1h", tz="UTC")
    synthetic = [(int(ts.timestamp() * 1000), 100, 101, 99, 100.5, 1) for ts in fetch_idx]

    def fake_request(url, params, label, attempts=4):
        calls.append(dict(params))
        lo, hi = int(params["startTime"]), int(params["endTime"])
        # Emulate XT's observed end-exclusive bounded-window behavior.
        return {"result": [list(r) for r in synthetic if lo <= r[0] < hi]}

    try:
        globals()["request_json"] = fake_request
        test_dir = Path("/tmp/hunter_v23_10_selftest")
        test_dir.mkdir(parents=True, exist_ok=True)
        old_cache = globals()["CACHE"]
        globals()["CACHE"] = test_dir
        fetched = fetch_kline("synthetic://futures", "selftest_usdt", "futures")
        assert len(fetched) == periods
        assert calls, "pagination made no calls"
        assert all(c["startTime"] <= c["endTime"] for c in calls)
        assert len(calls) <= 14, f"unexpected pagination count: {len(calls)}"
        globals()["CACHE"] = old_cache
    finally:
        globals()["request_json"] = old_request

    print("[SELFTEST] PASS — validation, gap rejection, features, panel, XT end-exclusive pagination")



# ========================= V24 RESEARCH ENGINE =========================
# This stage is exploratory by design. It does NOT fit a trading strategy.
# Features use only information available at signal time. Forward returns and
# path labels are used only as research targets and are never fed back into
# features. No parameter is selected automatically for deployment.

HORIZONS = (1, 4, 8, 12, 24)
FWD_HORIZONS = (4, 8, 12, 24)
TAIL_Q = 0.20
MIN_EVENT_N = 100

FEATURES = [
    "ret_1h", "ret_4h", "ret_12h", "ret_24h",
    "atr_pct", "rv_24", "rv_72", "ema20_gap", "ema50_gap",
    "ema200_gap", "trend_stack", "range_pct", "range_expansion",
    "volume_z24", "close_pos_24", "breakout_up24", "breakout_dn24",
    "basis", "basis_chg24", "basis_z", "basis_cs_rank",
    "fut_ret24_cs_rank", "vol_cs_rank",
]


def _safe_z(s, n):
    mu = s.rolling(n, min_periods=n).mean().shift(1)
    sd = s.rolling(n, min_periods=n).std(ddof=0).shift(1)
    return (s - mu) / sd.replace(0.0, np.nan)


def _rank_cs(frame):
    return frame.rank(axis=1, pct=True, method="average")


def build_research_streams(streams):
    out = {}
    for symbol, x0 in streams.items():
        x = x0.copy().sort_index()
        c = x["close"]
        h = x["high"]
        l = x["low"]
        v = x["volume"]

        for n in HORIZONS:
            x[f"ret_{n}h"] = c / c.shift(n) - 1.0

        x["atr_pct"] = x["atr"] / c
        x["rv_24"] = x["ret_1h"].rolling(24, min_periods=24).std().shift(1) * np.sqrt(24)
        x["rv_72"] = x["ret_1h"].rolling(72, min_periods=72).std().shift(1) * np.sqrt(72)

        ema20 = c.ewm(span=20, adjust=False, min_periods=20).mean()
        ema50 = c.ewm(span=50, adjust=False, min_periods=50).mean()
        ema200 = c.ewm(span=200, adjust=False, min_periods=200).mean()
        x["ema20_gap"] = c / ema20 - 1.0
        x["ema50_gap"] = c / ema50 - 1.0
        x["ema200_gap"] = c / ema200 - 1.0
        x["trend_stack"] = ((ema20 > ema50) & (ema50 > ema200)).astype(float) - ((ema20 < ema50) & (ema50 < ema200)).astype(float)

        prev_c = c.shift(1)
        x["range_pct"] = (h - l) / prev_c
        med_range = x["range_pct"].rolling(24, min_periods=24).median().shift(1)
        x["range_expansion"] = x["range_pct"] / med_range.replace(0.0, np.nan)

        vm = v.rolling(24, min_periods=24).mean().shift(1)
        vs = v.rolling(24, min_periods=24).std(ddof=0).shift(1)
        x["volume_z24"] = (v - vm) / vs.replace(0.0, np.nan)

        hi24 = h.rolling(24, min_periods=24).max().shift(1)
        lo24 = l.rolling(24, min_periods=24).min().shift(1)
        x["close_pos_24"] = (c - lo24) / (hi24 - lo24).replace(0.0, np.nan)
        x["breakout_up24"] = (c / hi24 - 1.0)
        x["breakout_dn24"] = (c / lo24 - 1.0)

        # Forward labels are deliberately computed separately from the feature set.
        for n in FWD_HORIZONS:
            x[f"fwd_close_{n}h"] = c.shift(-n) / c - 1.0
            # Path extremes begin on the next completed bar. This prevents the
            # signal candle itself from becoming a future outcome.
            future_high = pd.concat([h.shift(-j) for j in range(1, n + 1)], axis=1).max(axis=1)
            future_low = pd.concat([l.shift(-j) for j in range(1, n + 1)], axis=1).min(axis=1)
            x[f"mfe_{n}h"] = future_high / c - 1.0
            x[f"mae_{n}h"] = future_low / c - 1.0

        # First-hit path label at 1 ATR risk, 2R target. Entry is the next bar
        # open, matching the eventual backtest convention. Same-bar TP+SL is LOSS.
        x["rr2_long"] = np.nan
        x["rr2_short"] = np.nan
        for i in range(len(x) - 1):
            atr = float(x.iloc[i]["atr"])
            if not np.isfinite(atr) or atr <= 0:
                continue
            entry_raw = float(x.iloc[i + 1]["open"])
            long_entry = entry_raw * (1.0 + SLIP)
            short_entry = entry_raw * (1.0 - SLIP)
            ld = atr
            sd = atr
            lt = long_entry + 2.0 * ld
            ls = long_entry - ld
            st = short_entry - 2.0 * sd
            ss = short_entry + sd
            long_result = np.nan
            short_result = np.nan
            for j in range(i + 1, len(x)):
                bh = float(x.iloc[j]["high"]); bl = float(x.iloc[j]["low"])
                lh = bh >= lt; ll = bl <= ls
                sh = bh >= ss; sl = bl <= st
                if lh or ll:
                    long_result = 1.0 if (lh and not ll) else -1.0
                    break
            for j in range(i + 1, len(x)):
                bh = float(x.iloc[j]["high"]); bl = float(x.iloc[j]["low"])
                sh = bh >= ss; sl = bl <= st
                if sh or sl:
                    short_result = 1.0 if (sl and not sh) else -1.0
                    break
            x.iloc[i, x.columns.get_loc("rr2_long")] = long_result
            x.iloc[i, x.columns.get_loc("rr2_short")] = short_result

        out[symbol] = x.replace([np.inf, -np.inf], np.nan)
    return out


def build_research_panel(research_streams):
    idx = sorted(set().union(*(x.index for x in research_streams.values())))
    idx = pd.DatetimeIndex(idx).sort_values()
    rows = []
    for symbol, x in research_streams.items():
        y = x.copy()
        y["symbol"] = symbol
        y["timestamp"] = y.index
        rows.append(y.reset_index(drop=True))
    panel = pd.concat(rows, ignore_index=True)

    # Cross-sectional ranks are same-timestamp information only.
    for col, outcol in [("ret_24h", "fut_ret24_cs_rank"), ("atr_pct", "vol_cs_rank")]:
        panel[outcol] = panel.groupby("timestamp")[col].rank(pct=True, method="average")
    panel["basis_cs_rank"] = panel.groupby("timestamp")["basis"].rank(pct=True, method="average")
    return panel


def _split_name(ts, start, end):
    span = end - start
    p1 = start + span * 0.50
    p2 = start + span * 0.75
    if ts < p1:
        return "DISCOVERY_50"
    if ts < p2:
        return "DEVELOPMENT_25"
    return "VALIDATION_25"


def feature_report(panel, feature, label):
    d = panel[["timestamp", "symbol", feature, label]].dropna().copy()
    if len(d) < MIN_EVENT_N:
        return None
    q20 = d[feature].quantile(TAIL_Q)
    q80 = d[feature].quantile(1.0 - TAIL_Q)
    groups = {
        "LOW20": d[d[feature] <= q20],
        "MID60": d[(d[feature] > q20) & (d[feature] < q80)],
        "HIGH20": d[d[feature] >= q80],
    }
    result = {"feature": feature, "label": label, "n_total": len(d)}
    for name, g in groups.items():
        result[f"{name}_n"] = len(g)
        result[f"{name}_mean"] = float(g[label].mean()) if len(g) else np.nan
        result[f"{name}_median"] = float(g[label].median()) if len(g) else np.nan
        result[f"{name}_positive"] = float((g[label] > 0).mean()) if len(g) else np.nan
    if len(groups["LOW20"]) and len(groups["HIGH20"]):
        result["HIGH_minus_LOW"] = result["HIGH20_mean"] - result["LOW20_mean"]
    else:
        result["HIGH_minus_LOW"] = np.nan
    return result


def directional_report(panel, feature, label):
    d = panel[["timestamp", "symbol", feature, label]].dropna().copy()
    if len(d) < MIN_EVENT_N:
        return None
    rows = []
    for side_name, sign in (("LONG", 1), ("SHORT", -1)):
        value = sign * d[feature]
        q = value.quantile(1.0 - TAIL_Q)
        g = d[value >= q]
        rows.append({
            "feature": feature, "label": label, "side": side_name,
            "n": len(g), "mean": float(g[label].mean()) if len(g) else np.nan,
            "positive": float((g[label] > 0).mean()) if len(g) else np.nan,
        })
    return rows


def summarize_rr(panel):
    rows = []
    for split in ("DISCOVERY_50", "DEVELOPMENT_25", "VALIDATION_25", "ALL"):
        d = panel if split == "ALL" else panel[panel["split"] == split]
        for side, col in (("LONG", "rr2_long"), ("SHORT", "rr2_short")):
            v = d[col].dropna()
            if len(v) == 0:
                continue
            wins = int((v > 0).sum()); losses = int((v < 0).sum())
            rows.append({
                "split": split, "side": side, "n": len(v),
                "wins": wins, "losses": losses,
                "win_rate": wins / len(v),
                "mean_R": float(np.where(v > 0, 2.0, -1.0).mean()),
            })
    return pd.DataFrame(rows)


def main():
    print("HUNTER-V24-RESEARCH — EDGE DISCOVERY ENGINE")
    print("No strategy fitting. No parameter optimization. No future data in features.")
    print(f"XT 1H | {LOOKBACK_DAYS}d research + {WARMUP_DAYS}d warmup | fixed universe")

    _self_test()
    REPORT.mkdir(parents=True, exist_ok=True)
    streams = {}
    for i, symbol in enumerate(SYMBOLS, 1):
        print(f"\n[{i}/{len(SYMBOLS)}] {symbol}")
        futures = fetch_kline(FUT_URL, symbol, "futures")
        spot = fetch_kline(SPOT_URL, symbol, "spot")
        streams[symbol] = build_stream(futures, spot)

    research_streams = build_research_streams(streams)
    panel = build_research_panel(research_streams)
    start = panel.timestamp.min(); end = panel.timestamp.max()
    panel["split"] = panel["timestamp"].map(lambda x: _split_name(x, start, end))
    panel.to_csv(REPORT / "research_panel.csv", index=False)

    # Forward-return event study.
    feature_rows = []
    for feature in FEATURES:
        for h in FWD_HORIZONS:
            label = f"fwd_close_{h}h"
            r = feature_report(panel, feature, label)
            if r is not None:
                feature_rows.append(r)
    fr = pd.DataFrame(feature_rows)
    if not fr.empty:
        fr.sort_values(["label", "HIGH_minus_LOW"], ascending=[True, False]).to_csv(
            REPORT / "feature_tail_report.csv", index=False
        )

    # Directional conditional study using only signal-time feature values.
    dr = []
    for feature in FEATURES:
        for h in FWD_HORIZONS:
            label = f"fwd_close_{h}h"
            r = directional_report(panel, feature, label)
            if r:
                dr.extend(r)
    pd.DataFrame(dr).to_csv(REPORT / "directional_tail_report.csv", index=False)

    rr = summarize_rr(panel)
    rr.to_csv(REPORT / "rr2_path_report.csv", index=False)

    # A compact stability table: for each feature, compare HIGH-vs-LOW mean
    # forward return separately in each time split. This is descriptive only.
    stability = []
    for feature in FEATURES:
        for h in FWD_HORIZONS:
            label = f"fwd_close_{h}h"
            for split in ("DISCOVERY_50", "DEVELOPMENT_25", "VALIDATION_25"):
                d = panel[panel.split == split][[feature, label]].dropna()
                if len(d) < MIN_EVENT_N:
                    continue
                q1 = d[feature].quantile(TAIL_Q); q2 = d[feature].quantile(1-TAIL_Q)
                lo = d[d[feature] <= q1][label]; hi = d[d[feature] >= q2][label]
                stability.append({
                    "feature": feature, "label": label, "split": split,
                    "n_low": len(lo), "n_high": len(hi),
                    "low_mean": float(lo.mean()), "high_mean": float(hi.mean()),
                    "high_minus_low": float(hi.mean() - lo.mean()),
                })
    pd.DataFrame(stability).to_csv(REPORT / "stability_by_split.csv", index=False)

    print("\n" + "=" * 80)
    print("RESEARCH COMPLETE")
    print("=" * 80)
    print(f"Panel rows       : {len(panel):,}")
    print(f"Panel timestamps : {panel.timestamp.nunique():,}")
    print(f"Panel start      : {start}")
    print(f"Panel end        : {end}")
    print("Reports:")
    for p in sorted(REPORT.glob("*.csv")):
        print(f"  {p}")
    print("\nIMPORTANT: This run does not select a strategy. It produces evidence for the next locked hypothesis.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nFATAL RESEARCH ERROR")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
