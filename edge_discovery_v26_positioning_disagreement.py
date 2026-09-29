# V26 — POSITIONING DISAGREEMENT RESEARCH ENGINE
# Purpose: test a pre-registered perpetual-futures positioning disagreement hypothesis.
# Data: XT public futures 4H OHLCV + XT public spot 4H OHLCV + XT funding-rate history.
# No external derivatives vendor, no API key, no ML, no parameter fitting.
# Causal features only. Entry is next-bar open. RR is 1:2 with 1 ATR risk / 2 ATR target.
# Same-candle SL+TP = LOSS. No timeout. No BE/trailing/pyramiding.

import json
import math
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------
# Locked research configuration
# -----------------------------
XT_FUT = "https://fapi.xt.com"
XT_SPOT = "https://sapi.xt.com"
FUT_KLINE_PATH = "/future/market/v1/public/q/kline"
SPOT_KLINE_PATH = "/v4/public/kline"
FUNDING_PATH = "/future/market/v1/public/q/funding-rate-record"

SYMS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "SUI/USDT", "AVAX/USDT",
    "NEAR/USDT", "ADA/USDT", "BNB/USDT", "APT/USDT", "CRV/USDT",
    "ONDO/USDT", "PENDLE/USDT", "ICP/USDT", "WIF/USDT",
]

DAYS = 455
WARMUP_DAYS = 120
TIMEFRAME = "4h"
BAR_MS = 4 * 3600 * 1000
LIMIT = 1000
MIN_KLINE_COVERAGE = 0.97
MIN_FUNDING_AVAILABLE_DAYS = 120

# Funding state is normalized within each symbol using only prior 4H observations.
FUNDING_LOOKBACK_BARS = 90 * 6  # 90 calendar days on a 4H grid.
FUNDING_MIN_PERIODS = 180       # at least 30 days on the 4H grid.
MAX_FUNDING_AGE_HOURS = 24.0

# Fixed, pre-registered tails. These are research branches, not tuned after results.
TAIL_LEVELS = (0.10, 0.20)

# Barrier protocol.
ATR_PERIOD = 14
RR_STOP_ATR = 1.0
RR_TARGET_ATR = 2.0

# Execution-cost model used only for research economics, not for signal construction.
INITIAL_CAPITAL = 1000.0
TRADE_MARGIN = 100.0
LEVERAGE = 50.0
NOTIONAL = TRADE_MARGIN * LEVERAGE
FEE_RATE = 0.0007
SLIPPAGE = 0.0003
ROUND_TRIP_COST_USD = NOTIONAL * 2.0 * (FEE_RATE + SLIPPAGE)

OUT = Path("reports/xt_v26_positioning_disagreement")
OUT.mkdir(parents=True, exist_ok=True)


# -----------------------------
# HTTP / JSON helpers
# -----------------------------
def http_get(base: str, path: str, params: dict, retries: int = 4):
    last = None
    query = urllib.parse.urlencode(params, doseq=True, safe="")
    url = base + path + ("?" + query if query else "")
    headers = {
        "Accept": "application/json",
        "User-Agent": "score-hunter-v26/1.0",
    }

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=30) as response:
                body = response.read().decode("utf-8")
                payload = json.loads(body)

            if isinstance(payload, dict):
                code = payload.get("returnCode", payload.get("rc", payload.get("code", 0)))
                if str(code) not in ("0", "200"):
                    err = payload.get("error")
                    msg = payload.get("msg", payload.get("mc", "API error"))
                    if isinstance(err, dict):
                        msg = err.get("msg") or err.get("code") or msg
                    raise RuntimeError(f"API error {code}: {msg}")
            return payload
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"GET failed: {url}: {last}")


def normalize_ts(value):
    if value is None:
        return None
    try:
        ts = int(float(value))
    except Exception:
        return None
    if ts < 10_000_000_000:
        ts *= 1000
    return ts


def unwrap_list(payload):
    """Best-effort XT wrapper parser for list responses."""
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []

    result = payload.get("result")
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for key in ("items", "data", "rows", "list", "candles", "klines"):
            if isinstance(result.get(key), list):
                return result[key]

    for key in ("data", "rows", "list", "items", "candles", "klines"):
        if isinstance(payload.get(key), list):
            return payload[key]
    return []


def parse_kline_row(row):
    if isinstance(row, dict):
        ts = row.get("t", row.get("time", row.get("timestamp", row.get("ts"))))
        op = row.get("o", row.get("open"))
        hi = row.get("h", row.get("high"))
        lo = row.get("l", row.get("low"))
        cl = row.get("c", row.get("close"))
        vol = row.get("v", row.get("volume", row.get("q", row.get("amount", 0))))
    elif isinstance(row, (list, tuple)) and len(row) >= 6:
        ts, op, hi, lo, cl, vol = row[:6]
    else:
        return None

    ts = normalize_ts(ts)
    if ts is None or any(v is None for v in (op, hi, lo, cl)):
        return None
    try:
        return [ts, float(op), float(hi), float(lo), float(cl), float(vol or 0)]
    except Exception:
        return None


# -----------------------------
# XT futures / spot K-lines
# -----------------------------
def fetch_klines(base: str, path: str, symbol: str, start_ms: int, end_ms: int, label: str):
    rows = []
    cur = start_ms
    page = 0
    previous_max = None
    window = LIMIT * BAR_MS
    xt_symbol = symbol.replace("/", "_").lower()

    while cur < end_ms:
        page += 1
        if page > 1000:
            raise RuntimeError(f"{label} pagination guard exceeded: {symbol}")

        remaining = end_ms - cur
        final_window = remaining <= window
        window_start = cur
        window_end = end_ms if final_window else cur + window

        params = {
            "symbol": xt_symbol,
            "interval": TIMEFRAME,
            "startTime": int(window_start),
            "endTime": int(window_end),
            "limit": LIMIT,
        }
        payload = http_get(base, path, params)
        raw = unwrap_list(payload)
        if not isinstance(raw, list):
            raise RuntimeError(f"{label} {symbol}: unexpected kline response")

        parsed = []
        for row in raw:
            item = parse_kline_row(row)
            if item is None:
                continue
            if window_start <= item[0] < window_end:
                parsed.append(item)

        parsed.sort(key=lambda x: x[0])
        if not parsed:
            if final_window:
                break
            raise RuntimeError(
                f"{label} {symbol}: empty bounded page={page}; "
                f"window={pd.to_datetime(window_start, unit='ms', utc=True)} -> "
                f"{pd.to_datetime(window_end, unit='ms', utc=True)}"
            )

        max_ts = parsed[-1][0]
        if previous_max is not None and max_ts <= previous_max and not final_window:
            raise RuntimeError(f"{label} pagination stalled: {symbol} page={page}")

        rows.extend(parsed)
        previous_max = max(previous_max or max_ts, max_ts)
        print(
            f"[{label}] {symbol} page={page} rows={len(set(r[0] for r in rows))} "
            f"latest={pd.to_datetime(max_ts, unit='ms', utc=True)}"
        )

        if final_window:
            break
        next_cur = max_ts + BAR_MS
        if next_cur <= cur:
            raise RuntimeError(f"{label} pagination did not advance: {symbol}")
        cur = next_cur
        time.sleep(0.05)

    if not rows:
        raise RuntimeError(f"No {label} data: {symbol}")

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)

    # Completed candles only.
    now_ms = int(time.time() * 1000)
    if len(df) and int(df["ts"].iloc[-1]) + BAR_MS > now_ms:
        df = df.iloc[:-1].copy()

    return df


# -----------------------------
# XT funding history pagination
# -----------------------------
def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_funding_items(payload):
    result = payload.get("result", {}) if isinstance(payload, dict) else {}
    if not isinstance(result, dict):
        return [], False, False
    items = result.get("items", [])
    if not isinstance(items, list):
        items = []
    return items, as_bool(result.get("hasNext", False)), as_bool(result.get("hasPrev", False))


def parse_funding_row(row):
    if not isinstance(row, dict):
        return None
    ts = normalize_ts(row.get("createdTime", row.get("time", row.get("t"))))
    raw_rate = row.get("fundingRate", row.get("rate"))
    row_id = row.get("id")
    interval = row.get("collectionInternal")
    if ts is None or raw_rate is None or row_id is None:
        return None
    try:
        rate = float(raw_rate)
        rid = int(row_id)
        interval_sec = float(interval) if interval is not None else np.nan
    except Exception:
        return None
    return {
        "id": rid,
        "ts": ts,
        "funding_rate": rate,
        "collection_internal_sec": interval_sec,
    }


def fetch_funding(symbol: str, start_ms: int, end_ms: int):
    xt_symbol = symbol.replace("/", "_").lower()
    limit = 1000

    def request(direction, cursor=None):
        params = {"symbol": xt_symbol, "direction": direction, "limit": limit}
        if cursor is not None:
            params["id"] = int(cursor)
        return http_get(XT_FUT, FUNDING_PATH, params)

    first = request("NEXT")
    first_items, first_has_next, first_has_prev = parse_funding_items(first)
    parsed_all = []
    for item in first_items:
        p = parse_funding_row(item)
        if p is not None:
            parsed_all.append(p)

    if not parsed_all:
        raise RuntimeError(f"No XT funding records: {symbol}")

    visited = set()

    # Walk toward older records.
    current_items = first_items
    current_has_prev = first_has_prev
    while current_has_prev:
        parsed = [parse_funding_row(x) for x in current_items]
        parsed = [x for x in parsed if x is not None]
        if not parsed:
            break
        cursor = min(x["id"] for x in parsed)
        key = ("PREV", cursor)
        if key in visited:
            break
        visited.add(key)

        min_time = min(x["ts"] for x in parsed)
        if min_time <= start_ms:
            break

        payload = request("PREV", cursor)
        current_items, _, current_has_prev = parse_funding_items(payload)
        if not current_items:
            break
        for item in current_items:
            p = parse_funding_row(item)
            if p is not None:
                parsed_all.append(p)
        print(f"[XT-FUND] {symbol} PREV id={cursor} rows_total={len(parsed_all)}")
        time.sleep(0.05)

    # Walk toward newer records if the initial page was historical/oldest.
    current_items = first_items
    current_has_next = first_has_next
    while current_has_next:
        parsed = [parse_funding_row(x) for x in current_items]
        parsed = [x for x in parsed if x is not None]
        if not parsed:
            break
        cursor = max(x["id"] for x in parsed)
        key = ("NEXT", cursor)
        if key in visited:
            break
        visited.add(key)

        max_time = max(x["ts"] for x in parsed)
        if max_time >= end_ms:
            break

        payload = request("NEXT", cursor)
        current_items, current_has_next, _ = parse_funding_items(payload)
        if not current_items:
            break
        for item in current_items:
            p = parse_funding_row(item)
            if p is not None:
                parsed_all.append(p)
        print(f"[XT-FUND] {symbol} NEXT id={cursor} rows_total={len(parsed_all)}")
        time.sleep(0.05)

    df = pd.DataFrame(parsed_all)
    df = df.drop_duplicates("id").sort_values("ts").reset_index(drop=True)
    df = df[(df.ts >= start_ms) & (df.ts <= end_ms)].copy()

    if df.empty:
        raise RuntimeError(f"XT {symbol}: funding history empty after date filter")

    # Funding timestamps must be strictly increasing after de-duplication.
    if df.ts.duplicated().any():
        df = df.drop_duplicates("ts", keep="last").reset_index(drop=True)
    if not df.ts.is_monotonic_increasing:
        raise RuntimeError(f"XT {symbol}: funding timestamps are not monotonic")

    return df


# -----------------------------
# Feature construction
# -----------------------------
def build_features(fut, spot, funding):
    f = fut.copy()
    s = spot[["ts", "close", "volume"]].copy().rename(
        columns={"close": "spot_close", "volume": "spot_volume"}
    )

    f["dt"] = pd.to_datetime(f.ts, unit="ms", utc=True)
    s["dt"] = pd.to_datetime(s.ts, unit="ms", utc=True)

    # Exact 4H alignment; do not silently forward-fill spot prices.
    m = f.merge(s[["ts", "spot_close", "spot_volume"]], on="ts", how="left", validate="one_to_one")
    m["spot_missing"] = m.spot_close.isna()
    if m["spot_missing"].mean() > (1.0 - MIN_KLINE_COVERAGE):
        raise RuntimeError(
            f"Spot/futures alignment coverage failed: missing={m.spot_missing.mean():.4%}"
        )

    mf = funding[["ts", "funding_rate", "collection_internal_sec"]].copy().sort_values("ts")
    mf["fund_dt"] = pd.to_datetime(mf.ts, unit="ms", utc=True)

    # A signal is generated after the 4H bar closes, so only funding records
    # settled at or before that bar close are available.
    m["bar_close_ts"] = m.ts + BAR_MS
    m = pd.merge_asof(
        m.sort_values("bar_close_ts"),
        mf[["ts", "fund_dt", "funding_rate", "collection_internal_sec"]],
        left_on="bar_close_ts",
        right_on="ts",
        direction="backward",
        allow_exact_matches=True,
        suffixes=("", "_fund"),
    )
    m.rename(columns={"ts_fund": "funding_ts"}, inplace=True)

    m["bar_close_dt"] = pd.to_datetime(m.bar_close_ts, unit="ms", utc=True)
    m["funding_age_hours"] = (m.bar_close_dt - m.fund_dt).dt.total_seconds() / 3600.0
    m.loc[m.funding_age_hours < 0, "funding_age_hours"] = np.nan
    m.loc[m.funding_age_hours > MAX_FUNDING_AGE_HOURS, "funding_rate"] = np.nan

    # Price / spot / basis features.
    m["fut_ret_8h"] = m.close.pct_change(2)
    m["spot_ret_8h"] = m.spot_close.pct_change(2)
    m["basis"] = m.close / m.spot_close - 1.0
    m["basis_change_8h"] = m.basis - m.basis.shift(2)
    m["fut_minus_spot_ret_8h"] = m.fut_ret_8h - m.spot_ret_8h

    # Causal funding normalization: current value is compared only with prior
    # 90 calendar days on the 4H grid.
    prior = m.funding_rate.shift(1)
    m["funding_mean_90d"] = prior.rolling(
        FUNDING_LOOKBACK_BARS, min_periods=FUNDING_MIN_PERIODS
    ).mean()
    m["funding_std_90d"] = prior.rolling(
        FUNDING_LOOKBACK_BARS, min_periods=FUNDING_MIN_PERIODS
    ).std(ddof=0)
    m["funding_q10_90d"] = prior.rolling(
        FUNDING_LOOKBACK_BARS, min_periods=FUNDING_MIN_PERIODS
    ).quantile(0.10)
    m["funding_q20_90d"] = prior.rolling(
        FUNDING_LOOKBACK_BARS, min_periods=FUNDING_MIN_PERIODS
    ).quantile(0.20)
    m["funding_q80_90d"] = prior.rolling(
        FUNDING_LOOKBACK_BARS, min_periods=FUNDING_MIN_PERIODS
    ).quantile(0.80)
    m["funding_q90_90d"] = prior.rolling(
        FUNDING_LOOKBACK_BARS, min_periods=FUNDING_MIN_PERIODS
    ).quantile(0.90)
    m["funding_z_90d"] = (
        (m.funding_rate - m.funding_mean_90d) /
        m.funding_std_90d.replace(0, np.nan)
    )
    m["funding_change_8h"] = m.funding_rate - m.funding_rate.shift(2)

    # Volatility used only for barrier sizing and cost normalization.
    prev_close = m.close.shift(1)
    tr = pd.concat(
        [m.high - m.low, (m.high - prev_close).abs(), (m.low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    m["atr"] = tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()
    m["atr_pct"] = m.atr / m.close

    return m


# -----------------------------
# Barrier labels and economics
# -----------------------------
def find_barrier(x: pd.DataFrame, i: int, direction: str):
    n = len(x)
    if i + 1 >= n:
        return None

    atr = x.atr.iloc[i]
    entry = x.open.iloc[i + 1]
    if not np.isfinite(atr) or atr <= 0 or not np.isfinite(entry) or entry <= 0:
        return None

    if direction == "LONG":
        stop = entry - RR_STOP_ATR * atr
        target = entry + RR_TARGET_ATR * atr
    else:
        stop = entry + RR_STOP_ATR * atr
        target = entry - RR_TARGET_ATR * atr

    for j in range(i + 1, n):
        hi = x.high.iloc[j]
        lo = x.low.iloc[j]
        if direction == "LONG":
            # Conservative ambiguity rule: if both are hit in the same candle,
            # stop is taken first => LOSS.
            if lo <= stop and hi >= target:
                return {
                    "exit_idx": j,
                    "outcome": -1.0,
                    "exit_price": stop,
                    "reason": "BOTH_SAME_CANDLE_STOP_FIRST",
                    "entry": entry,
                    "atr": atr,
                    "risk_price": atr,
                }
            if lo <= stop:
                return {
                    "exit_idx": j,
                    "outcome": -1.0,
                    "exit_price": stop,
                    "reason": "STOP",
                    "entry": entry,
                    "atr": atr,
                    "risk_price": atr,
                }
            if hi >= target:
                return {
                    "exit_idx": j,
                    "outcome": 2.0,
                    "exit_price": target,
                    "reason": "TARGET",
                    "entry": entry,
                    "atr": atr,
                    "risk_price": atr,
                }
        else:
            if hi >= stop and lo <= target:
                return {
                    "exit_idx": j,
                    "outcome": -1.0,
                    "exit_price": stop,
                    "reason": "BOTH_SAME_CANDLE_STOP_FIRST",
                    "entry": entry,
                    "atr": atr,
                    "risk_price": atr,
                }
            if hi >= stop:
                return {
                    "exit_idx": j,
                    "outcome": -1.0,
                    "exit_price": stop,
                    "reason": "STOP",
                    "entry": entry,
                    "atr": atr,
                    "risk_price": atr,
                }
            if lo <= target:
                return {
                    "exit_idx": j,
                    "outcome": 2.0,
                    "exit_price": target,
                    "reason": "TARGET",
                    "entry": entry,
                    "atr": atr,
                    "risk_price": atr,
                }

    return {
        "exit_idx": None,
        "outcome": np.nan,
        "exit_price": np.nan,
        "reason": "CENSORED_END_OF_SAMPLE",
        "entry": entry,
        "atr": atr,
        "risk_price": atr,
    }


def funding_between(funding: pd.DataFrame, entry_ts: int, exit_ts: int, direction: str):
    # Position must already be open strictly before the funding settlement and
    # still be open strictly after it. This avoids charging a settlement that
    # occurs exactly at entry or exactly at exit.
    q = funding[(funding.ts > entry_ts) & (funding.ts < exit_ts)]
    if q.empty:
        return 0.0, 0
    signed = -1.0 if direction == "LONG" else 1.0
    funding_usd = float((NOTIONAL * q.funding_rate * signed).sum())
    return funding_usd, len(q)


def enrich_trade(x: pd.DataFrame, funding: pd.DataFrame, signal_idx: int, direction: str, barrier):
    entry_ts = int(x.ts.iloc[signal_idx + 1])
    exit_ts = int(x.ts.iloc[barrier["exit_idx"]]) if barrier["exit_idx"] is not None else None
    atr = barrier["atr"]
    entry = barrier["entry"]
    risk_usd = NOTIONAL * (atr / entry)
    cost_r = ROUND_TRIP_COST_USD / risk_usd if risk_usd > 0 else np.nan

    if exit_ts is None:
        return {
            "entry_ts": entry_ts,
            "exit_ts": np.nan,
            "holding_bars": np.nan,
            "gross_r": np.nan,
            "trading_cost_r": np.nan,
            "funding_r": np.nan,
            "net_r": np.nan,
            "funding_events": np.nan,
            "entry_price": entry,
            "exit_price": np.nan,
            "atr": atr,
            "risk_usd": risk_usd,
            "reason": barrier["reason"],
        }

    funding_usd, funding_events = funding_between(funding, entry_ts, exit_ts, direction)
    funding_r = funding_usd / risk_usd if risk_usd > 0 else np.nan
    net_r = barrier["outcome"] - cost_r + funding_r

    return {
        "entry_ts": entry_ts,
        "exit_ts": exit_ts,
        "holding_bars": barrier["exit_idx"] - signal_idx,
        "gross_r": barrier["outcome"],
        "trading_cost_r": cost_r,
        "funding_r": funding_r,
        "net_r": net_r,
        "funding_events": funding_events,
        "entry_price": entry,
        "exit_price": barrier["exit_price"],
        "atr": atr,
        "risk_usd": risk_usd,
        "reason": barrier["reason"],
    }


# -----------------------------
# Fixed pre-registered setups
# -----------------------------
def signal_masks(df: pd.DataFrame):
    low10 = (
        (df.funding_rate < 0)
        & (df.funding_rate <= df.funding_q10_90d)
        & (df.spot_ret_8h > 0)
    )
    high10 = (
        (df.funding_rate > 0)
        & (df.funding_rate >= df.funding_q90_90d)
        & (df.spot_ret_8h < 0)
    )
    low20 = (
        (df.funding_rate < 0)
        & (df.funding_rate <= df.funding_q20_90d)
        & (df.spot_ret_8h > 0)
    )
    high20 = (
        (df.funding_rate > 0)
        & (df.funding_rate >= df.funding_q80_90d)
        & (df.spot_ret_8h < 0)
    )

    # Basis-confirmed ablation: not a new tuned filter. It tests whether the
    # same disagreement survives when futures-vs-spot basis is repricing in the
    # same direction as spot.
    basis_l = df.basis_change_8h > 0
    basis_s = df.basis_change_8h < 0

    return {
        "CORE_T10": {"LONG": low10, "SHORT": high10},
        "BASIS_CONF_T10": {"LONG": low10 & basis_l, "SHORT": high10 & basis_s},
        "CORE_T20": {"LONG": low20, "SHORT": high20},
        "BASIS_CONF_T20": {"LONG": low20 & basis_l, "SHORT": high20 & basis_s},
    }


# -----------------------------
# Reporting helpers
# -----------------------------
def split_name(ts, split_a, split_b):
    if ts < split_a:
        return "DISCOVERY_50"
    if ts < split_b:
        return "DEVELOPMENT_25"
    return "VALIDATION_25"


def max_loss_streak(values):
    best = 0
    cur = 0
    for value in values:
        if value < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def summarize(group: pd.DataFrame, setup: str, direction: str, mode: str):
    closed = group[group.gross_r.notna()].copy()
    wins = int((closed.gross_r > 0).sum())
    losses = int((closed.gross_r < 0).sum())
    gross_pf_num = closed.loc[closed.gross_r > 0, "gross_r"].sum()
    gross_pf_den = -closed.loc[closed.gross_r < 0, "gross_r"].sum()
    net_pf_num = closed.loc[closed.net_r > 0, "net_r"].sum()
    net_pf_den = -closed.loc[closed.net_r < 0, "net_r"].sum()
    return {
        "setup": setup,
        "direction": direction,
        "mode": mode,
        "n_signals": len(group),
        "n_closed": len(closed),
        "n_censored": int(group.gross_r.isna().sum()),
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / len(closed)) if len(closed) else np.nan,
        "mean_gross_R": closed.gross_r.mean() if len(closed) else np.nan,
        "mean_net_R": closed.net_r.mean() if len(closed) else np.nan,
        "gross_profit_factor": (gross_pf_num / gross_pf_den) if gross_pf_den > 0 else np.nan,
        "net_profit_factor": (net_pf_num / net_pf_den) if net_pf_den > 0 else np.nan,
        "mean_trading_cost_R": closed.trading_cost_r.mean() if len(closed) else np.nan,
        "mean_funding_R": closed.funding_r.mean() if len(closed) else np.nan,
        "mean_holding_bars": closed.holding_bars.mean() if len(closed) else np.nan,
        "max_loss_streak_net": max_loss_streak(closed.net_r.tolist()) if len(closed) else np.nan,
        "raw_1R_2R_break_even_wr": 1.0 / 3.0,
    }


def assign_time_splits(panel):
    times = np.sort(panel.dt.unique())
    a = times[int(0.50 * len(times))]
    b = times[int(0.75 * len(times))]
    panel["split"] = panel.dt.apply(lambda x: split_name(x, a, b))
    return panel, a, b


def run_setup_on_symbol(df, funding, setup, direction, signal_mask):
    signal_indices = np.flatnonzero(signal_mask.to_numpy(dtype=bool))
    trades = []
    last_exit_idx = -1

    for i in signal_indices:
        barrier = find_barrier(df, int(i), direction)
        if barrier is None:
            continue

        enriched = enrich_trade(df, funding, int(i), direction, barrier)
        row = {
            "symbol": str(df["symbol"].iloc[0]) if "symbol" in df.columns and len(df) else "",
            "setup": setup,
            "direction": direction,
            "signal_idx": int(i),
            "signal_ts": int(df.ts.iloc[i]),
            "signal_dt": df.dt.iloc[i],
            "split": df.split.iloc[i],
            "locked_nonoverlap": False,
        }
        row.update(enriched)
        trades.append(row)

    # A second pass applies the causal one-position-per-symbol lock using the
    # future exit only as backtest state. It never changes the signal itself.
    for row in trades:
        i = row["signal_idx"]
        # Recover exit index from exit timestamp.
        if pd.notna(row["exit_ts"]):
            matches = np.flatnonzero(df.ts.to_numpy() == int(row["exit_ts"]))
            exit_idx = int(matches[0]) if len(matches) else None
        else:
            exit_idx = None

        if i > last_exit_idx:
            row["locked_nonoverlap"] = True
            if exit_idx is None:
                last_exit_idx = len(df) + 1
            else:
                last_exit_idx = exit_idx

    return trades


# -----------------------------
# Synthetic unit tests
# -----------------------------
def self_test():
    idx = pd.date_range("2025-01-01", periods=12, freq="4h", tz="UTC")
    # Long test: next-bar open 100, stop 99, target 102; candle 2 hits both.
    x = pd.DataFrame({
        "ts": (idx.view("int64") // 1_000_000).astype(np.int64),
        "open": [100] * 12,
        "high": [100, 100, 102, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        "low":  [100, 100,  98, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        "close": [100] * 12,
        "volume": [1] * 12,
        "atr": [1] * 12,
        "dt": idx,
        "split": ["DISCOVERY_50"] * 12,
    })
    x.attrs["symbol"] = "TEST/USDT"
    b = find_barrier(x, 0, "LONG")
    assert b is not None and b["outcome"] == -1.0

    # Short test: target hits before stop.
    x2 = x.copy()
    x2["high"] = 100.5
    x2["low"] = [100, 100, 97, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    b2 = find_barrier(x2, 0, "SHORT")
    assert b2 is not None and b2["outcome"] == 2.0

    # Funding sign test.
    f = pd.DataFrame({"ts": [idx[2].value // 1_000_000], "funding_rate": [0.001]})
    pnl, n = funding_between(f, int(idx[0].value // 1_000_000), int(idx[3].value // 1_000_000), "SHORT")
    assert n == 1 and pnl > 0
    pnl2, n2 = funding_between(f, int(idx[0].value // 1_000_000), int(idx[3].value // 1_000_000), "LONG")
    assert n2 == 1 and pnl2 < 0
    print("SELF-TEST PASS")


# -----------------------------
# Main research run
# -----------------------------
def main():
    self_test()

    end = pd.Timestamp.now(tz="UTC").floor(TIMEFRAME)
    start = end - pd.Timedelta(days=DAYS)
    fetch_start = start - pd.Timedelta(days=WARMUP_DAYS)

    a = int(fetch_start.timestamp() * 1000)
    b = int(end.timestamp() * 1000)

    panels = []
    funding_cov = []
    kline_cov = []
    all_trades = []

    for sym in SYMS:
        print("[DATA]", sym)
        fut = fetch_klines(XT_FUT, FUT_KLINE_PATH, sym, a, b, "XT-FUT")
        spot = fetch_klines(XT_SPOT, SPOT_KLINE_PATH, sym, a, b, "XT-SPOT")
        funding = fetch_funding(sym, a, b)

        fut = fut[(fut.ts >= int(start.timestamp() * 1000)) & (fut.ts < b)].copy()
        spot = spot[(spot.ts >= int(start.timestamp() * 1000)) & (spot.ts < b)].copy()
        funding = funding[(funding.ts >= a) & (funding.ts <= b)].copy()

        expected = int(DAYS * 24 / 4)
        fut_cov = len(fut) / expected if expected else 0.0
        spot_cov = len(spot) / expected if expected else 0.0
        if fut_cov < MIN_KLINE_COVERAGE:
            raise RuntimeError(f"Futures coverage failure {sym}: {len(fut)}/{expected}={fut_cov:.4f}")
        if spot_cov < MIN_KLINE_COVERAGE:
            raise RuntimeError(f"Spot coverage failure {sym}: {len(spot)}/{expected}={spot_cov:.4f}")

        diffs = funding.ts.diff().dropna() / 1000.0
        diffs = diffs[diffs > 0]
        median_interval = float(diffs.median()) if len(diffs) else np.nan
        max_gap_h = float(diffs.max() / 3600.0) if len(diffs) else np.nan

        # XT's public funding-history endpoint may expose a shorter historical
        # retention window than the OHLCV endpoint. Do not manufacture missing
        # funding observations and do not reject the entire experiment merely
        # because the requested 455-day OHLCV window is longer. Instead, audit
        # the actually available funding span and let causal features produce
        # signals only where funding is genuinely available.
        funding_first_ms = int(funding.ts.min())
        funding_last_ms = int(funding.ts.max())
        available_start_ms = max(int(start.timestamp() * 1000), funding_first_ms)
        available_end_ms = min(b, funding_last_ms)
        available_days = max(0.0, (available_end_ms - available_start_ms) / 86400000.0)
        expected_funding_available = (
            ((available_end_ms - available_start_ms) / 1000.0) / median_interval
            if np.isfinite(median_interval) and median_interval > 0 and available_end_ms > available_start_ms
            else np.nan
        )
        fund_cadence_cov = (
            len(funding) / expected_funding_available
            if np.isfinite(expected_funding_available) and expected_funding_available > 0
            else np.nan
        )
        if available_days < MIN_FUNDING_AVAILABLE_DAYS:
            raise RuntimeError(
                f"Funding history too short for {sym}: available_days={available_days:.1f}, "
                f"minimum={MIN_FUNDING_AVAILABLE_DAYS}"
            )

        print(
            f"[FUNDING] {sym} events={len(funding)} median_interval_h="
            f"{median_interval / 3600.0:.2f} max_gap_h={max_gap_h:.2f} "
            f"available_days={available_days:.1f} cadence_cov={fund_cadence_cov:.4f}"
        )

        df = build_features(fut, spot, funding)
        df["symbol"] = sym
        df = df.sort_values("dt").reset_index(drop=True)
        df.attrs["symbol"] = sym

        # Global time splits are identical across the fixed 4H panel.
        split_a_ts = start + (end - start) * 0.50
        split_b_ts = start + (end - start) * 0.75
        df["split"] = np.where(
            df.dt < split_a_ts,
            "DISCOVERY_50",
            np.where(df.dt < split_b_ts, "DEVELOPMENT_25", "VALIDATION_25")
        )

        # Exact research window and causal split assignment.
        df = df[(df.ts >= int(start.timestamp() * 1000)) & (df.ts < b)].copy().reset_index(drop=True)
        panels.append(df)

        kline_cov.append({
            "symbol": sym,
            "futures_rows": len(fut),
            "spot_rows": len(spot),
            "expected_4h_rows": expected,
            "futures_coverage": fut_cov,
            "spot_coverage": spot_cov,
            "spot_missing_after_join": float(df.spot_missing.mean()),
        })
        funding_cov.append({
            "symbol": sym,
            "funding_events": len(funding),
            "requested_research_days": DAYS,
            "median_interval_hours": median_interval / 3600.0 if np.isfinite(median_interval) else np.nan,
            "expected_events_available_span_approx": expected_funding_available,
            "funding_cadence_coverage_available_span": fund_cadence_cov,
            "available_funding_days": available_days,
            "requested_window_coverage": available_days / DAYS if DAYS > 0 else np.nan,
            "max_gap_hours": max_gap_h,
            "first_funding_dt": pd.to_datetime(funding.ts.min(), unit="ms", utc=True),
            "last_funding_dt": pd.to_datetime(funding.ts.max(), unit="ms", utc=True),
        })

        masks = signal_masks(df)
        for setup, by_dir in masks.items():
            for direction, mask in by_dir.items():
                trades = run_setup_on_symbol(df, funding, setup, direction, mask)
                all_trades.extend(trades)

    panel = pd.concat(panels, ignore_index=True).sort_values(["dt", "symbol"]).reset_index(drop=True)
    panel, split_a, split_b = assign_time_splits(panel)

    # Re-apply split labels to per-symbol signal rows through timestamp matching.
    split_lookup = panel[["symbol", "ts", "split"]].drop_duplicates().rename(columns={"ts": "signal_ts"})
    if all_trades:
        trades = pd.DataFrame(all_trades)
        trades = trades.drop(columns=["split"], errors="ignore").merge(
            split_lookup, on=["symbol", "signal_ts"], how="left"
        )
    else:
        trades = pd.DataFrame()

    # Reports.
    panel.to_csv(OUT / "research_panel.csv", index=False)
    pd.DataFrame(kline_cov).to_csv(OUT / "kline_coverage_audit.csv", index=False)
    pd.DataFrame(funding_cov).to_csv(OUT / "funding_coverage_audit.csv", index=False)

    # Signal labels on the panel itself, useful for audit/reproduction.
    panel_masks = []
    for sym, g in panel.groupby("symbol", sort=False):
        mm = signal_masks(g)
        q = g[["symbol", "ts"]].copy()
        for setup, dirs in mm.items():
            for direction, mask in dirs.items():
                q[f"signal_{setup}_{direction.lower()}"] = mask.to_numpy(dtype=bool)
        panel_masks.append(q)
    sig_df = pd.concat(panel_masks, ignore_index=True)
    panel = panel.merge(sig_df, on=["symbol", "ts"], how="left")
    panel.to_csv(OUT / "research_panel.csv", index=False)

    if trades.empty:
        raise RuntimeError("No research events were generated; inspect funding/feature coverage before proceeding.")

    trades["signal_dt"] = pd.to_datetime(trades.signal_ts, unit="ms", utc=True)
    trades["entry_dt"] = pd.to_datetime(trades.entry_ts, unit="ms", utc=True)
    trades["exit_dt"] = pd.to_datetime(trades.exit_ts, unit="ms", utc=True)

    trade_cols = [
        "symbol", "setup", "direction", "signal_idx", "signal_ts", "signal_dt", "split",
        "entry_ts", "entry_dt", "exit_ts", "exit_dt", "holding_bars", "entry_price",
        "exit_price", "atr", "risk_usd", "gross_r", "trading_cost_r", "funding_r",
        "net_r", "funding_events", "reason", "locked_nonoverlap",
    ]
    trades[trade_cols].to_csv(OUT / "trade_log.csv", index=False)

    report_rows = []
    for setup in trades.setup.unique():
        for direction in ("LONG", "SHORT"):
            for mode, q in {
                "ALL_EVENTS": trades[(trades.setup == setup) & (trades.direction == direction)],
                "LOCKED_NONOVERLAP": trades[(trades.setup == setup) & (trades.direction == direction) & trades.locked_nonoverlap],
            }.items():
                qq = q.copy()
                for split in ("DISCOVERY_50", "DEVELOPMENT_25", "VALIDATION_25"):
                    z = qq[qq.split == split]
                    report_rows.append(summarize(z, setup, direction, f"{mode}_{split}"))
                report_rows.append(summarize(qq, setup, direction, mode + "_ALL_SPLITS"))

    pd.DataFrame(report_rows).to_csv(OUT / "event_study_report.csv", index=False)

    # Per-symbol locked summary. This is descriptive, not a ranking.
    sym_rows = []
    for (setup, direction, symbol), q in trades.groupby(["setup", "direction", "symbol"]):
        qq = q[q.locked_nonoverlap].copy()
        sym_rows.append(summarize(qq, setup, direction, f"LOCKED_SYMBOL_{symbol}"))
    pd.DataFrame(sym_rows).to_csv(OUT / "per_symbol_locked_report.csv", index=False)

    # Funding/basis state snapshot report for the actual signal events.
    state_cols = [
        "symbol", "ts", "dt", "funding_rate", "funding_z_90d", "funding_change_8h",
        "funding_age_hours", "spot_ret_8h", "fut_ret_8h", "basis", "basis_change_8h",
        "fut_minus_spot_ret_8h", "atr_pct",
    ]
    event_state = panel.merge(
        trades[["symbol", "signal_ts", "setup", "direction", "locked_nonoverlap"]].rename(columns={"signal_ts": "ts"}),
        on=["symbol", "ts"], how="inner"
    )
    event_state[state_cols + ["setup", "direction", "locked_nonoverlap"]].to_csv(
        OUT / "event_state_snapshot.csv", index=False
    )

    notes = f"""# V26 Positioning Disagreement Research

Research window: {start} -> {end} UTC ({DAYS} days) with {WARMUP_DAYS} days warmup.
Universe: {len(SYMS)} fixed XT USDT perpetual symbols.
Timeframe: 4H.

## Data sources
- XT futures 4H OHLCV: {XT_FUT}{FUT_KLINE_PATH}
- XT spot 4H OHLCV: {XT_SPOT}{SPOT_KLINE_PATH}
- XT funding history: {XT_FUT}{FUNDING_PATH}

XT documents the funding-history endpoint as a public GET method that does not require a signature; it returns fundingRate, createdTime, collectionInternal and id with cursor pagination.

## Pre-registered hypothesis
An extreme funding state is potentially informative when the underlying spot market has already moved against the crowded side.
Long disagreement = negative extreme funding + positive 8H spot return.
Short disagreement = positive extreme funding + negative 8H spot return.
Two fixed tails are examined: 10% and 20%. A separate basis-confirmed ablation requires 8H futures-vs-spot basis change to have the same sign as spot.

## Barrier protocol
Entry = next-bar open. Stop = 1 ATR. Target = 2 ATR. Same-candle stop+target = LOSS. No timeout. No BE, trailing, pyramiding or future-data filters. Unresolved end-of-sample trades are censored and excluded from WR/mean-R, while still blocking later signals in the non-overlap backtest state.

## Economics
Initial capital = ${INITIAL_CAPITAL:.2f}; margin = ${TRADE_MARGIN:.2f}; leverage = {LEVERAGE:.0f}x; notional = ${NOTIONAL:.2f}.
Trading cost model = fee {FEE_RATE:.4%}/side + slippage {SLIPPAGE:.4%}/side. Funding is charged/credited from settled XT funding records strictly between entry and exit.

## No-lookahead controls
- Funding is merged only as of bar close; stale funding older than {MAX_FUNDING_AGE_HOURS:.0f}h is not used.
- Funding quantiles/z-scores use only prior observations (shift(1)).
- Spot is exact-timestamp joined; no spot gap fill is used.
- Signals are timestamped before entry; entry is the next candle open.
- Locked mode allows only one position per symbol and forbids same-candle re-entry.

Split boundaries: Discovery 50% = {pd.Timestamp(split_a)}, Development/Validation boundary = {pd.Timestamp(split_b)}.
"""

    (OUT / "README.md").write_text(notes, encoding="utf-8")

    print("V26 COMPLETE:", len(panel), "panel rows;", panel.symbol.nunique(), "symbols;", len(trades), "events")
    print("OUTPUT:", OUT.resolve())


if __name__ == "__main__":
    main()
