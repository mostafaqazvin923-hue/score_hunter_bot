#!/usr/bin/env python3
"""
HUNTER-V12 — STATE / FLOW / LIQUIDITY / REPRICING RESEARCH ENGINE

XT USDT-M PERPETUAL FUTURES
15m raw -> causal 1H / 4H / 1D

Research-first. No lookahead. No repainting. No fabricated OI/liquidations.

Core thesis:
    Liquidity sweep -> failed auction / absorption -> repricing
    confirmed by futures-vs-spot basis, volume participation and BTC-relative strength.

Important data-policy:
- Futures and spot OHLCV are downloaded from XT directly.
- Historical order-book snapshots and historical trade-by-trade aggressor flow are NOT
  assumed to exist for the full 365-day window. Therefore this engine does NOT fabricate
  CVD/OI/liquidation history from candles.
- `flow_pressure` below is explicitly a BAR-AGGREGATED PARTICIPATION PROXY, not true CVD.
  It is built from candle body/range * volume and is audited separately.
- Funding/OI are diagnostics only when an external historical file is supplied; they are
  never silently substituted with current values.

Execution:
- Initial equity $1000
- Isolated margin $100
- 50x leverage => $5000 notional
- RR 1:2
- no timeout
- no BE / trailing
- no overlapping trades
- no same-candle re-entry after a close
- same-candle SL+TP => LOSS
- position still open at dataset end => OPEN, not forced loss
- capital-aware: no new $100-margin trade if equity < $100

Selection:
- Train factor audit
- Predeclared small config grid
- Train -> validation selection
- untouched OOS
- no parameter changes based on OOS
"""

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests


SYMBOLS = [
    "btc_usdt", "eth_usdt", "sol_usdt", "sui_usdt", "avax_usdt",
    "near_usdt", "ada_usdt", "bnb_usdt", "apt_usdt", "crv_usdt",
    "ondo_usdt", "pendle_usdt", "icp_usdt", "wif_usdt",
]

FUTURES_URL = "https://fapi.xt.com/future/market/v1/public/q/kline"
SPOT_URL = "https://sapi.xt.com/v4/public/kline"

DATA_DIR = Path("data/xt_v12")
FUT_DIR = DATA_DIR / "futures"
SPOT_DIR = DATA_DIR / "spot"

DAYS = 365
WARMUP_DAYS = 90
INTERVAL_MS = 15 * 60 * 1000
FUTURES_LIMIT = 1500
SPOT_LIMIT = 1000

INITIAL_CAPITAL = 1000.0
MARGIN = 100.0
LEVERAGE = 50.0
NOTIONAL = MARGIN * LEVERAGE
RR = 2.0
FEE_RATE = 0.0007
SLIPPAGE = 0.0003

MIN_ROWS = 38000
MAX_GAP_MINUTES = 15.0

TRAIN_FRAC = 0.60
VAL_FRAC = 0.20

MIN_TRAIN_TRADES = 40
MIN_VALIDATION_TRADES = 25
MIN_OOS_TRADES = 150
MIN_OOS_WR = 50.0
MIN_OOS_PF = 1.20
MAX_OOS_STREAK = 4
MAX_OOS_DD_PCT = 50.0


@dataclass(frozen=True)
class Split:
    start: pd.Timestamp
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class Config:
    name: str
    sweep_lookback: int
    vol_z_min: float
    basis_z_max: float
    flow_min: float
    reclaim_min: float
    rr_atr: float


def env_refresh():
    return os.getenv("XT_REFRESH", "0") == "1"


def payload_rows(obj):
    if isinstance(obj, list):
        return obj
    if not isinstance(obj, dict):
        return None
    for k in ("result", "data", "rows", "list"):
        v = obj.get(k)
        if isinstance(v, list):
            return v
    return None


def parse_row(r, is_spot=False):
    try:
        if isinstance(r, dict):
            ts = r.get("t", r.get("timestamp"))
            o = r.get("o", r.get("open"))
            h = r.get("h", r.get("high"))
            l = r.get("l", r.get("low"))
            c = r.get("c", r.get("close"))
            # XT Spot kline: q = base/quote volume field used by CCXT.
            # XT Futures kline: a = volume field.
            v = r.get("q") if is_spot else r.get("a")
            if v is None:
                v = r.get("volume")
            if ts is None:
                return None
            return int(ts), float(o), float(h), float(l), float(c), float(v)
        if isinstance(r, (list, tuple)) and len(r) >= 6:
            return int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])
    except Exception:
        return None
    return None


def fetch_kline(symbol, base_url, cache_path, refresh=False):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    now = int(time.time() * 1000)
    raw_start_ms = now - int((DAYS + WARMUP_DAYS) * 86400 * 1000)
    start_ms = ((raw_start_ms + INTERVAL_MS - 1) // INTERVAL_MS) * INTERVAL_MS
    end_ms = (now // INTERVAL_MS) * INTERVAL_MS - 1

    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path)
        return validate(df, symbol, start_ms, end_ms)

    s = requests.Session()
    cursor = start_ms
    rows = []
    page = 0
    page_limit = SPOT_LIMIT if base_url.startswith("https://sapi.xt.com") else FUTURES_LIMIT

    while cursor <= end_ms:
        page += 1
        window_end = min(end_ms, cursor + page_limit * INTERVAL_MS - 1)
        params = {
            "symbol": symbol,
            "interval": "15m",
            "startTime": cursor,
            "endTime": window_end,
            "limit": page_limit,
        }
        last_error = None
        payload = None
        for attempt in range(4):
            try:
                r = s.get(base_url, params=params, timeout=30)
                r.raise_for_status()
                raw_json = r.json()
                payload = payload_rows(raw_json)
                if payload is None:
                    raise RuntimeError(
                        f"XT response has no kline list; response_keys={list(raw_json.keys())[:12] if isinstance(raw_json, dict) else type(raw_json).__name__}; "
                        f"response={str(raw_json)[:500]}"
                    )
                break
            except Exception as exc:
                last_error = exc
                time.sleep(1.0 + attempt)
        if payload is None:
            raise RuntimeError(f"{symbol}: page {page} failed: {last_error}")

        parsed = [x for x in (parse_row(z, is_spot=(base_url == SPOT_URL)) for z in payload) if x is not None]
        in_window = [x for x in parsed if cursor <= x[0] <= window_end]
        if not in_window:
            # XT may return an empty terminal page after the requested
            # historical window has already been collected.
            if rows:
                last_seen = max(int(x[0]) for x in rows)
                if last_seen >= end_ms - 2 * INTERVAL_MS:
                    break
            raise RuntimeError(f"{symbol}: page {page} produced no rows in requested window")

        rows.extend(in_window)
        mx = max(x[0] for x in in_window)
        next_cursor = mx + INTERVAL_MS
        if next_cursor <= cursor:
            raise RuntimeError(f"{symbol}: pagination made no progress")
        cursor = next_cursor

        if page % 10 == 0:
            print(f"[FETCH] {symbol} {page=} rows={len(rows)}")
        if mx >= end_ms:
            break

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[df.index < pd.Timestamp.now(tz="UTC").floor("15min")]
    out = validate(df.reset_index(), symbol, start_ms, end_ms, allow_gaps=base_url.startswith("https://sapi.xt.com"))
    out.to_csv(cache_path, index=False)
    return out


def validate(df, symbol, start_ms, end_ms, allow_gaps=False):
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"{symbol}: missing {required - set(df.columns)}")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    start = pd.to_datetime(start_ms, unit="ms", utc=True)
    end = pd.to_datetime(end_ms, unit="ms", utc=True)
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise RuntimeError(f"{symbol}: NaN in OHLCV")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise RuntimeError(f"{symbol}: non-positive prices")
    if (df["volume"] < 0).any():
        raise RuntimeError(f"{symbol}: negative volume")
    if len(df) < MIN_ROWS:
        raise RuntimeError(f"{symbol}: only {len(df)} rows")
    diffs = df["timestamp"].diff().dropna().dt.total_seconds() / 60.0
    bad = diffs > MAX_GAP_MINUTES + 1e-9
    gaps = int(bad.sum())
    if gaps:
        # Never fabricate OHLCV. Isolated one-candle holes can occur in
        # historical Spot data; those timestamps are naturally removed by
        # complete_resample()/cross-sectional timestamp intersection.
        missing_counts = [
            max(1, int(round(d / 15.0)) - 1)
            for d in diffs[bad].tolist()
        ]
        if any(m > 1 for m in missing_counts) and not allow_gaps:
            raise RuntimeError(
                f"{symbol}: {gaps} gaps, including a multi-candle gap; "
                "refusing to fabricate OHLCV"
            )
        if allow_gaps:
            print(
                f"[VALIDATE] {symbol}: {gaps} historical Spot gaps; "
                "no OHLCV fabricated; incomplete higher-timeframe buckets will be excluded"
            )
        else:
            print(
                f"[VALIDATE] {symbol}: {gaps} isolated 15m gaps; "
                "no OHLCV fabricated; affected higher-timeframe buckets will be excluded"
            )
    span = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400
    if span < 420:
        raise RuntimeError(f"{symbol}: span only {span:.1f} days")
    print(f"[VALIDATE] {symbol} rows={len(df)} first={df['timestamp'].iloc[0]} last={df['timestamp'].iloc[-1]} span={span:.1f}d gaps={gaps}")
    return df


def complete_resample(df15, rule, expected):
    x = df15.copy().set_index("timestamp").sort_index()
    out = pd.DataFrame({
        "open": x["open"].resample(rule).first(),
        "high": x["high"].resample(rule).max(),
        "low": x["low"].resample(rule).min(),
        "close": x["close"].resample(rule).last(),
        "volume": x["volume"].resample(rule).sum(),
        "count": x["close"].resample(rule).count(),
    })
    out = out[out["count"] == expected].drop(columns="count")
    return out


def atr(df, n=14):
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def safe_z(s, n):
    mu = s.rolling(n).mean()
    sd = s.rolling(n).std()
    return (s - mu) / sd.replace(0, np.nan)


def build_asset_features(f15, spot15, btc_h1=None):
    f = complete_resample(f15, "1h", 4)
    h4 = complete_resample(f15, "4h", 16)
    d1 = complete_resample(f15, "1d", 96)
    s = complete_resample(spot15, "1h", 4)

    a = f.copy()
    a["atr"] = atr(a, 14)
    rng = (a["high"] - a["low"]).replace(0, np.nan)
    body = a["close"] - a["open"]
    a["body_frac"] = body.abs() / rng
    a["close_loc"] = (a["close"] - a["low"]) / rng
    a["ret_6"] = a["close"].pct_change(6)
    a["ret_24"] = a["close"].pct_change(24)
    a["ret_72"] = a["close"].pct_change(72)
    a["vol_z"] = safe_z(a["volume"], 96)
    a["atr_z"] = safe_z(a["atr"], 96)
    a["eff_12"] = (a["close"] - a["close"].shift(12)).abs() / a["close"].diff().abs().rolling(12).sum()

    # Liquidity references use only completed prior bars.
    for n in (20, 24, 32):
        a[f"prior_hi_{n}"] = a["high"].shift(1).rolling(n).max()
        a[f"prior_lo_{n}"] = a["low"].shift(1).rolling(n).min()

    # Bar-aggregated participation proxy. Explicitly NOT CVD.
    a["flow_pressure"] = (body / rng) * a["volume"]
    a["flow_z"] = safe_z(a["flow_pressure"], 96)
    a["flow_6"] = a["flow_pressure"].rolling(6).sum()

    # Futures-vs-spot basis and volume participation.
    s = s.reindex(a.index)
    a["spot_close"] = s["close"]
    a["basis"] = a["close"] / a["spot_close"] - 1.0
    a["basis_z"] = safe_z(a["basis"], 192)
    a["spot_ret_24"] = s["close"].pct_change(24)
    a["fut_spot_ret_gap"] = a["ret_24"] - a["spot_ret_24"]
    a["volume_ratio"] = a["volume"] / s["volume"].replace(0, np.nan)
    a["volume_ratio_z"] = safe_z(np.log(a["volume_ratio"].clip(lower=1e-12)), 96)

    if btc_h1 is not None:
        btc = btc_h1["close"].reindex(a.index).ffill()
        a["btc_ret_24"] = btc.pct_change(24)
        a["rs24"] = a["ret_24"] - a["btc_ret_24"]
        a["rs_z"] = safe_z(a["rs24"], 96)
    else:
        a["rs24"] = np.nan
        a["rs_z"] = np.nan

    # Higher-timeframe structure.
    h4["atr"] = atr(h4, 14)
    h4["ema20"] = h4["close"].ewm(span=20, adjust=False).mean()
    h4["ema50"] = h4["close"].ewm(span=50, adjust=False).mean()
    h4["trend"] = np.where(h4["ema20"] > h4["ema50"], 1, -1)
    h4["slope"] = h4["ema20"].pct_change(5)

    d1["ema50"] = d1["close"].ewm(span=50, adjust=False).mean()
    d1["ema200"] = d1["close"].ewm(span=200, adjust=False).mean()
    d1["slope50"] = d1["ema50"].pct_change(5)
    d1["regime"] = np.where((d1["close"] > d1["ema200"]) & (d1["slope50"] > 0), 1,
                    np.where((d1["close"] < d1["ema200"]) & (d1["slope50"] < 0), -1, 0))

    # Critical causal alignment: signal at 1H timestamp sees the last COMPLETED
    # 4H/1D bar strictly before that 1H close.
    h4a = h4.shift(1).reindex(a.index, method="ffill")
    d1a = d1.shift(1).reindex(a.index, method="ffill")
    for col in ["atr", "ema20", "ema50", "trend", "slope"]:
        a["h4_" + col] = h4a[col]
    for col in ["ema50", "ema200", "slope50", "regime"]:
        a["d1_" + col] = d1a[col]
    return a, h4, d1


def split_timeline(frames):
    common = None
    for h1, _, _ in frames.values():
        idx = set(h1.index)
        common = idx if common is None else common.intersection(idx)
    times = pd.DatetimeIndex(sorted(common))
    if len(times) < 300:
        raise RuntimeError("Insufficient common timestamps")
    start, end = times[0], times[-1]
    span = end - start
    train_end = start + span * TRAIN_FRAC
    val_end = start + span * (TRAIN_FRAC + VAL_FRAC)
    return Split(start, train_end, val_end, end)


def spearman(x, y):
    z = pd.concat([x, y], axis=1).dropna()
    if len(z) < 50:
        return np.nan
    return z.iloc[:, 0].rank().corr(z.iloc[:, 1].rank())


def factor_audit(frames, split):
    factors = [
        "basis_z", "volume_ratio_z", "flow_z", "flow_6", "rs_z",
        "fut_spot_ret_gap", "vol_z", "atr_z", "eff_12",
    ]
    rows = []
    for sym, (h1, _, _) in frames.items():
        x = h1.loc[split.start:split.train_end]
        for fac in factors:
            if fac not in x:
                continue
            for horizon in (1, 4, 8, 24):
                y = x["close"].shift(-horizon) / x["close"] - 1
                rows.append((sym, fac, horizon, spearman(x[fac], y)))
    out = pd.DataFrame(rows, columns=["symbol", "factor", "horizon", "spearman"])
    if out.empty:
        return
    summ = out.groupby("factor")["spearman"].agg(["count", "mean", "median"]).sort_values("mean")
    print("\n================ TRAIN FACTOR AUDIT ================")
    print(summ.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=====================================================\n")


def asof_strict(df, ts):
    # Return last row strictly before ts.
    i = df.index.searchsorted(ts, side="left") - 1
    return None if i < 0 else df.iloc[i]


def candidate(ts, sym, f, cfg):
    if ts not in f.index:
        return None
    r = f.loc[ts]
    n = cfg.sweep_lookback
    hi_col = f"prior_hi_{n}"
    lo_col = f"prior_lo_{n}"
    need = ["atr", hi_col, lo_col, "basis_z", "volume_ratio_z", "flow_z",
            "rs_z", "h4_trend", "d1_regime", "close_loc", "vol_z", "eff_12"]
    if any(pd.isna(r.get(k)) for k in need):
        return None

    # Liquidity event must be a completed-bar event.
    prior_hi = r[hi_col]
    prior_lo = r[lo_col]
    long_sweep = bool(r["low"] < prior_lo and r["close"] > prior_lo)
    short_sweep = bool(r["high"] > prior_hi and r["close"] < prior_hi)
    if not (long_sweep or short_sweep):
        return None

    # Absorption/repricing logic:
    # Long: downside sweep, strong close back above liquidity, flow pressure
    # less bearish than the extreme sell impulse, spot confirms, futures basis
    # is not in a stretched premium.
    long_score = 0.0
    short_score = 0.0

    long_score += 2.0 * long_sweep
    reclaim_long = (r["close"] - prior_lo) / r["atr"]
    reclaim_short = (prior_hi - r["close"]) / r["atr"]
    long_score += 1.0 * (reclaim_long >= cfg.reclaim_min)
    long_score += 1.0 * (r["flow_z"] > cfg.flow_min)
    long_score += 1.0 * (r["volume_ratio_z"] > 0)
    long_score += 1.0 * (r["basis_z"] <= cfg.basis_z_max)
    long_score += 1.0 * (r["rs_z"] > -0.5)
    long_score += 0.75 * (r["close_loc"] >= 0.60)
    long_score += 0.75 * (r["eff_12"] < 0.55)
    long_score += 0.5 * (r["vol_z"] > cfg.vol_z_min)
    long_score += 0.5 * (r["h4_trend"] >= 0)
    long_score += 0.5 * (r["d1_regime"] >= 0)

    short_score += 2.0 * short_sweep
    short_score += 1.0 * (reclaim_short >= cfg.reclaim_min)
    short_score += 1.0 * (r["flow_z"] < -cfg.flow_min)
    short_score += 1.0 * (r["volume_ratio_z"] > 0)
    short_score += 1.0 * (r["basis_z"] >= -cfg.basis_z_max)
    short_score += 1.0 * (r["rs_z"] < 0.5)
    short_score += 0.75 * (r["close_loc"] <= 0.40)
    short_score += 0.75 * (r["eff_12"] < 0.55)
    short_score += 0.5 * (r["vol_z"] > cfg.vol_z_min)
    short_score += 0.5 * (r["h4_trend"] <= 0)
    short_score += 0.5 * (r["d1_regime"] <= 0)

    if long_score >= 6.0 and long_score > short_score:
        return {"symbol": sym, "side": 1, "score": long_score, "atr": float(r["atr"])}
    if short_score >= 6.0 and short_score > long_score:
        return {"symbol": sym, "side": -1, "score": short_score, "atr": float(r["atr"])}
    return None


def entry_price(p, side):
    return p * (1 + SLIPPAGE * side)


def exit_price(p, side):
    return p * (1 - SLIPPAGE * side)


def trade_pnl(side, entry, exitp):
    gross = (exitp - entry) / entry * NOTIONAL * side
    fees = NOTIONAL * FEE_RATE * 2.0
    return gross - fees


def run_backtest(frames, cfg, start, end, label):
    common = None
    for h1, _, _ in frames.values():
        idx = set(h1.loc[start:end].index)
        common = idx if common is None else common.intersection(idx)
    times = pd.DatetimeIndex(sorted(common))

    equity = INITIAL_CAPITAL
    peak = equity
    max_dd = 0.0
    active = None
    trades = []
    cooldown_until = None
    capital_blocked = False

    for ts in times:
        # Manage existing position first. No new entry on the close-reveal candle.
        if active is not None and ts >= active["entry_ts"]:
            h1 = frames[active["symbol"]][0]
            if ts in h1.index:
                b = h1.loc[ts]
                side = active["side"]
                sl_hit = b["low"] <= active["sl"] if side == 1 else b["high"] >= active["sl"]
                tp_hit = b["high"] >= active["tp"] if side == 1 else b["low"] <= active["tp"]
                if sl_hit or tp_hit:
                    outcome = "LOSS" if sl_hit else "WIN"
                    raw = active["sl"] if sl_hit else active["tp"]
                    pnl = trade_pnl(side, active["entry"], exit_price(raw, side))
                    # Fixed-margin accounting: the strategy cannot lose more than the
                    # isolated margin on one trade, but we do not fabricate a liquidation
                    # event. The stop outcome itself determines the loss.
                    pnl = max(-MARGIN, pnl)
                    equity = max(0.0, equity + pnl)
                    trades.append({"ts": ts, "symbol": active["symbol"], "side": side, "pnl": pnl, "outcome": outcome})
                    peak = max(peak, equity)
                    max_dd = max(max_dd, peak - equity)
                    active = None
                    cooldown_until = ts + pd.Timedelta(hours=1)
                    continue

        if active is not None:
            continue
        if cooldown_until is not None and ts < cooldown_until:
            continue
        if equity < MARGIN:
            capital_blocked = True
            continue

        candidates = []
        for sym, (h1, _, _) in frames.items():
            if ts not in h1.index:
                continue
            c = candidate(ts, sym, h1, cfg)
            if c is not None:
                candidates.append(c)
        if not candidates:
            continue
        candidates.sort(key=lambda x: (-x["score"], x["symbol"]))
        c = candidates[0]
        sym, side = c["symbol"], int(c["side"])
        h1 = frames[sym][0]
        pos = h1.index.searchsorted(ts, side="right")
        if pos >= len(h1):
            continue
        entry_ts = h1.index[pos]
        if entry_ts > end:
            continue
        ep = entry_price(float(h1.iloc[pos]["open"]), side)
        dist = float(c["atr"]) * cfg.rr_atr
        if not np.isfinite(dist) or dist <= 0:
            continue
        sl = ep - dist * side
        tp = ep + dist * RR * side
        active = {"symbol": sym, "side": side, "signal_ts": ts, "entry_ts": entry_ts, "entry": ep, "sl": sl, "tp": tp}

    vals = pd.Series([x["pnl"] for x in trades], dtype=float)
    wins = int((vals > 0).sum())
    losses = int((vals <= 0).sum())
    gp = float(vals[vals > 0].sum()) if wins else 0.0
    gl = float(-vals[vals <= 0].sum()) if losses else 0.0
    pf = gp / gl if gl > 0 else math.inf
    wr = 100 * wins / len(vals) if len(vals) else 0.0
    streak = max_streak = 0
    for v in vals:
        if v <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    days = max((end - start).total_seconds() / 86400, 1e-9)
    dd_pct = 100 * max_dd / INITIAL_CAPITAL
    return {
        "label": label, "config": cfg.name, "trades": len(vals), "wins": wins,
        "losses": losses, "wr": wr, "pf": pf, "pnl": float(vals.sum()) if len(vals) else 0.0,
        "dd": max_dd, "dd_pct": min(dd_pct, 100.0), "streak": max_streak,
        "tday": len(vals) / days, "open": active is not None,
        "capital_blocked": capital_blocked, "avg_win": gp / wins if wins else 0.0,
        "avg_loss": -gl / losses if losses else 0.0, "expectancy": float(vals.mean()) if len(vals) else 0.0,
        "median": float(vals.median()) if len(vals) else 0.0,
    }



def symbol_report(frames, cfg, start, end, label):
    rows = []
    for sym in frames:
        r = run_backtest({sym: frames[sym]}, cfg, start, end, label)
        rows.append((sym, r["trades"], r["wr"], r["pf"], r["pnl"], r["streak"], r["dd_pct"]))
    print(f"\n================ {label} PER-SYMBOL ================")
    print("symbol       trades     WR       PF        PnL       streak   DD%")
    for sym, n, wr, pf, pnl, streak, dd in rows:
        print(f"{sym:10s} {n:7d}  {wr:6.2f}%  {pf:7.3f}  ${pnl:9.2f}  {streak:6d}  {dd:6.2f}%")
    print("=====================================================")

def score(train, val):
    if train["trades"] < MIN_TRAIN_TRADES or val["trades"] < MIN_VALIDATION_TRADES:
        return -1e9
    if train["pnl"] <= 0 or val["pnl"] <= 0:
        return -1e8 + min(train["pnl"], 0) + min(val["pnl"], 0)
    # Reward stability and economic quality; do not optimize for OOS.
    return (
        2.0 * val["pf"]
        + 0.02 * val["wr"]
        + 0.001 * val["pnl"]
        - 0.01 * val["streak"]
        - 0.01 * val["dd_pct"]
        + 0.5 * train["pf"]
    )


def report(r, title):
    print(f"\n================ {title} ================")
    for k, lab in [
        ("trades", "Trades"), ("wins", "Wins"), ("losses", "Losses"),
        ("wr", "Win Rate"), ("pf", "Profit Factor"), ("pnl", "Net PnL"),
        ("dd", "Max Drawdown"), ("dd_pct", "Max Drawdown %"),
        ("streak", "Max Loss Streak"), ("tday", "Trades / Day"),
        ("avg_win", "Average Win"), ("avg_loss", "Average Loss"),
        ("expectancy", "Expectancy / Trade"), ("median", "Median Trade PnL"),
    ]:
        v = r[k]
        if k in {"wr", "dd_pct"}:
            print(f"{lab:24}: {v:.2f}%")
        elif k in {"pf", "tday"}:
            print(f"{lab:24}: {v:.4f}")
        elif k == "trades" or k == "wins" or k == "losses" or k == "streak":
            print(f"{lab:24}: {int(v)}")
        else:
            print(f"{lab:24}: ${v:,.2f}")
    print("==================================================")


def main():
    refresh = env_refresh()
    print("HUNTER-V12 — STATE / FLOW / LIQUIDITY / REPRICING")
    print("XT Futures + XT Spot | 15m -> 1H/4H/1D | causal")
    print(f"Capital=${INITIAL_CAPITAL:.0f} Margin=${MARGIN:.0f} Leverage={LEVERAGE:.0f}x RR=1:{RR:.0f}")

    raw_f = {}
    raw_s = {}
    for sym in SYMBOLS:
        raw_f[sym] = fetch_kline(sym, FUTURES_URL, FUT_DIR / f"{sym}.csv", refresh)
        raw_s[sym] = fetch_kline(sym, SPOT_URL, SPOT_DIR / f"{sym}.csv", refresh)

    # BTC is the cross-sectional benchmark.
    btc_h1 = complete_resample(raw_f["btc_usdt"], "1h", 4)
    frames = {}
    for sym in SYMBOLS:
        frames[sym] = build_asset_features(raw_f[sym], raw_s[sym], btc_h1)

    split = split_timeline(frames)
    print(f"\nSPLIT: train={split.start}..{split.train_end} | val={split.train_end}..{split.val_end} | OOS={split.val_end}..{split.end}")
    factor_audit(frames, split)

    grid = []
    # Small, predeclared research grid. It is intentionally not large enough to
    # brute-force the OOS period. Parameters describe market-state tolerances,
    # not outcome-fitting knobs.
    for sweep in (20, 32):
        for vz in (0.0, 0.75):
            for bz in (1.0, 2.0):
                for flow in (0.50, 0.75):
                    for reclaim in (0.15, 0.30):
                        grid.append(Config(f"L{sweep}_V{vz}_B{bz}_F{flow}_R{reclaim}", sweep, vz, bz, flow, reclaim, 1.5))

    results = []
    for cfg in grid:
        tr = run_backtest(frames, cfg, split.start, split.train_end, "TRAIN")
        va = run_backtest(frames, cfg, split.train_end, split.val_end, "VALIDATION")
        sc = score(tr, va)
        results.append((sc, cfg, tr, va))

    results.sort(key=lambda x: x[0], reverse=True)
    eligible = [x for x in results if x[0] > -1e8]
    if not eligible:
        raise RuntimeError("No train/validation-eligible configuration. Strategy has no validated edge.")

    _, cfg, tr, va = eligible[0]
    print(f"\n[SELECTED] {cfg.name}")
    report(tr, "SELECTED TRAIN")
    report(va, "SELECTED VALIDATION")

    oos = run_backtest(frames, cfg, split.val_end, split.end, "OOS")
    report(oos, "UNTOUCHED OOS")
    symbol_report(frames, cfg, split.val_end, split.end, "OOS")

    accepted = (
        oos["trades"] >= MIN_OOS_TRADES
        and oos["wr"] > MIN_OOS_WR
        and oos["pf"] > MIN_OOS_PF
        and oos["pnl"] > 0
        and oos["streak"] <= MAX_OOS_STREAK
        and oos["dd_pct"] < MAX_OOS_DD_PCT
    )
    print("\n================ ACCEPTANCE GATE ================")
    print(f"OOS trades >= {MIN_OOS_TRADES:<3}     : {'PASS' if oos['trades'] >= MIN_OOS_TRADES else 'FAIL'}")
    print(f"OOS WR > {MIN_OOS_WR:.1f}%           : {'PASS' if oos['wr'] > MIN_OOS_WR else 'FAIL'}")
    print(f"OOS PF > {MIN_OOS_PF:.2f}             : {'PASS' if oos['pf'] > MIN_OOS_PF else 'FAIL'}")
    print(f"OOS Net PnL > $0              : {'PASS' if oos['pnl'] > 0 else 'FAIL'}")
    print(f"OOS Max loss streak <= {MAX_OOS_STREAK}: {'PASS' if oos['streak'] <= MAX_OOS_STREAK else 'FAIL'}")
    print(f"OOS Max DD < {MAX_OOS_DD_PCT:.0f}%           : {'PASS' if oos['dd_pct'] < MAX_OOS_DD_PCT else 'FAIL'}")
    print(f"OOS open position at end      : {'YES' if oos['open'] else 'NO'} (informational)")
    print(f"Capital blocked (< ${MARGIN:.0f}) : {'YES' if oos['capital_blocked'] else 'NO'} (informational)")
    print(f"ACCEPTED                      : {accepted}")
    print("==================================================")
    print("[DECISION] ACCEPTED — robust edge criteria passed." if accepted else "[DECISION] REJECTED — no claim of robust edge.")


if __name__ == "__main__":
    main()
