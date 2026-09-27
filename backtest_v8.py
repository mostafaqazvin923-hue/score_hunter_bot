#!/usr/bin/env python3
"""
HUNTER-V10 — RESEARCH-FIRST FUTURES EDGE ENGINE

XT USDT-M Futures / 15m -> 1H / 4H / 1D
Research-first, causal, no lookahead, no repainting.

Design:
- Real XT Futures history only; no fabricated OI/funding/liquidation.
- Exact 15m pagination + hard validation.
- Complete 1H/4H/1D bars only.
- 1D regime + 4H structure + 1H event/entry.
- Predeclared strategy families and small parameter grid.
- TRAIN -> VALIDATION selection; untouched OOS.
- Manual Spearman calculation (no scipy dependency).
- One global position, no overlap.
- Entry on next 1H open.
- Fixed RR 1:2.
- No timeout, no BE, no trailing.
- Same-candle SL+TP = LOSS.
- Position still open at dataset end = OPEN.
- Entry and exit slippage modeled adversely.
- Fixed $100 margin, 50x leverage, $5,000 notional.
- If stop loss exceeds isolated margin, the loss is capped at margin
  (conservative liquidation proxy) and reported as LIQUIDATED.
  This prevents impossible equity below the available isolated margin.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from datetime import timezone
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
DATA_DIR = Path("data/xt_futures_v10")

DAYS = 365
WARMUP_DAYS = 90
INTERVAL_MS = 15 * 60 * 1000
LIMIT = 1500

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

MIN_VALIDATION_TRADES = 30
MIN_TRAIN_TRADES = 50

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
    family: str
    regime: str
    threshold: float
    atr_mult: float
    lookback: int


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--refresh", action="store_true")
    return p.parse_args()


def payload_rows(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ("result", "data", "rows", "items", "list"):
            if k in obj:
                x = payload_rows(obj[k])
                if x is not None:
                    return x
    return None


def parse_row(r):
    if isinstance(r, (list, tuple)):
        if len(r) < 6:
            return None
        vals = r[:6]
        try:
            return [int(float(vals[0])), *[float(x) for x in vals[1:6]]]
        except Exception:
            return None

    if not isinstance(r, dict):
        return None

    def pick(*keys):
        for k in keys:
            if k in r and r[k] is not None:
                return r[k]
        return None

    ts = pick("t", "time", "timestamp", "T")
    o = pick("o", "open")
    h = pick("h", "high")
    l = pick("l", "low")
    c = pick("c", "close")
    v = pick("v", "volume", "vol", "a")

    if any(x is None for x in (ts, o, h, l, c, v)):
        return None

    try:
        return [int(float(ts)), float(o), float(h), float(l), float(c), float(v)]
    except Exception:
        return None


def fetch_symbol(symbol: str, data_dir: Path, refresh=False):
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{symbol}.csv"

    now = int(time.time() * 1000)
    target_start = now - int((DAYS + WARMUP_DAYS) * 86400 * 1000)
    final_end = (now // INTERVAL_MS) * INTERVAL_MS - 1

    if path.exists() and not refresh:
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return validate(df, symbol, target_start, final_end)

    s = requests.Session()
    cursor = target_start
    all_rows = []
    page = 0

    while cursor <= final_end:
        page += 1
        window_end = min(final_end, cursor + LIMIT * INTERVAL_MS - 1)
        params = {
            "symbol": symbol,
            "interval": "15m",
            "startTime": cursor,
            "endTime": window_end,
            "limit": LIMIT,
        }

        last_error = None
        data = None
        for attempt in range(4):
            try:
                r = s.get(FUTURES_URL, params=params, timeout=30)
                r.raise_for_status()
                data = payload_rows(r.json())
                if data is None:
                    raise RuntimeError("XT response did not contain a kline list")
                break
            except Exception as e:
                last_error = e
                time.sleep(1.0 + attempt)

        if data is None:
            raise RuntimeError(f"{symbol}: page {page} failed: {last_error}")

        parsed = [x for x in (parse_row(z) for z in data) if x is not None]
        in_window = [x for x in parsed if cursor <= x[0] <= window_end]

        if not in_window:
            timestamps = [x[0] for x in parsed]
            if timestamps and max(timestamps) < cursor:
                break
            if not parsed:
                raise RuntimeError(f"{symbol}: page {page} returned no parseable rows")
            raise RuntimeError(
                f"{symbol}: page {page} returned rows but none in requested window"
            )

        all_rows.extend(in_window)
        mx = max(x[0] for x in in_window)

        if mx < cursor:
            raise RuntimeError(f"{symbol}: pagination moved backwards")
        next_cursor = mx + INTERVAL_MS
        if next_cursor <= cursor:
            raise RuntimeError(f"{symbol}: pagination made no progress")

        cursor = next_cursor

        if page % 10 == 0:
            print(f"[FETCH] {symbol} page={page} rows={len(all_rows)}")

        if mx >= final_end:
            break

    if not all_rows:
        raise RuntimeError(f"{symbol}: no data collected")

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Never trade the currently incomplete 15m candle.
    cutoff = pd.Timestamp.now(tz="UTC").floor("15min")
    df = df[df.index < cutoff]

    out = validate(df.reset_index(), symbol, target_start, final_end)
    out.to_csv(path, index=False)
    return out


def validate(df, symbol, target_start, final_end):
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"{symbol}: missing required columns")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    df = df.drop_duplicates("timestamp", keep="last")

    start = pd.to_datetime(target_start, unit="ms", utc=True)
    end = pd.to_datetime(final_end, unit="ms", utc=True)
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise RuntimeError(f"{symbol}: NaN in OHLCV")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise RuntimeError(f"{symbol}: non-positive price")
    if (df["volume"] < 0).any():
        raise RuntimeError(f"{symbol}: negative volume")

    if len(df) < MIN_ROWS:
        raise RuntimeError(f"{symbol}: only {len(df)} rows; history incomplete")

    diffs = df["timestamp"].diff().dropna().dt.total_seconds() / 60.0
    gaps = int((diffs > MAX_GAP_MINUTES + 1e-9).sum())
    if gaps:
        raise RuntimeError(f"{symbol}: {gaps} gaps detected")

    span = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400
    if span < 420:
        raise RuntimeError(f"{symbol}: span only {span:.1f}d")

    print(
        f"[VALIDATE] {symbol} rows={len(df)} first={df['timestamp'].iloc[0]} "
        f"last={df['timestamp'].iloc[-1]} span={span:.1f}d gaps={gaps}"
    )
    return df


def complete_resample(df15, rule, expected):
    x = df15.copy().set_index("timestamp").sort_index()
    o = x["open"].resample(rule, closed="left", label="right").first()
    h = x["high"].resample(rule, closed="left", label="right").max()
    l = x["low"].resample(rule, closed="left", label="right").min()
    c = x["close"].resample(rule, closed="left", label="right").last()
    v = x["volume"].resample(rule, closed="left", label="right").sum()
    n = x["close"].resample(rule, closed="left", label="right").count()

    out = pd.concat([o, h, l, c, v, n], axis=1)
    out.columns = ["open", "high", "low", "close", "volume", "count"]
    out = out[out["count"] == expected].drop(columns="count")
    return out.dropna()


def build_frames(raw):
    h1 = complete_resample(raw, "1h", 4)
    h4 = complete_resample(raw, "4h", 16)
    d1 = complete_resample(raw, "1d", 96)
    return h1, h4, d1


def atr(df, n=14):
    prev = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def add_features(h1, h4, d1):
    a = h1.copy()
    a["atr"] = atr(a, 14)
    a["ret_6"] = a["close"].pct_change(6)
    a["ret_24"] = a["close"].pct_change(24)
    a["ret_72"] = a["close"].pct_change(72)
    a["range_atr"] = (a["high"] - a["low"]) / a["atr"]

    body = (a["close"] - a["open"]).abs()
    a["body_atr"] = body / a["atr"]
    a["close_location"] = ((a["close"] - a["low"]) / (a["high"] - a["low"]).replace(0, np.nan))
    a["efficiency_12"] = (
        (a["close"] - a["close"].shift(12)).abs()
        / a["close"].diff().abs().rolling(12).sum()
    )
    a["vol_z"] = (
        (a["volume"] - a["volume"].rolling(48).mean())
        / a["volume"].rolling(48).std()
    )

    # Compression is measured from prior completed bars only.
    width = (
        2.0 * a["close"].rolling(20).std()
        / a["close"].rolling(20).mean()
    )
    a["bb_width"] = width
    a["bb_rank"] = width.shift(1).rolling(96).rank(pct=True)

    # Prior-bar liquidity reference.
    a["hh_24"] = a["high"].shift(1).rolling(24).max()
    a["ll_24"] = a["low"].shift(1).rolling(24).min()

    # 4H structure.
    b = h4.copy()
    b["atr"] = atr(b, 14)
    b["ema20"] = b["close"].ewm(span=20, adjust=False).mean()
    b["ema50"] = b["close"].ewm(span=50, adjust=False).mean()
    b["trend"] = np.where(b["ema20"] > b["ema50"], 1, -1)
    b["swing_hi"] = b["high"].shift(2).rolling(5).max()
    b["swing_lo"] = b["low"].shift(2).rolling(5).min()

    # 1D regime. EMA is computed from prior daily close, so a signal
    # during the current day can only see the last completed daily bar.
    d = d1.copy()
    d["ema50"] = d["close"].shift(1).ewm(span=50, adjust=False).mean()
    d["ema200"] = d["close"].shift(1).ewm(span=200, adjust=False).mean()
    d["slope20"] = d["ema50"].pct_change(5)
    d["regime"] = np.where(
        (d["close"].shift(1) > d["ema200"]) & (d["slope20"] > 0), 1,
        np.where((d["close"].shift(1) < d["ema200"]) & (d["slope20"] < 0), -1, 0)
    )
    return a, b, d


def asof(df, ts):
    if df.empty:
        return None
    i = df.index.searchsorted(ts, side="right") - 1
    if i < 0:
        return None
    return df.iloc[i]


def make_panel(symbol_frames):
    times = sorted(set().union(*[set(x[0].index) for x in symbol_frames.values()]))
    return pd.DatetimeIndex(times)


def split_timeline(common_times):
    common_times = pd.DatetimeIndex(common_times).sort_values()
    if len(common_times) < 300:
        raise RuntimeError("Not enough common 1H timestamps")
    start = common_times[0]
    end = common_times[-1]
    span = end - start
    train_end = start + span * TRAIN_FRAC
    val_end = start + span * (TRAIN_FRAC + VAL_FRAC)
    return Split(start, train_end, val_end, end)


def rank_spearman(x, y):
    z = pd.concat([x, y], axis=1).dropna()
    if len(z) < 30:
        return np.nan
    return z.iloc[:, 0].rank(method="average").corr(
        z.iloc[:, 1].rank(method="average")
    )


def factor_audit(frames, split):
    rows = []
    for sym, (h1, _, _) in frames.items():
        train = h1.loc[split.start:split.train_end].copy()
        for col in ["ret_6", "ret_24", "ret_72", "range_atr",
                    "body_atr", "close_location", "efficiency_12",
                    "vol_z", "bb_rank"]:
            if col not in train:
                continue
            for horizon in [1, 4, 8, 24]:
                fwd = train["close"].shift(-horizon) / train["close"] - 1
                ic = rank_spearman(train[col], fwd)
                rows.append((sym, col, horizon, ic))
    if not rows:
        return
    out = pd.DataFrame(rows, columns=["symbol", "factor", "horizon", "spearman"])
    summary = (
        out.groupby("factor")["spearman"]
        .agg(["count", "mean"])
        .assign(abs_mean=lambda z: z["mean"].abs())
        .sort_values("abs_mean", ascending=False)
    )
    print("\n================ TRAIN FACTOR AUDIT ================")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=====================================================\n")


def candidate_from_features(ts, sym, f, h4, d1, cfg):
    if ts not in f.index:
        return None
    row = f.loc[ts]
    r4 = asof(h4, ts)
    rd = asof(d1, ts)
    if r4 is None or rd is None:
        return None

    vals = [
        row["atr"], row["ret_24"], row["ret_72"], row["bb_rank"],
        row["range_atr"], row["body_atr"], row["close_location"],
        row["efficiency_12"], row["vol_z"], r4["ema20"], r4["ema50"],
        rd["ema200"], rd["regime"]
    ]
    if any(pd.isna(x) for x in vals):
        return None

    long_score = 0.0
    short_score = 0.0

    trend = int(r4["trend"])
    regime = int(rd["regime"])
    ret24 = float(row["ret_24"])
    ret72 = float(row["ret_72"])
    br = float(row["bb_rank"])
    rng = float(row["range_atr"])
    body = float(row["body_atr"])
    loc = float(row["close_location"])
    eff = float(row["efficiency_12"])
    vz = float(row["vol_z"])

    # Family A: continuation after compression + displacement.
    if cfg.family == "CONT":
        long_score = (
            1.0 * (trend == 1)
            + 1.0 * (regime == 1)
            + 1.0 * (br <= cfg.threshold)
            + 1.0 * (ret24 > 0)
            + 1.0 * (ret72 > 0)
            + 1.0 * (loc >= 0.65)
            + 1.0 * (body >= cfg.atr_mult)
            + 0.5 * (vz > 0)
        )
        short_score = (
            1.0 * (trend == -1)
            + 1.0 * (regime == -1)
            + 1.0 * (br <= cfg.threshold)
            + 1.0 * (ret24 < 0)
            + 1.0 * (ret72 < 0)
            + 1.0 * (loc <= 0.35)
            + 1.0 * (body >= cfg.atr_mult)
            + 0.5 * (vz > 0)
        )

    # Family B: failed auction / mean reversion after extreme displacement.
    elif cfg.family == "FADE":
        extreme = max(abs(ret24), abs(ret72))
        long_score = (
            1.0 * (trend == -1 or regime == -1)
            + 1.0 * (br >= cfg.threshold)
            + 1.0 * (loc <= 0.25)
            + 1.0 * (body >= cfg.atr_mult)
            + 1.0 * (ret24 < 0)
            + 0.5 * (eff < 0.45)
            + 0.5 * (extreme > 0)
        )
        short_score = (
            1.0 * (trend == 1 or regime == 1)
            + 1.0 * (br >= cfg.threshold)
            + 1.0 * (loc >= 0.75)
            + 1.0 * (body >= cfg.atr_mult)
            + 1.0 * (ret24 > 0)
            + 0.5 * (eff < 0.45)
            + 0.5 * (extreme > 0)
        )

    # Family C: volatility expansion with directional confirmation.
    elif cfg.family == "EXP":
        long_score = (
            1.0 * (trend == 1)
            + 1.0 * (regime == 1)
            + 1.0 * (rng >= cfg.atr_mult)
            + 1.0 * (loc >= 0.60)
            + 1.0 * (ret24 > 0)
            + 1.0 * (ret72 > 0)
            + 0.5 * (eff >= 0.50)
        )
        short_score = (
            1.0 * (trend == -1)
            + 1.0 * (regime == -1)
            + 1.0 * (rng >= cfg.atr_mult)
            + 1.0 * (loc <= 0.40)
            + 1.0 * (ret24 < 0)
            + 1.0 * (ret72 < 0)
            + 0.5 * (eff >= 0.50)
        )

    min_score = 6.0
    if long_score >= min_score and long_score > short_score:
        return {"symbol": sym, "side": 1, "score": long_score, "atr": float(row["atr"])}
    if short_score >= min_score and short_score > long_score:
        return {"symbol": sym, "side": -1, "score": short_score, "atr": float(row["atr"])}
    return None


def apply_entry_slippage(price, side):
    return price * (1 + SLIPPAGE * side)


def apply_exit_slippage(price, side):
    # Adverse slippage: long exits lower, short exits higher.
    return price * (1 - SLIPPAGE * side)


def pnl_for_exit(side, entry, exit_price):
    gross = (exit_price - entry) / entry * NOTIONAL * side
    fees = NOTIONAL * FEE_RATE * 2.0
    return gross - fees


def run_backtest(frames, split, cfg, start, end, label):
    # Common timestamps are required for cross-sectional selection.
    common = None
    for h1, _, _ in frames.values():
        t = h1.loc[start:end].index
        common = set(t) if common is None else common.intersection(set(t))
    times = pd.DatetimeIndex(sorted(common))

    active = None
    trades = []
    equity = INITIAL_CAPITAL
    peak = equity
    max_dd = 0.0
    cooldown_until = None

    for ts in times:
        # Manage only after the actual next-bar entry.
        if active is not None and ts >= active["entry_ts"]:
            h1 = frames[active["symbol"]][0]
            if ts in h1.index:
                bar = h1.loc[ts]
                side = active["side"]

                stop_hit = (
                    bar["low"] <= active["sl"] if side == 1
                    else bar["high"] >= active["sl"]
                )
                tp_hit = (
                    bar["high"] >= active["tp"] if side == 1
                    else bar["low"] <= active["tp"]
                )

                if stop_hit or tp_hit:
                    # Conservative same-candle resolution.
                    outcome = "LOSS" if stop_hit else "WIN"
                    raw_exit = active["sl"] if stop_hit else active["tp"]
                    exit_price = apply_exit_slippage(raw_exit, side)
                    pnl = pnl_for_exit(side, active["entry"], exit_price)

                    # Isolated-margin cap / liquidation proxy.
                    if pnl < -MARGIN:
                        pnl = -MARGIN
                        outcome = "LIQUIDATED"

                    equity += pnl
                    trades.append({
                        "time": ts, "symbol": active["symbol"],
                        "side": side, "pnl": pnl, "outcome": outcome
                    })
                    peak = max(peak, equity)
                    max_dd = max(max_dd, peak - equity)
                    active = None
                    cooldown_until = ts + pd.Timedelta(hours=1)
                    # Never re-enter on the candle that reveals the result.
                    continue

        if active is not None:
            continue
        if cooldown_until is not None and ts < cooldown_until:
            continue

        candidates = []
        for sym, (h1, h4, d1) in frames.items():
            if ts < start or ts > end or ts not in h1.index:
                continue
            c = candidate_from_features(ts, sym, h1, h4, d1, cfg)
            if c is not None:
                candidates.append(c)

        if not candidates:
            continue

        candidates.sort(key=lambda x: (-x["score"], x["symbol"], -x["side"]))
        c = candidates[0]
        sym = c["symbol"]
        h1 = frames[sym][0]

        # Signal at completed 1H close; entry at the next completed 1H open.
        pos = h1.index.searchsorted(ts, side="right")
        if pos >= len(h1.index):
            continue
        entry_ts = h1.index[pos]
        if entry_ts > end:
            continue

        raw_entry = float(h1.iloc[pos]["open"])
        side = int(c["side"])
        entry = apply_entry_slippage(raw_entry, side)
        stop_dist = float(c["atr"]) * 1.5
        sl = entry - stop_dist * side
        tp = entry + stop_dist * RR * side

        active = {
            "symbol": sym,
            "side": side,
            "signal_ts": ts,
            "entry_ts": entry_ts,
            "entry": entry,
            "sl": sl,
            "tp": tp,
        }

    vals = pd.Series([x["pnl"] for x in trades], dtype=float)
    wins = int((vals > 0).sum())
    losses = int((vals <= 0).sum())
    gross_profit = float(vals[vals > 0].sum()) if wins else 0.0
    gross_loss = float(-vals[vals <= 0].sum()) if losses else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else math.inf
    wr = 100.0 * wins / len(vals) if len(vals) else 0.0

    streak = 0
    max_streak = 0
    for x in vals:
        if x <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    days = max((end - start).total_seconds() / 86400, 1e-9)
    dd_pct = 100.0 * max_dd / INITIAL_CAPITAL

    return {
        "label": label,
        "config": cfg.name,
        "trades": len(vals),
        "wins": wins,
        "losses": losses,
        "wr": wr,
        "pf": pf,
        "pnl": float(vals.sum()) if len(vals) else 0.0,
        "dd": max_dd,
        "dd_pct": dd_pct,
        "streak": max_streak,
        "tday": len(vals) / days,
        "open": active is not None,
        "avg_win": gross_profit / wins if wins else 0.0,
        "avg_loss": -gross_loss / losses if losses else 0.0,
        "expectancy": float(vals.mean()) if len(vals) else 0.0,
        "median": float(vals.median()) if len(vals) else 0.0,
    }


def score_config(train, val):
    # Predeclared selection rule. No OOS information is used.
    if train["trades"] < MIN_TRAIN_TRADES or val["trades"] < MIN_VALIDATION_TRADES:
        return -1e9

    if not np.isfinite(train["pf"]) or not np.isfinite(val["pf"]):
        return -1e9

    # Reward consistency, sample size and risk-adjusted profitability.
    if train["pnl"] <= 0 or val["pnl"] <= 0:
        return -1e8 + min(train["pnl"], val["pnl"])

    wr_gap = abs(train["wr"] - val["wr"])
    pf_gap = abs(train["pf"] - val["pf"])

    return (
        min(train["pf"], val["pf"]) * 10.0
        + min(train["wr"], val["wr"]) * 0.15
        + math.log1p(min(train["trades"], val["trades"]))
        - wr_gap * 0.10
        - pf_gap * 1.0
        - max(train["dd_pct"], val["dd_pct"]) * 0.05
    )


def report(r, title):
    print(f"\n================ {title} ================")
    print(f"Trades              : {r['trades']}")
    print(f"Wins                : {r['wins']}")
    print(f"Losses              : {r['losses']}")
    print(f"Win Rate            : {r['wr']:.2f}%")
    print(f"Profit Factor       : {r['pf']:.3f}" if np.isfinite(r["pf"]) else "Profit Factor       : inf")
    print(f"Net PnL             : ${r['pnl']:.2f}")
    print(f"Max Drawdown        : ${r['dd']:.2f}")
    print(f"Max Drawdown %      : {r['dd_pct']:.2f}%")
    print(f"Max Loss Streak     : {r['streak']}")
    print(f"Trades / Day        : {r['tday']:.4f}")
    print(f"Average Win         : ${r['avg_win']:.2f}")
    print(f"Average Loss        : ${r['avg_loss']:.2f}")
    print(f"Expectancy / Trade  : ${r['expectancy']:.2f}")
    print(f"Median Trade PnL    : ${r['median']:.2f}")
    print("=" * 50)


def main():
    a = args()
    data_dir = Path(a.data_dir)

    raw = {}
    frames = {}

    print("[DATA] Collecting XT Futures history...")
    for sym in SYMBOLS:
        df = fetch_symbol(sym, data_dir, a.refresh)
        raw[sym] = df
        h1, h4, d1 = build_frames(df)
        h1, h4, d1 = add_features(h1, h4, d1)
        frames[sym] = (h1, h4, d1)

    common = None
    for h1, _, _ in frames.values():
        t = set(h1.index)
        common = t if common is None else common.intersection(t)

    split = split_timeline(pd.DatetimeIndex(sorted(common)))
    print(f"[TIMELINE] Train: {split.start} -> {split.train_end}")
    print(f"[TIMELINE] Validation: {split.train_end} -> {split.val_end}")
    print(f"[TIMELINE] OOS: {split.val_end} -> {split.end}")

    factor_audit(frames, split)

    configs = []
    for family in ["CONT", "FADE", "EXP"]:
        for threshold in ([0.25, 0.35] if family == "CONT" else [0.65, 0.75] if family == "FADE" else [0.70, 0.85]):
            for mult in [0.60, 0.80, 1.00]:
                name = f"{family}_thr{threshold:.2f}_m{mult:.2f}"
                configs.append(Config(name, family, "REGIME", threshold, mult, 24))

    print(f"[RESEARCH] Predeclared configs: {len(configs)}")

    selected = []
    for cfg in configs:
        tr = run_backtest(frames, split, cfg, split.start, split.train_end, "TRAIN")
        va = run_backtest(frames, split, cfg, split.train_end, split.val_end, "VAL")
        score = score_config(tr, va)
        selected.append((score, cfg, tr, va))

    selected.sort(key=lambda x: (-x[0], x[1].name))
    print("\n================ MODEL SELECTION ================")
    for score, cfg, tr, va in selected[:10]:
        print(
            f"{cfg.name:24s} score={score:8.3f} "
            f"TRAIN n={tr['trades']:4d} WR={tr['wr']:5.1f} PF={tr['pf']:5.2f} "
            f"VAL n={va['trades']:4d} WR={va['wr']:5.1f} PF={va['pf']:5.2f}"
        )
    print("==================================================")

    eligible = [x for x in selected if x[0] > -1e8]
    if not eligible:
        print("[DECISION] REJECTED — no configuration passed pre-OOS research.")
        return

    _, cfg, tr, va = eligible[0]
    print(f"\n[SELECTED] {cfg.name}")
    report(tr, "SELECTED TRAIN")
    report(va, "SELECTED VALIDATION")

    oos = run_backtest(frames, split, cfg, split.val_end, split.end, "OOS")
    report(oos, "UNTOUCHED OOS")

    accepted = (
        oos["trades"] >= MIN_OOS_TRADES
        and oos["wr"] > MIN_OOS_WR
        and oos["pf"] > MIN_OOS_PF
        and oos["pnl"] > 0
        and oos["streak"] <= MAX_OOS_STREAK
        and oos["dd_pct"] < MAX_OOS_DD_PCT
        and not oos["open"]
    )

    print("\n================ ACCEPTANCE GATE ================")
    print(f"OOS trades >= {MIN_OOS_TRADES}       : {'PASS' if oos['trades'] >= MIN_OOS_TRADES else 'FAIL'}")
    print(f"OOS WR > {MIN_OOS_WR:.1f}%              : {'PASS' if oos['wr'] > MIN_OOS_WR else 'FAIL'}")
    print(f"OOS PF > {MIN_OOS_PF:.2f}              : {'PASS' if oos['pf'] > MIN_OOS_PF else 'FAIL'}")
    print(f"OOS Net PnL > $0                 : {'PASS' if oos['pnl'] > 0 else 'FAIL'}")
    print(f"OOS Max loss streak <= {MAX_OOS_STREAK}: {'PASS' if oos['streak'] <= MAX_OOS_STREAK else 'FAIL'}")
    print(f"OOS Max DD < {MAX_OOS_DD_PCT:.0f}%              : {'PASS' if oos['dd_pct'] < MAX_OOS_DD_PCT else 'FAIL'}")
    print(f"OOS closed at end                : {'PASS' if not oos['open'] else 'FAIL'}")
    print(f"ACCEPTED                          : {accepted}")
    print("==================================================")

    if accepted:
        print("[DECISION] ACCEPTED — candidate survived untouched OOS.")
    else:
        print("[DECISION] REJECTED — no claim of robust edge.")


if __name__ == "__main__":
    main()
