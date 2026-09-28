#!/usr/bin/env python3
"""
HUNTER-V16 — NEW STRATEGY / WALK-FORWARD VALIDATION ENGINE
XT USDT-M PERPETUAL FUTURES
15m raw -> causal 1H / 4H / 1D

Strategy family:
  AUCTION-DISPLACEMENT / RETEST

This is a deliberate break from HUNTER-V13..V15's liquidity-sweep/reclaim
family.  V16 does not use the old sweep score or its directional scoring.
The hypothesis is:
  1) price compresses inside a prior range,
  2) a decisive displacement candle closes outside that range with
     participation and directional efficiency,
  3) the next completed 1H bars provide a causal retest/hold of the broken
     level,
  4) entry occurs on the NEXT 1H open after the retest is confirmed.

Validation protocol:
  - expanding walk-forward folds;
  - each fold selects parameters using only its TRAIN + VALIDATION windows;
  - the following OOS window is then evaluated once;
  - a final latest holdout is reserved and is never used for selection;
  - no OOS metric is used to choose a parameter.

Execution / integrity:
  - strict no-lookahead / no-repaint;
  - one global non-overlapping position;
  - next 1H open entry;
  - no same-candle re-entry;
  - same-candle SL+TP => LOSS;
  - fixed RR 1:2;
  - no timeout, break-even or trailing;
  - incomplete futures candles removed;
  - futures gaps are fatal; spot gaps are never fabricated and affected
    buckets are excluded;
  - open position at a segment end is not realized and is not carried over.

Important:
  This script is an evaluation engine, not a claim that V16 has an edge.
  It should be judged by the aggregated walk-forward OOS and the final
  untouched holdout, not by train performance.
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

# Expanding walk-forward schedule in common 1H timestamps.
# Multiple historical OOS folds are used for robustness; the latest 45 days are
# reserved as a final untouched holdout after the walk-forward process.
FOLD_TRAIN_DAYS = 150
FOLD_VAL_DAYS = 30
FOLD_OOS_DAYS = 30
FINAL_HOLDOUT_DAYS = 45

MIN_TRAIN_TRADES = 25
MIN_VAL_TRADES = 10
MIN_FOLD_OOS_TRADES = 12

# Acceptance is deliberately about robustness, not a single WR threshold.
MIN_AGG_OOS_TRADES = 50
MIN_AGG_OOS_PF = 1.15
MAX_AGG_OOS_DD_PCT = 50.0
MAX_AGG_OOS_STREAK = 7
MIN_POSITIVE_OOS_FOLDS = 2


@dataclass(frozen=True)
class Config:
    name: str
    range_n: int
    displacement_atr: float
    body_frac: float
    volume_z: float
    retest_atr: float
    stop_atr: float
    rank_floor: float


@dataclass(frozen=True)
class Window:
    start: pd.Timestamp
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    oos_end: pd.Timestamp


@dataclass(frozen=True)
class Fold:
    number: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    oos_end: pd.Timestamp


def payload_rows(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ("result", "data", "rows", "list"):
            if isinstance(obj.get(k), list):
                return obj[k]
    return None


def parse_row(row, spot=False):
    try:
        if isinstance(row, dict):
            ts = row.get("t", row.get("timestamp"))
            o = row.get("o", row.get("open"))
            h = row.get("h", row.get("high"))
            l = row.get("l", row.get("low"))
            c = row.get("c", row.get("close"))
            v = row.get("q") if spot else row.get("a")
            if v is None:
                v = row.get("volume")
            if None in (ts, o, h, l, c, v):
                return None
            return int(ts), float(o), float(h), float(l), float(c), float(v)
        if isinstance(row, (list, tuple)) and len(row) >= 6:
            return int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])
    except Exception:
        return None
    return None


def validate(df, symbol, start_ms, end_ms, *, futures=True):
    need = {"timestamp", "open", "high", "low", "close", "volume"}
    if not need.issubset(df.columns):
        raise RuntimeError(f"{symbol}: missing columns {sorted(need - set(df.columns))}")
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    x = x.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    start = pd.to_datetime(start_ms, unit="ms", utc=True)
    end = pd.to_datetime(end_ms, unit="ms", utc=True)
    x = x[(x.timestamp >= start) & (x.timestamp <= end)].copy()
    for c in ["open", "high", "low", "close", "volume"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    if x[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise RuntimeError(f"{symbol}: NaN in OHLCV")
    if (x[["open", "high", "low", "close"]] <= 0).any().any():
        raise RuntimeError(f"{symbol}: non-positive price")
    if (x.volume < 0).any():
        raise RuntimeError(f"{symbol}: negative volume")
    if len(x) < 38000:
        raise RuntimeError(f"{symbol}: only {len(x)} rows; expected at least 38000")
    diffs = x.timestamp.diff().dropna().dt.total_seconds().div(900.0)
    gaps = diffs[diffs > 1.0 + 1e-9]
    if futures and len(gaps):
        raise RuntimeError(f"{symbol}: {len(gaps)} futures gaps; no OHLCV fabrication allowed")
    if not futures and len(gaps):
        print(f"[VALIDATE] {symbol}: {len(gaps)} spot gap(s); no OHLCV fabricated; affected buckets excluded")
    span = (x.timestamp.iloc[-1] - x.timestamp.iloc[0]).total_seconds() / 86400.0
    if span < 420:
        raise RuntimeError(f"{symbol}: span only {span:.1f} days")
    print(f"[VALIDATE] {'FUT' if futures else 'SPOT'} {symbol} rows={len(x)} first={x.timestamp.iloc[0]} last={x.timestamp.iloc[-1]} span={span:.1f}d gaps={len(gaps)}")
    return x.reset_index(drop=True)


def fetch_xt(symbol, url, cache, refresh=False, futures=True):
    cache.parent.mkdir(parents=True, exist_ok=True)
    now = int(time.time() * 1000)
    start_raw = now - int((DAYS + WARMUP_DAYS) * 86400 * 1000)
    start_ms = ((start_raw + INTERVAL_MS - 1) // INTERVAL_MS) * INTERVAL_MS
    end_ms = (now // INTERVAL_MS) * INTERVAL_MS - 1
    if cache.exists() and not refresh:
        return validate(pd.read_csv(cache), symbol, start_ms, end_ms, futures=futures)

    limit = FUTURES_LIMIT if futures else SPOT_LIMIT
    session = requests.Session()
    cursor = start_ms
    all_rows = []
    page = 0
    while cursor <= end_ms:
        page += 1
        window_end = min(end_ms, cursor + limit * INTERVAL_MS - 1)
        params = {"symbol": symbol, "interval": "15m", "startTime": cursor, "endTime": window_end, "limit": limit}
        payload = None
        err = None
        for attempt in range(4):
            try:
                resp = session.get(url, params=params, timeout=30)
                resp.raise_for_status()
                payload = payload_rows(resp.json())
                if payload is None:
                    raise RuntimeError("XT response contains no kline list")
                break
            except Exception as exc:
                err = exc
                time.sleep(1.0 + attempt)
        if payload is None:
            raise RuntimeError(f"{symbol}: page {page} failed: {err}")
        parsed = [p for p in (parse_row(r, spot=not futures) for r in payload) if p is not None]
        inside = [p for p in parsed if cursor <= p[0] <= window_end]
        if not inside:
            if window_end >= end_ms:
                print(f"[FETCH] {'FUT' if futures else 'SPOT'} {symbol} page={page} empty final window; stopping")
                break
            raise RuntimeError(f"{symbol}: page {page} returned no rows in [{cursor},{window_end}]")
        all_rows.extend(inside)
        mx = max(p[0] for p in inside)
        nxt = mx + INTERVAL_MS
        if nxt <= cursor:
            raise RuntimeError(f"{symbol}: pagination stalled")
        cursor = nxt
        if page % 10 == 0:
            print(f"[FETCH] {'FUT' if futures else 'SPOT'} {symbol} page={page} rows={len(all_rows)}")
        if mx >= end_ms:
            break
        time.sleep(0.05)

    x = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    x["timestamp"] = pd.to_datetime(x.pop("ts"), unit="ms", utc=True)
    x = x.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    x = x[x.timestamp < pd.Timestamp.now(tz="UTC").floor("15min")]
    x = validate(x, symbol, start_ms, end_ms, futures=futures)
    x.to_csv(cache, index=False)
    return x


def complete_resample(df15, rule, expected):
    x = df15.set_index("timestamp").sort_index()
    y = pd.DataFrame({
        "open": x.open.resample(rule).first(),
        "high": x.high.resample(rule).max(),
        "low": x.low.resample(rule).min(),
        "close": x.close.resample(rule).last(),
        "volume": x.volume.resample(rule).sum(),
        "n": x.close.resample(rule).count(),
    })
    return y[y.n == expected].drop(columns="n")


def atr(x, n=14):
    prev = x.close.shift(1)
    tr = pd.concat([(x.high-x.low), (x.high-prev).abs(), (x.low-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def zscore(s, n):
    mu = s.rolling(n).mean()
    sd = s.rolling(n).std().replace(0, np.nan)
    return (s-mu)/sd


def build_features(f15, s15, btc_h1):
    h1 = complete_resample(f15, "1h", 4)
    h4 = complete_resample(f15, "4h", 16)
    d1 = complete_resample(f15, "1d", 96)
    spot = complete_resample(s15, "1h", 4)

    a = h1.copy()
    a["atr"] = atr(a)
    a["atr_pct"] = a.atr / a.close
    rng = (a.high-a.low).replace(0, np.nan)
    a["body_frac"] = (a.close-a.open).abs()/rng
    a["close_loc"] = (a.close-a.low)/rng
    a["ret_6"] = a.close.pct_change(6, fill_method=None)
    a["ret_12"] = a.close.pct_change(12, fill_method=None)
    a["ret_24"] = a.close.pct_change(24, fill_method=None)
    a["vol_z"] = zscore(np.log1p(a.volume), 96)
    a["vol_ratio"] = a.volume/a.volume.rolling(24).median()
    a["flow"] = ((a.close-a.open)/rng)*a.volume
    a["flow_z"] = zscore(a["flow"], 96)
    a["eff"] = (a.close-a.close.shift(12)).abs()/a.close.diff().abs().rolling(12).sum()

    # Structure levels are strictly prior to the current 1H candle.
    for n in (12, 20, 32, 48):
        a[f"hi_{n}"] = a.high.shift(2).rolling(n).max()
        a[f"lo_{n}"] = a.low.shift(2).rolling(n).min()

    # Prior-hour body/range features for displacement detection.
    a["prev_range"] = (a.high.shift(1)-a.low.shift(1))
    a["prev_body"] = (a.close.shift(1)-a.open.shift(1)).abs()
    a["prev_body_frac"] = a.prev_body/a.prev_range.replace(0,np.nan)
    a["prev_close_loc"] = (a.close.shift(1)-a.low.shift(1))/a.prev_range.replace(0,np.nan)
    a["prev_atr"] = a.atr.shift(1)
    a["prev_vol_z"] = a.vol_z.shift(1)
    a["prev_eff"] = a.eff.shift(1)

    spot = spot.reindex(a.index)
    a["spot_close"] = spot.close
    a["basis"] = a.close/a.spot_close-1.0
    a["basis_z"] = zscore(a.basis, 192)
    a["spot_ret_24"] = spot.close.pct_change(24, fill_method=None)
    a["fut_spot_gap"] = a.ret_24-a.spot_ret_24
    a["spot_available"] = a.spot_close.notna()

    btc = btc_h1.close.reindex(a.index)
    a["btc_ret_24"] = btc.pct_change(24)
    a["rs24"] = a.ret_24-a.btc_ret_24
    a["rs_z"] = zscore(a.rs24, 96)

    h4["ema20"] = h4.close.ewm(span=20, adjust=False).mean()
    h4["ema50"] = h4.close.ewm(span=50, adjust=False).mean()
    h4["atr"] = atr(h4)
    h4["trend"] = np.where(h4.ema20 > h4.ema50, 1, -1)
    h4["slope"] = h4.ema20.pct_change(5)

    d1["ema50"] = d1.close.ewm(span=50, adjust=False).mean()
    d1["ema200"] = d1.close.ewm(span=200, adjust=False).mean()
    d1["slope50"] = d1.ema50.pct_change(5)
    d1["regime"] = np.where((d1.close>d1.ema200)&(d1.slope50>0),1,np.where((d1.close<d1.ema200)&(d1.slope50<0),-1,0))

    # Completed higher-timeframe bars only.
    h4a = h4.shift(1).reindex(a.index, method="ffill")
    d1a = d1.shift(1).reindex(a.index, method="ffill")
    a["h4_trend"] = h4a.trend
    a["h4_slope"] = h4a.slope
    a["h4_atr"] = h4a.atr
    a["d1_regime"] = d1a.regime
    a["d1_slope"] = d1a.slope50
    return a


def common_index(frames):
    common = None
    for x in frames.values():
        s = set(x.index)
        common = s if common is None else common.intersection(s)
    return pd.DatetimeIndex(sorted(common))


def make_folds(frames):
    t = common_index(frames)
    if len(t) < 700:
        raise RuntimeError("Insufficient common 1H timestamps for walk-forward")
    start = t[0]
    step = pd.Timedelta(days=FOLD_OOS_DAYS)
    train_end = start + pd.Timedelta(days=FOLD_TRAIN_DAYS)
    val_end = train_end + pd.Timedelta(days=FOLD_VAL_DAYS)
    folds=[]
    n=1
    final_start = t[-1] - pd.Timedelta(days=FINAL_HOLDOUT_DAYS)
    while val_end + pd.Timedelta(days=FOLD_OOS_DAYS) <= final_start:
        folds.append(Fold(n,start,train_end,val_end,val_end+pd.Timedelta(days=FOLD_OOS_DAYS)))
        n += 1
        train_end = val_end
        val_end = train_end + pd.Timedelta(days=FOLD_VAL_DAYS)
    if not folds:
        raise RuntimeError("No valid walk-forward folds")
    return folds, t


def factor_audit(frames, fold):
    facs=["rs_z","flow_z","vol_z","basis_z","fut_spot_gap","eff","ret_24"]
    rows=[]
    for sym,x in frames.items():
        tr=x[(x.index>=fold.train_start)&(x.index<fold.train_end)]
        for fac in facs:
            z=pd.concat([tr[fac],tr.close.pct_change(8,fill_method=None).shift(-8)],axis=1).dropna()
            if len(z)>=50:
                rows.append((sym,fac,z.iloc[:,0].rank().corr(z.iloc[:,1].rank())))
    q=pd.DataFrame(rows,columns=["symbol","factor","spearman"])
    print("\n================ FIRST-FOLD TRAIN FACTOR AUDIT ================")
    if q.empty:
        print("No sufficient factor observations.")
    else:
        print(q.groupby("factor").spearman.agg(["count","mean","median","max","min"]).round(4).to_string())
    print("===============================================================\n")


def candidate(ts, sym, x, cfg):
    if ts not in x.index:
        return None
    r=x.loc[ts]
    n=cfg.range_n
    req=[f"hi_{n}",f"lo_{n}","atr","atr_pct","prev_atr","prev_body_frac","prev_close_loc",
         "prev_vol_z","prev_eff","h4_trend","d1_regime","rs_z","basis_z","eff"]
    if any(pd.isna(r.get(k)) for k in req):
        return None

    upper=float(r[f"hi_{n}"])
    lower=float(r[f"lo_{n}"])
    atrv=float(r.atr)
    prev_atr=float(r.prev_atr)
    prev_range=float(r.prev_range)
    if atrv<=0 or prev_atr<=0 or prev_range<=0:
        return None

    # Displacement happened on the immediately completed 1H candle.
    # The actual prior candle is read directly from the causal shifted position.
    prev_close=float(x.close.shift(1).loc[ts])
    prev_open=float(x.open.shift(1).loc[ts])
    prev_high=float(x.high.shift(1).loc[ts])
    prev_low=float(x.low.shift(1).loc[ts])

    long_break = prev_close > upper
    short_break = prev_close < lower
    long_disp = (
        long_break
        and (prev_close-prev_open) > cfg.displacement_atr*prev_atr
        and float(r.prev_body_frac) >= cfg.body_frac
        and float(r.prev_vol_z) >= cfg.volume_z
        and float(r.prev_eff) >= 0.35
        and float(r.prev_close_loc) >= 0.70
    )
    short_disp = (
        short_break
        and (prev_open-prev_close) > cfg.displacement_atr*prev_atr
        and float(r.prev_body_frac) >= cfg.body_frac
        and float(r.prev_vol_z) >= cfg.volume_z
        and float(r.prev_eff) >= 0.35
        and float(r.prev_close_loc) <= 0.30
    )
    if not (long_disp or short_disp):
        return None

    # Current 1H candle is the causal retest/hold candle. It may wick through
    # the broken level, but must close back on the correct side.
    long_retest = r.low <= upper + cfg.retest_atr*atrv and r.close > upper and r.close > r.open
    short_retest = r.high >= lower - cfg.retest_atr*atrv and r.close < lower and r.close < r.open

    # Context is a directional filter, not a score. Neutral daily regime is
    # allowed; opposite daily regime is rejected. This prevents the old
    # additive-score failure mode where weak factors could outvote structure.
    long_context = int(r.h4_trend) >= 0 and int(r.d1_regime) >= 0 and float(r.rs_z) > -1.5
    short_context = int(r.h4_trend) <= 0 and int(r.d1_regime) <= 0 and float(r.rs_z) < 1.5

    long_ok = long_disp and long_retest and long_context
    short_ok = short_disp and short_retest and short_context
    if not (long_ok or short_ok):
        return None

    # If both directions somehow qualify, prefer the side with the stronger
    # displacement normalized by its prior ATR. This is deterministic and uses
    # only completed information.
    long_strength=((prev_close-prev_open)/prev_atr) if long_ok else -np.inf
    short_strength=((prev_open-prev_close)/prev_atr) if short_ok else -np.inf
    side=1 if long_strength>=short_strength else -1
    score=float(max(long_strength,short_strength))

    return {
        "symbol":sym,"side":side,"score":score,"atr":atrv,
        "d1_regime":int(r.d1_regime),"h4_trend":int(r.h4_trend),
        "vol_z":float(r.prev_vol_z),"rs_z":float(r.rs_z),"eff":float(r.prev_eff),
        "basis_z":float(r.basis_z),"body_frac":float(r.prev_body_frac),
        "close_loc":float(r.prev_close_loc),"ret_24":float(r.ret_24),
        "range_n":n,
    }


def entry_price(openp, side):
    return openp*(1+SLIPPAGE*side)


def exit_price(p, side):
    return p*(1-SLIPPAGE*side)


def pnl_for(side, entry, exitp):
    gross=(exitp-entry)/entry*NOTIONAL*side
    fees=NOTIONAL*FEE_RATE*2.0
    return gross-fees


def run_backtest(frames,cfg,start,end,label,collect_trades=False):
    times=common_index({k:v[(v.index>=start)&(v.index<end)] for k,v in frames.items()})
    equity=INITIAL_CAPITAL
    peak=equity
    max_dd=0.0
    active=None
    trades=[]
    blocked=0
    last_close_ts=None

    for ts in times:
        if active is not None:
            x=frames[active["symbol"]]
            if ts>=active["entry_ts"] and ts in x.index:
                b=x.loc[ts]
                side=active["side"]
                sl=b.low<=active["sl"] if side==1 else b.high>=active["sl"]
                tp=b.high>=active["tp"] if side==1 else b.low<=active["tp"]
                if sl or tp:
                    outcome="LOSS" if sl else "WIN"
                    exit_reason="SL" if sl else "TP"
                    raw=active["sl"] if sl else active["tp"]
                    xp=exit_price(raw,side)
                    p=max(-MARGIN,pnl_for(side,active["entry"],xp))
                    equity=max(0.0,equity+p)
                    peak=max(peak,equity)
                    max_dd=max(max_dd,peak-equity)
                    trades.append({**active,"exit_ts":ts,"exit_price":xp,"exit_reason":exit_reason,
                                   "outcome":outcome,"pnl":p,"equity_after":equity})
                    active=None
                    last_close_ts=ts
                    continue
        if active is not None:
            continue
        if last_close_ts is not None and ts<=last_close_ts:
            continue
        if equity<MARGIN:
            blocked+=1
            continue

        cand=[]
        scores=[]
        for sym,x in frames.items():
            c=candidate(ts,sym,x,cfg)
            if c is None:
                scores.append(0.0)
            else:
                cand.append(c); scores.append(float(c["score"]))
        if not cand:
            continue

        cand.sort(key=lambda c:(c["score"],c["symbol"]),reverse=True)
        top=cand[0]
        rank_pct=float(np.mean(np.asarray(scores)<top["score"]))
        if rank_pct < cfg.rank_floor:
            continue

        sym=top["symbol"]; side=top["side"]; x=frames[sym]
        i=x.index.searchsorted(ts,side="right")
        if i>=len(x):
            continue
        entry_ts=x.index[i]
        if entry_ts<start or entry_ts>=end:
            continue
        ep=entry_price(float(x.iloc[i].open),side)
        dist=float(top["atr"])*cfg.stop_atr
        if not np.isfinite(dist) or dist<=0:
            continue
        active={
            "symbol":sym,"side":side,"signal_ts":ts,"entry_ts":entry_ts,
            "entry":ep,"sl":ep-dist*side,"tp":ep+dist*RR*side,
            "signal_score":top["score"],"atr":top["atr"],
            "d1_regime":top["d1_regime"],"h4_trend":top["h4_trend"],
            "vol_z":top["vol_z"],"rs_z":top["rs_z"],"eff":top["eff"],
            "basis_z":top["basis_z"],"body_frac":top["body_frac"],
            "close_loc":top["close_loc"],"ret_24":top["ret_24"],
            "candidate_count":len(cand),"rank_pct":rank_pct,
        }

    vals=pd.Series([t["pnl"] for t in trades],dtype=float)
    wins=int((vals>0).sum()); losses=int((vals<=0).sum())
    gp=float(vals[vals>0].sum()) if wins else 0.0
    gl=float(-vals[vals<=0].sum()) if losses else 0.0
    pf=gp/gl if gl else (math.inf if wins else 0.0)
    wr=100*wins/len(vals) if len(vals) else 0.0
    streak=cur=0
    for v in vals:
        if v<=0:
            cur+=1; streak=max(streak,cur)
        else:
            cur=0
    days=max((end-start).total_seconds()/86400.0,1e-9)
    return {
        "label":label,"config":cfg.name,"trades":len(vals),"wins":wins,"losses":losses,
        "wr":wr,"pf":pf,"pnl":float(vals.sum()) if len(vals) else 0.0,"dd":max_dd,
        "dd_pct":100*max_dd/peak if peak>0 else 100.0,"streak":streak,
        "tday":len(vals)/days,"open":active is not None,"blocked":blocked,
        "avg_win":gp/wins if wins else 0.0,"avg_loss":-gl/losses if losses else 0.0,
        "expectancy":float(vals.mean()) if len(vals) else 0.0,
        "median":float(vals.median()) if len(vals) else 0.0,"final_equity":equity,
        "trades_log":trades if collect_trades else None,
    }


def select_score(train,val):
    if train["trades"]<MIN_TRAIN_TRADES or val["trades"]<MIN_VAL_TRADES:
        return -1e9
    if train["pnl"]<=0 or val["pnl"]<=0:
        return -1e8
    if train["pf"]<1.05 or val["pf"]<1.05:
        return -1e7
    # Reward validation quality, but penalize instability and drawdown.
    return (3.0*val["pf"] + 0.02*val["wr"] + 0.001*val["pnl"]
            + 0.75*train["pf"] - 0.03*val["streak"] - 0.02*val["dd_pct"])


def report(r,title):
    print(f"\n================ {title} ================")
    print(f"Trades                 : {r['trades']}")
    print(f"Wins / Losses          : {r['wins']} / {r['losses']}")
    print(f"Win Rate               : {r['wr']:.2f}%")
    print(f"Profit Factor          : {r['pf']:.4f}")
    print(f"Net PnL                : ${r['pnl']:,.2f}")
    print(f"Final Realized Equity  : ${r['final_equity']:,.2f}")
    print(f"Max Drawdown           : ${r['dd']:,.2f}")
    print(f"Max Drawdown %         : {r['dd_pct']:.2f}%")
    print(f"Max Loss Streak        : {r['streak']}")
    print(f"Trades / Day           : {r['tday']:.4f}")
    print(f"Average Win            : ${r['avg_win']:,.2f}")
    print(f"Average Loss           : ${r['avg_loss']:,.2f}")
    print(f"Expectancy / Trade     : ${r['expectancy']:,.2f}")
    print(f"Median Trade PnL       : ${r['median']:,.2f}")
    print(f"Capital-blocked checks : {r['blocked']}")
    print(f"Open position at end   : {r['open']}")
    print("==================================================")


def aggregate_trade_stats(trades, start, end):
    vals=pd.Series([t["pnl"] for t in trades],dtype=float)
    wins=int((vals>0).sum()); losses=int((vals<=0).sum())
    gp=float(vals[vals>0].sum()) if wins else 0.0
    gl=float(-vals[vals<=0].sum()) if losses else 0.0
    pf=gp/gl if gl else (math.inf if wins else 0.0)
    streak=cur=0
    equity=INITIAL_CAPITAL; peak=equity; dd=0.0
    for v in vals:
        equity=max(0.0,equity+float(v)); peak=max(peak,equity); dd=max(dd,peak-equity)
        if v<=0:
            cur+=1; streak=max(streak,cur)
        else: cur=0
    return {
        "trades":len(vals),"wins":wins,"losses":losses,"wr":100*wins/len(vals) if len(vals) else 0.0,
        "pf":pf,"pnl":float(vals.sum()) if len(vals) else 0.0,"dd":dd,
        "dd_pct":100*dd/peak if peak else 100.0,"streak":streak,
        "expectancy":float(vals.mean()) if len(vals) else 0.0,"final_equity":equity,
    }


def print_side_report(trades,title):
    df=pd.DataFrame(trades)
    print(f"\n================ {title} ================")
    if df.empty:
        print("No trades."); print("=================================================="); return
    df["side_name"]=df.side.map({1:"LONG",-1:"SHORT"})
    rows=[]
    for key,g in df.groupby("side_name"):
        vals=g.pnl.astype(float); wins=(vals>0).sum(); gp=vals[vals>0].sum(); gl=-vals[vals<=0].sum()
        rows.append((key,len(g),100*wins/len(g),gp/gl if gl else math.inf,g.pnl.sum(),g.pnl.mean()))
    out=pd.DataFrame(rows,columns=["side","trades","WR","PF","PnL","Expectancy"])
    print(out.to_string(index=False,formatters={"WR":"{:.2f}".format,"PF":"{:.3f}".format,"PnL":"{:.2f}".format,"Expectancy":"{:.2f}".format}))
    print("==================================================")


def save_forensic(trades,outdir):
    outdir.mkdir(parents=True,exist_ok=True)
    if not trades:
        return
    df=pd.DataFrame(trades)
    for c in ("signal_ts","entry_ts","exit_ts"):
        df[c]=pd.to_datetime(df[c],utc=True)
    df["month"]=df.exit_ts.dt.to_period("M").astype(str)
    df["side_name"]=df.side.map({1:"LONG",-1:"SHORT"})
    df["regime_name"]=df.d1_regime.map({1:"BULL",0:"NEUTRAL",-1:"BEAR"})
    df.to_csv(outdir/"walk_forward_oos_trades.csv",index=False)
    print(f"[FORENSIC] {outdir/'walk_forward_oos_trades.csv'}")


def main():
    refresh=os.getenv("XT_REFRESH","0")=="1"
    root=Path("data/xt_v16")
    outdir=Path("reports/xt_v16")
    print("HUNTER-V16 — AUCTION DISPLACEMENT / RETEST + WALK-FORWARD")
    print(f"Capital=${INITIAL_CAPITAL:.0f} Margin=${MARGIN:.0f} Leverage={LEVERAGE:.0f}x RR=1:{RR:.0f}")
    print("One global position; no overlap; next 1H open; no timeout/BE/trailing.")

    fut={}; spot={}
    for sym in SYMBOLS:
        fut[sym]=fetch_xt(sym,FUTURES_URL,root/"futures"/f"{sym}.csv",refresh,True)
        spot[sym]=fetch_xt(sym,SPOT_URL,root/"spot"/f"{sym}.csv",refresh,False)

    btc_h1=complete_resample(fut["btc_usdt"],"1h",4)
    frames={sym:build_features(fut[sym],spot[sym],btc_h1) for sym in SYMBOLS}
    folds,t=make_folds(frames)
    print(f"COMMON 1H: {t[0]} -> {t[-1]} | folds={len(folds)} | final_holdout={FINAL_HOLDOUT_DAYS}d")
    factor_audit(frames,folds[0])

    # Small predeclared grid. It is structural, not an OOS-tuned search.
    grid=[]
    for rn in (20,32,48):
        for da in (0.8,1.0):
            for bf in (0.55,0.70):
                for vz in (0.0,0.5):
                    for rt in (0.15,0.30):
                        for sa in (1.25,1.50):
                            for rf in (0.0,0.5):
                                grid.append(Config(f"N{rn}_D{da}_B{bf}_V{vz}_R{rt}_S{sa}_RF{rf}",rn,da,bf,vz,rt,sa,rf))

    fold_records=[]
    all_oos_trades=[]
    for fold in folds:
        print(f"\n################ WALK-FORWARD FOLD {fold.number} ################")
        print(f"TRAIN [{fold.train_start},{fold.train_end})")
        print(f"VAL   [{fold.train_end},{fold.val_end})")
        print(f"OOS   [{fold.val_end},{fold.oos_end})")
        ranked=[]
        for cfg in grid:
            tr=run_backtest(frames,cfg,fold.train_start,fold.train_end,"TRAIN")
            va=run_backtest(frames,cfg,fold.train_end,fold.val_end,"VALIDATION")
            ranked.append((select_score(tr,va),cfg,tr,va))
        ranked.sort(key=lambda z:z[0],reverse=True)
        eligible=[z for z in ranked if z[0]>-1e8]
        if not eligible:
            raise RuntimeError(f"Fold {fold.number}: no eligible train/validation configuration")
        _,cfg,tr,va=eligible[0]
        oos=run_backtest(frames,cfg,fold.val_end,fold.oos_end,"OOS",collect_trades=True)
        print(f"[FOLD {fold.number} SELECTED] {cfg.name}")
        report(tr,f"FOLD {fold.number} TRAIN")
        report(va,f"FOLD {fold.number} VALIDATION")
        report(oos,f"FOLD {fold.number} OOS")
        print_side_report(oos["trades_log"],f"FOLD {fold.number} OOS BY SIDE")
        if oos["trades"] < MIN_FOLD_OOS_TRADES:
            print(f"[FOLD {fold.number}] WARNING: only {oos['trades']} OOS trades")
        fold_records.append((fold,cfg,tr,va,oos))
        all_oos_trades.extend(oos["trades_log"] or [])

    # Final holdout is evaluated only with the configuration selected from the
    # immediately preceding walk-forward validation process. We deliberately do
    # NOT search the final holdout or change parameters from its result.
    last_cfg=fold_records[-1][1]
    final_start=t[-1]-pd.Timedelta(days=FINAL_HOLDOUT_DAYS)
    final_oos=run_backtest(frames,last_cfg,final_start,t[-1]+pd.Timedelta(hours=1),"FINAL_HOLDOUT",collect_trades=True)
    report(final_oos,"FINAL UNTOUCHED HOLDOUT")
    print_side_report(final_oos["trades_log"],"FINAL HOLDOUT BY SIDE")

    agg=aggregate_trade_stats(all_oos_trades,folds[0].val_end,folds[-1].oos_end)
    print("\n================ AGGREGATED WALK-FORWARD OOS ================")
    for k,v in agg.items():
        if isinstance(v,float): print(f"{k:20s}: {v:.4f}")
        else: print(f"{k:20s}: {v}")
    print("==============================================================")

    positive_folds=sum(1 for _,_,_,_,o in fold_records if o["pnl"]>0)
    checks={
        "Agg OOS trades":agg["trades"]>=MIN_AGG_OOS_TRADES,
        "Agg OOS PF":agg["pf"]>=MIN_AGG_OOS_PF,
        "Agg OOS DD":agg["dd_pct"]<=MAX_AGG_OOS_DD_PCT,
        "Agg OOS streak":agg["streak"]<=MAX_AGG_OOS_STREAK,
        "Positive OOS folds":positive_folds>=MIN_POSITIVE_OOS_FOLDS,
    }
    print("\n================ WALK-FORWARD GATE ================")
    for k,v in checks.items(): print(f"{k:22s}: {'PASS' if v else 'FAIL'}")
    accepted=all(checks.values())
    print(f"ROBUST_WALK_FORWARD     : {accepted}")
    print("====================================================")

    save_forensic(all_oos_trades,outdir)
    save_forensic(final_oos["trades_log"],outdir/"final_holdout")

    if accepted:
        print("[DECISION] WALK-FORWARD SURVIVED. Final holdout is reported separately; no production claim is made from this run alone.")
    else:
        print("[DECISION] REJECTED. No stable edge demonstrated; close this strategy family and move to a new hypothesis.")


if __name__=="__main__":
    main()
