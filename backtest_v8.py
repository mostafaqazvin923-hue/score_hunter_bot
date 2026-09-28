#!/usr/bin/env python3
"""
HUNTER-V14 — FORENSIC / CAUSAL / SPLIT-ISOLATED BACKTEST ENGINE
XT USDT-M PERPETUAL FUTURES
15m raw -> causal 1H / 4H / 1D

V14 purpose:
  - Preserve V13 signal architecture.
  - Make Train / Validation / OOS intervals half-open and isolated.
  - Never allow a signal whose NEXT-1H entry lies outside the current segment.
  - Keep OOS untouched by parameter selection.
  - Record a forensic trade log for diagnosis.
  - Produce OOS breakdowns by side, symbol, regime, score bucket,
    exit reason, and month.
  - Remove the fake rank_floor behavior: rank_floor is now a real
    cross-sectional percentile threshold.

No lookahead, repainting, fabricated OHLCV, OI, CVD, or liquidation data.
No timeout, BE, or trailing. Fixed RR 1:2.
Same-candle SL+TP => LOSS.
Open position at segment end is not realized and is not carried into the next segment.
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

TRAIN_FRAC = 0.60
VAL_FRAC = 0.20

MIN_ROWS = 38000
MIN_TRAIN_TRADES = 30
MIN_VAL_TRADES = 15

# These are reporting/acceptance gates only. They do not alter OOS selection.
MIN_OOS_TRADES = 100
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
    sweep: int
    min_score: float
    stop_atr: float
    rank_floor: float
    vol_floor: float


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
    if len(x) < MIN_ROWS:
        raise RuntimeError(f"{symbol}: only {len(x)} rows; expected at least {MIN_ROWS}")
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
    rng = (a.high-a.low).replace(0, np.nan)
    a["body_frac"] = (a.close-a.open).abs()/rng
    a["close_loc"] = (a.close-a.low)/rng
    a["ret_12"] = a.close.pct_change(12, fill_method=None)
    a["ret_24"] = a.close.pct_change(24, fill_method=None)
    a["ret_72"] = a.close.pct_change(72, fill_method=None)
    a["vol_z"] = zscore(np.log1p(a.volume), 96)
    a["vol_ratio"] = a.volume/a.volume.rolling(24).median()
    a["flow"] = ((a.close-a.open)/rng)*a.volume
    a["flow_z"] = zscore(a["flow"], 96)
    a["eff"] = (a.close-a.close.shift(12)).abs()/a.close.diff().abs().rolling(12).sum()
    for n in (20, 32, 48):
        a[f"hi_{n}"] = a.high.shift(1).rolling(n).max()
        a[f"lo_{n}"] = a.low.shift(1).rolling(n).min()

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

    # Shift the completed higher-timeframe bar before mapping it onto 1H.
    h4a = h4.shift(1).reindex(a.index, method="ffill")
    d1a = d1.shift(1).reindex(a.index, method="ffill")
    a["h4_trend"] = h4a.trend
    a["h4_slope"] = h4a.slope
    a["h4_atr"] = h4a.atr
    a["d1_regime"] = d1a.regime
    a["d1_slope"] = d1a.slope50
    return a


def split_timeline(frames):
    common = None
    for x in frames.values():
        common = set(x.index) if common is None else common.intersection(x.index)
    t = pd.DatetimeIndex(sorted(common))
    if len(t) < 500:
        raise RuntimeError("Insufficient common 1H timestamps")
    span = t[-1]-t[0]
    return Split(t[0], t[0]+span*TRAIN_FRAC, t[0]+span*(TRAIN_FRAC+VAL_FRAC), t[-1])


def spearman(x, y):
    z = pd.concat([x,y],axis=1).dropna()
    if len(z)<50:
        return np.nan
    return z.iloc[:,0].rank().corr(z.iloc[:,1].rank())


def factor_audit(frames, split):
    facs = ["rs_z","flow_z","vol_z","basis_z","fut_spot_gap","eff","ret_24"]
    rows=[]
    for sym,x in frames.items():
        tr=x[(x.index >= split.start) & (x.index < split.train_end)]
        for fac in facs:
            for h in (1,4,8,24):
                rows.append((sym,fac,h,spearman(tr[fac],tr.close.shift(-h)/tr.close-1)))
    q=pd.DataFrame(rows,columns=["symbol","factor","horizon","spearman"])
    print("\n================ TRAIN FACTOR AUDIT ================")
    print(q.groupby("factor").spearman.agg(["count","mean","median","max","min"]).round(4).to_string())
    print("=====================================================\n")


def asset_profile(sym, row):
    if sym in {"btc_usdt","eth_usdt","bnb_usdt"}:
        return 1.00
    if sym in {"sol_usdt","avax_usdt","sui_usdt","near_usdt","icp_usdt"}:
        return 1.05
    return 1.10


def candidate(ts, sym, x, cfg):
    if ts not in x.index:
        return None
    r=x.loc[ts]
    n=cfg.sweep
    req=[f"hi_{n}",f"lo_{n}","atr","h4_trend","d1_regime","vol_z","flow_z","rs_z","eff","close_loc","ret_24"]
    if any(pd.isna(r.get(k)) for k in req):
        return None
    hi,lo=float(r[f"hi_{n}"]),float(r[f"lo_{n}"])
    atrv=float(r.atr)
    long_sweep=r.low<lo and r.close>lo
    short_sweep=r.high>hi and r.close<hi
    if not (long_sweep or short_sweep):
        return None

    long_ok = r.d1_regime >= 0 and r.h4_trend >= 0
    short_ok = r.d1_regime <= 0 and r.h4_trend <= 0
    lr=(r.close-lo)/atrv
    sr=(hi-r.close)/atrv
    base_long = (2.0*long_sweep + 1.0*(lr>=0.15) + 1.0*(r.close_loc>=0.62)
                 + 0.8*(r.flow_z>0) + 0.8*(r.vol_z>cfg.vol_floor)
                 + 0.8*(r.eff<0.55) + 0.8*(r.rs_z>0) + 0.6*long_ok
                 + 0.4*(r.basis_z<2.5 if pd.notna(r.basis_z) else False))
    base_short = (2.0*short_sweep + 1.0*(sr>=0.15) + 1.0*(r.close_loc<=0.38)
                  + 0.8*(r.flow_z<0) + 0.8*(r.vol_z>cfg.vol_floor)
                  + 0.8*(r.eff<0.55) + 0.8*(r.rs_z<0) + 0.6*short_ok
                  + 0.4*(r.basis_z>-2.5 if pd.notna(r.basis_z) else False))
    score=max(base_long,base_short)
    if score < cfg.min_score:
        return None
    side=1 if base_long>base_short else -1
    if side==1 and not long_sweep:
        return None
    if side==-1 and not short_sweep:
        return None
    if not np.isfinite(atrv) or atrv<=0:
        return None
    return {
        "symbol":sym, "side":side, "score":float(score), "atr":atrv,
        "profile":asset_profile(sym,r), "d1_regime":int(r.d1_regime),
        "h4_trend":int(r.h4_trend), "vol_z":float(r.vol_z),
        "flow_z":float(r.flow_z), "rs_z":float(r.rs_z), "eff":float(r.eff),
        "basis_z":float(r.basis_z) if pd.notna(r.basis_z) else np.nan,
        "close_loc":float(r.close_loc), "ret_24":float(r.ret_24),
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
    # Half-open segment [start, end). This prevents boundary candles from being
    # shared between Train/Validation/OOS. A signal is eligible only if its NEXT
    # 1H entry is also strictly inside this segment.
    common=None
    for x in frames.values():
        idx=set(x.index[(x.index >= start) & (x.index < end)])
        common=idx if common is None else common.intersection(idx)
    times=pd.DatetimeIndex(sorted(common))

    equity=INITIAL_CAPITAL
    peak=equity
    max_dd=0.0
    active=None
    trades=[]
    blocked=0
    last_close_ts=None

    for ts in times:
        # First process an already-open position on the current candle.
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
                    p=pnl_for(side,active["entry"],xp)
                    p=max(-MARGIN,p)
                    equity=max(0.0,equity+p)
                    peak=max(peak,equity)
                    max_dd=max(max_dd,peak-equity)
                    trades.append({
                        **active,
                        "exit_ts":ts,
                        "exit_price":xp,
                        "exit_reason":exit_reason,
                        "outcome":outcome,
                        "pnl":p,
                        "equity_after":equity,
                    })
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
        cross_scores=[]
        for sym,x in frames.items():
            c=candidate(ts,sym,x,cfg)
            if c is None:
                cross_scores.append(0.0)
            else:
                cand.append(c)
                cross_scores.append(float(c["score"]))
        if not cand:
            continue

        # Cross-sectional percentile is measured against the WHOLE universe,
        # assigning score=0 to symbols with no candidate at this timestamp.
        # This makes rank_floor materially active instead of always being 1.0
        # for the top candidate. Percentile uses <= so ties receive the same
        # percentile and cannot manufacture an advantage from sort order.
        cand.sort(key=lambda c:(c["score"], c["symbol"]), reverse=True)
        top=cand[0]
        top_score=float(top["score"])
        rank_pct=float(np.mean(np.asarray(cross_scores) <= top_score))
        if rank_pct < cfg.rank_floor:
            continue

        sym=top["symbol"]
        side=top["side"]
        x=frames[sym]
        i=x.index.searchsorted(ts,side="right")
        if i>=len(x):
            continue
        entry_ts=x.index[i]
        # Crucial split-isolation rule: no signal can create a position outside
        # the current segment.
        if entry_ts < start or entry_ts >= end:
            continue
        ep=entry_price(float(x.iloc[i].open),side)
        dist=float(top["atr"])*cfg.stop_atr*float(top["profile"])
        if not np.isfinite(dist) or dist<=0:
            continue
        active={
            "symbol":sym,"side":side,"signal_ts":ts,"entry_ts":entry_ts,
            "entry":ep,"sl":ep-dist*side,"tp":ep+dist*RR*side,
            "signal_score":top["score"],"atr":top["atr"],
            "profile":top["profile"],"d1_regime":top["d1_regime"],
            "h4_trend":top["h4_trend"],"vol_z":top["vol_z"],
            "flow_z":top["flow_z"],"rs_z":top["rs_z"],"eff":top["eff"],
            "basis_z":top["basis_z"],"close_loc":top["close_loc"],
            "ret_24":top["ret_24"],"candidate_count":len(cand),
        }

    vals=pd.Series([t["pnl"] for t in trades],dtype=float)
    wins=int((vals>0).sum())
    losses=int((vals<=0).sum())
    gp=float(vals[vals>0].sum()) if wins else 0.0
    gl=float(-vals[vals<=0].sum()) if losses else 0.0
    pf=gp/gl if gl else (math.inf if wins else 0.0)
    wr=100*wins/len(vals) if len(vals) else 0.0
    streak=cur=0
    for v in vals:
        if v<=0:
            cur+=1
            streak=max(streak,cur)
        else:
            cur=0
    days=max((end-start).total_seconds()/86400.0,1e-9)
    result={
        "label":label,"config":cfg.name,"trades":len(vals),"wins":wins,"losses":losses,
        "wr":wr,"pf":pf,"pnl":float(vals.sum()) if len(vals) else 0.0,"dd":max_dd,
        "dd_pct":100*max_dd/peak if peak>0 else 100.0,"streak":streak,
        "tday":len(vals)/days,"open":active is not None,"blocked":blocked,
        "avg_win":gp/wins if wins else 0.0,"avg_loss":-gl/losses if losses else 0.0,
        "expectancy":float(vals.mean()) if len(vals) else 0.0,
        "median":float(vals.median()) if len(vals) else 0.0,
        "final_equity":equity,
        "trades_log":trades if collect_trades else None,
    }
    return result


def score(train,val):
    if train["trades"]<MIN_TRAIN_TRADES or val["trades"]<MIN_VAL_TRADES:
        return -1e9
    if train["pnl"]<=0 or val["pnl"]<=0:
        return -1e8
    if train["streak"]>10 or val["streak"]>8:
        return -1e7
    return (2.5*val["pf"] + 0.03*val["wr"] + 0.001*val["pnl"]
            -0.02*val["streak"] -0.015*val["dd_pct"] +0.75*train["pf"])


def report(r,title):
    print(f"\n================ {title} ================")
    print(f"Trades                 : {r['trades']}")
    print(f"Wins / Losses          : {r['wins']} / {r['losses']}")
    print(f"Win Rate               : {r['wr']:.2f}%")
    print(f"Profit Factor          : {r['pf']:.4f}")
    print(f"Net PnL                : ${r['pnl']:,.2f}")
    print(f"Final Realized Equity  : ${r['final_equity']:,.2f}")
    print(f"Max Drawdown           : ${r['dd']:,.2f}")
    print(f"Max Drawdown %         : {r['dd_pct']:.2f}% (DD / peak equity)")
    print(f"Max Loss Streak        : {r['streak']}")
    print(f"Trades / Day           : {r['tday']:.4f}")
    print(f"Average Win            : ${r['avg_win']:,.2f}")
    print(f"Average Loss           : ${r['avg_loss']:,.2f}")
    print(f"Expectancy / Trade     : ${r['expectancy']:,.2f}")
    print(f"Median Trade PnL       : ${r['median']:,.2f}")
    print(f"Capital-blocked checks : {r['blocked']} (informational)")
    print(f"Open position at end   : {r['open']} (informational)")
    print("==================================================")


def print_group_report(df, group_col, title):
    print(f"\n================ {title} ================")
    if df.empty:
        print("No realized trades.")
        print("==================================================")
        return
    rows=[]
    for key,g in df.groupby(group_col, dropna=False):
        wins=(g.pnl>0).sum()
        losses=(g.pnl<=0).sum()
        gp=g.loc[g.pnl>0,"pnl"].sum()
        gl=-g.loc[g.pnl<=0,"pnl"].sum()
        pf=gp/gl if gl else (math.inf if wins else 0.0)
        rows.append((str(key),len(g),100*wins/len(g),pf,g.pnl.sum(),g.pnl.mean()))
    out=pd.DataFrame(rows,columns=[group_col,"trades","WR","PF","PnL","Expectancy"])
    print(out.sort_values("PnL",ascending=False).to_string(index=False,formatters={
        "WR":"{:.2f}".format,"PF":"{:.3f}".format,
        "PnL":"{:.2f}".format,"Expectancy":"{:.2f}".format}))
    print("==================================================")


def forensic_report(trades, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    if not trades:
        print("\n[FORENSIC] No realized OOS trades.")
        return
    df=pd.DataFrame(trades)
    df["signal_ts"]=pd.to_datetime(df["signal_ts"],utc=True)
    df["entry_ts"]=pd.to_datetime(df["entry_ts"],utc=True)
    df["exit_ts"]=pd.to_datetime(df["exit_ts"],utc=True)
    df["month"]=df.exit_ts.dt.to_period("M").astype(str)
    df["side_name"]=df.side.map({1:"LONG",-1:"SHORT"})
    df["regime_name"]=df.d1_regime.map({1:"BULL",0:"NEUTRAL",-1:"BEAR"})
    df["trend_name"]=df.h4_trend.map({1:"UP",-1:"DOWN"})
    df["score_bucket"]=pd.cut(df.signal_score,
                               bins=[-np.inf,6.0,6.5,7.0,7.5,np.inf],
                               labels=["<6.0","6.0-6.5","6.5-7.0","7.0-7.5",">=7.5"],
                               right=False)
    path=outdir/"oos_trade_log.csv"
    df.to_csv(path,index=False)
    print(f"\n[FORENSIC] Full OOS trade log: {path}")
    for col,title in [
        ("side_name","OOS BY SIDE"),
        ("symbol","OOS BY SYMBOL"),
        ("regime_name","OOS BY 1D REGIME"),
        ("trend_name","OOS BY 4H TREND"),
        ("score_bucket","OOS BY SCORE BUCKET"),
        ("exit_reason","OOS BY EXIT REASON"),
        ("month","OOS BY MONTH"),
    ]:
        print_group_report(df,col,title)

    # Long/short x symbol is useful for identifying concentrated directional failure.
    pivot=[]
    for (sym,side),g in df.groupby(["symbol","side_name"]):
        wins=(g.pnl>0).sum(); losses=(g.pnl<=0).sum()
        gp=g.loc[g.pnl>0,"pnl"].sum(); gl=-g.loc[g.pnl<=0,"pnl"].sum()
        pf=gp/gl if gl else (math.inf if wins else 0.0)
        pivot.append((sym,side,len(g),100*wins/len(g),pf,g.pnl.sum()))
    px=pd.DataFrame(pivot,columns=["symbol","side","trades","WR","PF","PnL"])
    print("\n================ OOS BY SYMBOL × SIDE ================")
    print(px.sort_values("PnL",ascending=False).to_string(index=False,formatters={
        "WR":"{:.2f}".format,"PF":"{:.3f}".format,"PnL":"{:.2f}".format}))
    print("========================================================")


def main():
    refresh=os.getenv("XT_REFRESH","0")=="1"
    root=Path("data/xt_v14")
    outdir=Path("reports/xt_v14")

    print("HUNTER-V14 — FORENSIC / SPLIT-ISOLATED / CAUSAL")
    print("XT Futures + auxiliary Spot | 15m -> 1H/4H/1D")
    print(f"Capital=${INITIAL_CAPITAL:.0f} Margin=${MARGIN:.0f} Leverage={LEVERAGE:.0f}x RR=1:{RR:.0f}")
    print("One global position; no overlap; no timeout; no BE/trailing.")
    print("Segment policy: [start,end), no boundary trade carry-over.")

    fut={}; spot={}
    for sym in SYMBOLS:
        fut[sym]=fetch_xt(sym,FUTURES_URL,root/"futures"/f"{sym}.csv",refresh,True)
        spot[sym]=fetch_xt(sym,SPOT_URL,root/"spot"/f"{sym}.csv",refresh,False)

    btc_h1=complete_resample(fut["btc_usdt"],"1h",4)
    frames={sym:build_features(fut[sym],spot[sym],btc_h1) for sym in SYMBOLS}
    split=split_timeline(frames)
    print(f"\nSPLIT train=[{split.start},{split.train_end}) | validation=[{split.train_end},{split.val_end}) | OOS=[{split.val_end},{split.end})")
    factor_audit(frames,split)

    # Same small predeclared grid as V13. No OOS metrics enter selection.
    grid=[Config(f"S{s}_Q{q}_A{a}_R{r}_V{v}",s,q,a,r,v)
          for s in (20,32,48) for q in (5.8,6.4,7.0) for a in (1.25,1.50)
          for r in (0.50,0.75) for v in (0.0,0.5)]

    results=[]
    for cfg in grid:
        tr=run_backtest(frames,cfg,split.start,split.train_end,"TRAIN")
        va=run_backtest(frames,cfg,split.train_end,split.val_end,"VALIDATION")
        results.append((score(tr,va),cfg,tr,va))
    results.sort(key=lambda z:z[0],reverse=True)
    eligible=[z for z in results if z[0]>-1e8]
    if not eligible:
        raise RuntimeError("No train/validation eligible configuration; no validated edge.")

    _,cfg,tr,va=eligible[0]
    print(f"\n[SELECTED] {cfg.name}")
    print(f"[SELECTED PARAMS] sweep={cfg.sweep} min_score={cfg.min_score} stop_atr={cfg.stop_atr} rank_floor={cfg.rank_floor} vol_floor={cfg.vol_floor}")
    report(tr,"SELECTED TRAIN")
    report(va,"SELECTED VALIDATION")

    oos=run_backtest(frames,cfg,split.val_end,split.end,"OOS",collect_trades=True)
    report(oos,"UNTOUCHED OOS")
    forensic_report(oos["trades_log"],outdir)

    checks={
        "OOS trades":oos["trades"]>=MIN_OOS_TRADES,
        "OOS WR":oos["wr"]>MIN_OOS_WR,
        "OOS PF":oos["pf"]>MIN_OOS_PF,
        "OOS PnL":oos["pnl"]>0,
        "OOS loss streak":oos["streak"]<=MAX_OOS_STREAK,
        "OOS DD":oos["dd_pct"]<MAX_OOS_DD_PCT,
    }
    print("\n================ ACCEPTANCE GATE ================")
    for k,v in checks.items():
        print(f"{k:20s}: {'PASS' if v else 'FAIL'}")
    accepted=all(checks.values())
    print(f"ACCEPTED              : {accepted}")
    print("==================================================")
    print("[DECISION] ACCEPTED — all predeclared OOS criteria passed." if accepted else "[DECISION] REJECTED — no claim of robust edge.")


if __name__=="__main__":
    main()
