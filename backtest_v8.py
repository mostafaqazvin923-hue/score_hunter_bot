#!/usr/bin/env python3
"""
HUNTER-V144 — MTF CRYPTO FACTOR RESEARCH
1D regime + 4H context + 1H signal.
Research-only, causal, no lookahead, no timeout, no overlap.
Fixed RR 1:2. XT Futures 15m data -> 1H/4H/1D.

Families:
A) Cross-sectional momentum + relative strength
B) Short-term reversal + market/regime filter
C) Volatility regime + momentum/reversal switch
D) Funding/basis proxy is NOT fabricated: only enabled if historical XT funding/basis
   data is explicitly supplied. OHLCV alone cannot honestly recreate it.
"""
from __future__ import annotations
import argparse, time, math
from pathlib import Path
import numpy as np
import pandas as pd
import requests

SYMBOLS=["BTC","ETH","SOL","SUI","AVAX","NEAR","ADA","BNB","APT","CRV","ONDO","PENDLE","ICP","WIF"]
DATA_DIR=Path("data/xt_futures_v141")
OUT_DIR=DATA_DIR/"results"
INITIAL_EQUITY=1000.0; TRADE_MARGIN=100.0; LEVERAGE=50.0
FEE_RATE=0.0007; SLIPPAGE=0.0003; RR=2.0; MAX_OPEN_POSITIONS=3

BASE="https://fapi.xt.com"
LIMIT=1500; INTERVAL="15m"; DAYS=365; WARMUP_DAYS=35
MIN_ROWS=30000
MAX_REQUEST_RETRIES=6
S=requests.Session()
S.headers.update({"User-Agent":"HUNTER-backtest/143"})

def resolve_symbols():
    u=f"{BASE}/future/market/v1/public/symbol/list"
    r=S.get(u,timeout=20); r.raise_for_status()
    data=r.json().get("result",r.json())
    rows=data.get("symbols",[]) if isinstance(data,dict) else data
    out={}
    for x in rows:
        if not isinstance(x,dict): continue
        raw=str(x.get("symbol",x.get("name",""))).upper()
        compact=raw.replace("_","").replace("-","").replace("/","").replace(":","")
        for a in SYMBOLS:
            aliases={a+"USDT", a+"_USDT", a+"/USDT", a+"/USDT:USDT"}
            if compact in {z.replace("_","").replace("-","").replace("/","").replace(":","") for z in aliases}:
                out[a]=x.get("symbol") or x.get("name")
    return out

def _get_json(url, params):
    last=None
    for attempt in range(MAX_REQUEST_RETRIES):
        try:
            r=S.get(url,params=params,timeout=20)
            r.raise_for_status()
            js=r.json()
            # XT can return an HTTP-200 API error/rate-limit payload.
            if isinstance(js,dict):
                code=js.get("code")
                if code not in (None,0,"0",200,"200"):
                    raise RuntimeError(f"XT API code={code}: {js.get('msg',js.get('message',''))}")
            return js
        except Exception as e:
            last=e
            time.sleep(min(2.0,0.25*(2**attempt)))
    raise RuntimeError(f"XT request failed after {MAX_REQUEST_RETRIES} retries: {last}")

def _parse_kline(js):
    raw=js.get("result",js) if isinstance(js,dict) else js
    if isinstance(raw,dict):
        arr=raw.get("data",raw.get("rows",raw.get("list",[])))
    else:
        arr=raw
    if not isinstance(arr,list):
        return []
    out=[]
    for x in arr:
        if isinstance(x,dict):
            t=x.get("t",x.get("timestamp",x.get("time")))
            o=x.get("o",x.get("open")); h=x.get("h",x.get("high"))
            l=x.get("l",x.get("low")); c=x.get("c",x.get("close"))
            v=x.get("a",x.get("volume",x.get("q",0)))
        else:
            if len(x)<6: continue
            t,o,h,l,c,v=x[:6]
        try:
            out.append([int(t),float(o),float(h),float(l),float(c),float(v)])
        except (TypeError,ValueError):
            continue
    return out

def fetch_symbol(asset, xt_symbol, days):
    """
    Robust XT 15m collector.

    Important: XT documents both startTime and endTime for /q/kline.  We
    paginate FORWARD with startTime+endTime windows instead of relying on a
    single moving endTime.  This avoids the partial-page behaviour that caused
    the previous run to receive only 15 BTC candles.
    """
    now_ms=int(time.time()*1000)
    start_ms=now_ms-int((days+WARMUP_DAYS)*86400*1000)
    # Never use a still-forming candle.
    interval_ms=15*60*1000
    end_ms=(now_ms//interval_ms)*interval_ms-1

    best=[]
    endpoint=f"{BASE}/future/market/v1/public/q/kline"

    for whole_try in range(5):
        rows=[]
        cursor=start_ms
        guard=0
        failed=False

        while cursor < end_ms and guard < 2000:
            guard += 1
            # 1500 x 15m = 15.625 days. Keep a tiny overlap and dedupe later.
            window_end=min(end_ms, cursor + LIMIT*interval_ms - 1)
            params={
                "symbol":xt_symbol,
                "interval":INTERVAL,
                "startTime":cursor,
                "endTime":window_end,
                "limit":LIMIT,
            }

            batch=[]
            last_err=None
            for attempt in range(6):
                try:
                    batch=_parse_kline(_get_json(endpoint,params))
                    if batch:
                        break
                except Exception as e:
                    last_err=e
                time.sleep(min(2.0,0.35*(attempt+1)))

            if not batch:
                failed=True
                break

            # Keep only valid candles inside the requested window.
            batch=[x for x in batch if start_ms <= x[0] <= end_ms]
            if not batch:
                failed=True
                break

            rows.extend(batch)
            mn=min(x[0] for x in batch)
            mx=max(x[0] for x in batch)

            # Hard progress check. Never loop on the same XT page.
            if mx < cursor:
                failed=True
                break

            # Normal case: move just past the last candle received.
            next_cursor=mx+1
            if next_cursor <= cursor:
                failed=True
                break
            cursor=next_cursor

            # If XT returned fewer than LIMIT rows, the next forward request
            # is still valid; do not treat a short page as a fatal error.
            time.sleep(0.08)

        if rows:
            df=pd.DataFrame(
                rows,
                columns=["timestamp","open","high","low","close","volume"]
            )
            df=df.drop_duplicates("timestamp").sort_values("timestamp")
            df=df[(df.timestamp>=start_ms)&(df.timestamp<=end_ms)]

            if len(df)>len(best):
                best=df

            if len(df)>=MIN_ROWS:
                # Require broad time coverage, not merely row count.
                span=int(df.timestamp.max()-df.timestamp.min())
                required_span=int(days*86400*1000*0.90)
                if span >= required_span:
                    df["timestamp"]=pd.to_datetime(df.timestamp,unit="ms",utc=True)
                    return df.set_index("timestamp").dropna()

        time.sleep(1.0*(whole_try+1))

    got=len(best)
    if got:
        first=pd.to_datetime(int(best.timestamp.min()),unit="ms",utc=True)
        last=pd.to_datetime(int(best.timestamp.max()),unit="ms",utc=True)
        detail=f"; range={first} -> {last}"
    else:
        detail=""
    raise RuntimeError(
        f"Insufficient XT data for {asset}: got {got} rows{detail}; "
        f"refusing to run a partial backtest"
    )

def ensure_data(data_dir,days):
    data_dir.mkdir(parents=True,exist_ok=True); mapping=resolve_symbols()
    print(f"Verified XT symbols: {len(mapping)} / {len(SYMBOLS)}")
    if len(mapping)!=len(SYMBOLS): raise RuntimeError("XT symbol mapping incomplete")
    out={}
    for a in SYMBOLS:
        path=data_dir/f"{a}_USDT_15m.csv"
        try:
            df=fetch_symbol(a,mapping[a],days)
            if len(df)<MIN_ROWS:
                raise RuntimeError(f"Validation failed for {a}: only {len(df)} rows")
            df.to_csv(path)
            out[a]=df
            gaps=int(df.index.to_series().diff().dt.total_seconds().div(900).sub(1).clip(lower=0).sum())
            print(f"{a:<8} rows={len(df)} gaps={gaps}")
        except Exception as e:
            raise RuntimeError(f"XT download failed for {a}: {e}") from e
    return out

def resample(df,rule):
    return df.resample(rule,label="right",closed="right").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()

def features(all15):
    out={}
    h1={a:resample(df,"1h") for a,df in all15.items()}
    h4={a:resample(df,"4h") for a,df in all15.items()}
    d1={a:resample(df,"1D") for a,df in all15.items()}
    for a in SYMBOLS:
        x=h1[a].copy(); q=h4[a]; d=d1[a]
        x["ret24"]=x.close.pct_change(24); x["ret72"]=x.close.pct_change(72)
        x["atr"]=pd.concat([(x.high-x.low),(x.high-x.close.shift()).abs(),(x.low-x.close.shift()).abs()],axis=1).max(axis=1).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
        x["rv"]=x.close.pct_change().rolling(24).std()
        x["vol_z"]=(x.volume-x.volume.rolling(48).mean())/x.volume.rolling(48).std()
        x["ema20"]=x.close.ewm(span=20,adjust=False).mean()
        x["ema50"]=x.close.ewm(span=50,adjust=False).mean()
        x["dev"]=(x.close-x.ema20)/x.atr
        q["ema50"]=q.close.ewm(span=50,adjust=False).mean(); q["ema200"]=q.close.ewm(span=200,adjust=False).mean()
        q["trend"]=np.where((q.close>q.ema200)&(q.ema50>q.ema200),1,np.where((q.close<q.ema200)&(q.ema50<q.ema200),-1,0))
        d["ema50"]=d.close.ewm(span=50,adjust=False).mean(); d["ema200"]=d.close.ewm(span=200,adjust=False).mean()
        d["regime"]=np.where((d.close>d.ema200)&(d.ema50>d.ema200),1,np.where((d.close<d.ema200)&(d.ema50<d.ema200),-1,0))
        x["h4_trend"]=q["trend"].reindex(x.index,method="ffill")
        x["d1_regime"]=d["regime"].reindex(x.index,method="ffill")
        out[a]=x
    return out

def add_cross_sectional(f):
    times=sorted(set().union(*(x.index for x in f.values())))
    for ts in times:
        vals=[]
        for a,x in f.items():
            if ts in x.index and np.isfinite(x.loc[ts,"ret24"]): vals.append((a,x.loc[ts,"ret24"]))
        vals.sort(key=lambda z:z[1])
        n=len(vals)
        for rank,(a,_) in enumerate(vals):
            f[a].loc[ts,"mom_rank"]=rank/(n-1) if n>1 else .5
    return f

def signals(f):
    sig=[]
    for a,x in f.items():
        for i in range(1,len(x)-1):
            r=x.iloc[i]; ts=x.index[i]
            if not all(np.isfinite(r.get(k,np.nan)) for k in ["atr","ret24","ret72","mom_rank","vol_z","dev"]): continue
            # A: relative-strength momentum continuation
            longA=r.d1_regime==1 and r.h4_trend==1 and r.mom_rank>=.75 and r.ret24>0 and r.ret72>0 and r.close>r.ema20
            shortA=r.d1_regime==-1 and r.h4_trend==-1 and r.mom_rank<=.25 and r.ret24<0 and r.ret72<0 and r.close<r.ema20
            # B: controlled short-term reversal, only inside non-neutral daily regime
            longB=r.d1_regime==1 and r.h4_trend>=0 and r.dev<=-1.5 and r.vol_z>-1.0
            shortB=r.d1_regime==-1 and r.h4_trend<=0 and r.dev>=1.5 and r.vol_z>-1.0
            # C: volatility regime switch; momentum when volatility expands, reversal when compressed
            vol_q=x["rv"].rolling(240,min_periods=120).rank(pct=True).iloc[i]
            longC=(r.d1_regime==1 and r.h4_trend==1 and vol_q>=.70 and r.ret24>0 and r.mom_rank>=.60) or (r.d1_regime==1 and vol_q<=.25 and r.dev<=-1.75)
            shortC=(r.d1_regime==-1 and r.h4_trend==-1 and vol_q>=.70 and r.ret24<0 and r.mom_rank<=.40) or (r.d1_regime==-1 and vol_q<=.25 and r.dev>=1.75)
            for fam,side in [("A_MOMENTUM", "LONG" if longA else "SHORT" if shortA else None),("B_REVERSAL","LONG" if longB else "SHORT" if shortB else None),("C_REGIME_SWITCH","LONG" if longC else "SHORT" if shortC else None)]:
                if side: sig.append({"asset":a,"signal_ts":ts,"family":fam,"side":side,"atr":float(r.atr)})
    return sig

def backtest(f, sig):
    byts={}
    for s in sig: byts.setdefault(s["signal_ts"],[]).append(s)
    all_ts=sorted(set().union(*(x.index for x in f.values())))
    positions=[]; trades=[]; equity=INITIAL_EQUITY; peak=equity; last_close=pd.Timestamp.min.tz_localize("UTC")
    for ts in all_ts:
        # close positions before considering new signal; no same-close/signal candle reuse
        for p in positions[:]:
            x=f[p["asset"]]
            if ts not in x.index: continue
            row=x.loc[ts]; hit_sl=(row.low<=p["sl"] if p["side"]=="LONG" else row.high>=p["sl"]); hit_tp=(row.high>=p["tp"] if p["side"]=="LONG" else row.low<=p["tp"])
            if hit_sl or hit_tp:
                outcome="LOSS" if hit_sl else "WIN"; ex=p["sl"] if hit_sl else p["tp"]
                gross=((ex-p["entry"])/p["entry"] if p["side"]=="LONG" else (p["entry"]-ex)/p["entry"])*TRADE_MARGIN*LEVERAGE
                pnl=gross-(TRADE_MARGIN*LEVERAGE*FEE_RATE*2); equity+=pnl; peak=max(peak,equity)
                trades.append({**p,"exit_ts":ts,"outcome":outcome,"pnl":pnl,"equity":equity}); positions.remove(p); last_close=ts
        if ts<=last_close: continue
        if len(positions)>=MAX_OPEN_POSITIONS: continue
        # one best candidate per family/asset, rank by cross-sectional strength magnitude
        for s in sorted(byts.get(ts,[]),key=lambda z:abs(z["atr"]),reverse=True):
            if len(positions)>=MAX_OPEN_POSITIONS: break
            if any(p["asset"]==s["asset"] for p in positions): continue
            x=f[s["asset"]]; future=x.index[x.index>ts]
            if len(future)==0: continue
            ets=future[0]; entry=float(x.loc[ets,"open"])*(1+SLIPPAGE if s["side"]=="LONG" else 1-SLIPPAGE)
            risk=1.5*s["atr"]; sl=entry-risk if s["side"]=="LONG" else entry+risk; tp=entry+RR*risk if s["side"]=="LONG" else entry-RR*risk
            positions.append({"asset":s["asset"],"family":s["family"],"side":s["side"],"signal_ts":ts,"entry_ts":ets,"entry":entry,"sl":sl,"tp":tp})
    # mark unresolved as OPEN; no forced loss
    return trades,positions

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--days",type=int,default=DAYS); ap.add_argument("--data-dir",default=str(DATA_DIR)); args=ap.parse_args()
    print("="*88); print("HUNTER-V144 — MTF FACTOR RESEARCH"); print("="*88)
    all15=ensure_data(Path(args.data_dir),args.days); f=add_cross_sectional(features(all15)); print("\nRunning frozen factor families...")
    sig=signals(f); trades,openp=backtest(f,sig)
    for fam in ["A_MOMENTUM","B_REVERSAL","C_REGIME_SWITCH"]:
        t=[z for z in trades if z["family"]==fam]; wins=sum(z["outcome"]=="WIN" for z in t); losses=len(t)-wins
        grossw=sum(z["pnl"] for z in t if z["pnl"]>0); grossl=-sum(z["pnl"] for z in t if z["pnl"]<0)
        eq=INITIAL_EQUITY; peak=eq; dd=0; streak=mx=0
        for z in sorted(t,key=lambda q:q["exit_ts"]):
            eq+=z["pnl"]; peak=max(peak,eq); dd=min(dd,eq-peak); streak=streak+1 if z["outcome"]=="LOSS" else 0; mx=max(mx,streak)
        print(f"{fam}: Trades={len(t)} Open={sum(p['family']==fam for p in openp)} WR={(wins/len(t)*100 if t else 0):.2f}% PF={(grossw/grossl if grossl else 0):.3f} NetPnL=${sum(z['pnl'] for z in t):,.2f} DD=${dd:,.2f} MaxLossStreak={mx}")
    print(f"Total signals={len(sig)} Realized={len(trades)} Open={len(openp)}")
if __name__=="__main__": main()
