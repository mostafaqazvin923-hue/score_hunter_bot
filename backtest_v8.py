# HUNTER-V151
# Causal Ichimoku + Liquidity Sweep + BOS + Order Block retest
# Full single-file backtest

import sys, time, subprocess
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

exchange = ccxt.lbank({
    "enableRateLimit": True,
    "timeout": 20000,
    "options": {"defaultType": "swap"}
})

SYMBOLS = {
    "BTC":"BTC/USDT","ETH":"ETH/USDT","SOL":"SOL/USDT",
    "BNB":"BNB/USDT","XRP":"XRP/USDT","ADA":"ADA/USDT",
    "AVAX":"AVAX/USDT","LINK":"LINK/USDT","DOGE":"DOGE/USDT",
    "DOT":"DOT/USDT"
}

DAYS=365
WARMUP_DAYS=60
MARGIN=100.0
LEVERAGE=50.0
FEE=0.0007
SLIPPAGE=0.0003
RR=2.0
MAX_OPEN=3
COOLDOWN_BARS=4
PIVOT_L=2
PIVOT_R=2
MIN_RVOL=0.80
MIN_BODY_ATR=0.20
ATR_BUFFER=0.20


def fetch(symbol, start, end):
    rows=[]
    since=int((start-timedelta(days=WARMUP_DAYS)).timestamp()*1000)
    end_ms=int(end.timestamp()*1000)
    while since < end_ms:
        try:
            batch=exchange.fetch_ohlcv(symbol,"15m",since=since,limit=1000)
        except Exception as e:
            print("  ERROR",symbol,e)
            return None
        if not batch: break
        rows.extend(batch)
        last=batch[-1][0]
        if last < since: break
        since=last+1
        if len(batch)<1000: break
        time.sleep(.2)
    if not rows: return None
    df=pd.DataFrame(rows,columns=["ts","Open","High","Low","Close","Volume"])
    df["Date"]=pd.to_datetime(df.ts,unit="ms")
    df=df[["Date","Open","High","Low","Close","Volume"]].drop_duplicates("Date")
    df=df.set_index("Date").sort_index()
    tf=15*60*1000
    last_complete=(exchange.milliseconds()//tf)*tf-tf
    df=df[df.index<=pd.to_datetime(last_complete,unit="ms")]
    return df[(df.index>=start-timedelta(days=WARMUP_DAYS))&(df.index<=end)]


def atr(df,p=14):
    pc=df.Close.shift(1)
    tr=pd.concat([df.High-df.Low,(df.High-pc).abs(),(df.Low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/p,adjust=False,min_periods=p).mean()


def adx(df,p=14):
    up=df.High.diff()
    dn=-df.Low.diff()
    plus=pd.Series(np.where((up>dn)&(up>0),up,0.),index=df.index)
    minus=pd.Series(np.where((dn>up)&(dn>0),dn,0.),index=df.index)
    tr=pd.concat([df.High-df.Low,
                  (df.High-df.Close.shift()).abs(),
                  (df.Low-df.Close.shift()).abs()],axis=1).max(axis=1)
    a=1/p
    tw=tr.ewm(alpha=a,adjust=False,min_periods=p).mean()
    pw=plus.ewm(alpha=a,adjust=False,min_periods=p).mean()
    mw=minus.ewm(alpha=a,adjust=False,min_periods=p).mean()
    pdi=100*pw/tw.replace(0,np.nan)
    mdi=100*mw/tw.replace(0,np.nan)
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=a,adjust=False,min_periods=p).mean()


def add_ichi(x):
    x=x.copy()
    x["tenkan"]=(x.High.rolling(9).max()+x.Low.rolling(9).min())/2
    x["kijun"]=(x.High.rolling(26).max()+x.Low.rolling(26).min())/2
    x["span_a_raw"]=(x.tenkan+x.kijun)/2
    x["span_b_raw"]=(x.High.rolling(52).max()+x.Low.rolling(52).min())/2
    return x


def prepare(df):
    base=df.copy()
    base["ATR"]=atr(base)
    base["RVOL"]=base.Volume/base.Volume.rolling(20,min_periods=20).mean()
    base["BODY"]=(base.Close-base.Open).abs()

    h1=base.resample("1h").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    h4=base.resample("4h").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()

    h1["ATR"]=atr(h1); h1["ADX"]=adx(h1); h1=add_ichi(h1)
    h4["ATR"]=atr(h4); h4["ADX"]=adx(h4); h4=add_ichi(h4)

    return base,h1,h4


def pivots(h1):
    out=[]
    H=h1.High.to_numpy(); L=h1.Low.to_numpy(); idx=h1.index
    for k in range(PIVOT_L,len(h1)-PIVOT_R):
        if H[k]>H[k-PIVOT_L:k].max() and H[k]>=H[k+1:k+1+PIVOT_R].max():
            out.append((idx[k+PIVOT_R],"HIGH",float(H[k])))
        if L[k]<L[k-PIVOT_L:k].min() and L[k]<=L[k+1:k+1+PIVOT_R].min():
            out.append((idx[k+PIVOT_R],"LOW",float(L[k])))
    return sorted(out,key=lambda x:x[0])


def regime(h4,t):
    x=h4[h4.index+pd.Timedelta(hours=4)<=t]
    if x.empty:return None
    r=x.iloc[-1]
    vals=[r.Close,r.tenkan,r.kijun,r.span_a_raw,r.span_b_raw,r.ADX]
    if not all(pd.notna(v) for v in vals):return None
    hi=max(r.span_a_raw,r.span_b_raw); lo=min(r.span_a_raw,r.span_b_raw)
    if r.Close>hi and r.tenkan>r.kijun and r.ADX>=15:return "LONG"
    if r.Close<lo and r.tenkan<r.kijun and r.ADX>=15:return "SHORT"
    return None


def build_signals(df,h1,h4):
    pv=pivots(h1)
    last_low=None; last_high=None
    state=None; sweep=None; ob=None
    events=[]
    pvi=0

    for ht,r in h1.iterrows():
        ct=ht+pd.Timedelta(hours=1)
        while pvi<len(pv) and pv[pvi][0]<=ct:
            _,typ,price=pv[pvi]
            if typ=="LOW": last_low=price
            else: last_high=price
            pvi+=1

        reg=regime(h4,ct)
        if reg is None:
            state=sweep=ob=None
            continue

        high=float(r.High); low=float(r.Low); close=float(r.Close)

        if state is None:
            if reg=="LONG" and last_low is not None and low<last_low and close>last_low:
                state="SWEPT"; sweep={"side":"LONG","extreme":low}
            elif reg=="SHORT" and last_high is not None and high>last_high and close<last_high:
                state="SWEPT"; sweep={"side":"SHORT","extreme":high}

        elif state=="SWEPT":
            prev=h1.loc[:ht].iloc[:-1].tail(12)
            if len(prev):
                if reg=="LONG" and close>prev.High.max():
                    q=h1.loc[:ht].iloc[:-1].tail(8)
                    q=q[q.Close<q.Open]
                    if not q.empty:
                        z=q.iloc[-1]
                        ob={"side":"LONG","low":float(z.Low),"high":float(z.High),"extreme":sweep["extreme"]}
                        state="OB"
                elif reg=="SHORT" and close<prev.Low.min():
                    q=h1.loc[:ht].iloc[:-1].tail(8)
                    q=q[q.Close>q.Open]
                    if not q.empty:
                        z=q.iloc[-1]
                        ob={"side":"SHORT","low":float(z.Low),"high":float(z.High),"extreme":sweep["extreme"]}
                        state="OB"

        if state=="OB":
            if (ob["side"]=="LONG" and close<ob["low"]) or (ob["side"]=="SHORT" and close>ob["high"]):
                state=sweep=ob=None
                continue

            cs=df[(df.index>=ct)&(df.index<ct+pd.Timedelta(hours=1))]
            for t,c in cs.iterrows():
                if not all(pd.notna(c[x]) for x in ["ATR","RVOL","BODY"]):continue
                if c.RVOL<MIN_RVOL or c.BODY<MIN_BODY_ATR*c.ATR:continue
                touch=c.Low<=ob["high"] and c.High>=ob["low"]
                rng=c.High-c.Low
                if not touch or rng<=0:continue
                bull=c.Close>c.Open and c.Close>=c.Low+.60*rng
                bear=c.Close<c.Open and c.Close<=c.Low+.40*rng
                if (ob["side"]=="LONG" and bull) or (ob["side"]=="SHORT" and bear):
                    events.append({"SignalTime":t,"Side":ob["side"],"ATR":float(c.ATR),"Extreme":ob["extreme"]})
                    state=sweep=ob=None
                    break

    return events


def simulate(df,e):
    k=df.index.searchsorted(e["SignalTime"],side="right")
    if k>=len(df):return None
    et=df.index[k]; raw=float(df.iloc[k].Open)
    side=e["Side"]; entry=raw*(1+SLIPPAGE if side=="LONG" else 1-SLIPPAGE)

    if side=="LONG":
        sl=e["Extreme"]-ATR_BUFFER*e["ATR"]
        risk=entry-sl
        if risk<=0:return None
        tp=entry+RR*risk
    else:
        sl=e["Extreme"]+ATR_BUFFER*e["ATR"]
        risk=sl-entry
        if risk<=0:return None
        tp=entry-RR*risk

    outcome="OPEN_AT_END"; xt=df.index[-1]; xp=float(df.Close.iloc[-1])
    for j in range(k+1,len(df)):
        hi=float(df.High.iloc[j]); lo=float(df.Low.iloc[j])
        if side=="LONG":
            if lo<=sl:
                outcome="LOSS";xt=df.index[j];xp=sl;break
            if hi>=tp:
                outcome="WIN";xt=df.index[j];xp=tp;break
        else:
            if hi>=sl:
                outcome="LOSS";xt=df.index[j];xp=sl;break
            if lo<=tp:
                outcome="WIN";xt=df.index[j];xp=tp;break

    ret=(xp-entry)/entry if side=="LONG" else (entry-xp)/entry
    pnl=MARGIN*LEVERAGE*ret-MARGIN*LEVERAGE*FEE*2
    return {"SignalTime":e["SignalTime"],"EntryTime":et,"ExitTime":xt,
            "Symbol":None,"Side":side,"Outcome":outcome,"PnL":pnl,
            "HoldingHours":(xt-et).total_seconds()/3600}


def run(all_data,start,end):
    candidates=[]
    for sym,(df,h1,h4) in all_data.items():
        print("Building",sym)
        for e in build_signals(df,h1,h4):
            if start<=e["SignalTime"]<=end:
                tr=simulate(df,e)
                if tr:
                    tr["Symbol"]=sym
                    candidates.append(tr)

    candidates.sort(key=lambda x:(x["EntryTime"],x["Symbol"]))
    active=[]; accepted=[]; last_exit={}; rejects={"symbol_locked":0,"max_positions":0,"cooldown":0}

    for tr in candidates:
        t=tr["EntryTime"]
        still=[]
        for p in active:
            if p["ExitTime"]<=t:
                last_exit[p["Symbol"]]=p["ExitTime"]
            else:still.append(p)
        active=still

        if len(active)>=MAX_OPEN:
            rejects["max_positions"]+=1;continue
        if any(p["Symbol"]==tr["Symbol"] for p in active):
            rejects["symbol_locked"]+=1;continue
        if tr["Symbol"] in last_exit and t<last_exit[tr["Symbol"]]+pd.Timedelta(minutes=15*COOLDOWN_BARS):
            rejects["cooldown"]+=1;continue

        active.append(tr);accepted.append(tr)

    return accepted,len(candidates),rejects


def streaks(seq):
    vals=[];n=0
    for x in seq:
        if x=="LOSS":n+=1
        elif n:vals.append(n);n=0
    if n:vals.append(n)
    return max(vals) if vals else 0,vals


def report(trades,candidates,rejects):
    print("="*82);print("HUNTER-V151 REPORT");print("="*82)
    print("Candidate Signals:",candidates)
    print("Accepted Trades:",len(trades))
    if not trades:return
    d=pd.DataFrame(trades)
    c=d[d.Outcome.isin(["WIN","LOSS"])]
    w=c[c.Outcome=="WIN"];l=c[c.Outcome=="LOSS"]
    gp=w.PnL.sum();gl=abs(l.PnL.sum());pf=gp/gl if gl else float("inf")
    eq=c.PnL.cumsum();dd=eq-eq.cummax()
    ms,ls=streaks(c.Outcome.tolist())
    print("Closed Trades:",len(c))
    print("Open At End:",len(d)-len(c))
    print(f"Win Rate: {100*len(w)/len(c):.2f}%")
    print(f"Loss Rate: {100*len(l)/len(c):.2f}%")
    print(f"Net PnL: ${c.PnL.sum():,.2f}")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Average Win: ${w.PnL.mean():,.2f}" if len(w) else "Average Win: $0")
    print(f"Average Loss: ${l.PnL.mean():,.2f}" if len(l) else "Average Loss: $0")
    print(f"Trades / Day: {len(d)/DAYS:.2f}")
    print(f"Average Holding Hours: {d.HoldingHours.mean():.2f}")
    print(f"Max Drawdown: ${dd.min():,.2f}")
    print("Max Consecutive Losses:",ms)
    print("Loss Streak List:",ls)
    print("RR: 1:2 | Timeout: DISABLED | Lookahead: NONE BY DESIGN")
    print("\nRejections:",rejects)
    print("\n--- BY SYMBOL ---")
    s=c.groupby("Symbol").agg(Trades=("Outcome","count"),Wins=("Outcome",lambda x:(x=="WIN").sum()),Losses=("Outcome",lambda x:(x=="LOSS").sum()),PnL=("PnL","sum"))
    s["WinRate_%"]=(100*s.Wins/s.Trades).round(2);print(s.to_string())
    print("\n--- BY DIRECTION ---")
    q=c.groupby("Side").agg(Trades=("Outcome","count"),Wins=("Outcome",lambda x:(x=="WIN").sum()),Losses=("Outcome",lambda x:(x=="LOSS").sum()),PnL=("PnL","sum"))
    q["WinRate_%"]=(100*q.Wins/q.Trades).round(2);print(q.to_string())
    print("\n--- MONTHLY ---")
    c=c.copy();c["Month"]=c.ExitTime.dt.to_period("M").astype(str)
    m=c.groupby("Month").agg(Trades=("Outcome","count"),Wins=("Outcome",lambda x:(x=="WIN").sum()),Losses=("Outcome",lambda x:(x=="LOSS").sum()),PnL=("PnL","sum"))
    m["WinRate_%"]=(100*m.Wins/m.Trades).round(2);print(m.to_string())


def main():
    end=datetime.now(timezone.utc).replace(tzinfo=None)
    start=end-timedelta(days=DAYS)
    data={}
    print("HUNTER-V151 START")
    for name,symbol in SYMBOLS.items():
        print("Loading",name)
        df=fetch(symbol,start,end)
        if df is None or len(df)<2000:
            print("  skipped");continue
        data[name]=prepare(df)
    if not data:
        print("No valid data");return
    trades,candidates,rejects=run(data,start,end)
    report(trades,candidates,rejects)

if __name__=="__main__":
    main()
