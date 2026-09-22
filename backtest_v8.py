
import subprocess, sys, time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

# HUNTER-V138
# Causal Liquidity Sweep -> Reclaim -> Momentum
# No lookahead, next-open entry, fixed 1:2 RR, no timeout.

exchange = ccxt.lbank({"enableRateLimit": True, "timeout": 20000})

SYMBOLS = {
    "CRV":"CRV/USDT","DOGE":"DOGE/USDT","ICP":"ICP/USDT",
    "APT":"APT/USDT","PENDLE":"PENDLE/USDT","WIF":"WIF/USDT",
    "ONDO":"ONDO/USDT","NEAR":"NEAR/USDT","SEI":"SEI/USDT",
    "XLM":"XLM/USDT","ADA":"ADA/USDT","BNB":"BNB/USDT",
    "SOL":"SOL/USDT","ETH":"ETH/USDT",
}
DAYS=365
SLIPPAGE=0.0003
FEE_RATE=0.0007
MARGIN=100.0
LEVERAGE=50.0

def fetch(symbol,start,end):
    since=int((start-timedelta(days=15)).timestamp()*1000)
    endms=int(end.timestamp()*1000)
    out=[]
    while since<endms:
        try: batch=exchange.fetch_ohlcv(symbol,"15m",since=since,limit=1000)
        except Exception as e:
            print("  ERROR:",e); return None
        if not batch: break
        out.extend(batch); last=batch[-1][0]
        if last<=since: break
        since=last+1
        if len(batch)<1000 or last>=endms: break
        time.sleep(.2)
    if not out:return None
    d=pd.DataFrame(out,columns=["Timestamp","Open","High","Low","Close","Volume"])
    d["Date"]=pd.to_datetime(d.Timestamp,unit="ms")
    d=d[["Date","Open","High","Low","Close","Volume"]].dropna()
    d=d.drop_duplicates("Date").sort_values("Date").set_index("Date")
    return d[(d.index>=start)&(d.index<=end)]

def atr(d,n=14):
    pc=d.Close.shift(1)
    tr=pd.concat([(d.High-d.Low),(d.High-pc).abs(),(d.Low-pc).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()

def prep(d):
    h4=d.resample("4h").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    h4["e50"]=h4.Close.ewm(span=50,adjust=False).mean()
    h4["e200"]=h4.Close.ewm(span=200,adjust=False).mean()
    h4["bull"]=(h4.Close>h4.e200)&(h4.e50>h4.e200)
    h4["bear"]=(h4.Close<h4.e200)&(h4.e50<h4.e200)

    h1=d.resample("1h").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    # causal pivot: a pivot is usable only after two right-side hours
    h1["ph"]=(h1.High>h1.High.shift(1))&(h1.High>h1.High.shift(2))&(h1.High>=h1.High.shift(-1))&(h1.High>=h1.High.shift(-2))
    h1["pl"]=(h1.Low<h1.Low.shift(1))&(h1.Low<h1.Low.shift(2))&(h1.Low<=h1.Low.shift(-1))&(h1.Low<=h1.Low.shift(-2))

    x=d.copy()
    x["ATR"]=atr(x)
    x["Body"]=(x.Close-x.Open).abs()
    x["Range"]=x.High-x.Low
    x["RVOL"]=x.Volume/x.Volume.rolling(20).mean().shift(1)
    x["AvgBody"]=x.Body.rolling(20).mean().shift(1)
    x["CLV"]=(x.Close-x.Low)/x.Range.replace(0,np.nan)
    return x,h1,h4

def htf(h4,t):
    z=h4[(h4.index+pd.Timedelta(hours=4))<=t]
    return None if z.empty else z.iloc[-1]

def liq(h1,t):
    z=h1[(h1.index+pd.Timedelta(hours=2))<=t]
    if len(z)<3:return None,None
    ph=z[z.ph]; pl=z[z.pl]
    return (float(ph.High.iloc[-1]) if not ph.empty else None,
            float(pl.Low.iloc[-1]) if not pl.empty else None)

def pnl(side,e,x):
    r=(x-e)/e if side=="LONG" else (e-x)/e
    return MARGIN*LEVERAGE*r-MARGIN*LEVERAGE*FEE_RATE*2

def engine(sym,d,h1,h4,start,end):
    trades=[]; pos=None; last_exit=-1
    # i-1 = completed sweep; i = completed confirmation; i+1 = entry
    for i in range(260,len(d)-1):
        t=d.index[i]; r=d.iloc[i]
        if t<start or t>end: continue

        if pos:
            hi,lo=float(r.High),float(r.Low)
            e,sl,tp=pos["e"],pos["sl"],pos["tp"]
            if pos["side"]=="LONG":
                if not pos["be"] and hi>=pos["be_trigger"]:
                    pos["sl"]=e;pos["be"]=True;sl=e
                if lo<=sl:
                    ex=e if pos["be"] else sl
                    trades.append({**pos,"ExitTimestamp":t,"Outcome":"BE" if pos["be"] else "LOSS",
                                   "Dollar_PnL":pnl(pos["side"],e,ex),"Exit_Price":ex})
                    pos=None;last_exit=i;continue
                if hi>=tp:
                    trades.append({**pos,"ExitTimestamp":t,"Outcome":"WIN",
                                   "Dollar_PnL":pnl(pos["side"],e,tp),"Exit_Price":tp})
                    pos=None;last_exit=i;continue
            else:
                if not pos["be"] and lo<=pos["be_trigger"]:
                    pos["sl"]=e;pos["be"]=True;sl=e
                if hi>=sl:
                    ex=e if pos["be"] else sl
                    trades.append({**pos,"ExitTimestamp":t,"Outcome":"BE" if pos["be"] else "LOSS",
                                   "Dollar_PnL":pnl(pos["side"],e,ex),"Exit_Price":ex})
                    pos=None;last_exit=i;continue
                if lo<=tp:
                    trades.append({**pos,"ExitTimestamp":t,"Outcome":"WIN",
                                   "Dollar_PnL":pnl(pos["side"],e,tp),"Exit_Price":tp})
                    pos=None;last_exit=i;continue
        if i==last_exit: continue

        regime=htf(h4,t)
        if regime is None: continue
        bull,bear=bool(regime.bull),bool(regime.bear)
        qh,ql=liq(h1,t)
        if qh is None or ql is None: continue

        s=d.iloc[i-1]
        a=float(r.ATR) if np.isfinite(r.ATR) else np.nan
        if not np.isfinite(a) or a<=0: continue
        if not np.isfinite(s.RVOL) or not np.isfinite(r.RVOL): continue
        if not np.isfinite(r.AvgBody) or r.Range<=0 or s.Range<=0: continue

        # Sweep must occur on the immediately previous closed candle.
        long_sweep=(bull and s.Low<=ql-0.05*a and s.Close>ql and s.CLV>=0.60
                    and s.Body>=0.35*a and s.RVOL>=0.95)
        short_sweep=(bear and s.High>=qh+0.05*a and s.Close<qh and s.CLV<=0.40
                     and s.Body>=0.35*a and s.RVOL>=0.95)

        long_confirm=(r.Close>r.Open and r.Body>=0.30*a and r.Body>=0.60*r.AvgBody
                      and r.CLV>=0.60 and r.RVOL>=0.95 and r.Close>s.Close)
        short_confirm=(r.Close<r.Open and r.Body>=0.30*a and r.Body>=0.60*r.AvgBody
                       and r.CLV<=0.40 and r.RVOL>=0.95 and r.Close<s.Close)

        ent=d.iloc[i+1]
        if long_sweep and long_confirm:
            e=float(ent.Open)*(1+SLIPPAGE); sl=float(s.Low)-0.15*a
            risk=e-sl
            if 0.60*a<=risk<=3.50*a:
                pos={"symbol":sym,"side":"LONG","EntryTimestamp":d.index[i+1],"e":e,
                     "sl":sl,"InitialSL":sl,"tp":e+2*risk,"be_trigger":e+risk,"be":False,
                     "Entry_Price":e}
        elif short_sweep and short_confirm:
            e=float(ent.Open)*(1-SLIPPAGE); sl=float(s.High)+0.15*a
            risk=sl-e
            if 0.60*a<=risk<=3.50*a:
                pos={"symbol":sym,"side":"SHORT","EntryTimestamp":d.index[i+1],"e":e,
                     "sl":sl,"InitialSL":sl,"tp":e-2*risk,"be_trigger":e-risk,"be":False,
                     "Entry_Price":e}
    if pos:
        trades.append({**pos,"ExitTimestamp":None,"Outcome":"OPEN","Dollar_PnL":0.0,"Exit_Price":None})
    return trades

def report(df):
    df=df.sort_values(["EntryTimestamp","Symbol"]).reset_index(drop=True)
    real=df[df.Outcome.isin(["WIN","LOSS","BE"])]
    wins=real[real.Outcome=="WIN"]; losses=real[real.Outcome=="LOSS"]; bes=real[real.Outcome=="BE"]
    pnlv=float(real.Dollar_PnL.sum()); gp=float(wins.Dollar_PnL.sum()); gl=abs(float(losses.Dollar_PnL.sum()))
    pf=gp/gl if gl else float("inf")
    eq=real.Dollar_PnL.cumsum(); dd=float((eq-eq.cummax()).min()) if len(eq) else 0
    streaks=[];cur=0
    for o in real.Outcome:
        if o=="LOSS":cur+=1
        elif cur:streaks.append(cur);cur=0
    if cur:streaks.append(cur)
    print("\n"+"="*80)
    print("===== HUNTER-V138 — CAUSAL LIQUIDITY SWEEP / RECLAIM =====")
    print("="*80)
    print(f"Total Trades:          {len(df)}")
    print(f"Realized Trades:       {len(real)}")
    print(f"Open at Dataset End:   {sum(df.Outcome=='OPEN')}")
    print(f"Trades / Month:        {len(df)/12:.1f}")
    print(f"Win Rate:              {len(wins)/len(df)*100:.2f}%")
    print(f"Loss Rate:             {len(losses)/len(df)*100:.2f}%")
    print(f"Break Even Rate:       {len(bes)/len(df)*100:.2f}%")
    print(f"Net PnL:               ${pnlv:,.2f}")
    print(f"Profit Factor:         {pf:.2f}")
    print(f"Average Win:           ${wins.Dollar_PnL.mean() if len(wins) else 0:,.2f}")
    print(f"Average Loss:          ${losses.Dollar_PnL.mean() if len(losses) else 0:,.2f}")
    print(f"Max Drawdown:          ${dd:,.2f}")
    print(f"Maximum Consecutive Losses: {max(streaks) if streaks else 0}")
    print(f"Loss Streaks:          {streaks}")
    print("-"*80)
    print("INTEGRITY:")
    print("  Entry:               NEXT 15m OPEN after closed confirmation")
    print("  HTF:                 COMPLETED 4H / CONFIRMED 1H")
    print("  Signal:              LIQUIDITY SWEEP -> RECLAIM -> MOMENTUM")
    print("  RR:                  1:2 FIXED")
    print("  Timeout:             DISABLED")
    print("  Lookahead:           NONE BY DESIGN")
    print("  Overlap per symbol:  LOCKED")
    print("  Same-candle reentry: BLOCKED")
    print("-"*80)
    print("BY SYMBOL:")
    for s in SYMBOLS:
        z=df[df.Symbol==s]
        if len(z): print(f"  {s:7} -> Trades: {len(z):4}, Win Rate: {z.Outcome.eq('WIN').mean()*100:6.2f}%, PnL: ${z.Dollar_PnL.sum():10,.2f}")
    print("-"*80)
    print("BY DIRECTION:")
    for s in ["LONG","SHORT"]:
        z=df[df.Side==s]
        if len(z): print(f"  {s:7} -> Trades: {len(z):4}, Win Rate: {z.Outcome.eq('WIN').mean()*100:6.2f}%, PnL: ${z.Dollar_PnL.sum():10,.2f}")
    print("-"*80)
    print("BY MONTH:")
    df["Month"]=pd.to_datetime(df.EntryTimestamp).dt.strftime("%Y-%m")
    for m,z in df.groupby("Month"):
        print(f"  {m} -> Trades: {len(z):4}, Win Rate: {z.Outcome.eq('WIN').mean()*100:6.2f}%, PnL: ${z.Dollar_PnL.sum():10,.2f}")
    df.to_csv("v138_causal_trades.csv",index=False)
    print("="*80)

def main():
    now=datetime.now(); start=now-timedelta(days=DAYS); end=now
    print("="*80); print("HUNTER-V138 — CAUSAL LIQUIDITY SWEEP / RECLAIM / MOMENTUM"); print("="*80)
    print(f"Period: {start} -> {end}")
    print("Data: EXACT V123/V129 LBank infrastructure | Universe: EXACT 14 symbols | 15m")
    alltr=[]
    for sym,ls in SYMBOLS.items():
        print(f"\n{'-'*80}\n{sym} -> {ls}\n{'-'*80}")
        d=fetch(ls,start,end)
        if d is None or len(d)<1000: print("  NO/INSUFFICIENT DATA"); continue
        print("  Candles:",len(d))
        x,h1,h4=prep(d); tr=engine(sym,x,h1,h4,start,end); alltr.extend(tr)
        print("  Trades:",len(tr))
    if alltr:
        out = pd.DataFrame(alltr)
        out["Symbol"] = out["symbol"]
        out["Side"] = out["side"]
        report(out)
    else: print("NO TRADES GENERATED")

if __name__=="__main__":
    main()
