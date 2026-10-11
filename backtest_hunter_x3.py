#!/usr/bin/env python3
"""HUNTER-X3: portfolio-aware research backtest on real Binance UM futures klines.

Hypothesis: volatility compression -> range breakout -> retest/hold -> continuation.
Tests 15m, 1h and 4h using the same 15m source data. Entries are next-bar open.
No lookahead, no same-symbol re-entry on the exit bar, one position per symbol,
portfolio margin capacity enforced, notional capped at $5,000, with position size reduced to a 1% equity risk budget.
Trades whose stop loss plus estimated round-trip fees exceed a conservative $75
budget are skipped, because $100 isolated margin must not be treated as $400 risk.
SL/TP same candle: SL first. Gap through stop: exit at worse bar open, plus slippage.
No timeout, trailing stop or fabricated candles. Funding/liquidation are not fully
simulated; this is research code and not production execution software.
"""
from __future__ import annotations
import io, time, zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests

BASE='https://data.binance.vision/data/futures/um'
SYMBOLS=['BTCUSDT','ETHUSDT','SOLUSDT','SUIUSDT','AVAXUSDT','NEARUSDT','ADAUSDT','BNBUSDT','APTUSDT','CRVUSDT','ONDOUSDT','PENDLEUSDT','ICPUSDT','WIFUSDT']
BASE_TF='15m'; TIMEFRAMES={'15m':'15min','1h':'1h','4h':'4h'}
TEST_DAYS=365; WARMUP_DAYS=90
INITIAL_CAPITAL=1000.0; MARGIN=100.0; LEVERAGE=50.0; NOTIONAL=MARGIN*LEVERAGE
RR=2.0; FEE=0.0007; SLIPPAGE=0.0003
# Do not risk the full isolated margin. Includes a round-trip fee estimate.
RISK_PER_TRADE_PCT=0.01
ATR_N=14; EMA_FAST=50; EMA_REGIME=200
SWING_LOOKBACK=8; ATR_STOP=1.2
MIN_STOP_PCT=0.001; MAX_STOP_PCT=0.025
OUT=Path('hunter_x3_results'); OUT.mkdir(exist_ok=True)
S=requests.Session(); S.headers['User-Agent']='HUNTER-X3-research/1.0'

def now(): return datetime.now(timezone.utc)
def fetch(url,retries=3):
    for k in range(retries):
        try:
            r=S.get(url,timeout=35)
            if r.status_code==404: return None
            r.raise_for_status(); return r.content
        except requests.RequestException:
            if k==retries-1: return None
            time.sleep(1+k)

def parse_zip(blob):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names=[n for n in z.namelist() if n.lower().endswith('.csv')]
        if not names:return pd.DataFrame()
        with z.open(names[0]) as f: d=pd.read_csv(f,header=None)
    if d.empty or d.shape[1]<6:return pd.DataFrame()
    d=d[pd.to_numeric(d.iloc[:,0],errors='coerce').notna()].iloc[:,:6].copy()
    d.columns=['timestamp','open','high','low','close','volume']
    ts=pd.to_numeric(d.timestamp,errors='coerce')
    if ts.dropna().median()>1e14: ts=ts/1000
    d['timestamp']=pd.to_datetime(ts,unit='ms',utc=True,errors='coerce')
    for c in ['open','high','low','close','volume']:d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna().set_index('timestamp').sort_index()
    return d[~d.index.duplicated(keep='last')]

def download(symbol,start,end):
    chunks=[]; month=datetime(start.year,start.month,1,tzinfo=timezone.utc)
    while month<end:
        nxt=(month.replace(day=28)+timedelta(days=4)).replace(day=1); tag=month.strftime('%Y-%m')
        url=f'{BASE}/monthly/klines/{symbol}/{BASE_TF}/{symbol}-{BASE_TF}-{tag}.zip'; blob=fetch(url)
        if blob:
            d=parse_zip(blob)
            if not d.empty:chunks.append(d)
        else:
            day=month
            while day<nxt and day<end:
                if day.date()>=start.date():
                    tagd=day.strftime('%Y-%m-%d'); u=f'{BASE}/daily/klines/{symbol}/{BASE_TF}/{symbol}-{BASE_TF}-{tagd}.zip'; b=fetch(u)
                    if b:
                        d=parse_zip(b)
                        if not d.empty:chunks.append(d)
                day+=timedelta(days=1)
        month=nxt
    if not chunks:return pd.DataFrame()
    d=pd.concat(chunks).sort_index(); d=d[~d.index.duplicated(keep='last')]
    d=d[(d.index>=pd.Timestamp(start))&(d.index<pd.Timestamp(end))]
    # 15m candle must have closed by now.
    return d[d.index+pd.Timedelta(minutes=15)<=pd.Timestamp(now())]

def resample(d,rule):
    n={'15min':1,'1h':4,'4h':16}[rule]
    x=d.resample(rule,origin='epoch',label='left',closed='left').agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum'),n=('close','count'))
    return x[x.n==n].drop(columns='n').dropna()

def indicators(d):
    x=d.copy(); prev=x.close.shift(1)
    tr=pd.concat([x.high-x.low,(x.high-prev).abs(),(x.low-prev).abs()],axis=1).max(axis=1)
    x['atr']=tr.ewm(alpha=1/ATR_N,adjust=False,min_periods=ATR_N).mean()
    x['ema_fast']=x.close.ewm(span=EMA_FAST,adjust=False,min_periods=EMA_FAST).mean()
    x['ema_regime']=x.close.ewm(span=EMA_REGIME,adjust=False,min_periods=EMA_REGIME).mean()
    # Breakout levels only use candles that had closed before the breakout candle.
    x['range_high']=x.high.shift(1).rolling(20,min_periods=20).max()
    x['range_low']=x.low.shift(1).rolling(20,min_periods=20).min()
    x['swing_low']=x.low.shift(1).rolling(SWING_LOOKBACK,min_periods=SWING_LOOKBACK).min()
    x['swing_high']=x.high.shift(1).rolling(SWING_LOOKBACK,min_periods=SWING_LOOKBACK).max()
    x['atr_pct']=x.atr/x.close
    x['atr_pct_q40']=x.atr_pct.rolling(100,min_periods=60).quantile(0.40)
    x['vol_median']=x.volume.rolling(20,min_periods=20).median()
    x['compression_prior']=(x.atr_pct.shift(2) <= x.atr_pct_q40.shift(2))
    x['prev2_vol_median']=x.vol_median.shift(2)
    x['clv']=(x.close-x.low)/(x.high-x.low).replace(0,np.nan)
    x['ema_slope']=x.ema_regime-x.ema_regime.shift(10)
    for col in ['close','open','high','low','atr','range_high','range_low','volume','vol_median','atr_pct','atr_pct_q40','ema_fast','ema_regime','ema_slope']:
        x['prev_'+col]=x[col].shift(1)
    return x

def get_signal(r):
    keys=['atr','ema_fast','ema_regime','range_high','range_low','swing_low','swing_high','atr_pct','atr_pct_q40','vol_median','clv','ema_slope']
    if any(pd.isna(r[k]) for k in keys): return None
    if not 0.001 <= r.atr_pct <= 0.04: return None
    # Compression is evaluated before the breakout candle; no current/future data is used.
    if pd.isna(r.atr_pct_q40): return None
    # Previous candle must break the prior 20-bar boundary; current candle retests and holds it.
    # `prev_*` fields are prepared in main indicator frame from the prior completed bar.
    required=['prev_close','prev_open','prev_high','prev_low','prev_atr','prev_range_high','prev_range_low','prev_volume','prev_vol_median','prev_atr_pct','prev_atr_pct_q40','prev_ema_fast','prev_ema_regime','prev_ema_slope']
    if any(pd.isna(r[k]) for k in required): return None
    compressed = bool(r.compression_prior)
    if not compressed: return None
    volume_ok = r.prev_volume >= 1.2*r.prev2_vol_median
    if not volume_ok: return None
    bull_regime = r.close > r.ema_regime and r.ema_fast > r.ema_regime and r.ema_slope > 0
    bear_regime = r.close < r.ema_regime and r.ema_fast < r.ema_regime and r.ema_slope < 0
    long_level=float(r.prev_range_high); short_level=float(r.prev_range_low)
    if bull_regime and r.prev_close > long_level + 0.15*r.prev_atr and r.low <= long_level + 0.25*r.atr and r.close > long_level and r.close > r.open and r.clv >= 0.60:
        return 'LONG'
    if bear_regime and r.prev_close < short_level - 0.15*r.prev_atr and r.high >= short_level - 0.25*r.atr and r.close < short_level and r.close < r.open and r.clv <= 0.40:
        return 'SHORT'
    return None

def slip_entry(o,side):return o*(1+SLIPPAGE if side=='LONG' else 1-SLIPPAGE)
def slip_exit(px,side):return px*(1-SLIPPAGE if side=='LONG' else 1+SLIPPAGE)

def simulate(tf,frames,tstart,tend):
    # Global event loop: all symbols share a timeline and the same collateral pool.
    all_times=sorted(set(t for d in frames.values() for t in d.index if t>=tstart and t<tend))
    if not all_times:return [],{'skipped_margin':0,'skipped_stop_budget':0,'skipped_invalid':0}
    ixmaps={s:{t:i for i,t in enumerate(d.index)} for s,d in frames.items()}
    positions={}; trades=[]; balance=INITIAL_CAPITAL; stats={'skipped_margin':0,'skipped_stop_budget':0,'skipped_invalid':0}
    # Signals are evaluated on the just-closed candle; executed at this timestamp's open.
    for ts in all_times:
        closed_this_bar=set()
        # Existing positions: process this candle's OHLC for stop/target. On gaps, use open if worse.
        for sym in list(positions):
            pos=positions[sym]; d=frames[sym]; j=ixmaps[sym].get(ts)
            if j is None or ts<=pos['entry_time']:continue
            bar=d.iloc[j]; side=pos['side']; stop=pos['stop']; target=pos['target']
            if side=='LONG':
                sl_hit=bar.low<=stop; tp_hit=bar.high>=target
                if sl_hit:
                    raw=min(stop,float(bar.open)) if bar.open<stop else stop; reason='SL_BOTH' if tp_hit else 'SL'
                elif tp_hit:raw=max(target,float(bar.open)) if bar.open>target else target; reason='TP'
                else:continue
            else:
                sl_hit=bar.high>=stop; tp_hit=bar.low<=target
                if sl_hit:
                    raw=max(stop,float(bar.open)) if bar.open>stop else stop; reason='SL_BOTH' if tp_hit else 'SL'
                elif tp_hit:raw=min(target,float(bar.open)) if bar.open<target else target; reason='TP'
                else:continue
            xp=slip_exit(float(raw),side); direction=1 if side=='LONG' else -1
            gross=direction*(xp-pos['entry'])*pos['qty']; fees=FEE*pos['qty']*(pos['entry']+xp); net=gross-fees
            balance+=net
            trades.append({**{k:pos[k] for k in ['timeframe','symbol','side','signal_time','entry_time','entry','stop','target','qty','risk_budget_est']},'exit_time':ts.isoformat(),'exit_price':xp,'exit_reason':reason,'gross_pnl':gross,'fees':fees,'net_pnl':net,'bars_held':j-pos['entry_index']+1,'balance_after_close':balance})
            del positions[sym]
            closed_this_bar.add(sym)
        # Candidate signals from prior closed candle for each symbol at this bar's open.
        candidates=[]
        for sym,d in frames.items():
            j=ixmaps[sym].get(ts)
            if j is None or sym in positions or sym in closed_this_bar or j<1:continue
            prev=d.iloc[j-1]
            if d.index[j-1]>=tend or d.index[j-1]<tstart:continue
            side=get_signal(prev)
            if side:candidates.append((sym,d,j,prev,side))
        # Deterministic ordering avoids look-ahead; symbols sorted alphabetically for simultaneous signals.
        for sym,d,j,r,side in sorted(candidates,key=lambda z:z[0]):
            if sym in positions:continue
            raw_open=float(d.open.iloc[j]); entry=slip_entry(raw_open,side); atr=float(r.atr)
            if side=='LONG':
                stop=min(float(r.swing_low),float(r.low),entry-ATR_STOP*atr); risk=entry-stop; target=entry+RR*risk
            else:
                stop=max(float(r.swing_high),float(r.high),entry+ATR_STOP*atr); risk=stop-entry; target=entry-RR*risk
            if risk<=0 or not MIN_STOP_PCT<=risk/entry<=MAX_STOP_PCT:
                stats['skipped_invalid']+=1;continue
            risk_cash=max(1.0, balance*RISK_PER_TRADE_PCT)
            cost_per_unit=risk + FEE*(entry+stop) + 2*SLIPPAGE*entry
            qty=min(NOTIONAL/entry, risk_cash/max(cost_per_unit, 1e-12))
            estimated=qty*cost_per_unit
            margin_used=qty*entry/LEVERAGE
            used_margin=sum(float(p['margin_used']) for p in positions.values())
            reserved_risk=sum(float(p['risk_budget_est']) for p in positions.values())
            if used_margin+margin_used > min(balance, len(SYMBOLS)*MARGIN) or balance-used_margin-reserved_risk < margin_used+estimated:
                stats['skipped_margin']+=1;continue
            positions[sym]={'timeframe':tf,'symbol':sym,'side':side,'signal_time':d.index[j-1].isoformat(),'entry_time':ts,'entry_index':j,'entry':entry,'stop':stop,'target':target,'qty':qty,'risk_budget_est':estimated,'margin_used':margin_used}
            # Entry occurs at this candle's open, so this candle's high/low must be checked.
            # Never postpone first-bar SL/TP evaluation until the next candle.
            bar=d.iloc[j]
            if side=='LONG':
                sl_hit=bar.low<=stop; tp_hit=bar.high>=target
                if sl_hit:
                    raw=min(stop,float(bar.open)) if bar.open<stop else stop; reason='SL_BOTH' if tp_hit else 'SL'
                elif tp_hit:
                    raw=max(target,float(bar.open)) if bar.open>target else target; reason='TP'
                else: raw=None
            else:
                sl_hit=bar.high>=stop; tp_hit=bar.low<=target
                if sl_hit:
                    raw=max(stop,float(bar.open)) if bar.open>stop else stop; reason='SL_BOTH' if tp_hit else 'SL'
                elif tp_hit:
                    raw=min(target,float(bar.open)) if bar.open<target else target; reason='TP'
                else: raw=None
            if raw is not None:
                xp=slip_exit(float(raw),side); direction=1 if side=='LONG' else -1
                gross=direction*(xp-entry)*qty; fees=FEE*qty*(entry+xp); net=gross-fees; balance+=net
                pos=positions.pop(sym)
                closed_this_bar.add(sym)
                trades.append({**{k:pos[k] for k in ['timeframe','symbol','side','signal_time','entry_time','entry','stop','target','qty','risk_budget_est']},'exit_time':ts.isoformat(),'exit_price':xp,'exit_reason':reason,'gross_pnl':gross,'fees':fees,'net_pnl':net,'bars_held':1,'balance_after_close':balance})
    # Mark remaining open positions at final available close, explicitly; apply exit slip and fees.
    for sym,pos in list(positions.items()):
        d=frames[sym]; eligible=d[(d.index>=pos['entry_time'])&(d.index<tend)]
        if eligible.empty:continue
        ts=eligible.index[-1]; raw=float(eligible.close.iloc[-1]); xp=slip_exit(raw,pos['side']); direction=1 if pos['side']=='LONG' else -1
        gross=direction*(xp-pos['entry'])*pos['qty']; fees=FEE*pos['qty']*(pos['entry']+xp); net=gross-fees; balance+=net
        trades.append({**{k:pos[k] for k in ['timeframe','symbol','side','signal_time','entry_time','entry','stop','target','qty','risk_budget_est']},'exit_time':ts.isoformat(),'exit_price':xp,'exit_reason':'DATA_END_MARK','gross_pnl':gross,'fees':fees,'net_pnl':net,'bars_held':len(eligible),'balance_after_close':balance})
    return trades,stats

def summarize(tf,trades,stats):
    if not trades:return {'timeframe':tf,'trades':0,'wins':0,'losses':0,'win_rate_pct':0,'net_pnl':0,'final_equity':INITIAL_CAPITAL,'profit_factor':0,'max_drawdown_pct':0,'max_loss_streak':0,**stats}
    d=pd.DataFrame(trades).sort_values(['exit_time','symbol']); p=d.net_pnl; wins=int((p>0).sum()); losses=int((p<=0).sum()); gp=float(p[p>0].sum()); gl=float(-p[p<0].sum())
    eq=INITIAL_CAPITAL+p.cumsum(); peak=eq.cummax().clip(lower=INITIAL_CAPITAL); dd=(peak-eq)/peak.replace(0,np.nan)
    streak=mx=0
    for v in p:
        if v<=0:streak+=1;mx=max(mx,streak)
        else:streak=0
    return {'timeframe':tf,'trades':len(d),'wins':wins,'losses':losses,'win_rate_pct':100*wins/len(d),'net_pnl':float(p.sum()),'final_equity':float(INITIAL_CAPITAL+p.sum()),'profit_factor':gp/gl if gl else float('inf'),'max_drawdown_pct':float(dd.max()*100),'max_loss_streak':mx,'fees_total':float(d.fees.sum()),'data_end_marks':int((d.exit_reason=='DATA_END_MARK').sum()),**stats}

def main():
    end=now().replace(minute=0,second=0,microsecond=0); start=end-timedelta(days=TEST_DAYS+WARMUP_DAYS); tstart=pd.Timestamp(end-timedelta(days=TEST_DAYS)); tend=pd.Timestamp(end)
    print(f'HUNTER-X3 test={tstart} -> {tend}; capital=${INITIAL_CAPITAL}; max_margin/trade=${MARGIN}; max_leverage={LEVERAGE}x; max_notional=${NOTIONAL}; risk/trade={RISK_PER_TRADE_PCT:.1%} equity; RR=1:2')
    base={}
    for n,sym in enumerate(SYMBOLS,1):
        d=download(sym,start,end)
        if d.empty:print(f'[{n}/{len(SYMBOLS)}] {sym}: NO DATA');continue
        base[sym]=d;print(f'[{n}/{len(SYMBOLS)}] {sym}: {len(d):,} 15m bars')
    if not base:raise RuntimeError('No real Binance archive data downloaded; no synthetic fallback.')
    summaries=[]
    for tf,rule in TIMEFRAMES.items():
        frames={s:resample(d,rule) for s,d in base.items()}; frames={s:indicators(d) for s,d in frames.items() if len(d)>EMA_REGIME+50}
        print(f'\nRunning {tf} across {len(frames)} symbols')
        trades,stats=simulate(tf,frames,tstart,tend)
        pd.DataFrame(trades).to_csv(OUT/f'trades_{tf}.csv',index=False)
        sm=summarize(tf,trades,stats);summaries.append(sm);print(pd.Series(sm).to_string())
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
    print(f'\nSaved results: {OUT.resolve()}')
    print('Limits: funding and exchange-specific liquidation/maintenance-margin rules are not simulated; do not use for live trading.')
if __name__=='__main__':main()
