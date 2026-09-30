import time,json
from pathlib import Path
import numpy as np,pandas as pd,requests

BASE='https://fapi.xt.com'
PATH='/future/market/v1/public/q/kline'
SYMBOLS=['btc_usdt','eth_usdt','sol_usdt','sui_usdt','avax_usdt','near_usdt','ada_usdt','bnb_usdt','apt_usdt','crv_usdt','ondo_usdt','pendle_usdt','icp_usdt','wif_usdt']
LEADERS=['btc_usdt','eth_usdt','sol_usdt']
TARGETS=[s for s in SYMBOLS if s not in LEADERS]
DAYS=455
WARMUP_DAYS=30
BAR=3600*1000
INTERVAL='1h'
LIMIT=1000
MIN_COV=.97
MARGIN=100.0
NOTIONAL=5000.0
FEE=.0007
SLIP=.0003
LEADER_Z_N=168
LEADER_Z_MIN=1.0
TARGET_MIN_ABS_RET_Z=0.0
STOP_ATR=1.0
TARGET_ATR=2.0
ATR_N=14
OUT=Path('reports/xt_v28_lead_lag')
OUT.mkdir(parents=True,exist_ok=True)


def get(params):
    last=None
    for i in range(4):
        try:
            r=requests.get(BASE+PATH,params=params,timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last=e
            time.sleep(1.5*(i+1))
    raise RuntimeError(f'XT request failed: {last}')


def rows(x):
    d=x.get('result',x.get('data',x.get('rows',x.get('list')))) if isinstance(x,dict) else x
    out=[]
    for a in d or []:
        try:
            if isinstance(a,dict):
                t=a.get('t',a.get('time',a.get('timestamp')))
                o=a.get('o',a.get('open')); h=a.get('h',a.get('high')); l=a.get('l',a.get('low')); c=a.get('c',a.get('close')); v=a.get('v',a.get('volume',a.get('vol')))
            else:
                t,o,h,l,c,v=a[:6]
            out.append((int(t),float(o),float(h),float(l),float(c),float(v)))
        except Exception:
            pass
    return out


def fetch(sym,start,end):
    window_bars=LIMIT-1
    window_ms=window_bars*BAR
    cur=start; allr=[]; page=0
    while cur<end:
        page+=1
        win_end=min(cur+window_ms,end)
        b=rows(get({'symbol':sym,'interval':INTERVAL,'startTime':cur,'endTime':win_end,'limit':LIMIT}))
        b=sorted(set(b)); b=[x for x in b if cur<=x[0]<win_end]
        expected=int(np.ceil((win_end-cur)/BAR))
        print(f'[XT-FUT] {sym} page={page} window={pd.to_datetime(cur,unit="ms",utc=True)} -> {pd.to_datetime(win_end,unit="ms",utc=True)} rows={len(b)}/{expected} total={len(allr)+len(b)}')
        if not b:
            raise RuntimeError(f'XT returned no candles for {sym}: {cur}->{win_end}')
        if len(b)<expected:
            raise RuntimeError(f'XT window coverage failure {sym}: got={len(b)} expected={expected} page={page}')
        allr.extend(b)
        cur=win_end
        time.sleep(.10)
    d=pd.DataFrame(allr,columns=['ts','open','high','low','close','volume']).drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
    gap=d.ts.diff().dropna()
    if (gap>BAR).any():
        bad=d.loc[gap[gap>BAR].index[0],'ts']
        raise RuntimeError(f'Gap {sym} near {pd.to_datetime(bad,unit="ms",utc=True)}')
    return d


def atr(g):
    pc=g.close.shift()
    tr=pd.concat([(g.high-g.low),(g.high-pc).abs(),(g.low-pc).abs()],axis=1).max(axis=1)
    return tr.rolling(ATR_N,min_periods=ATR_N).mean()


def split(ts,s,e):
    f=(ts-s)/(e-s)
    return 'DISCOVERY' if f<.5 else ('DEVELOPMENT' if f<.75 else 'VALIDATION')


def build_panel(raw):
    frames=[]
    for sym,g in raw.groupby('symbol'):
        g=g.sort_values('ts').copy()
        g['ret1']=g.close.pct_change(1)
        g['atr']=atr(g)
        frames.append(g)
    p=pd.concat(frames,ignore_index=True).sort_values(['ts','symbol']).reset_index(drop=True)
    leader=p[p.symbol.isin(LEADERS)].pivot(index='ts',columns='symbol',values='ret1').sort_index()
    leader_basket=leader.mean(axis=1,min_count=len(LEADERS))
    mu=leader_basket.rolling(LEADER_Z_N,min_periods=LEADER_Z_N).mean()
    sd=leader_basket.rolling(LEADER_Z_N,min_periods=LEADER_Z_N).std(ddof=0)
    leader_z=(leader_basket-mu)/sd.replace(0,np.nan)
    lb=pd.DataFrame({'ts':leader_basket.index,'leader_ret_1h':leader_basket.values,'leader_z':leader_z.values})
    p=p.merge(lb,on='ts',how='left')
    # Negative lead-lag hypothesis: large-coin shock predicts the next-hour
    # move of smaller coins in the opposite direction.
    p=p[p.symbol.isin(TARGETS)].copy()
    p['direction']=np.where(p.leader_z>=LEADER_Z_MIN,'SHORT',np.where(p.leader_z<=-LEADER_Z_MIN,'LONG',''))
    p['setup']=np.where(p.direction!='','LARGE_SMALL_SEESAW','')
    return p


def make_signals(p,s,e):
    q=p[p.direction!=''].copy()
    q['split']=q.ts.map(lambda x:split(x,s,e))
    # Fixed, pre-registered rule. No parameter is selected from the data.
    return q[['ts','symbol','leader_ret_1h','leader_z','atr','direction','setup','split']].copy()


def run(p,sig):
    bars={sym:g.sort_values('ts').reset_index(drop=True) for sym,g in p.groupby('symbol')}
    idx={sym:{int(t):i for i,t in enumerate(g.ts)} for sym,g in bars.items()}
    sm={(r.symbol,int(r.ts)):r for r in sig.itertuples()}
    pos={}; trades=[]
    for ts in sorted(p.ts.unique()):
        ts=int(ts)
        # Exits are processed before new entries on the same timestamp.
        for sym in list(pos):
            z=pos[sym]; i=idx[sym].get(ts)
            if i is None or i<=z['entry_i']:
                continue
            b=bars[sym].iloc[i]
            sl,tp=z['sl'],z['tp']
            if z['direction']=='LONG':
                hs=b.low<=sl; ht=b.high>=tp
                if not(hs or ht): continue
                if hs and ht:
                    rr=-1.; reason='SL_AND_TP_SAME_CANDLE_LOSS'; xp=sl
                elif hs:
                    rr=-1.; reason='SL'; xp=sl
                else:
                    rr=2.; reason='TP'; xp=tp
            else:
                hs=b.high>=sl; ht=b.low<=tp
                if not(hs or ht): continue
                if hs and ht:
                    rr=-1.; reason='SL_AND_TP_SAME_CANDLE_LOSS'; xp=sl
                elif hs:
                    rr=-1.; reason='SL'; xp=sl
                else:
                    rr=2.; reason='TP'; xp=tp
            fees=NOTIONAL*FEE*2
            slip=NOTIONAL*SLIP*2
            net=rr*MARGIN-fees-slip
            trades.append({**z,'exit_ts':int(b.ts),'exit_price':xp,'gross_r':rr,'fees_usd':fees,'slippage_usd':slip,'net_usd':net,'net_r':net/MARGIN,'exit_reason':reason})
            del pos[sym]
        for (sym,st),r in sm.items():
            if st!=ts or sym in pos:
                continue
            i=idx[sym].get(ts)
            if i is None or i+1>=len(bars[sym]):
                continue
            eb=bars[sym].iloc[i+1]
            a=float(r.atr); entry=float(eb.open)
            if not(np.isfinite(a) and a>0 and np.isfinite(entry) and entry>0):
                continue
            if r.direction=='LONG':
                sl=entry-STOP_ATR*a; tp=entry+TARGET_ATR*a
            else:
                sl=entry+STOP_ATR*a; tp=entry-TARGET_ATR*a
            pos[sym]={'symbol':sym,'signal_ts':ts,'entry_ts':int(eb.ts),'entry_i':i+1,'entry_price':entry,'sl':sl,'tp':tp,'split':r.split,'direction':r.direction,'setup':r.setup,'leader_z':float(r.leader_z)}
    return pd.DataFrame(trades)


def streak(x):
    m=c=0
    for v in x:
        c=c+1 if v<0 else 0
        m=max(m,c)
    return m


def summary(t):
    if t.empty:
        return pd.DataFrame()
    out=[]
    for (sp,d),g in t.groupby(['split','direction']):
        wp=g.loc[g.net_r>0,'net_r'].sum(); wl=-g.loc[g.net_r<0,'net_r'].sum()
        out.append({'split':sp,'direction':d,'n_closed':len(g),'wins':int((g.net_r>0).sum()),'losses':int((g.net_r<0).sum()),'win_rate':float((g.net_r>0).mean()),'mean_net_r':float(g.net_r.mean()),'profit_factor':float(wp/wl) if wl else np.inf,'net_r_sum':float(g.net_r.sum()),'net_usd_sum':float(g.net_usd.sum()),'max_loss_streak':streak(g.net_r.tolist())})
    return pd.DataFrame(out)


def self_test():
    assert TARGET_ATR==2*STOP_ATR
    assert set(LEADERS).issubset(set(SYMBOLS))
    assert not(set(LEADERS)&set(TARGETS))
    assert FEE>0 and SLIP>0 and BAR==3600*1000
    print('SELF-TEST PASS')


def main():
    self_test()
    end=int(time.time()*1000); end-=end%BAR
    research_start=end-DAYS*86400000
    fetch_start=research_start-WARMUP_DAYS*86400000
    expected=(end-research_start)//BAR
    frames=[]; audits=[]
    for sym in SYMBOLS:
        print('[DATA]',sym)
        d=fetch(sym,fetch_start,end)
        core=d[(d.ts>=research_start)&(d.ts<end)]
        cov=len(core)/expected
        audits.append({'symbol':sym,'role':'LEADER' if sym in LEADERS else 'TARGET','expected':expected,'actual':len(core),'coverage':cov,'first_ts':int(d.ts.min()),'last_ts':int(d.ts.max())})
        if cov<MIN_COV:
            raise RuntimeError(f'Coverage failure {sym}: {cov:.4f}')
        d['symbol']=sym; frames.append(d)
    raw=pd.concat(frames,ignore_index=True)
    p=build_panel(raw)
    p=p[(p.ts>=research_start)&(p.ts<end)].copy()
    s=int(p.ts.min()); e=int(p.ts.max())
    sig=make_signals(p,s,e)
    t=run(p,sig)
    summ=summary(t)
    pd.DataFrame(audits).to_csv(OUT/'kline_coverage_audit.csv',index=False)
    p.to_csv(OUT/'research_panel.csv',index=False)
    sig.to_csv(OUT/'signals.csv',index=False)
    t.to_csv(OUT/'trade_log.csv',index=False)
    summ.to_csv(OUT/'split_report.csv',index=False)
    meta={'strategy':'V28_LARGE_SMALL_NEGATIVE_LEAD_LAG_SEESAW','research_days':DAYS,'warmup_days':WARMUP_DAYS,'timeframe':'1h','leaders':LEADERS,'targets':TARGETS,'no_lookahead':True,'next_bar_execution':True,'rr':'1:2','stop_atr':STOP_ATR,'target_atr':TARGET_ATR,'no_timeout':True,'same_candle_sl_tp_loss':True,'per_symbol_overlap_lock':True,'parameters_locked':{'leader_z_window_hours':LEADER_Z_N,'leader_z_threshold':LEADER_Z_MIN},'hypothesis':'large-coin one-hour return shock predicts next-hour target-coin return in the opposite direction'}
    (OUT/'run_metadata.json').write_text(json.dumps(meta,indent=2))
    print(f'V28 COMPLETE: panel_rows={len(p)} signals={len(sig)} closed_trades={len(t)}')
    print('OUTPUT:',OUT)

if __name__=='__main__':
    main()
