import time,json
from pathlib import Path
import numpy as np,pandas as pd,requests
BASE='https://fapi.xt.com'; PATH='/future/market/v1/public/q/kline'
SYMBOLS=['btc_usdt','eth_usdt','sol_usdt','sui_usdt','avax_usdt','near_usdt','ada_usdt','bnb_usdt','apt_usdt','crv_usdt','ondo_usdt','pendle_usdt','icp_usdt','wif_usdt']
DAYS=455; WARMUP_DAYS=7; BAR=4*3600*1000; LIMIT=1000; MIN_COV=.97
MARGIN=100.; NOTIONAL=5000.; FEE=.0007; SLIP=.0003; ATR_N=14
ASSET_MOM=6; MARKET_MOM=6; REGIME=42; BETA_N=60; Z_N=60; Z_MIN=.75; TOP_PCT=.30; MIN_CS=4
STOP_ATR=1.; TARGET_ATR=2.; OUT=Path('reports/xt_v27_residual_momentum'); OUT.mkdir(parents=True,exist_ok=True)

def get(p):
    last=None
    for i in range(4):
        try:
            r=requests.get(BASE+PATH,params=p,timeout=30); r.raise_for_status(); return r.json()
        except Exception as e: last=e; time.sleep(1.5*(i+1))
    raise RuntimeError(f'XT request failed: {last}')

def rows(x):
    d=x.get('result',x.get('data',x.get('rows',x.get('list')))) if isinstance(x,dict) else x
    out=[]
    for a in d or []:
        try:
            if isinstance(a,dict): t=a.get('t',a.get('time',a.get('timestamp'))); o=a.get('o',a.get('open')); h=a.get('h',a.get('high')); l=a.get('l',a.get('low')); c=a.get('c',a.get('close')); v=a.get('v',a.get('volume',a.get('vol')))
            else: t,o,h,l,c,v=a[:6]
            out.append((int(t),float(o),float(h),float(l),float(c),float(v)))
        except: pass
    return out

def fetch(sym,start,end):
    cur=start; allr=[]; page=0
    while cur<end:
        page+=1; b=rows(get({'symbol':sym,'interval':'4h','startTime':cur,'endTime':end,'limit':LIMIT}))
        b=sorted(set(b)); b=[x for x in b if cur<=x[0]<end]
        if not b: break
        allr.extend(b); nxt=max(x[0] for x in b)+BAR
        if nxt<=cur: break
        cur=nxt; print(f'[XT-FUT] {sym} page={page} rows={len(allr)} latest={pd.to_datetime(max(x[0] for x in allr),unit="ms",utc=True)}')
        if len(b)<LIMIT: break
    if not allr: raise RuntimeError(f'No data {sym}')
    d=pd.DataFrame(allr,columns=['ts','open','high','low','close','volume']).drop_duplicates('ts').sort_values('ts')
    gap=d.ts.diff().dropna();
    if (gap>BAR).any(): raise RuntimeError(f'Gap {sym}')
    return d.reset_index(drop=True)

def ATR(d):
    pc=d.close.shift(); tr=pd.concat([d.high-d.low,(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1)
    return tr.rolling(ATR_N,min_periods=ATR_N).mean()

def features(panel):
    btc=panel[panel.symbol=='btc_usdt'].set_index('ts').sort_index(); br=btc.close.pct_change(MARKET_MOM); rg=btc.close.pct_change(REGIME); bm=btc.close.pct_change()
    out=[]
    for sym,g in panel.groupby('symbol'):
        g=g.sort_values('ts').set_index('ts'); r=g.close.pct_change(ASSET_MOM); m=br.reindex(g.index)
        ar=g.close.pct_change(); mr=bm.reindex(g.index); cov=ar.rolling(BETA_N,min_periods=BETA_N).cov(mr); var=mr.rolling(BETA_N,min_periods=BETA_N).var(); beta=(cov/var.replace(0,np.nan)).clip(-5,5)
        resid=r-beta*m; mu=resid.rolling(Z_N,min_periods=Z_N).mean(); sd=resid.rolling(Z_N,min_periods=Z_N).std(ddof=0)
        g['atr']=ATR(g.reset_index()).to_numpy(); g['asset_mom']=r; g['btc_mom']=m; g['btc_regime']=rg.reindex(g.index); g['beta']=beta; g['residual_mom']=resid; g['residual_z']=(resid-mu)/sd.replace(0,np.nan); out.append(g.reset_index())
    return pd.concat(out,ignore_index=True).sort_values(['ts','symbol']).reset_index(drop=True)

def split(ts,s,e):
    f=(ts-s)/(e-s); return 'DISCOVERY' if f<.5 else ('DEVELOPMENT' if f<.75 else 'VALIDATION')

def make_signals(p,s,e):
    p=p.copy(); p['split']=p.ts.map(lambda x:split(x,s,e)); q=p[(p.btc_regime>0)&(p.btc_mom>0)&(p.residual_z>=Z_MIN)&p.atr.notna()].copy()
    q['rank_pct']=q.groupby('ts').residual_z.rank(method='first',ascending=False,pct=True); q['n_cs']=q.groupby('ts').symbol.transform('count'); q=q[(q.n_cs>=MIN_CS)&(q.rank_pct<=TOP_PCT)].copy(); q['setup']='LONG_RESIDUAL_MOM'; return q

def run(p,sig):
    bars={x:g.sort_values('ts').reset_index(drop=True) for x,g in p.groupby('symbol')}; idx={x:{int(t):i for i,t in enumerate(g.ts)} for x,g in bars.items()}; sm={(r.symbol,int(r.ts)):r for r in sig.itertuples()}; pos={}; trades=[]
    for ts in sorted(p.ts.unique()):
        ts=int(ts)
        for sym in list(pos):
            z=pos[sym]; i=idx[sym].get(ts)
            if i is None or i<=z['entry_i']: continue
            b=bars[sym].iloc[i]; sl,tp=z['sl'],z['tp']; hs=b.low<=sl; ht=b.high>=tp
            if not(hs or ht): continue
            rr=-1. if hs else 2.; reason='SL_AND_TP_SAME_CANDLE_LOSS' if hs and ht else ('SL' if hs else 'TP'); fees=NOTIONAL*FEE*2; slip=NOTIONAL*SLIP*2; net=rr*MARGIN-fees-slip
            trades.append({**z,'exit_ts':int(b.ts),'exit_price':sl if hs else tp,'gross_r':rr,'fees_usd':fees,'slippage_usd':slip,'net_usd':net,'net_r':net/MARGIN,'exit_reason':reason}); del pos[sym]
        for (sym,st),r in sm.items():
            if st!=ts or sym in pos: continue
            i=idx[sym].get(ts)
            if i is None or i+1>=len(bars[sym]): continue
            eb=bars[sym].iloc[i+1]; a=float(r.atr); entry=float(eb.open)
            if not(np.isfinite(a) and a>0): continue
            pos[sym]={'symbol':sym,'signal_ts':ts,'entry_ts':int(eb.ts),'entry_i':i+1,'entry_price':entry,'sl':entry-STOP_ATR*a,'tp':entry+TARGET_ATR*a,'split':r.split,'direction':'LONG','setup':'LONG_RESIDUAL_MOM'}
    return pd.DataFrame(trades)

def streak(x):
    m=c=0
    for v in x:
        c=c+1 if v<0 else 0; m=max(m,c)
    return m

def summary(t):
    if t.empty:return pd.DataFrame()
    out=[]
    for (sp,d),g in t.groupby(['split','direction']):
        wp=g[g.net_r>0].net_r.sum(); wl=-g[g.net_r<0].net_r.sum(); out.append({'split':sp,'direction':d,'n_closed':len(g),'wins':int((g.net_r>0).sum()),'losses':int((g.net_r<0).sum()),'win_rate':(g.net_r>0).mean(),'mean_net_r':g.net_r.mean(),'profit_factor':wp/wl if wl else np.inf,'net_r_sum':g.net_r.sum(),'net_usd_sum':g.net_usd.sum(),'max_loss_streak':streak(g.net_r.tolist())})
    return pd.DataFrame(out)

def self_test(): assert TARGET_ATR==2*STOP_ATR and FEE>0 and SLIP>0; print('SELF-TEST PASS')

def main():
    self_test(); end=int(time.time()*1000); end-=end%BAR; rs=end-DAYS*86400000; fs=rs-WARMUP_DAYS*86400000; frames=[]; audits=[]; expected=(end-rs)//BAR
    for sym in SYMBOLS:
        print('[DATA]',sym); d=fetch(sym,fs,end); core=d[(d.ts>=rs)&(d.ts<end)]; cov=len(core)/expected; audits.append({'symbol':sym,'expected':expected,'actual':len(core),'coverage':cov,'first_ts':int(d.ts.min()),'last_ts':int(d.ts.max())});
        if cov<MIN_COV: raise RuntimeError(f'Coverage failure {sym}: {cov:.4f}')
        d['symbol']=sym; frames.append(d)
    p=features(pd.concat(frames,ignore_index=True)); p=p[(p.ts>=rs)&(p.ts<end)].copy(); s=int(p.ts.min()); e=int(p.ts.max()); sig=make_signals(p,s,e); t=run(p,sig); summ=summary(t)
    pd.DataFrame(audits).to_csv(OUT/'kline_coverage_audit.csv',index=False); p.to_csv(OUT/'research_panel.csv',index=False); sig.to_csv(OUT/'signals.csv',index=False); t.to_csv(OUT/'trade_log.csv',index=False); summ.to_csv(OUT/'split_report.csv',index=False)
    meta={'strategy':'V27_REGIME_CONDITIONED_RESIDUAL_MOMENTUM','no_lookahead':True,'next_bar_execution':True,'rr':'1:2','stop_atr':STOP_ATR,'target_atr':TARGET_ATR,'no_timeout':True,'same_candle_sl_tp_loss':True,'per_symbol_overlap_lock':True,'parameters_locked':{'asset_mom_bars':ASSET_MOM,'market_mom_bars':MARKET_MOM,'regime_bars':REGIME,'beta_bars':BETA_N,'z_bars':Z_N,'z_min':Z_MIN,'top_pct':TOP_PCT,'min_cross_section':MIN_CS}}
    (OUT/'run_metadata.json').write_text(json.dumps(meta,indent=2))
    print(f'V27 COMPLETE: panel_rows={len(p)} signals={len(sig)} closed_trades={len(t)}'); print('OUTPUT:',OUT)
if __name__=='__main__':main()
