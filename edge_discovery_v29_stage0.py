import json,time,urllib.parse,urllib.request
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
XT='https://fapi.xt.com'; SYMS=['BTC/USDT','ETH/USDT','SOL/USDT','SUI/USDT','AVAX/USDT','NEAR/USDT','ADA/USDT','BNB/USDT','APT/USDT','CRV/USDT','ONDO/USDT','PENDLE/USDT','ICP/USDT','WIF/USDT']
DAYS=455; WARMUP=30; BAR=4*3600*1000; LIMIT=1000; MIN_COV=.97; ZW=42; MOM=6; H=24; Z=1.0
OUT=Path('reports/xt_v29_stage0_liquidity_momentum'); OUT.mkdir(parents=True,exist_ok=True)
def get(u,p):
    last=None
    for i in range(4):
        try:
            q=urllib.parse.urlencode(p); r=urllib.request.Request(u+'?'+q,headers={'User-Agent':'score-hunter-v29-stage0/1.0'})
            with urllib.request.urlopen(r,timeout=30) as x:j=json.loads(x.read().decode())
            if isinstance(j,dict) and str(j.get('code','0')) not in ('0','200'): raise RuntimeError(j.get('msg','API error'))
            return j
        except Exception as e:last=e; time.sleep(1.5*(i+1))
    raise RuntimeError(f'GET failed: {last}')
def xt(sym,a,b):
    rows=[]; cur=a; prev=None; page=0; W=LIMIT*BAR; s=sym.replace('/','_').lower()
    while cur<b:
        page+=1
        if page>1000: raise RuntimeError('pagination guard')
        final=b-cur<=W; ws=cur; we=b if final else cur+W
        j=get(XT+'/future/market/v1/public/q/kline',{'symbol':s,'interval':'4h','startTime':ws,'endTime':we,'limit':LIMIT})
        d=j
        if isinstance(d,dict):
            d=d.get('result',d.get('data',d.get('rows',d.get('list',[]))))
            if isinstance(d,dict): d=d.get('data',d.get('rows',d.get('list',d.get('items',[]))))
        if not isinstance(d,list): raise RuntimeError(f'bad response {sym}')
        got=[]
        for a1 in d:
            try:
                if isinstance(a1,dict): z=[a1.get('time',a1.get('timestamp',a1.get('ts',a1.get('t')))),a1.get('open',a1.get('o')),a1.get('high',a1.get('h')),a1.get('low',a1.get('l')),a1.get('close',a1.get('c')),a1.get('volume',a1.get('vol',a1.get('v',a1.get('amount',0))))]
                elif isinstance(a1,(list,tuple)) and len(a1)>=6:z=list(a1[:6])
                else:continue
                if any(v is None for v in z[:5]):continue
                ts=int(float(z[0])); ts=ts*1000 if ts<10_000_000_000 else ts
                if ws<=ts<we:got.append([ts,float(z[1]),float(z[2]),float(z[3]),float(z[4]),float(z[5] or 0)])
            except:continue
        if not got:
            if final:break
            raise RuntimeError(f'empty page {sym}')
        mx=max(x[0] for x in got)
        if prev is not None and mx<=prev and not final:raise RuntimeError(f'stalled {sym}')
        rows+=got; prev=max(prev or mx,mx)
        if final:break
        cur=mx+BAR
    df=pd.DataFrame(rows,columns=['ts','open','high','low','close','volume']).drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
    now=int(time.time()*1000)
    if len(df) and df.ts.iloc[-1]+BAR>now:df=df.iloc[:-1].copy()
    if len(df)<int((DAYS+WARMUP)*6*.95):raise RuntimeError(f'insufficient {sym}: {len(df)}')
    return df
def cz(s,n):
    m=s.shift(1).rolling(n,min_periods=n).mean(); sd=s.shift(1).rolling(n,min_periods=n).std(ddof=0); return (s-m)/sd.replace(0,np.nan)
def feat(x):
    x=x.copy(); pc=x.close.shift(1); tr=pd.concat([x.high-x.low,(x.high-pc).abs(),(x.low-pc).abs()],axis=1).max(axis=1); x['atr']=tr.rolling(14,min_periods=14).mean(); x['ret']=x.close.pct_change(); x['mom']=x.close.pct_change(MOM); x['dv']=x.close*x.volume; x['amihud']=x.ret.abs()/x.dv.replace(0,np.nan)*1e6; x['liqz']=cz(np.log(x.dv.replace(0,np.nan)),ZW); x['illiqz']=cz(np.log(x.amihud),ZW); x['rv']=x.ret.rolling(ZW,min_periods=ZW).std(); x['rvz']=cz(np.log(x.rv.replace(0,np.nan)),ZW); return x
def rr(x):
    out=np.full(len(x),np.nan)
    for i in range(len(x)-1):
        a=x.atr.iloc[i]; e=x.open.iloc[i+1]
        if not np.isfinite(a) or a<=0:continue
        for j in range(i+1,min(len(x),i+1+H)):
            sl=x.low.iloc[j]<=e-a; tp=x.high.iloc[j]>=e+2*a
            if sl and tp:out[i]=-1;break
            if sl:out[i]=-1;break
            if tp:out[i]=2;break
    return out
def main():
    end=pd.Timestamp(datetime.now(timezone.utc)).floor('4h'); start=end-pd.Timedelta(days=DAYS); fetch=start-pd.Timedelta(days=WARMUP); a=int(fetch.timestamp()*1000); b=int(end.timestamp()*1000); ps=[]; cov=[]
    for sym in SYMS:
        print('[DATA]',sym); x=feat(xt(sym,a,b)); x=x[(x.ts>=int(start.timestamp()*1000))&(x.ts<b)].copy(); expected=DAYS*6; c=len(x)/expected
        if c<MIN_COV:raise RuntimeError(f'coverage {sym} {c:.4f}')
        x['rr2']=rr(x); x['fwd24']=x.close.shift(-MOM)/x.close-1; x['symbol']=sym; x['dt']=pd.to_datetime(x.ts,unit='ms',utc=True); ps.append(x); cov.append({'symbol':sym,'rows':len(x),'expected':expected,'coverage':c})
    p=pd.concat(ps,ignore_index=True).sort_values(['dt','symbol']); p.to_csv(OUT/'research_panel.csv',index=False); pd.DataFrame(cov).to_csv(OUT/'kline_coverage_audit.csv',index=False)
    times=np.sort(p.dt.unique()); splits={'DISCOVERY_50':(0,.5),'DEVELOPMENT_25':(.5,.75),'VALIDATION_25':(.75,1)}; rows=[]
    states={'ALL':pd.Series(True,index=p.index),'LIQ_IMPROVING':p.liqz>=Z,'LIQ_STRESSED':p.illiqz>=Z,'VOL_STRESSED':p.rvz>=Z,'LIQ_IMPROVING_NOT_VOL_STRESSED':(p.liqz>=Z)&~(p.rvz>=Z),'LIQ_STRESSED_NOT_VOL_STRESSED':(p.illiqz>=Z)&~(p.rvz>=Z),'LIQ_IMPROVING_AND_VOL_STRESSED':(p.liqz>=Z)&(p.rvz>=Z),'LIQ_STRESSED_AND_VOL_STRESSED':(p.illiqz>=Z)&(p.rvz>=Z)}
    dirs={'LONG_MOM':p.mom>0,'SHORT_MOM':p.mom<0}
    for name,(u,v) in splits.items():
        q=p[(p.dt>=times[int(u*len(times))])&(p.dt<=times[min(len(times)-1,int(v*len(times))-1)])]
        for sn,sm in states.items():
            for dn,dm in dirs.items():
                zq=q[sm.loc[q.index]&dm.loc[q.index]]; y=zq.fwd24.dropna(); r=zq.rr2.dropna()
                if len(y)==0 and len(r)==0:continue
                rows.append({'split':name,'state':sn,'direction':dn,'n_forward':len(y),'mean_fwd24':y.mean() if len(y) else np.nan,'positive_fwd_rate':(y>0).mean() if len(y) else np.nan,'n_rr2':len(r),'rr_win_rate':(r>0).mean() if len(r) else np.nan,'rr_mean':r.mean() if len(r) else np.nan,'pf':(r[r>0].sum()/abs(r[r<0].sum())) if (r<0).any() else np.nan})
    pd.DataFrame(rows).to_csv(OUT/'event_study_report.csv',index=False)
    with open(OUT/'run_metadata.json','w') as f:json.dump({'version':'V29-STAGE0','days':DAYS,'warmup':WARMUP,'symbols':SYMS,'momentum_bars':MOM,'state_z':Z,'state_window':ZW,'rr':'1:2','stop_atr':1,'target_atr':2,'lookahead':'rolling stats shifted 1; next-bar open entry','same_candle_sl_tp':'loss','tuning_after_results':False},f,indent=2)
    print('V29-STAGE0 COMPLETE',len(p),'rows',len(rows),'event cells')
if __name__=='__main__':main()
