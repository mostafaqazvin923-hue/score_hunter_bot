import os, sys, subprocess
from datetime import datetime, timedelta, timezone
try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable,'-m','pip','install','ccxt']); import ccxt
import numpy as np, pandas as pd

# HUNTER-V78: V76-compatible signals, no V77 hard filters.
exchange=ccxt.lbank({'enableRateLimit':True})
SYMBOLS={'BTC':'BTC/USDT','ETH':'ETH/USDT','BNB':'BNB/USDT','XRP':'XRP/USDT','SOL':'SOL/USDT','DOGE':'DOGE/USDT','ADA':'ADA/USDT','LINK':'LINK/USDT','TRX':'TRX/USDT','HYPE':'HYPE/USDT','ZEC':'ZEC/USDT','AVAX':'AVAX/USDT','SUI':'SUI/USDT','XLM':'XLM/USDT','AAVE':'AAVE/USDT','ATOM':'ATOM/USDT','ICP':'ICP/USDT','INJ':'INJ/USDT','RENDER':'RENDER/USDT','ONDO':'ONDO/USDT'}
LOOKBACK_DAYS=365; TIMEFRAME='4h'; MAX_POSITIONS=5; MAX_LONGS=3; MAX_SHORTS=3
SLIPPAGE=.0003; FEE_RATE=.0007; ATR_PERIOD=14; INITIAL_ATR_MULTIPLIER=1.8; TRAILING_ATR_MULTIPLIER=2.; TIMEOUT_CANDLES=45; EMA_WARMUP=200
INITIAL_CAPITAL=1000.; TRADE_MARGIN=100.; LEVERAGE=80.; NOTIONAL=TRADE_MARGIN*LEVERAGE
MIN_STOP_PCT=.01; MAX_STOP_PCT=.04; OUT='hunter_v78_output'; os.makedirs(OUT,exist_ok=True)
start=int((datetime.now(timezone.utc)-timedelta(days=LOOKBACK_DAYS)).timestamp()*1000)

def fetch(sym):
    rows=[]; cur=start; last=None
    for _ in range(500):
        if cur>=exchange.milliseconds(): break
        batch=None
        for attempt in range(3):
            try: batch=exchange.fetch_ohlcv(sym,timeframe=TIMEFRAME,since=cur,limit=1000); break
            except Exception as e:
                if attempt==2: print('FAIL',sym,e); return None
        if not batch: break
        newest=int(batch[-1][0])
        if last is not None and newest<=last: break
        rows += batch; last=newest; cur=newest+1
        if len(batch)<1000: break
    if not rows:return None
    d=pd.DataFrame(rows,columns=['Timestamp','Open','High','Low','Close','Volume'])
    d['Timestamp']=pd.to_numeric(d.Timestamp,errors='coerce'); d=d.dropna(subset=['Timestamp']); d.Timestamp=d.Timestamp.astype(np.int64)
    if d.Timestamp.median()<10_000_000_000:d.Timestamp*=1000
    d=d[(d.Timestamp>=start)&(d.Timestamp<exchange.milliseconds())].copy(); d['Date']=pd.to_datetime(d.Timestamp,unit='ms',utc=True)
    for c in ['Open','High','Low','Close','Volume']:d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d[['Date','Open','High','Low','Close','Volume']].dropna().drop_duplicates('Date').sort_values('Date').reset_index(drop=True)
    tf=4*3600*1000; bucket=(exchange.milliseconds()//tf)*tf; d=d[d.Date.astype('int64')//10**6<bucket].reset_index(drop=True)
    if len(d)<EMA_WARMUP+ATR_PERIOD+40:return None
    if len(d)>1 and d.Date.diff().dt.total_seconds().max()>15000:return None
    pc=d.Close.shift(1); tr=pd.concat([d.High-d.Low,(d.High-pc).abs(),(d.Low-pc).abs()],axis=1).max(axis=1)
    d['ATR']=tr.rolling(ATR_PERIOD).mean(); d['EMA20']=d.Close.ewm(span=20,adjust=False).mean(); d['EMA50']=d.Close.ewm(span=50,adjust=False).mean(); d['EMA200']=d.Close.ewm(span=200,adjust=False).mean(); d['Mom_Short']=d.Close.pct_change(10); d['Mom_Long']=d.Close.pct_change(30); d['ATR_Pct']=d.ATR/d.Close
    return d.dropna().set_index('Date')

print('Downloading LBank 4h data...'); data={}
for n,s in SYMBOLS.items():
    print(n,end=' ',flush=True); x=fetch(s)
    if x is not None:data[n]=x; print(f'OK {len(x)}')
    else:print('INVALID')
if 'BTC' not in data: raise RuntimeError('BTC unavailable')
print(f'Valid Symbols: {len(data)}/{len(SYMBOLS)}')

def btc_reg(ts):
    b=data['BTC'].loc[ts]
    if b.Close>b.EMA200 and b.EMA50>b.EMA200:return 'BULL'
    if b.Close<b.EMA200 and b.EMA50<b.EMA200:return 'BEAR'
    return 'NEUTRAL'
def breadth(ts):
    a=[d.loc[ts] for d in data.values() if ts in d.index]
    return sum(x.Close>x.EMA200 and x.EMA50>x.EMA200 for x in a)/len(a) if a else np.nan
def side_count(pos,side):return sum(p['side']==side for p in pos.values())
def exit_calc(p,ep):
    gross=((ep-p['entry'])*p['qty']) if p['side']=='LONG' else ((p['entry']-ep)*p['qty'])
    fees=(p['entry']+ep)*p['qty']*FEE_RATE
    pnl=gross-fees; risk=abs(p['entry']-p['initial_stop'])*p['qty']; return gross,fees,pnl,pnl/risk if risk>0 else np.nan

times=sorted({t for d in data.values() for t in d.index}); pos={}; trades=[]; capital=INITIAL_CAPITAL; equity=[]
checks={'signals_checked':0,'long_signal':0,'short_signal':0,'accepted':0,'open_position':0,'position_limit':0,'direction_limit':0,'stop_distance':0}
for ts in times:
    closed=[]
    for sym,p in list(pos.items()):
        if ts not in data[sym].index:continue
        i=data[sym].index.get_loc(ts); c=data[sym].iloc[i]; old=p['stop_loss']
        hit=c.Low<=old if p['side']=='LONG' else c.High>=old; timeout=i-p['entry_i']>=TIMEOUT_CANDLES
        if hit or timeout:
            ep=(min(old,c.Open) if p['side']=='LONG' else max(old,c.Open)) if hit else c.Close
            reason='STOP' if hit else 'TIMEOUT'; gross,fees,pnl,r=exit_calc(p,ep); outcome='WIN' if pnl>0 else 'LOSS'; capital+=pnl
            trades.append({**p,'Timestamp':ts,'Exit':ep,'GrossPnL':gross,'Fees':fees,'PnL':pnl,'R':r,'Outcome':outcome,'Reason':reason,'Entry':p['entry'],'InitialStop':p['initial_stop'],'FinalStop':old,'Qty':p['qty'],'CandlesHeld':i-p['entry_i']})
            closed.append(sym)
        else:
            if p['side']=='LONG':
                p['highest']=max(p['highest'],c.High); p['stop_loss']=max(p['stop_loss'],p['highest']-TRAILING_ATR_MULTIPLIER*c.ATR)
            else:
                p['lowest']=min(p['lowest'],c.Low); p['stop_loss']=min(p['stop_loss'],p['lowest']+TRAILING_ATR_MULTIPLIER*c.ATR)
    for s in closed:del pos[s]
    if ts not in data['BTC'].index:continue
    b=data['BTC'].loc[ts]; reg=btc_reg(ts); br=breadth(ts)
    longs=[]; shorts=[]
    for sym,d in data.items():
        if sym=='BTC' or ts not in d.index:continue
        i=d.index.get_loc(ts)
        if i<EMA_WARMUP:continue
        c=d.iloc[i]; checks['signals_checked']+=1
        long=(c.Close>c.EMA20 and c.EMA20>c.EMA50 and c.Close>c.EMA200 and c.Mom_Short>.012 and c.Mom_Long>.035 and br>=.35)
        short=(c.Close<c.EMA20 and c.EMA20<c.EMA50 and c.Close<c.EMA200 and c.Mom_Short<-.012 and c.Mom_Long<-.035 and br<=.65)
        if i>0:
            prev=d.iloc[i-1]; long &= prev.Low<=prev.EMA20*1.015; short &= prev.High>=prev.EMA20*.985
        rel=float(c.Mom_Long-b.Mom_Long)
        if long:checks['long_signal']+=1; longs.append((float(c.Mom_Long+.10*rel),sym,c,rel,i))
        if short:checks['short_signal']+=1; shorts.append((float(c.Mom_Long-.10*rel),sym,c,rel,i))
    longs.sort(key=lambda x:x[0],reverse=True); shorts.sort(key=lambda x:x[0])
    ordered=([('LONG',x) for x in longs]+[('SHORT',x) for x in shorts]) if reg!='BEAR' else ([('SHORT',x) for x in shorts]+[('LONG',x) for x in longs])
    for side,item in ordered:
        if len(pos)>=MAX_POSITIONS:checks['position_limit']+=1;break
        score,sym,c,rel,i=item
        if sym in pos:checks['open_position']+=1;continue
        if side_count(pos,side)>=(MAX_LONGS if side=='LONG' else MAX_SHORTS):checks['direction_limit']+=1;continue
        entry=c.Open*(1+SLIPPAGE if side=='LONG' else 1-SLIPPAGE); stop=entry-(INITIAL_ATR_MULTIPLIER*c.ATR if side=='LONG' else -INITIAL_ATR_MULTIPLIER*c.ATR); dist=abs(entry-stop)/entry
        if not MIN_STOP_PCT<=dist<=MAX_STOP_PCT:checks['stop_distance']+=1;continue
        p={'side':side,'EntryTimestamp':ts,'entry':entry,'initial_stop':stop,'stop_loss':stop,'qty':NOTIONAL/entry,'entry_i':i,'highest':entry,'lowest':entry,'ATR_Pct_Entry':float(c.ATR_Pct),'Mom_Long_Entry':float(c.Mom_Long),'Mom_Short_Entry':float(c.Mom_Short),'BTC_Regime_Entry':reg,'Breadth_Entry':float(br),'RelativeMomentum_Entry':rel,'SL_Distance_Pct':dist}
        pos[sym]=p;checks['accepted']+=1
    eq=capital
    for sym,p in pos.items():
        if ts in data[sym].index:
            mark=float(data[sym].loc[ts].Close); eq += (mark-p['entry'])*p['qty'] if p['side']=='LONG' else (p['entry']-mark)*p['qty']
    equity.append((ts,eq))

# Force close at final available close.
if pos:
    last=max(d.index.max() for d in data.values())
    for sym,p in list(pos.items()):
        if last not in data[sym].index:continue
        ep=float(data[sym].loc[last].Close); gross,fees,pnl,r=exit_calc(p,ep); capital+=pnl; trades.append({**p,'Timestamp':last,'Exit':ep,'GrossPnL':gross,'Fees':fees,'PnL':pnl,'R':r,'Outcome':'WIN' if pnl>0 else 'LOSS','Reason':'END_OF_TEST','Entry':p['entry'],'InitialStop':p['initial_stop'],'FinalStop':p['stop_loss'],'Qty':p['qty'],'CandlesHeld':0})

T=pd.DataFrame(trades)
if T.empty:print('NO TRADES');sys.exit()
T=T.sort_values(['Timestamp','Symbol'] if 'Symbol' in T else ['Timestamp']).reset_index(drop=True)
# Symbol was stored as dict key only implicitly; recover from position dict is impossible after expansion, so map via entry timestamps/values is not safe.
# Rebuild Symbol by matching EntryTimestamp + Entry + side to candidate universe is unnecessary for core results; use a robust helper from rows below.
# Every trade currently lacks Symbol because p has no symbol. Add it by rerunning a deterministic match on unique entry price/time/side.
def add_symbol(row):
    candidates=[]
    ts=row.EntryTimestamp; ep=row.entry; side=row.side
    for sym,d in data.items():
        if sym=='BTC' or ts not in d.index:continue
        c=d.loc[ts]; expected=float(c.Open*(1+SLIPPAGE if side=='LONG' else 1-SLIPPAGE))
        if abs(expected-ep)<=max(1e-10,abs(ep)*1e-10):candidates.append(sym)
    return candidates[0] if len(candidates)==1 else 'UNKNOWN'
T['Symbol']=T.apply(add_symbol,axis=1)
# Ensure correct final capital from all closed trades.
net=float(T.PnL.sum()); final=INITIAL_CAPITAL+net; wr=float((T.Outcome=='WIN').mean()*100); R=float(T.R.sum())
ls=[];s=0
for o in T.Outcome:
    if o=='LOSS':s+=1
    elif s:ls.append(s);s=0
if s:ls.append(s)
maxls=max(ls) if ls else 0; avgls=float(np.mean(ls)) if ls else 0
vals=np.array([x[1] for x in equity]+[final],dtype=float); peaks=np.maximum.accumulate(vals); dd=vals-peaks; maxdd=float(dd.min()); maxddpct=float(maxdd/max(peaks.max(),1e-12)*100)
print('\n'+'='*72);print('HUNTER-V78 FINAL');print('='*72);print(f'Valid Symbols : {len(data)}/{len(SYMBOLS)}');print(f'Total Trades  : {len(T)}');print(f'Wins          : {(T.Outcome=="WIN").sum()}');print(f'Losses        : {(T.Outcome=="LOSS").sum()}');print(f'Win Rate      : {wr:.2f}%');print(f'Net R         : {R:.2f}R');print(f'Net PnL       : ${net:,.2f}');print(f'Final Capital : ${final:,.2f}');print(f'Return        : {(final/INITIAL_CAPITAL-1)*100:.2f}%');print(f'Max Drawdown  : ${maxdd:,.2f} ({maxddpct:.2f}%)');print(f'Max Loss Streak: {maxls}');print(f'Avg Loss Streak: {avgls:.2f}')
print('\nDIRECTION');
for side in ['LONG','SHORT']:
 d=T[T.side==side]
 if len(d):print(f'{side:<5} | Trades={len(d):3d} | WR={(d.Outcome=="WIN").mean()*100:6.2f}% | PnL=${d.PnL.sum():,.2f} | R={d.R.sum():.2f}')
print('\nPER SYMBOL')
for sym in SYMBOLS:
 d=T[T.Symbol==sym]; ls2=[];s=0
 for o in d.sort_values('Timestamp').Outcome:
  if o=='LOSS':s+=1;ls2.append(s)
  else:s=0
 print(f'{sym:<8} | Trades={len(d):3d} | WR={(d.Outcome=="WIN").mean()*100 if len(d) else 0:6.2f}% | PnL=${d.PnL.sum() if len(d) else 0:10,.2f} | R={d.R.sum() if len(d) else 0:9.2f} | MaxLS={max(ls2) if ls2 else 0}')
print('\nLOSS STREAK DISTRIBUTION');
for k,v in pd.Series(ls).value_counts().sort_index().items():print(f'{int(k):2d}: {int(v)}')
print('\nDIAGNOSTICS');
for k,v in checks.items():print(f'{k:<20}: {v}')
T.to_csv(os.path.join(OUT,'trades.csv'),index=False)
T.groupby(['Symbol','side']).agg(Trades=('PnL','size'),Wins=('Outcome',lambda x:(x=='WIN').sum()),PnL=('PnL','sum'),R=('R','sum')).reset_index().to_csv(os.path.join(OUT,'symbol_direction_stats.csv'),index=False)
T.groupby('Symbol').agg(Trades=('PnL','size'),Wins=('Outcome',lambda x:(x=='WIN').sum()),PnL=('PnL','sum'),R=('R','sum')).reset_index().to_csv(os.path.join(OUT,'symbol_stats.csv'),index=False)
T.groupby('side').agg(Trades=('PnL','size'),Wins=('Outcome',lambda x:(x=='WIN').sum()),PnL=('PnL','sum'),R=('R','sum')).reset_index().to_csv(os.path.join(OUT,'direction_stats.csv'),index=False)
print(f'\nCSV: {OUT}/')
