import os, time, json, urllib.parse, urllib.request
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

XT='https://fapi.xt.com'
CG='https://open-api-v4.coinglass.com'
SYMS=['BTC/USDT','ETH/USDT','SOL/USDT','SUI/USDT','AVAX/USDT','NEAR/USDT','ADA/USDT','BNB/USDT','APT/USDT','CRV/USDT','ONDO/USDT','PENDLE/USDT','ICP/USDT','WIF/USDT']
DAYS=455; WARMUP=7; BAR=4*3600*1000; LIMIT=1000; MIN_COV=.97
OUT=Path('reports/xt_v25_derivatives_research'); OUT.mkdir(parents=True,exist_ok=True)
KEY = os.getenv("COINGLASS_API_KEY", "")

# Secret diagnostics are deliberately non-secret:
# we NEVER print the key itself or any substring of it.
# Strip only UTF-8 BOM, surrounding ASCII whitespace, and one matching
# pair of surrounding quotes. Do not silently change characters inside
# the key.
KEY = KEY.lstrip("\ufeff").strip()
if len(KEY) >= 2 and KEY[0] == KEY[-1] and KEY[0] in ("'", '"'):
    KEY = KEY[1:-1].strip()

if not KEY:
    raise RuntimeError(
        "COINGLASS_API_KEY is missing/empty. "
        "Set the GitHub Actions repository secret COINGLASS_API_KEY."
    )

bad = [(i, ord(ch)) for i, ch in enumerate(KEY) if ord(ch) > 127]
if bad:
    positions = ",".join(str(i) for i, _ in bad[:20])
    raise RuntimeError(
        "COINGLASS_API_KEY has non-ASCII characters at character position(s) "
        f"{positions} (0-based). The key itself is NOT printed. "
        "This usually means the secret contains copied text/characters "
        "that are not part of the raw API key. Replace the secret with the "
        "exact raw API key from CoinGlass."
    )

# Detect accidental internal whitespace without revealing the secret.
internal_ws = [i for i, ch in enumerate(KEY) if ch.isspace()]
if internal_ws:
    positions = ",".join(str(i) for i in internal_ws[:20])
    raise RuntimeError(
        "COINGLASS_API_KEY contains whitespace inside the value at "
        f"character position(s) {positions} (0-based). "
        "The key itself is NOT printed. Use the raw API key only."
    )
def get(url, params):
    last = None
    for i in range(4):
        try:
            query = urllib.parse.urlencode(params, doseq=True, safe="")
            full_url = url + ("?" + query if query else "")
            headers = {
                "Accept": "application/json",
                "User-Agent": "score-hunter-v25/1.0",
            }
            if url.startswith(CG):
                headers["CG-API-KEY"] = KEY

            req = urllib.request.Request(full_url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=30) as response:
                j = json.loads(response.read().decode("utf-8"))

            if isinstance(j, dict) and str(j.get("code", "0")) not in ("0", "200"):
                raise RuntimeError(j.get("msg", "API error"))
            return j
        except Exception as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"GET failed: {url} {params}: {last}")

def xt(sym, a, b):
    rows = []
    cur = a
    symbol = sym.replace("/", "_").lower()

    # XT's futures kline endpoint is treated as end-exclusive here.
    # We advance from the actual newest candle returned, never by assumed
    # page size, and never make a tiny tail request that can trigger the
    # empty-page boundary seen in earlier XT pagination work.
    page = 0
    previous_max = None
    WINDOW = LIMIT * BAR

    while cur < b:
        page += 1
        if page > 1000:
            raise RuntimeError(f"XT pagination guard exceeded: {sym}")

        remaining = b - cur
        final_window = remaining <= WINDOW

        if final_window:
            window_start = cur
            window_end = b
        else:
            window_start = cur
            window_end = cur + WINDOW

        params = {
            "symbol": symbol,
            "interval": "4h",
            "startTime": int(window_start),
            "endTime": int(window_end),
            "limit": LIMIT,
        }

        j = get(f"{XT}/future/market/v1/public/q/kline", params)

        # XT has returned the kline list under different wrappers.
        d = j
        if isinstance(d, dict):
            d = d.get("result", d.get("data", d.get("rows", d.get("list", []))))
            if isinstance(d, dict):
                d = d.get("data", d.get("rows", d.get("list", d.get("items", []))))

        if not isinstance(d, list):
            raise RuntimeError(
                f"XT {sym} page={page}: unexpected kline response type: {type(d).__name__}"
            )

        parsed = []
        for x in d:
            try:
                if isinstance(x, dict):
                    ts = x.get("time", x.get("timestamp", x.get("ts", x.get("t"))))
                    o = x.get("open", x.get("o"))
                    h = x.get("high", x.get("h"))
                    l = x.get("low", x.get("l"))
                    c = x.get("close", x.get("c"))
                    v = x.get("volume", x.get("vol", x.get("v", x.get("amount", 0))))
                    z = [ts, o, h, l, c, v]
                elif isinstance(x, (list, tuple)) and len(x) >= 6:
                    z = list(x[:6])
                else:
                    continue

                if any(v is None for v in z[:5]):
                    continue

                ts = int(float(z[0]))
                if ts < 10_000_000_000:
                    ts *= 1000

                if window_start <= ts < window_end:
                    parsed.append(
                        [ts, float(z[1]), float(z[2]), float(z[3]), float(z[4]), float(z[5] or 0)]
                    )
            except Exception:
                continue

        parsed.sort(key=lambda r: r[0])

        if not parsed:
            if final_window:
                break
            raise RuntimeError(
                f"XT {sym}: empty bounded page={page}; "
                f"window={pd.to_datetime(window_start, unit='ms', utc=True)} -> "
                f"{pd.to_datetime(window_end, unit='ms', utc=True)}"
            )

        max_ts = parsed[-1][0]

        if previous_max is not None and max_ts <= previous_max and not final_window:
            raise RuntimeError(f"XT pagination stalled: {sym} page={page}")

        rows.extend(parsed)
        previous_max = max(previous_max or max_ts, max_ts)

        print(
            f"[XT] {sym} page={page} rows={len(set(r[0] for r in rows))} "
            f"latest={pd.to_datetime(max_ts, unit='ms', utc=True)}"
        )

        if final_window:
            break

        next_cur = max_ts + BAR
        if next_cur <= cur:
            raise RuntimeError(f"XT pagination did not advance: {sym} page={page}")
        cur = next_cur
        time.sleep(0.05)

    if not rows:
        raise RuntimeError(f"No XT data: {sym}")

    df = pd.DataFrame(
        rows,
        columns=["ts", "open", "high", "low", "close", "volume"],
    )
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)

    # Completed candles only.
    now_ms = int(time.time() * 1000)
    if len(df) and int(df["ts"].iloc[-1]) + BAR > now_ms:
        df = df.iloc[:-1].copy()

    # V25 uses 4h candles: 6 candles/day.
    # DAYS is the research horizon in calendar days, so the expected
    # completed-candle count is DAYS * (24/4), not DAYS * 24.
    expected_4h_rows = int((DAYS + WARMUP) * 24 / 4)
    min_required = int(expected_4h_rows * 0.95)

    if len(df) < min_required:
        span = (
            (df["ts"].iloc[-1] - df["ts"].iloc[0]) / 86_400_000
            if len(df) > 1 else 0
        )
        raise RuntimeError(
            f"XT {sym}: insufficient 4h history: rows={len(df)}, "
            f"need>={min_required}, expected={expected_4h_rows}, span={span:.1f}d"
        )

    return df

def cg(path,sym,a,b,ex=None):
    rows=[]; cur=a
    while cur<b:
        p={'symbol':sym,'interval':'4h','limit':LIMIT,'start_time':cur,'end_time':min(b,cur+LIMIT*BAR)}
        if ex:p['exchange_list']=ex
        d=get(CG+path,p).get('data',[]); d=d.get('data',[]) if isinstance(d,dict) else d
        if not d: break
        rows+=d; last=max(int(x['time']) for x in d); cur=last+BAR
        if len(d)<2: break
    if not rows: raise RuntimeError(f'No CoinGlass data: {path} {sym}')
    return pd.DataFrame(rows).drop_duplicates('time').sort_values('time')

def z(s,n=24):
    m=s.shift(1).rolling(n,min_periods=n).mean(); sd=s.shift(1).rolling(n,min_periods=n).std(ddof=0)
    return (s-m)/sd.replace(0,np.nan)

def features(x):
    p=x.close.shift(1); tr=pd.concat([x.high-x.low,(x.high-p).abs(),(x.low-p).abs()],axis=1).max(axis=1)
    x['atr']=tr.rolling(14,min_periods=14).mean(); x['atr_pct']=x.atr/x.close; x['ret4']=x.close.pct_change(); x['ret24']=x.close.pct_change(6)
    x['rv24']=x.ret4.rolling(6,min_periods=6).std(); x['rv72']=x.ret4.rolling(18,min_periods=18).std()
    for n in (20,50,200): x[f'ema{n}']=x.close.ewm(span=n,adjust=False,min_periods=n).mean(); x[f'ema{n}_gap']=x.close/x[f'ema{n}']-1
    x['trend_stack']=((x.close>x.ema20)&(x.ema20>x.ema50)&(x.ema50>x.ema200)).astype(int)
    x['range_expansion']=(x.high-x.low)/x.atr.replace(0,np.nan); x['volume_z24']=z(x.volume)
    lo=x.low.rolling(24,min_periods=24).min().shift(1); hi=x.high.rolling(24,min_periods=24).max().shift(1); x['close_pos24']=(x.close-lo)/(hi-lo).replace(0,np.nan)
    return x

def add_deriv(x,oi,liq,tak):
    oi=oi.rename(columns={'time':'ts','open':'oi_open','high':'oi_high','low':'oi_low','close':'oi_close'})
    liq=liq.rename(columns={'time':'ts','aggregated_long_liquidation_usd':'liq_long','aggregated_short_liquidation_usd':'liq_short'})
    tak=tak.rename(columns={'time':'ts','aggregated_buy_volume_usd':'taker_buy','aggregated_sell_volume_usd':'taker_sell'})
    for d in (oi,liq,tak): d['ts']=d.ts.astype('int64')
    x=x.merge(oi[['ts','oi_open','oi_high','oi_low','oi_close']],on='ts',how='left',validate='one_to_one').merge(liq[['ts','liq_long','liq_short']],on='ts',how='left',validate='one_to_one').merge(tak[['ts','taker_buy','taker_sell']],on='ts',how='left',validate='one_to_one')
    x['oi_ret']=x.oi_close.pct_change(); x['oi_range']=(x.oi_high-x.oi_low)/x.oi_close.replace(0,np.nan); x['taker_imb']=(x.taker_buy-x.taker_sell)/(x.taker_buy+x.taker_sell).replace(0,np.nan); x['liq_total']=x.liq_long+x.liq_short; x['liq_imb']=(x.liq_short-x.liq_long)/x.liq_total.replace(0,np.nan)
    for c in ('oi_close','oi_ret','taker_imb','liq_total','liq_imb'): x[c+'_z24']=z(x[c])
    return x

def labels(x,h=24):
    n=len(x); L=np.full(n,np.nan); S=np.full(n,np.nan)
    for i in range(n-1):
        a=x.atr.iloc[i]; e=x.open.iloc[i+1]
        if not np.isfinite(a) or a<=0: continue
        end=min(n,i+1+h)
        for j in range(i+1,end):
            if x.low.iloc[j]<=e-a and x.high.iloc[j]>=e+2*a: L[i]=0; break
            if x.low.iloc[j]<=e-a: L[i]=-1; break
            if x.high.iloc[j]>=e+2*a: L[i]=1; break
        for j in range(i+1,end):
            if x.high.iloc[j]>=e+a and x.low.iloc[j]<=e-2*a: S[i]=0; break
            if x.high.iloc[j]>=e+a: S[i]=-1; break
            if x.low.iloc[j]<=e-2*a: S[i]=1; break
    return L,S

def tail(df,fs,label):
    out=[]
    for f in fs:
        v=df[[f,label]].dropna()
        if len(v)<1000: continue
        q1,q3=v[f].quantile([.25,.75]); lo=v.loc[v[f]<=q1,label]; hi=v.loc[v[f]>=q3,label]
        out.append({'feature':f,'label':label,'n_low':len(lo),'n_high':len(hi),'low_mean':lo.mean(),'high_mean':hi.mean(),'high_minus_low':hi.mean()-lo.mean()})
    return pd.DataFrame(out)

def main():
    end=pd.Timestamp(datetime.now(timezone.utc)).floor('4h'); start=end-pd.Timedelta(days=DAYS); fetch=start-pd.Timedelta(days=WARMUP); a=int(fetch.timestamp()*1000); b=int(end.timestamp()*1000)
    panels=[]; cov=[]
    for sym in SYMS:
        print('[DATA]',sym)
        x=features(xt(sym,a,b)); cs=sym.split('/')[0]
        oi=cg('/api/futures/open-interest/aggregated-history',cs,a,b); liq=cg('/api/futures/liquidation/aggregated-history',cs,a,b,'Binance,OKX,Bybit'); tak=cg('/api/futures/aggregated-taker-buy-sell-volume/history',cs,a,b,'Binance,OKX,Bybit')
        x=add_deriv(x,oi,liq,tak); x=x[(x.ts>=int(start.timestamp()*1000))&(x.ts<b)].copy()
        cv={'oi_close':x.oi_close.notna().mean(),'liq_total':x.liq_total.notna().mean(),'taker_buy':x.taker_buy.notna().mean()}
        if min(cv.values())<MIN_COV: raise RuntimeError(f'Coverage failure {sym}: {cv}')
        x['rr_long'],x['rr_short']=labels(x); x['symbol']=sym; panels.append(x); cov.append({'symbol':sym,'rows':len(x),**{f'coverage_{k}':v for k,v in cv.items()}})
    p=pd.concat(panels,ignore_index=True); p['dt']=pd.to_datetime(p.ts,unit='ms',utc=True); p=p.sort_values(['dt','symbol']); p.to_csv(OUT/'research_panel.csv',index=False); pd.DataFrame(cov).to_csv(OUT/'coverage_audit.csv',index=False)
    fs=['atr_pct','ret4','ret24','rv24','rv72','ema20_gap','ema50_gap','ema200_gap','trend_stack','range_expansion','volume_z24','close_pos24','oi_ret','oi_range','oi_close_z24','oi_ret_z24','taker_imb','taker_imb_z24','liq_total_z24','liq_imb_z24']
    pd.concat([tail(p,fs,'rr_long'),tail(p,fs,'rr_short')],ignore_index=True).to_csv(OUT/'feature_tail_report.csv',index=False)
    times=np.sort(p.dt.unique()); rows=[]
    for name,(u,v) in {'DISCOVERY_50':(0,.5),'DEVELOPMENT_25':(.5,.75),'VALIDATION_25':(.75,1)}.items():
        q=p[(p.dt>=times[int(u*len(times))])&(p.dt<=times[min(len(times)-1,int(v*len(times))-1)])]
        for lab in ('rr_long','rr_short'):
            y=q[lab].dropna(); rows.append({'split':name,'direction':lab,'n':len(y),'wins':int((y>0).sum()),'losses':int((y<0).sum()),'win_rate':(y>0).mean(),'mean_R':y.mean()})
    pd.DataFrame(rows).to_csv(OUT/'rr2_path_report.csv',index=False)
    st=[]
    for name,(u,v) in {'DISCOVERY_50':(0,.5),'DEVELOPMENT_25':(.5,.75),'VALIDATION_25':(.75,1)}.items():
        q=p[(p.dt>=times[int(u*len(times))])&(p.dt<=times[min(len(times)-1,int(v*len(times))-1)])]
        t=pd.concat([tail(q,fs,'rr_long'),tail(q,fs,'rr_short')],ignore_index=True); t['split']=name; st.append(t)
    pd.concat(st,ignore_index=True).to_csv(OUT/'stability_by_split.csv',index=False)
    print('V25 COMPLETE:',len(p),'rows;',p.symbol.nunique(),'symbols')

if __name__=='__main__': main()
