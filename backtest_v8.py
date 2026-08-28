import time
from datetime import datetime, timezone, timedelta
import requests
import pandas as pd
import numpy as np

BINANCE_FUTURES_URL = "https://fapi.binance.com/fapi/v1/klines"

SYMBOLS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
    "XRP": "XRPUSDT", "ADA": "ADAUSDT", "LINK": "LINKUSDT",
    "DOGE": "DOGEUSDT",
}

DAYS = 450
WARMUP_DAYS = 80
LIMIT = 1500

EMA_FAST, EMA_MID, EMA_SLOW = 20, 50, 200
ATR_PERIOD, ADX_PERIOD, RSI_PERIOD = 14, 14, 14
ADX_MIN = 18.0
SLOPE_LOOKBACK = 5
MIN_EMA50_SLOPE_ATR = 0.05

PULLBACK_LOOKBACK = 6
PULLBACK_ATR_DISTANCE = 0.65
LONG_RSI_MIN, LONG_RSI_MAX = 48.0, 68.0
SHORT_RSI_MIN, SHORT_RSI_MAX = 32.0, 52.0

ATR_EXPANSION_LOOKBACK = 20
ATR_EXPANSION_MIN = 0.90
MIN_BODY_RATIO = 0.45
MIN_CLOSE_LOCATION = 0.65
MAX_ENTRY_DISTANCE_ATR = 1.00

SL_ATR = 1.20
TP_R = 1.35
STRUCTURE_LOOKBACK = 6
STRUCTURE_BUFFER_ATR = 0.10

COOLDOWN_BARS_AFTER_LOSS = 6
MAX_CONSECUTIVE_LOSSES = 2

FEE_PER_SIDE = 0.0004
SLIPPAGE_PER_SIDE = 0.0002
RISK_PER_TRADE = 0.01


def get_klines(symbol, interval, start_ms, end_ms):
    rows, cursor = [], start_ms
    while cursor < end_ms:
        r = requests.get(
            BINANCE_FUTURES_URL,
            params={"symbol": symbol, "interval": interval,
                    "startTime": cursor, "endTime": end_ms, "limit": LIMIT},
            timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        nxt = int(batch[-1][0]) + 1
        if nxt <= cursor:
            break
        cursor = nxt
        if len(batch) < LIMIT:
            break
        time.sleep(0.12)

    if not rows:
        raise RuntimeError(f"No data for {symbol} {interval}")

    df = pd.DataFrame(rows, columns=[
        "time","open","high","low","close","volume","close_time",
        "quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"])
    df = df.drop_duplicates("time").sort_values("time")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df[["time","open","high","low","close","volume"]].reset_index(drop=True)


def download_symbol(name):
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=DAYS + WARMUP_DAYS)
    a = int(start.timestamp()*1000)
    b = int(now.timestamp()*1000)
    return (
        get_klines(SYMBOLS[name], "1h", a, b),
        get_klines(SYMBOLS[name], "4h", a, b)
    )


def true_range(df):
    pc = df["close"].shift(1)
    return pd.concat([
        df["high"]-df["low"],
        (df["high"]-pc).abs(),
        (df["low"]-pc).abs()], axis=1).max(axis=1)


def add_indicators(df):
    x = df.copy()
    x["ema20"] = x["close"].ewm(span=EMA_FAST, adjust=False).mean()
    x["ema50"] = x["close"].ewm(span=EMA_MID, adjust=False).mean()
    x["ema200"] = x["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    tr = true_range(x)
    x["atr"] = tr.ewm(alpha=1/ATR_PERIOD, adjust=False).mean()

    d = x["close"].diff()
    gain, loss = d.clip(lower=0), -d.clip(upper=0)
    ag = gain.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    al = loss.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    x["rsi"] = (100 - 100/(1 + ag/al.replace(0,np.nan))).fillna(50)

    up, down = x["high"].diff(), -x["low"].diff()
    plus = pd.Series(np.where((up>down)&(up>0),up,0.0), index=x.index)
    minus = pd.Series(np.where((down>up)&(down>0),down,0.0), index=x.index)
    atrv = tr.ewm(alpha=1/ADX_PERIOD, adjust=False).mean()
    pdi = 100*plus.ewm(alpha=1/ADX_PERIOD,adjust=False).mean()/atrv.replace(0,np.nan)
    mdi = 100*minus.ewm(alpha=1/ADX_PERIOD,adjust=False).mean()/atrv.replace(0,np.nan)
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    x["adx"] = dx.ewm(alpha=1/ADX_PERIOD,adjust=False).mean().fillna(0)

    x["atr_mean"] = x["atr"].rolling(ATR_EXPANSION_LOOKBACK).mean()
    x["ema50_slope"] = x["ema50"] - x["ema50"].shift(SLOPE_LOOKBACK)
    rng = (x["high"]-x["low"]).replace(0,np.nan)
    x["body_ratio"] = (x["close"]-x["open"]).abs()/rng
    x["long_loc"] = (x["close"]-x["low"])/rng
    x["short_loc"] = (x["high"]-x["close"])/rng
    return x


def regime(r):
    if any(pd.isna(r[c]) for c in ["ema50","ema200","atr","adx","ema50_slope"]):
        return None
    slope = r["ema50_slope"]/max(r["atr"],1e-12)
    if (r["close"]>r["ema200"] and r["ema50"]>r["ema200"] and
        r["adx"]>=ADX_MIN and slope>=MIN_EMA50_SLOPE_ATR):
        return "LONG"
    if (r["close"]<r["ema200"] and r["ema50"]<r["ema200"] and
        r["adx"]>=ADX_MIN and slope<=-MIN_EMA50_SLOPE_ATR):
        return "SHORT"
    return None


def prepare(df1, df4):
    h4 = add_indicators(df4)
    h4["regime"] = h4.apply(regime, axis=1)
    x = pd.merge_asof(
        df1.sort_values("time"),
        h4[["time","regime"]].sort_values("time"),
        on="time", direction="backward")
    cutoff = x["time"].max() - pd.Timedelta(days=DAYS)
    return x[x["time"]>=cutoff].reset_index(drop=True)


def pullback_recent(df, i, direction):
    start = max(0, i-PULLBACK_LOOKBACK)
    for j in range(start, i):
        r = df.iloc[j]
        if pd.isna(r["atr"]):
            continue
        if direction=="LONG":
            touched = r["low"] <= r["ema20"]+r["atr"]*PULLBACK_ATR_DISTANCE or                       r["low"] <= r["ema50"]+r["atr"]*PULLBACK_ATR_DISTANCE
            aligned = r["close"] > r["ema200"]
        else:
            touched = r["high"] >= r["ema20"]-r["atr"]*PULLBACK_ATR_DISTANCE or                       r["high"] >= r["ema50"]-r["atr"]*PULLBACK_ATR_DISTANCE
            aligned = r["close"] < r["ema200"]
        if touched and aligned:
            return True
    return False


def signal(df, i):
    if i < EMA_SLOW+20:
        return None
    r, p = df.iloc[i], df.iloc[i-1]
    d = r["regime"]
    if d not in ("LONG","SHORT"):
        return None
    cols = ["close","open","high","low","ema20","ema50","ema200","atr","rsi","adx","atr_mean"]
    if any(pd.isna(r[c]) for c in cols):
        return None

    if d=="LONG":
        if not (r["close"]>r["ema20"]>r["ema50"]>r["ema200"]): return None
        if not (LONG_RSI_MIN<=r["rsi"]<=LONG_RSI_MAX and r["rsi"]>p["rsi"]): return None
        candle_ok = r["close"]>r["open"] and r["body_ratio"]>=MIN_BODY_RATIO and r["long_loc"]>=MIN_CLOSE_LOCATION
        trigger = r["close"]>p["high"]
    else:
        if not (r["close"]<r["ema20"]<r["ema50"]<r["ema200"]): return None
        if not (SHORT_RSI_MIN<=r["rsi"]<=SHORT_RSI_MAX and r["rsi"]<p["rsi"]): return None
        candle_ok = r["close"]<r["open"] and r["body_ratio"]>=MIN_BODY_RATIO and r["short_loc"]>=MIN_CLOSE_LOCATION
        trigger = r["close"]<p["low"]

    if r["adx"]<ADX_MIN or not pullback_recent(df,i,d): return None
    if r["atr"]/max(r["atr_mean"],1e-12)<ATR_EXPANSION_MIN: return None
    if not candle_ok or not trigger: return None
    if abs(r["close"]-r["ema20"])/max(r["atr"],1e-12)>MAX_ENTRY_DISTANCE_ATR: return None

    recent = df.iloc[max(0,i-STRUCTURE_LOOKBACK+1):i+1]
    if d=="LONG":
        sl=min(r["close"]-r["atr"]*SL_ATR, recent["low"].min()-r["atr"]*STRUCTURE_BUFFER_ATR)
        risk=r["close"]-sl
        tp=r["close"]+risk*TP_R
    else:
        sl=max(r["close"]+r["atr"]*SL_ATR, recent["high"].max()+r["atr"]*STRUCTURE_BUFFER_ATR)
        risk=sl-r["close"]
        tp=r["close"]-risk*TP_R
    if risk<=0: return None
    return {"direction":d,"signal_time":r["time"],"entry":float(r["close"]),
            "sl":float(sl),"tp":float(tp),"risk_pct":float(risk/r["close"])}


def simulate(df):
    trades, pos, cooldown, losses = [], None, 0, 0
    for i in range(1,len(df)):
        r=df.iloc[i]
        if pos:
            d=pos["direction"]
            hit_sl = r["low"]<=pos["sl"] if d=="LONG" else r["high"]>=pos["sl"]
            hit_tp = r["high"]>=pos["tp"] if d=="LONG" else r["low"]<=pos["tp"]
            if hit_sl or hit_tp:
                result="SL" if hit_sl else "TP"
                px=pos["sl"] if hit_sl else pos["tp"]
                if d=="LONG": px*=1-SLIPPAGE_PER_SIDE
                else: px*=1+SLIPPAGE_PER_SIDE
                gross=(px-pos["entry"])/pos["entry"] if d=="LONG" else (pos["entry"]-px)/pos["entry"]
                net=gross-2*FEE_PER_SIDE
                pos["exit_time"],pos["result"],pos["net_return"]=r["time"],result,net
                pos["pnl_r"]=net/max(pos["risk_pct"],1e-12)
                trades.append(pos)
                if result=="SL":
                    losses+=1; cooldown=COOLDOWN_BARS_AFTER_LOSS
                else: losses=0
                if losses>=MAX_CONSECUTIVE_LOSSES:
                    cooldown=max(cooldown,COOLDOWN_BARS_AFTER_LOSS*2)
                pos=None
            continue
        if cooldown>0:
            cooldown-=1
            continue
        s=signal(df,i)
        if s:
            s["entry"] = s["entry"]*(1+SLIPPAGE_PER_SIDE if s["direction"]=="LONG" else 1-SLIPPAGE_PER_SIDE)
            pos=s
    return pd.DataFrame(trades)


def metrics(t):
    if t.empty:
        return dict(trades=0,wins=0,losses=0,win_rate=0,avg_r=0,pf=0,max_dd=0,signals_day=0)
    wins=(t.result=="TP").sum()
    losses=(t.result=="SL").sum()
    rs=t.pnl_r.astype(float)
    gp=rs[rs>0].sum()
    gl=-rs[rs<0].sum()
    eq=[1.0]
    for x in rs: eq.append(eq[-1]*(1+x*RISK_PER_TRADE))
    peak=eq[0]; dd=0
    for x in eq:
        peak=max(peak,x); dd=max(dd,(peak-x)/peak)
    days=max((t.exit_time.max()-t.signal_time.min()).total_seconds()/86400,1)
    return dict(trades=len(t),wins=int(wins),losses=int(losses),
                win_rate=100*wins/len(t),avg_r=rs.mean(),
                pf=gp/gl if gl else float("inf"),max_dd=100*dd,
                signals_day=len(t)/days)


def main():
    print("\nSCORE HUNTER — TREND PULSE v1")
    print("4H regime + 1H pullback/recovery + volatility expansion")
    print("Conservative intrabar rule: if TP and SL occur in same candle, SL wins.")
    all_t=[]
    for name in SYMBOLS:
        try:
            print(f"\nDownloading {name}...")
            d1,d4=download_symbol(name)
            d1=add_indicators(d1)
            df=prepare(d1,d4)
            t=simulate(df)
            if not t.empty:
                t["symbol"]=name; all_t.append(t)
            m=metrics(t)
            print(f"{name}: {m['trades']} trades | WR {m['win_rate']:.2f}% | PF {m['pf']:.2f} | AvgR {m['avg_r']:.3f}")
        except Exception as e:
            print(f"{name} ERROR: {type(e).__name__}: {e}")

    if not all_t:
        print("\nNO TRADES FOUND"); return

    t=pd.concat(all_t,ignore_index=True).sort_values("signal_time").reset_index(drop=True)
    m=metrics(t)
    days=t["signal_time"].dt.date.nunique()
    raw=len(t)/max(days,1)

    print("\n"+"="*65)
    print("PORTFOLIO RESULT")
    print("="*65)
    print(f"Trades:          {m['trades']}")
    print(f"Wins:            {m['wins']}")
    print(f"Losses:          {m['losses']}")
    print(f"Win rate:        {m['win_rate']:.2f}%")
    print(f"Average R:       {m['avg_r']:.3f}")
    print(f"Profit factor:   {m['pf']:.2f}")
    print(f"Max drawdown:    {m['max_dd']:.2f}%")
    print(f"Signals/day:     {raw:.2f}")

    split=t["signal_time"].max()-pd.Timedelta(days=90)
    oos=t[t["signal_time"]>=split]
    om=metrics(oos)
    print("\nOOS — LAST 90 DAYS")
    print("-"*65)
    print(f"Trades:          {om['trades']}")
    print(f"Win rate:        {om['win_rate']:.2f}%")
    print(f"Average R:       {om['avg_r']:.3f}")
    print(f"Profit factor:   {om['pf']:.2f}")
    print(f"Max drawdown:    {om['max_dd']:.2f}%")
    print(f"Signals/day:     {om['signals_day']:.2f}")

    t.to_csv("trend_pulse_trades.csv",index=False)
    print("\nSaved: trend_pulse_trades.csv")


if __name__=="__main__":
    main()
