import os
import json
import requests
from datetime import datetime, timezone

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

KRAKEN_URL = "https://api.kraken.com/0/public/OHLC"
STATE_FILE = "state.json"

INTERVAL = 240
REQUIRED_SCORE = 5
TP_PERCENT = 1.0
SL_PERCENT = 0.50

# Fixed-TP market-space filter
TP_SPACE_LOOKBACK = 20
TP_SPACE_BUFFER_PERCENT = 0.10
ENTRY_INTERVAL = 60
STRUCTURE_LOOKBACK = 20
VOLUME_MULTIPLIER = 1.20
SWEEP_LOOKBACK = 10
RETEST_TOLERANCE_PERCENT = 0.20
RANGE_EMA_GAP_PERCENT = 0.15
MIN_ADVANCED_SCORE = 8

COINS = {
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "BTC": "XBTUSDT",
    "ADA": "ADAUSDT",
    "LINK": "LINKUSDT",
    "DOGE": "DOGEUSDT",
}

def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    response = requests.post(
        url,
        data={"chat_id": CHAT_ID, "text": message},
        timeout=20,
    )
    print("Telegram:", response.status_code)
    print(response.text)
    response.raise_for_status()

def get_4h_data(symbol):
    print(f"\nGetting {symbol} 4H candles...")
    response = requests.get(
        KRAKEN_URL,
        params={"pair": COINS[symbol], "interval": INTERVAL},
        timeout=20,
    )
    print(f"{symbol} Kraken:", response.status_code)
    response.raise_for_status()

    payload = response.json()
    if payload.get("error"):
        raise RuntimeError(f"{symbol} Kraken API error: {payload['error']}")

    result = payload.get("result", {})
    pair_key = next((key for key in result if key != "last"), None)
    if pair_key is None:
        raise RuntimeError(f"{symbol}: no candle data returned")

    candles = []
    for row in result[pair_key]:
        candles.append({
            "time": int(row[0]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[6]),
        })

    if len(candles) > 1:
        candles = candles[:-1]

    if len(candles) < 210:
        raise RuntimeError(f"{symbol}: only {len(candles)} candles available")

    print(f"{symbol}: {len(candles)} closed 4H candles")
    return candles

def ema(values, period):
    if len(values) < period:
        return None
    value = sum(values[:period]) / period
    multiplier = 2.0 / (period + 1)
    for price in values[period:]:
        value = (price - value) * multiplier + value
    return value

def sma(values, period):
    if len(values) < period:
        return None
    return sum(values[-period:]) / period

def rsi(values, period=14):
    if len(values) <= period:
        return None

    gains = []
    losses = []
    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def atr(highs, lows, closes, period=14):
    if len(closes) <= period:
        return None

    true_ranges = []
    for i in range(1, len(closes)):
        true_ranges.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))

    return sum(true_ranges[-period:]) / period

def has_tp_space(candles, direction, entry):
    # Check recent opposing structure before allowing the fixed 1% TP.
    if len(candles) < TP_SPACE_LOOKBACK + 2:
        return False

    # Exclude the current signal candle.
    lookback = candles[-(TP_SPACE_LOOKBACK + 1):-1]
    buffer = TP_SPACE_BUFFER_PERCENT / 100.0

    if direction == "LONG":
        tp = entry * (1 + TP_PERCENT / 100.0)
        resistance = max(c["high"] for c in lookback)
        return resistance >= tp * (1 + buffer)

    tp = entry * (1 - TP_PERCENT / 100.0)
    support = min(c["low"] for c in lookback)
    return support <= tp * (1 - buffer)


def ema_series(values, period):
    if len(values) < period: return []
    m=2.0/(period+1); v=sum(values[:period])/period
    out=[None]*(period-1)+[v]
    for x in values[period:]:
        v=(x-v)*m+v; out.append(v)
    return out

def rsi_series(values, period=14):
    if len(values)<=period: return []
    gains=[]; losses=[]
    for i in range(1,len(values)):
        d=values[i]-values[i-1]; gains.append(max(d,0)); losses.append(max(-d,0))
    ag=sum(gains[:period])/period; al=sum(losses[:period])/period
    out=[None]*period
    out.append(100.0 if al==0 else 100-100/(1+ag/al))
    for i in range(period,len(gains)):
        ag=(ag*(period-1)+gains[i])/period; al=(al*(period-1)+losses[i])/period
        out.append(100.0 if al==0 else 100-100/(1+ag/al))
    return out

def atr_series(candles, period=14):
    if len(candles)<=period: return []
    tr=[]
    for i in range(1,len(candles)):
        h,l,pc=candles[i]['high'],candles[i]['low'],candles[i-1]['close']
        tr.append(max(h-l,abs(h-pc),abs(l-pc)))
    first=sum(tr[:period])/period; out=[None]*period+[first]; v=first
    for x in tr[period:]: v=(v*(period-1)+x)/period; out.append(v)
    return out

def get_1h_data(symbol):
    print(f"Getting {symbol} 1H confirmation candles...")
    r=requests.get(KRAKEN_URL,params={'pair':COINS[symbol],'interval':ENTRY_INTERVAL},timeout=20)
    print(f"{symbol} Kraken 1H:",r.status_code); r.raise_for_status()
    payload=r.json()
    if payload.get('error'): raise RuntimeError(f"{symbol} Kraken 1H API error: {payload['error']}")
    result=payload.get('result',{}); key=next((k for k in result if k!='last'),None)
    if key is None: raise RuntimeError(f"{symbol}: no 1H candle data")
    candles=[{'time':int(x[0]),'open':float(x[1]),'high':float(x[2]),'low':float(x[3]),'close':float(x[4]),'volume':float(x[6])} for x in result[key]]
    candles.sort(key=lambda x:x['time'])
    if len(candles)>1: candles=candles[:-1]
    if len(candles)<210: raise RuntimeError(f"{symbol}: only {len(candles)} closed 1H candles")
    print(f"{symbol}: {len(candles)} closed 1H candles")
    return candles

def market_structure_4h(candles):
    closes=[c['close'] for c in candles]
    e20,e50,e200=(ema_series(closes,p)[-1] for p in (20,50,200))
    recent=candles[-STRUCTURE_LOOKBACK:]; prior=candles[-2*STRUCTURE_LOOKBACK:-STRUCTURE_LOOKBACK]
    rh=max(c['high'] for c in recent); rl=min(c['low'] for c in recent)
    ph=max(c['high'] for c in prior); pl=min(c['low'] for c in prior); close=closes[-1]
    bull=close>e200 and e20>e50 and rh>=ph and rl>=pl
    bear=close<e200 and e20<e50 and rh<=ph and rl<=pl
    return 'LONG' if bull and not bear else 'SHORT' if bear and not bull else 'NONE'

def bos_event(candles,direction):
    cur=candles[-1]; recent=candles[-7:-1]; prior=candles[-13:-7]
    rh=max(c['high'] for c in recent); rl=min(c['low'] for c in recent)
    ph=max(c['high'] for c in prior); pl=min(c['low'] for c in prior)
    return (cur['close']>rh and cur['close']>cur['open']) if direction=='LONG' else (cur['close']<rl and cur['close']<cur['open']) or (cur['close']<pl and cur['close']<cur['open'])

def liquidity_sweep(candles,direction):
    w=candles[-(SWEEP_LOOKBACK+1):-1]; cur=candles[-1]
    if direction=='LONG':
        level=min(c['low'] for c in w); return cur['low']<level and cur['close']>level
    level=max(c['high'] for c in w); return cur['high']>level and cur['close']<level

def retest_confirmation(candles,direction):
    cur=candles[-1]; w=candles[-8:-2]
    if direction=='LONG':
        level=max(c['high'] for c in w); tol=level*RETEST_TOLERANCE_PERCENT/100
        return cur['low']<=level+tol and cur['close']>level
    level=min(c['low'] for c in w); tol=level*RETEST_TOLERANCE_PERCENT/100
    return cur['high']>=level-tol and cur['close']<level

def candle_confirmation(c,direction):
    rng=c['high']-c['low']
    if rng<=0:return False
    body=abs(c['close']-c['open'])/rng
    return (c['close']>c['open'] and body>=0.50) if direction=='LONG' else (c['close']<c['open'] and body>=0.50)

def volume_confirmation(candles):
    avg=sum(c['volume'] for c in candles[-21:-1])/20
    return candles[-1]['volume']>=avg*VOLUME_MULTIPLIER

def range_market(candles):
    closes=[c['close'] for c in candles]; e20=ema_series(closes,20)[-1]; e50=ema_series(closes,50)[-1]; e200=ema_series(closes,200)[-1]
    p=closes[-1]; return abs(e20-e50)/p*100<RANGE_EMA_GAP_PERCENT and abs(e50-e200)/p*100<RANGE_EMA_GAP_PERCENT

def has_tp_space(candles,direction,entry):
    w=candles[-(TP_SPACE_LOOKBACK+1):-1]; b=TP_SPACE_BUFFER_PERCENT/100
    if direction=='LONG': return max(c['high'] for c in w)>=entry*(1+TP_PERCENT/100)*(1+b)
    return min(c['low'] for c in w)<=entry*(1-TP_PERCENT/100)*(1-b)

def calculate_signal(candles, symbol):
    if len(candles)<210:return None
    one=get_1h_data(symbol); closes4=[c['close'] for c in candles]; r4=rsi_series(closes4); a4=atr_series(candles)
    closes1=[c['close'] for c in one]; e20=ema_series(closes1,20)[-1]; e50=ema_series(closes1,50)[-1]; e200=ema_series(closes1,200)[-1]; r1=rsi_series(closes1)
    structure=market_structure_4h(candles); vol=volume_confirmation(candles); rng=range_market(candles); volat=a4[-1]/closes4[-1]>=0.002
    results={}
    for d in ('LONG','SHORT'):
        trend=(structure==d and ((closes1[-1]>e200 and e20>e50) if d=='LONG' else (closes1[-1]<e200 and e20<e50)))
        rsiok=(r4[-1]>50 and r4[-1]<72 and r4[-1]>r4[-2]) if d=='LONG' else (r4[-1]<50 and r4[-1]>28 and r4[-1]<r4[-2])
        bos=bos_event(one,d); sweep=liquidity_sweep(one,d); retest=retest_confirmation(one,d); candle=candle_confirmation(one[-1],d)
        score=2*int(trend)+2*int(bos)+int(rsiok)+int(vol)+int(sweep)+int(retest)+int(candle)+int(volat)
        results[d]=(score,trend,rsiok,vol,bos,sweep,retest,candle,volat)
    print(f"\n===== {symbol} ADVANCED 4H/1H =====\n4H Structure: {structure}\n4H RSI: {r4[-1]:.2f}\nVolume >= {VOLUME_MULTIPLIER}x MA: {vol}\nRange market: {rng}")
    for d,x in results.items(): print(f"{d}: score={x[0]}/11 trend={int(x[1])} rsi={int(x[2])} volume={int(x[3])} BOS={int(x[4])} sweep={int(x[5])} retest={int(x[6])} candle={int(x[7])} volatility={int(x[8])}")
    if rng:return None
    candidates=[]
    for d,x in results.items():
        if x[0]>=MIN_ADVANCED_SCORE and x[1] and x[4] and x[6]:
            entry=one[-1]['close']
            if has_tp_space(one,d,entry): candidates.append({'direction':d,'score':x[0],'price':entry})
            else: print(f"{symbol}: {d} rejected - insufficient TP space")
    return max(candidates,key=lambda x:x['score']) if candidates else None

def create_message(symbol, signal):
    direction = signal["direction"]
    score = signal["score"]
    entry = signal["price"]

    if direction == "LONG":
        tp = entry * (1 + TP_PERCENT / 100)
        sl = entry * (1 - SL_PERCENT / 100)
        return (
            "🚨 SCORE HUNTER 4H 🚨\n\n"
            f"💰 {symbol}USDT\n"
            "📊 🟢 LONG\n"
            f"⭐ Score: {score}/7\n"
            f"💵 Entry: {entry:.8f}\n"
            f"🎯 TP: {tp:.8f} (+1%)\n"
            f"🛑 SL: {sl:.8f} (-0.5%)\n\n"
            "⏱ Timeframe: 4H\n"
            "🕯 Closed candle confirmation\n"
            "⚠️ Manage risk."
        )

    tp = entry * (1 - TP_PERCENT / 100)
    sl = entry * (1 + SL_PERCENT / 100)
    return (
        "🚨 SCORE HUNTER 4H 🚨\n\n"
        f"💰 {symbol}USDT\n"
        "📊 🔴 SHORT\n"
        f"⭐ Score: {score}/7\n"
        f"💵 Entry: {entry:.8f}\n"
        f"🎯 TP: {tp:.8f} (-1%)\n"
        f"🛑 SL: {sl:.8f} (+0.5%)\n\n"
        "⏱ Timeframe: 4H\n"
        "🕯 Closed candle confirmation\n"
        "⚠️ Manage risk."
    )

def main():
    print("🟢 SCORE HUNTER PRO ADVANCED 4H/1H")
    print("🔎 BOS + LIQUIDITY SWEEP + RETEST + CANDLE CONFIRMATION")
    print(f"⭐ Advanced minimum score: {MIN_ADVANCED_SCORE}/11")
    print("🕯 CLOSED CANDLE ONLY - NO MID-CANDLE SIGNAL")
    print("♾️ DAILY SIGNAL LIMIT: DISABLED")
    print("📊 Coins: " + " / ".join(COINS.keys()))
    print("⏱ Timeframe: 4H")
    print(f"⭐ Legacy score threshold retained: {REQUIRED_SCORE}/7")
    print(f"🎯 TP: {TP_PERCENT}%")
    print(f"🛑 SL: {SL_PERCENT}%")
    print(f"📐 TP SPACE FILTER: ON | lookback={TP_SPACE_LOOKBACK} | buffer={TP_SPACE_BUFFER_PERCENT}%")

    state = load_state()

    for symbol in COINS:
        print(f"\n\n========== {symbol} ==========")

        try:
            candles = get_4h_data(symbol)
            latest_candle_time = candles[-1]["time"]
            coin_state = state.get(symbol, {})
            previous_candle_time = coin_state.get("last_checked_candle")

            print(
                f"{symbol} latest CLOSED 4H candle: "
                f"{latest_candle_time}"
            )

            if previous_candle_time == latest_candle_time:
                print(f"{symbol}: No new 4H candle.")
                continue

            coin_state["last_checked_candle"] = latest_candle_time

            signal = calculate_signal(candles, symbol)

            if signal is None:
                print(f"{symbol}: No valid signal.")
                state[symbol] = coin_state
                save_state(state)
                continue

            message = create_message(symbol, signal)
            send_telegram(message)

            coin_state["last_signal"] = signal
            coin_state["signal_candle"] = latest_candle_time
            coin_state["signal_time"] = int(
                datetime.now(timezone.utc).timestamp()
            )

            state[symbol] = coin_state
            save_state(state)

            print(f"🚨 {symbol}: SIGNAL SENT")

        except Exception as e:
            print(
                f"❌ {symbol} ERROR: "
                f"{type(e).__name__}: {e}"
            )
            continue

    save_state(state)
    print("\n✅ ALL COINS SCANNED")

if __name__ == "__main__":
    main()
