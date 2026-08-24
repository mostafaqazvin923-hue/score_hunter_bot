import requests
import time
from datetime import datetime, timezone, timedelta

# ============================================================
# SCORE HUNTER PRO v8.16 (REALISTIC REAL-WORLD EDITION)
# ============================================================

COINS = {
    "ETH": "eth_usdt",
    "SOL": "sol_usdt",
    "XRP": "xrp_usdt",
    "BTC": "btc_usdt",
    "ADA": "ada_usdt",
    "LINK": "link_usdt",
    "DOGE": "doge_usdt",
}

HISTORY_DAYS = 365
OOS_DAYS = 90
WARMUP_DAYS = 60

INTERVAL_1H = 60
INTERVAL_4H = 240

SECONDS_1H = 3600
SECONDS_4H = 14400

EMA20 = 20
EMA50 = 50
EMA200 = 200

RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14

ADX_MIN = 20.0

STRUCTURE_LOOKBACK = 5
REVERSAL_LOOKBACK = 6

SL_ATR = 1.50
STRUCTURE_BUFFER_ATR = 0.10
MAX_SL_ATR = 3.50

TP_R_MULTIPLE = 2.0
MIN_RR = 1.50

MIN_BODY_RATIO = 0.55
MIN_CLOSE_LOCATION = 0.70

REQUEST_TIMEOUT = 20
MAX_RETRIES = 3
LBANK_MAX_SIZE = 2000

# REALISTIC TRADING COSTS
FEE_TAKER = 0.0006      # 0.06% Taker Fee
SLIPPAGE_EST = 0.0002   # 0.02% Estimated Slippage

LBANK_URL = "https://api.lbkex.com/v2/kline.do"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json"
})

def utc_now():
    return datetime.now(timezone.utc)

def timestamp_seconds(dt):
    return int(dt.timestamp())

def safe_get(url, params=None, retries=MAX_RETRIES):
    for attempt in range(1, retries + 1):
        try:
            res = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
            res.raise_for_status()
            return res
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"API Error: {exc}")
            time.sleep(1.0)

def normalize_candle(t, o, h, l, c, v):
    try:
        t = float(t)
        if t > 10_000_000_000: t /= 1000.0
        o, h, l, c = float(o), float(h), float(l), float(c)
        v = float(v) if v else 0.0
        if o <= 0 or h <= 0 or l <= 0 or c <= 0 or h < l: return None
        return {"time": int(t), "open": o, "high": h, "low": l, "close": c, "volume": v}
    except Exception:
        return None

def deduplicate(candles):
    unique = {c["time"]: c for c in candles if c is not None}
    res = list(unique.values())
    res.sort(key=lambda x: x["time"])
    return res

def lbank_request_page(symbol, interval, cursor):
    lbank_type = "hour1" if interval == INTERVAL_1H else "hour4"
    params = {"symbol": COINS[symbol], "size": LBANK_MAX_SIZE, "type": lbank_type, "time": int(cursor)}
    res = safe_get(LBANK_URL, params=params).json()
    raw = res.get("data", [])
    candles = []
    for r in raw:
        c = normalize_candle(r[0], r[1], r[2], r[3], r[4], r[5])
        if c: candles.append(c)
    return deduplicate(candles)

def get_lbank_klines(symbol, interval, start_dt, end_dt):
    req_start = timestamp_seconds(start_dt)
    req_end = timestamp_seconds(end_dt)
    all_c = []
    cursor = req_start
    while cursor <= req_end:
        try:
            page = lbank_request_page(symbol, interval, cursor)
        except Exception:
            break
        if not page: break
        page_newest = max(c["time"] for c in page)
        before = len(all_c)
        all_c.extend(page)
        all_c = deduplicate(all_c)
        if len(all_c) == before or page_newest >= req_end: break
        cursor = page_newest + (interval * 60)
        time.sleep(0.05)
    return [c for c in all_c if req_start <= c["time"] <= req_end]

def load_market_data(symbol, start_dt, end_dt):
    try:
        c4 = get_lbank_klines(symbol, INTERVAL_4H, start_dt, end_dt)
        c1 = get_lbank_klines(symbol, INTERVAL_1H, start_dt, end_dt)
        if c4 and c1 and len(c4) > 250 and len(c1) > 250:
            return c4, c1, "LBANK"
        return None, None, "FAILED"
    except Exception:
        return None, None, "FAILED"

def ema(values, period):
    if len(values) < period: return None
    val = sum(values[:period]) / period
    mult = 2.0 / (period + 1)
    for p in values[period:]:
        val = ((p - val) * mult) + val
    return val

def atr(candles, period=14):
    if len(candles) < (period + 1): return None
    trs = []
    for i in range(1, len(candles)):
        c, p = candles[i], candles[i - 1]
        tr = max(c["high"] - c["low"], abs(c["high"] - p["close"]), abs(c["low"] - p["close"]))
        trs.append(tr)
    val = sum(trs[:period]) / period
    for tr in trs[period:]:
        val = ((val * (period - 1)) + tr) / period
    return val

def rsi(candles, period=14):
    if len(candles) < (period + 1): return None
    closes = [c["close"] for c in candles]
    gains, losses = [], []
    for i in range(1, len(closes)):
        chg = closes[i] - closes[i - 1]
        gains.append(max(chg, 0.0))
        losses.append(max(-chg, 0.0))
    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
    if avg_l == 0: return 100.0
    return 100.0 - (100.0 / (1.0 + (avg_g / avg_l)))

def adx(candles, period=14):
    if len(candles) < (period * 2 + 5): return None
    tr_l, p_dm_l, m_dm_l = [], [], []
    for i in range(1, len(candles)):
        c, p = candles[i], candles[i - 1]
        tr = max(c["high"] - c["low"], abs(c["high"] - p["close"]), abs(c["low"] - p["close"]))
        up = c["high"] - p["high"]
        down = p["low"] - c["low"]
        tr_l.append(tr)
        p_dm_l.append(up if up > down and up > 0 else 0.0)
        m_dm_l.append(down if down > up and down > 0 else 0.0)
    atr_v, p_v, m_v = sum(tr_l[:period])/period, sum(p_dm_l[:period])/period, sum(m_dm_l[:period])/period
    dx_v = []
    for i in range(period, len(tr_l)):
        atr_v = (atr_v * (period - 1) + tr_l[i]) / period
        p_v = (p_v * (period - 1) + p_dm_l[i]) / period
        m_v = (m_v * (period - 1) + m_dm_l[i]) / period
        if atr_v <= 0: continue
        p_di, m_di = 100.0 * p_v / atr_v, 100.0 * m_v / atr_v
        den = p_di + m_di
        dx_v.append(100.0 * abs(p_di - m_di) / den if den > 0 else 0.0)
    if len(dx_v) < period: return None
    adx_v = sum(dx_v[:period]) / period
    for v in dx_v[period:]:
        adx_v = (adx_v * (period - 1) + v) / period
    return adx_v

def get_4h_direction(candles):
    if len(candles) < EMA200: return None
    closes = [c["close"] for c in candles]
    c = closes[-1]
    e20, e50, e200 = ema(closes, EMA20), ema(closes, EMA50), ema(closes, EMA200)
    if not (e20 and e50 and e200): return None
    if c > e200 and e20 > e50 > e200: return "LONG"
    if c < e200 and e20 < e50 < e200: return "SHORT"
    return None

def get_closed_4h_for_entry(candles_4h, entry_candle):
    entry_close_time = entry_candle["time"] + SECONDS_1H
    return [c for c in candles_4h if (c["time"] + SECONDS_4H) <= entry_close_time]

# VOLUME FILTER APPLIED
def detect_breakout_long(candles):
    if len(candles) < (STRUCTURE_LOOKBACK + 21): return False
    cur = candles[-1]
    prev = candles[-STRUCTURE_LOOKBACK - 1:-1]
    res = max(c["high"] for c in prev)
    if cur["close"] <= res or cur["close"] <= cur["open"]: return False
    rng = cur["high"] - cur["low"]
    if rng <= 0: return False
    if (abs(cur["close"] - cur["open"]) / rng) < MIN_BODY_RATIO: return False
    if ((cur["close"] - cur["low"]) / rng) < MIN_CLOSE_LOCATION: return False
    
    vol_sma = sum(c["volume"] for c in candles[-21:-1]) / 20.0
    return cur["volume"] > vol_sma

def detect_breakout_short(candles):
    if len(candles) < (STRUCTURE_LOOKBACK + 21): return False
    cur = candles[-1]
    prev = candles[-STRUCTURE_LOOKBACK - 1:-1]
    sup = min(c["low"] for c in prev)
    if cur["close"] >= sup or cur["close"] >= cur["open"]: return False
    rng = cur["high"] - cur["low"]
    if rng <= 0: return False
    if (abs(cur["close"] - cur["open"]) / rng) < MIN_BODY_RATIO: return False
    if ((cur["high"] - cur["close"]) / rng) < MIN_CLOSE_LOCATION: return False
    
    vol_sma = sum(c["volume"] for c in candles[-21:-1]) / 20.0
    return cur["volume"] > vol_sma

def detect_reversal_long(candles, trend, adx_v, rsi_v):
    if trend != "SHORT" or len(candles) < 60: return False
    cur = candles[-1]
    prev = candles[-REVERSAL_LOOKBACK - 1:-1]
    if cur["close"] <= max(c["high"] for c in prev) or cur["close"] <= cur["open"]: return False
    closes = [c["close"] for c in candles]
    e20, e50 = ema(closes, EMA20), ema(closes, EMA50)
    if not (e20 and e50) or e20 <= e50 or cur["close"] <= e50: return False
    return adx_v >= ADX_MIN and rsi_v >= 50

def detect_reversal_short(candles, trend, adx_v, rsi_v):
    if trend != "LONG" or len(candles) < 60: return False
    cur = candles[-1]
    prev = candles[-REVERSAL_LOOKBACK - 1:-1]
    if cur["close"] >= min(c["low"] for c in prev) or cur["close"] >= cur["open"]: return False
    closes = [c["close"] for c in candles]
    e20, e50 = ema(closes, EMA20), ema(closes, EMA50)
    if not (e20 and e50) or e20 >= e50 or cur["close"] >= e50: return False
    return adx_v >= ADX_MIN and rsi_v <= 50

def calculate_long_levels(candles, entry, atr_v):
    if not atr_v or atr_v <= 0: return None
    prev = candles[-STRUCTURE_LOOKBACK - 1:-1]
    sl = min(entry - atr_v * SL_ATR, min(c["low"] for c in prev) - atr_v * STRUCTURE_BUFFER_ATR)
    risk = entry - sl
    if risk <= 0 or (risk / atr_v) > MAX_SL_ATR: return None
    tp = entry + risk * TP_R_MULTIPLE
    return {"tp": tp, "sl": sl, "risk": risk, "rr": (tp - entry) / risk} if ((tp - entry) / risk) >= MIN_RR else None

def calculate_short_levels(candles, entry, atr_v):
    if not atr_v or atr_v <= 0: return None
    prev = candles[-STRUCTURE_LOOKBACK - 1:-1]
    sl = max(entry + atr_v * SL_ATR, max(c["high"] for c in prev) + atr_v * STRUCTURE_BUFFER_ATR)
    risk = sl - entry
    if risk <= 0 or (risk / atr_v) > MAX_SL_ATR: return None
    tp = entry - risk * TP_R_MULTIPLE
    return {"tp": tp, "sl": sl, "risk": risk, "rr": (entry - tp) / risk} if ((entry - tp) / risk) >= MIN_RR else None

def analyze_at_index(candles_4h, candles_1h):
    if len(candles_1h) < (EMA200 + 10) or len(candles_4h) < EMA200: return None
    cur = candles_1h[-1]
    entry = cur["close"]
    trend = get_4h_direction(candles_4h)
    if not trend: return None
    
    atr_v, rsi_v, adx_v = atr(candles_1h, ATR_PERIOD), rsi(candles_1h, RSI_PERIOD), adx(candles_1h, ADX_PERIOD)
    if not (atr_v and rsi_v and adx_v) or adx_v < ADX_MIN: return None

    if trend == "LONG" and 50 <= rsi_v <= 85 and detect_breakout_long(candles_1h):
        levels = calculate_long_levels(candles_1h, entry, atr_v)
        if levels: return {"direction": "LONG", "entry_time": cur["time"], "entry": entry, "tp": levels["tp"], "sl": levels["sl"]}

    if trend == "SHORT" and 15 <= rsi_v <= 50 and detect_breakout_short(candles_1h):
        levels = calculate_short_levels(candles_1h, entry, atr_v)
        if levels: return {"direction": "SHORT", "entry_time": cur["time"], "entry": entry, "tp": levels["tp"], "sl": levels["sl"]}

    if detect_reversal_long(candles_1h, trend, adx_v, rsi_v):
        levels = calculate_long_levels(candles_1h, entry, atr_v)
        if levels: return {"direction": "LONG", "entry_time": cur["time"], "entry": entry, "tp": levels["tp"], "sl": levels["sl"]}

    if detect_reversal_short(candles_1h, trend, adx_v, rsi_v):
        levels = calculate_short_levels(candles_1h, entry, atr_v)
        if levels: return {"direction": "SHORT", "entry_time": cur["time"], "entry": entry, "tp": levels["tp"], "sl": levels["sl"]}

    return None

# REALISTIC RESULT CHECK: NEXT CANDLE OPEN + FEE/SLIPPAGE DEDUCTION
def check_trade_result(candles, entry_index, signal):
    direction = signal["direction"]
    tp, sl = signal["tp"], signal["sl"]

    if entry_index + 1 >= len(candles): return None, None, 0.0

    actual_entry = candles[entry_index + 1]["open"]
    risk_dist = (actual_entry - sl) if direction == "LONG" else (sl - actual_entry)
    if risk_dist <= 0: return None, None, 0.0

    total_cost_pct = (FEE_TAKER * 2) + SLIPPAGE_EST
    fee_in_r = (total_cost_pct * actual_entry) / risk_dist

    for i in range(entry_index + 1, len(candles)):
        c = candles[i]
        hit_tp = (c["high"] >= tp) if direction == "LONG" else (c["low"] <= tp)
        hit_sl = (c["low"] <= sl) if direction == "LONG" else (c["high"] >= sl)

        if hit_tp and hit_sl: return "SL", i, -1.0 - fee_in_r
        if hit_sl: return "SL", i, -1.0 - fee_in_r
        if hit_tp: return "TP", i, TP_R_MULTIPLE - fee_in_r

    return None, None, 0.0

def backtest_coin(symbol, candles_4h, candles_1h, strategy_start, oos_start):
    trades = []
    strat_start_ts = timestamp_seconds(strategy_start)
    oos_start_ts = timestamp_seconds(oos_start)
    i = 0

    while i < len(candles_1h):
        entry_c = candles_1h[i]
        if entry_c["time"] < strat_start_ts:
            i += 1
            continue

        usable_4h = get_closed_4h_for_entry(candles_4h, entry_c)
        if len(usable_4h) < EMA200:
            i += 1
            continue

        signal = analyze_at_index(usable_4h, candles_1h[:i + 1])
        if signal is None:
            i += 1
            continue

        res, exit_idx, net_r = check_trade_result(candles_1h, i, signal)
        if res is None: break

        trades.append({
            "symbol": symbol, "direction": signal["direction"],
            "entry_time": signal["entry_time"], "result": res, "R": net_r,
            "period": "OOS" if signal["entry_time"] >= oos_start_ts else "IS"
        })
        i = exit_idx + 1

    return trades

def calculate_stats(trades):
    if not trades: return {"total": 0, "win_rate": 0.0, "net_r": 0.0, "profit_factor": 0.0}
    total = len(trades)
    wins = sum(1 for t in trades if t["result"] == "TP")
    r_vals = [t["R"] for t in trades]
    net_r = sum(r_vals)
    gross_p = sum(r for r in r_vals if r > 0)
    gross_l = abs(sum(r for r in r_vals if r < 0))
    pf = (gross_p / gross_l) if gross_l > 0 else 0.0
    return {"total": total, "win_rate": (wins / total) * 100.0, "net_r": net_r, "profit_factor": pf}

def main():
    end_dt = utc_now()
    strategy_start = end_dt - timedelta(days=HISTORY_DAYS)
    data_start = strategy_start - timedelta(days=WARMUP_DAYS)
    oos_start = end_dt - timedelta(days=OOS_DAYS)

    all_trades = []
    for symbol in COINS:
        c4, c1, src = load_market_data(symbol, data_start, end_dt)
        if c4 and c1:
            all_trades.extend(backtest_coin(symbol, c4, c1, strategy_start, oos_start))

    is_stats = calculate_stats([t for t in all_trades if t["period"] == "IS"])
    oos_stats = calculate_stats([t for t in all_trades if t["period"] == "OOS"])

    print(f"IS Win Rate  : {is_stats['win_rate']:.2f}% | IS Net R  : {is_stats['net_r']:+.2f}R | IS PF  : {is_stats['profit_factor']:.3f}")
    print(f"OOS Win Rate : {oos_stats['win_rate']:.2f}% | OOS Net R : {oos_stats['net_r']:+.2f}R | OOS PF : {oos_stats['profit_factor']:.3f}")

if __name__ == "__main__":
    main()
