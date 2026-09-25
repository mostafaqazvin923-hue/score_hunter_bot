#!/usr/bin/env python3
"""
HUNTER-V2-XT-STAGE1 — TREND PULLBACK / VOLATILITY EXPANSION

Stage-1 strategy logic preserved from HUNTER-V2-STAGE1.
DATA PIPELINE replaced with the validated XT Futures collector:
- https://fapi.xt.com
- 15m candles
- backward pagination, 1500 candles/request
- 365d test window + 35d warmup
- 14 XT USDT-M symbols
- no forward-fill; gaps are preserved

Execution: next 15m open after a completed signal candle.
RR = 1:2, no timeout, no trailing, same-candle SL+TP => LOSS.
"""

from __future__ import annotations

import math
from pathlib import Path
from datetime import datetime, timedelta, timezone

import time
import numpy as np
import pandas as pd
import requests

BASE = "https://fapi.xt.com"
SYMBOL_LIST_URL = f"{BASE}/future/market/v1/public/symbol/list"
KLINE_URL = f"{BASE}/future/market/v1/public/q/kline"

INTERVAL = "15m"
LIMIT = 1500
TIMEOUT = 20
RETRIES = 5
SLEEP = 0.12

SYMBOLS = [
    "BTC", "ETH", "SOL", "SUI", "AVAX", "NEAR", "ADA",
    "BNB", "APT", "CRV", "ONDO", "PENDLE", "ICP", "WIF",
]

DATA_DIR = Path("data/xt_futures_15m")
OUT_DIR = DATA_DIR / "backtest_v2_stage1"

INITIAL_CAPITAL = 1000.0
TRADE_MARGIN = 100.0
LEVERAGE = 50.0
FEE_RATE = 0.0007
SLIPPAGE = 0.0003
RR = 2.0
MAX_OPEN_POSITIONS = 3
MAX_STRUCTURAL_RISK = 0.025
MAX_LOSS_STREAK_PAUSE = 4
PAUSE_BARS = 16
DAYS = 365
WARMUP_DAYS = 35

# Stage-1 signal parameters — preserved from V2.
EMA_FAST_4H = 50
EMA_SLOW_4H = 200
ADX_LEN = 14
EMA_PULL_FAST = 20
EMA_PULL_SLOW = 50
PULLBACK_LOOKBACK_H = 6
PULLBACK_ATR_BUFFER = 0.20
EXPANSION_ATR_MULT = 1.05
EXPANSION_BODY_RATIO = 0.60
VOLUME_MULT = 1.05
STOP_ATR_BUFFER = 0.20

# ============================================================
# GENERIC HELPERS
# ============================================================

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def api_json(session, url, params=None):
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = session.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            payload = r.json()
            if not isinstance(payload, dict):
                raise RuntimeError(f"unexpected response type: {type(payload).__name__}")
            rc = payload.get("returnCode")
            if rc not in (None, 0, "0"):
                raise RuntimeError(
                    f"XT API error: {payload.get('error') or payload.get('msgInfo') or payload}"
                )
            return payload
        except Exception as exc:
            last = exc
            if attempt < RETRIES:
                time.sleep(min(2 * attempt, 5))
    raise RuntimeError(f"API request failed after {RETRIES} attempts: {last}")


def extract_list(payload):
    result = payload.get("result")
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for key in ("items", "list", "data", "symbols"):
            if isinstance(result.get(key), list):
                return result[key]
    raise RuntimeError("Could not find symbol list in XT response")


def discover_symbols(session):
    items = extract_list(api_json(session, SYMBOL_LIST_URL))
    found = {}
    for item in items:
        if isinstance(item, str):
            raw = item
        elif isinstance(item, dict):
            raw = item.get("symbol") or item.get("s") or item.get("name") or item.get("pair")
        else:
            continue
        if raw:
            found[str(raw).strip().upper()] = item
    if not found:
        raise RuntimeError("XT symbol list returned zero usable symbols")
    return found


def resolve_symbol(asset, discovered):
    wanted = asset.upper()
    candidates = [f"{wanted}_USDT", f"{wanted}/USDT", f"{wanted}-USDT", wanted]
    for c in candidates:
        if c.upper() in discovered:
            return c.upper()

    def norm(x):
        return str(x).upper().replace("/", "_").replace("-", "_")

    matches = [x for x in discovered if norm(x) == f"{wanted}_USDT"]
    return matches[0] if len(matches) == 1 else None


def normalize_rows(rows, symbol):
    records = []
    for row in rows:
        if isinstance(row, dict):
            required = ("t", "o", "h", "l", "c", "a")
            if not all(k in row for k in required):
                raise RuntimeError(f"{symbol}: malformed dict kline")
            records.append({
                "Timestamp": int(row["t"]),
                "Open": float(row["o"]),
                "High": float(row["h"]),
                "Low": float(row["l"]),
                "Close": float(row["c"]),
                "Volume": float(row["a"]),
            })
        elif isinstance(row, (list, tuple)) and len(row) >= 6:
            records.append({
                "Timestamp": int(row[0]),
                "Open": float(row[1]),
                "High": float(row[2]),
                "Low": float(row[3]),
                "Close": float(row[4]),
                "Volume": float(row[5]),
            })
        else:
            raise RuntimeError(f"{symbol}: unsupported kline row")

    df = pd.DataFrame(records)
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    return df


def fetch_batch(session, symbol, start_ms, end_ms):
    params = {
        "symbol": symbol.strip().lower(),
        "interval": INTERVAL,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": LIMIT,
    }
    payload = api_json(session, KLINE_URL, params)
    result = payload.get("result")
    if not isinstance(result, list):
        raise RuntimeError("XT kline result is not a list")
    return normalize_rows(result, symbol)


def audit(df, symbol):
    if df.empty:
        raise RuntimeError(f"{symbol}: empty dataset")

    bad = (
        (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
        | (df["Volume"] < 0)
        | (df["High"] < df[["Open", "Close", "Low"]].max(axis=1))
        | (df["Low"] > df[["Open", "Close", "High"]].min(axis=1))
    )
    if bad.any():
        raise RuntimeError(f"{symbol}: {int(bad.sum())} invalid OHLCV rows")

    diffs = df["Date"].diff().dropna()
    gaps = diffs[diffs > pd.Timedelta(minutes=15)]
    return {
        "rows": int(len(df)),
        "start": df["Date"].iloc[0].isoformat(),
        "end": df["Date"].iloc[-1].isoformat(),
        "duplicates": int(df["Date"].duplicated().sum()),
        "gaps": int(len(gaps)),
        "max_gap_minutes": int(gaps.max().total_seconds() / 60) if len(gaps) else 15,
    }


def download_symbol(session, symbol, start_dt, end_dt):
    start_ms = to_ms(start_dt)
    cursor_end = to_ms(end_dt)
    batches = []
    calls = 0

    while cursor_end >= start_ms:
        batch = fetch_batch(session, symbol, start_ms, cursor_end)
        calls += 1
        if batch.empty:
            break

        batch = batch.sort_values("Date").reset_index(drop=True)
        first = int(batch["Timestamp"].iloc[0])
        last = int(batch["Timestamp"].iloc[-1])

        if first < start_ms:
            batch = batch[batch["Timestamp"] >= start_ms].copy()
            if batch.empty:
                break
            first = int(batch["Timestamp"].iloc[0])
            last = int(batch["Timestamp"].iloc[-1])

        if last > cursor_end:
            raise RuntimeError(f"{symbol}: API returned candle beyond requested end")

        batches.append(batch)
        print(
            f"  {symbol}: request={calls:02d} rows={len(batch):4d} "
            f"{batch['Date'].iloc[0]} -> {batch['Date'].iloc[-1]}"
        )

        if first <= start_ms:
            break

        next_end = first - 1
        if next_end >= cursor_end:
            raise RuntimeError(f"{symbol}: pagination made no progress")
        cursor_end = next_end

        if calls > 1000:
            raise RuntimeError(f"{symbol}: pagination safety stop")
        time.sleep(SLEEP)

    if not batches:
        raise RuntimeError(f"{symbol}: zero historical candles")

    df = pd.concat(batches, ignore_index=True)
    df = df.drop_duplicates("Date", keep="last").sort_values("Date").reset_index(drop=True)
    df = df[(df["Date"] >= pd.Timestamp(start_dt)) & (df["Date"] <= pd.Timestamp(end_dt))].copy()

    current = pd.Timestamp.now(tz="UTC")
    if not df.empty and df["Date"].iloc[-1] + pd.Timedelta(minutes=15) > current:
        df = df.iloc[:-1].copy()

    rep = audit(df, symbol)
    rep["api_requests"] = calls
    return df, rep



def resample_ohlcv(df, rule):
    return df.resample(rule, label='right', closed='right').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()


def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def atr(df, n=14):
    prev = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev).abs(),
        (df['Low'] - prev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def adx_di(df, n=14):
    up = df['High'].diff()
    down = -df['Low'].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift(1)).abs(),
        (df['Low'] - df['Close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    atrv = tr.rolling(n, min_periods=n).mean()
    pdi = 100 * plus_dm.rolling(n, min_periods=n).mean() / atrv.replace(0, np.nan)
    mdi = 100 * minus_dm.rolling(n, min_periods=n).mean() / atrv.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adxv = dx.rolling(n, min_periods=n).mean()
    return adxv, pdi, mdi


def prepare(df):
    m15 = df.copy()
    m15['ATR'] = atr(m15, 14)
    m15['Body'] = (m15['Close'] - m15['Open']).abs()
    m15['Range'] = (m15['High'] - m15['Low']).replace(0, np.nan)
    m15['BodyRatio'] = m15['Body'] / m15['Range']
    m15['AvgRange'] = m15['Range'].rolling(20, min_periods=20).mean()
    m15['VolMed'] = m15['Volume'].rolling(20, min_periods=20).median()
    m15['PrevHigh'] = m15['High'].shift(1)
    m15['PrevLow'] = m15['Low'].shift(1)

    h1 = resample_ohlcv(df, '1h')
    h1['EMA20'] = ema(h1['Close'], EMA_PULL_FAST)
    h1['EMA50'] = ema(h1['Close'], EMA_PULL_SLOW)
    h1['ATR'] = atr(h1, 14)
    h1['ADX'], h1['DIp'], h1['DIm'] = adx_di(h1, ADX_LEN)
    h1['PullLow'] = h1['Low'].rolling(PULLBACK_LOOKBACK_H, min_periods=PULLBACK_LOOKBACK_H).min()
    h1['PullHigh'] = h1['High'].rolling(PULLBACK_LOOKBACK_H, min_periods=PULLBACK_LOOKBACK_H).max()
    h1['TouchedLong'] = h1['Low'] <= (h1['EMA20'] + PULLBACK_ATR_BUFFER * h1['ATR'])
    h1['TouchedShort'] = h1['High'] >= (h1['EMA20'] - PULLBACK_ATR_BUFFER * h1['ATR'])
    h1['LongReclaim'] = (h1['Close'] > h1['EMA20']) & (h1['Close'] > h1['Open'])
    h1['ShortReject'] = (h1['Close'] < h1['EMA20']) & (h1['Close'] < h1['Open'])
    h1['RecentLongTouch'] = h1['TouchedLong'].rolling(4, min_periods=1).max().astype(bool)
    h1['RecentShortTouch'] = h1['TouchedShort'].rolling(4, min_periods=1).max().astype(bool)
    h1['LongPullbackLow'] = h1['Low'].rolling(4, min_periods=1).min()
    h1['ShortPullbackHigh'] = h1['High'].rolling(4, min_periods=1).max()

    h4 = resample_ohlcv(df, '4h')
    h4['EMA50'] = ema(h4['Close'], EMA_FAST_4H)
    h4['EMA200'] = ema(h4['Close'], EMA_SLOW_4H)
    h4['ATR'] = atr(h4, 14)
    h4['ADX'], h4['DIp'], h4['DIm'] = adx_di(h4, ADX_LEN)
    h4['EMA50Slope'] = h4['EMA50'] - h4['EMA50'].shift(3)
    h4['Bull'] = (h4['Close'] > h4['EMA200']) & (h4['EMA50'] > h4['EMA200']) & (h4['EMA50Slope'] > 0) & (h4['DIp'] > h4['DIm'])
    h4['Bear'] = (h4['Close'] < h4['EMA200']) & (h4['EMA50'] < h4['EMA200']) & (h4['EMA50Slope'] < 0) & (h4['DIm'] > h4['DIp'])

    return m15, h1, h4


def latest_completed(htf, ts):
    # Resampled candles are timestamped at their right edge. At a 15m
    # candle opening at ts, only htf timestamps strictly before ts are complete.
    x = htf.loc[htf.index < ts]
    if x.empty:
        return None
    return x.iloc[-1]


def h1_context(h1, ts):
    x = h1.loc[h1.index < ts]
    if len(x) < 10:
        return None
    return x.iloc[-1], x.iloc[-min(4, len(x)):]


def signal_at(m15, h1, h4, i):
    ts = m15.index[i]
    r = m15.iloc[i]
    if not np.isfinite(r['ATR']) or r['ATR'] <= 0:
        return None

    regime = latest_completed(h4, ts)
    hc = h1_context(h1, ts)
    if regime is None or hc is None:
        return None
    hr, recent_h = hc

    # Stage-1 deliberately uses a compact core: trend + pullback/reclaim + expansion.
    bull = bool(regime.get('Bull', False))
    bear = bool(regime.get('Bear', False))
    if not bull and not bear:
        return None

    if not np.isfinite(hr['EMA20']) or not np.isfinite(hr['EMA50']) or not np.isfinite(hr['ATR']):
        return None

    vol_ok = np.isfinite(r['VolMed']) and r['Volume'] >= r['VolMed'] * VOLUME_MULT
    expansion = (r['Range'] >= r['ATR'] * EXPANSION_ATR_MULT and
                 r['BodyRatio'] >= EXPANSION_BODY_RATIO and
                 r['Range'] >= r['AvgRange'] * 1.05 if np.isfinite(r['AvgRange']) else False)

    # Trigger is a closed-candle break of the immediately preceding candle.
    long_trigger = r['Close'] > r['PrevHigh'] and r['Close'] > r['Open']
    short_trigger = r['Close'] < r['PrevLow'] and r['Close'] < r['Open']

    if bull:
        # Pullback must have occurred recently and the completed 1H candle must reclaim EMA20.
        pullback = bool(hr['RecentLongTouch']) and bool(hr['LongReclaim']) and hr['Close'] >= hr['EMA50'] * 0.985
        if pullback and long_trigger and expansion and vol_ok:
            stop_anchor = float(hr['LongPullbackLow'])
            sl = stop_anchor - STOP_ATR_BUFFER * float(r['ATR'])
            entry_ref = float(r['Close'])
            if sl < entry_ref:
                return {'side': 'LONG', 'sl_ref': sl, 'signal_close': entry_ref}

    if bear:
        pullback = bool(hr['RecentShortTouch']) and bool(hr['ShortReject']) and hr['Close'] <= hr['EMA50'] * 1.015
        if pullback and short_trigger and expansion and vol_ok:
            stop_anchor = float(hr['ShortPullbackHigh'])
            sl = stop_anchor + STOP_ATR_BUFFER * float(r['ATR'])
            entry_ref = float(r['Close'])
            if sl > entry_ref:
                return {'side': 'SHORT', 'sl_ref': sl, 'signal_close': entry_ref}

    return None


def cluster_of(symbol):
    for c, syms in CLUSTERS.items():
        if symbol in syms:
            return c
    return symbol


def fee(notional):
    return notional * FEE_RATE * 2.0


def make_trade(symbol, side, entry, sl, signal_ts, entry_ts):
    risk_pct = abs(entry - sl) / entry if entry else np.inf
    if risk_pct <= 0 or risk_pct > MAX_STRUCTURAL_RISK:
        return None
    risk_dist = abs(entry - sl)
    if side == 'LONG':
        tp = entry + RR * risk_dist
    else:
        tp = entry - RR * risk_dist
    notional = TRADE_MARGIN * LEVERAGE
    return {
        'symbol': symbol, 'side': side, 'signal_ts': signal_ts, 'entry_ts': entry_ts,
        'entry': entry, 'sl': sl, 'tp': tp, 'notional': notional,
        'fee': fee(notional), 'cluster': cluster_of(symbol),
        'status': 'OPEN'
    }



def main():
    print("=" * 90)
    print("HUNTER-V2-XT-STAGE1 — TREND PULLBACK / VOLATILITY EXPANSION")
    print("VALIDATED XT DATA PIPELINE / STRICT NO-LOOKAHEAD BACKTEST")
    print("=" * 90)

    now = datetime.now(timezone.utc)
    test_end = pd.Timestamp(now).tz_convert("UTC")
    target_start = test_end - pd.Timedelta(days=DAYS)
    download_start = test_end - pd.Timedelta(days=DAYS + WARMUP_DAYS)

    data = {}
    valid = []
    session = requests.Session()
    discovered = discover_symbols(session)

    print("XT data collection: 14 symbols / 365d + 35d warmup")
    for asset in SYMBOLS:
        print(f"\n{asset}: downloading XT Futures 15m ...")
        try:
            xt_symbol = resolve_symbol(asset, discovered)
            if not xt_symbol:
                raise RuntimeError(f"XT symbol not found: {asset}")
            df, rep = download_symbol(session, xt_symbol, download_start.to_pydatetime(), test_end.to_pydatetime())
            if len(df) < 20000:
                raise RuntimeError(f"{asset}: insufficient XT rows: {len(df)}")
            # Save the exact verified dataset for reproducibility.
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            out = df[["Timestamp", "Open", "High", "Low", "Close", "Volume", "Date"]].copy()
            out.to_csv(DATA_DIR / f"{asset}_USDT_15m.csv", index=False)
            print(f"{asset:8s} rows={rep['rows']} gaps={rep['gaps']} requests={rep['api_requests']}")
            # Strategy computations use all downloaded warmup data; the test loop starts at target_start.
            m15, h1, h4 = prepare(df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]])
            data[asset] = {"m15": m15, "h1": h1, "h4": h4}
            valid.append(asset)
        except Exception as exc:
            print(f"{asset:8s} !! {exc}")

    print(f"\nVerified XT symbols: {len(valid)} / {len(SYMBOLS)}")
    if not valid:
        raise RuntimeError("No valid XT datasets available; aborting backtest.")

    print("\nBuilding Stage-1 candidate signals ...")
    signals = []
    raw_signal_count = 0
    for symbol in valid:
        d = data[symbol]
        m15 = d['m15']
        start_i = int(m15.index.searchsorted(target_start))
        for i in range(max(1, start_i), len(m15) - 1):
            s = signal_at(m15, d['h1'], d['h4'], i)
            if not s:
                continue
            raw_signal_count += 1
            entry_ts = m15.index[i + 1]
            raw_open = float(m15.iloc[i + 1]['Open'])
            entry = raw_open * (1 + SLIPPAGE) if s['side'] == 'LONG' else raw_open * (1 - SLIPPAGE)
            tr = make_trade(symbol, s['side'], entry, s['sl_ref'], m15.index[i], entry_ts)
            if tr:
                signals.append(tr)

    signals.sort(key=lambda x: (x['entry_ts'], x['symbol']))
    print(f"Raw signal candidates: {raw_signal_count}")
    print(f"Risk-valid candidate signals: {len(signals)}")

    open_positions = []
    closed = []
    equity = INITIAL_CAPITAL
    peak_equity = INITIAL_CAPITAL
    max_dd = 0.0
    pause_until = None

    unique_times = sorted(set(s['entry_ts'] for s in signals))
    by_time = {}
    for s in signals:
        by_time.setdefault(s['entry_ts'], []).append(s)

    def candle_for(pos, ts):
        df = data[pos['symbol']]['m15']
        return df.loc[ts] if ts in df.index else None

    for ts in unique_times:
        still = []
        for pos in open_positions:
            row = candle_for(pos, ts)
            if row is None:
                still.append(pos)
                continue
            hi, lo = float(row['High']), float(row['Low'])
            hit_sl = lo <= pos['sl'] if pos['side'] == 'LONG' else hi >= pos['sl']
            hit_tp = hi >= pos['tp'] if pos['side'] == 'LONG' else lo <= pos['tp']
            if hit_sl or hit_tp:
                win = bool(hit_tp and not hit_sl)
                gross = TRADE_MARGIN * RR if win else -TRADE_MARGIN
                pnl = gross - pos['fee']
                equity += pnl
                pos.update({'exit_ts': ts, 'result': 'WIN' if win else 'LOSS', 'pnl': pnl})
                closed.append(pos)
                if not win and sum(1 for x in reversed(closed) if x['result']=='LOSS') >= MAX_LOSS_STREAK_PAUSE:
                    pause_until = ts + pd.Timedelta(minutes=15 * PAUSE_BARS)
            else:
                still.append(pos)
        open_positions = still
        peak_equity = max(peak_equity, equity)
        max_dd = min(max_dd, equity - peak_equity)

        candidates = by_time.get(ts, [])
        if not candidates or (pause_until is not None and ts < pause_until):
            continue
        available = MAX_OPEN_POSITIONS - len(open_positions)
        if available <= 0 or equity < 0:
            continue
        used_clusters = {p['cluster'] for p in open_positions}
        used_symbols = {p['symbol'] for p in open_positions}
        exited_symbols = {p['symbol'] for p in closed if p.get('exit_ts') == ts}
        for cand in candidates:
            if available <= 0:
                break
            if cand['symbol'] in used_symbols or cand['cluster'] in used_clusters or cand['symbol'] in exited_symbols:
                continue
            open_positions.append(cand)
            used_symbols.add(cand['symbol'])
            used_clusters.add(cand['cluster'])
            available -= 1

    wins = [x for x in closed if x['result'] == 'WIN']
    losses = [x for x in closed if x['result'] == 'LOSS']
    gross_wins = sum(x['pnl'] for x in wins)
    gross_losses = -sum(x['pnl'] for x in losses)
    pf = gross_wins / gross_losses if gross_losses > 0 else (float('inf') if gross_wins > 0 else 0.0)
    wr = len(wins) / len(closed) * 100 if closed else 0.0
    net = sum(x['pnl'] for x in closed)
    streaks, cur = [], 0
    for x in sorted(closed, key=lambda z: z['exit_ts']):
        if x['result'] == 'LOSS':
            cur += 1
        else:
            if cur: streaks.append(cur)
            cur = 0
    if cur: streaks.append(cur)
    max_streak = max(streaks) if streaks else 0

    print("\n" + "=" * 90)
    print("===== HUNTER-V2-XT-STAGE1 RESULT =====")
    print("=" * 90)
    print(f"XT verified symbols:     {len(valid)}/{len(SYMBOLS)}")
    print(f"Raw signal candidates:   {raw_signal_count}")
    print(f"Risk-valid candidates:   {len(signals)}")
    print(f"Realized trades:         {len(closed)}")
    print(f"Open at dataset end:     {len(open_positions)}")
    print(f"Win rate:                {wr:.2f}%")
    print(f"Profit factor:           {pf:.3f}")
    print(f"Net PnL:                 ${net:,.2f}")
    print(f"Final equity:             ${INITIAL_CAPITAL + net:,.2f}")
    print(f"Max drawdown:             ${max_dd:,.2f}")
    print(f"Max loss streak:          {max_streak}")
    print(f"Trades / day:             {len(closed) / DAYS:.2f}")
    print(f"RR nominal:               2.00")
    print("\nPer-symbol:")
    for asset in sorted(valid):
        arr = [x for x in closed if x['symbol'] == asset]
        w = sum(x['result'] == 'WIN' for x in arr)
        pnl = sum(x['pnl'] for x in arr)
        print(f"  {asset:8s} trades={len(arr):4d} WR={(w/len(arr)*100 if arr else 0):6.2f}% PnL=${pnl:,.2f}")
    print("\nLoss streak sequence:")
    print(", ".join(map(str, streaks)) if streaks else "None")
    print("=" * 90)


if __name__ == '__main__':
    main()
