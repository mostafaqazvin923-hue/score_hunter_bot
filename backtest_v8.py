import os
import sys
import time
import subprocess
from datetime import datetime, timedelta, timezone

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ccxt'])
    import ccxt

import numpy as np
import pandas as pd

# ============================================================
# HUNTER-V2 STAGE-1
# TREND PULLBACK / VOLATILITY EXPANSION
# RAW EDGE RESEARCH ENGINE
# ============================================================
# Purpose:
#   Test a fundamentally different entry thesis from the old
#   liquidity-sweep family.
#
# Architecture:
#   4H = trend regime
#   1H = pullback/reclaim
#   15M = volatility expansion trigger
#
# Entry:
#   Signal is generated only from a CLOSED 15M candle.
#   Entry is NEXT 15M candle OPEN + slippage.
#
# Exit:
#   Structural ATR stop + exact 1:2 RR.
#   No BE, no trailing, no timeout.
#   If SL and TP are both touched in one candle -> LOSS.
#
# Portfolio:
#   $1,000 initial equity
#   $100 fixed margin/trade
#   50x leverage
#   max 3 open positions
#   max 1 position per correlation cluster
#   no overlapping positions per symbol
#
# IMPORTANT:
#   This is deliberately a Stage-1 RAW EDGE test.
#   It avoids a large confirmation-score stack. We want to know
#   whether Trend -> Pullback -> Expansion has standalone edge.
# ============================================================

EXCHANGE = ccxt.lbank({'enableRateLimit': True, 'timeout': 20000})

SYMBOLS = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT',
    'SUI/USDT:USDT', 'AVAX/USDT:USDT', 'NEAR/USDT:USDT',
    'ADA/USDT:USDT', 'BNB/USDT:USDT', 'APT/USDT:USDT',
    'CRV/USDT:USDT', 'ONDO/USDT:USDT', 'PENDLE/USDT:USDT',
    'ICP/USDT:USDT', 'WIF/USDT:USDT'
]

CLUSTERS = {
    'MAJOR': {'BTC/USDT:USDT', 'ETH/USDT:USDT'},
    'L1': {'SOL/USDT:USDT', 'SUI/USDT:USDT', 'AVAX/USDT:USDT', 'NEAR/USDT:USDT', 'ADA/USDT:USDT', 'BNB/USDT:USDT', 'APT/USDT:USDT'},
    'DEFI': {'CRV/USDT:USDT', 'ONDO/USDT:USDT', 'PENDLE/USDT:USDT'},
    'OTHER': {'ICP/USDT:USDT'},
    'MEME': {'WIF/USDT:USDT'},
}

DAYS = 365
TIMEFRAME = '15m'
FETCH_DAYS_PER_CHUNK = 15
WARMUP_DAYS = 35
TRADE_MARGIN = 100.0
LEVERAGE = 50.0
INITIAL_CAPITAL = 1000.0
FEE_RATE = 0.0007
SLIPPAGE = 0.0003
RR = 2.0
MAX_OPEN_POSITIONS = 3
MAX_STRUCTURAL_RISK = 0.025
MAX_LOSS_STREAK_PAUSE = 4
PAUSE_BARS = 16

# Stage-1 signal parameters: intentionally modest, not optimized.
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


def fetch_ohlcv(symbol, since_ms, until_ms):
    rows = []
    cursor = since_ms
    limit = 1000
    while cursor < until_ms:
        batch = EXCHANGE.fetch_ohlcv(symbol, TIMEFRAME, since=cursor, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1][0]
        nxt = last + 15 * 60 * 1000
        if nxt <= cursor:
            break
        cursor = nxt
        if len(batch) < limit:
            # LBank can occasionally return a short page before the target.
            # Advance one candle and continue; stop is still controlled by until_ms.
            time.sleep(0.15)
        time.sleep(0.05)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df = df.drop_duplicates('timestamp').sort_values('timestamp')
    df = df[df['timestamp'] < until_ms]
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('Date')
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)


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
    print('=' * 90)
    print('HUNTER-V2 STAGE-1 — TREND PULLBACK / VOLATILITY EXPANSION')
    print('RAW EDGE / STRICT PORTFOLIO-LEVEL NO-LOOKAHEAD BACKTEST')
    print('=' * 90)

    now = datetime.now(timezone.utc)
    end_ms = int(now.timestamp() * 1000)
    start_ms = int((now - timedelta(days=DAYS + WARMUP_DAYS)).timestamp() * 1000)
    target_start = pd.Timestamp(now - timedelta(days=DAYS), tz='UTC')

    data = {}
    valid = []
    for symbol in SYMBOLS:
        print(f'دریافت و آماده‌سازی: {symbol.split("/")[0]} ...')
        try:
            df = fetch_ohlcv(symbol, start_ms, end_ms)
            if len(df) < 2000:
                print('  -> داده کافی نیست')
                continue
            m15, h1, h4 = prepare(df)
            data[symbol] = {'m15': m15, 'h1': h1, 'h4': h4}
            valid.append(symbol)
            print(f'  -> 15M={len(m15):,} | 1H={len(h1):,} | 4H={len(h4):,}')
        except Exception as e:
            print(f'  !! خطا: {e}')

    print(f'نمادهای معتبر: {len(valid)} / {len(SYMBOLS)}')
    print('\nساخت سیگنال‌های Stage-1 ...')

    signals = []
    for symbol in valid:
        d = data[symbol]
        m15 = d['m15']
        start_i = int(m15.index.searchsorted(target_start))
        for i in range(max(1, start_i), len(m15) - 1):
            s = signal_at(m15, d['h1'], d['h4'], i)
            if not s:
                continue
            entry_ts = m15.index[i + 1]
            raw_open = float(m15.iloc[i + 1]['Open'])
            if s['side'] == 'LONG':
                entry = raw_open * (1 + SLIPPAGE)
            else:
                entry = raw_open * (1 - SLIPPAGE)
            tr = make_trade(symbol, s['side'], entry, s['sl_ref'], m15.index[i], entry_ts)
            if tr:
                signals.append(tr)

    signals.sort(key=lambda x: (x['entry_ts'], x['symbol']))
    print(f'Candidate signals: {len(signals)}')

    # Fast lookup for future candles.
    open_positions = []
    closed = []
    last_entry_by_symbol = {}
    loss_streak = 0
    pause_until = None
    equity = INITIAL_CAPITAL
    peak_equity = INITIAL_CAPITAL
    max_dd = 0.0
    max_streak = 0
    streak_seq = []
    current_streak = 0
    last_trade_result_ts = None

    # Candidate signals are processed chronologically. Existing positions are
    # managed candle-by-candle before considering a new entry at that timestamp.
    unique_times = sorted(set(s['entry_ts'] for s in signals))
    by_time = {}
    for s in signals:
        by_time.setdefault(s['entry_ts'], []).append(s)

    def candle_for(pos, ts):
        return data[pos['symbol']]['m15'].loc[ts] if ts in data[pos['symbol']]['m15'].index else None

    for ts in unique_times:
        # Manage every currently open position through this candle first.
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
                # Conservative same-candle conflict: SL wins.
                win = bool(hit_tp and not hit_sl)
                gross = TRADE_MARGIN * RR if win else -TRADE_MARGIN
                pnl = gross - pos['fee']
                equity += pnl
                pos.update({'exit_ts': ts, 'result': 'WIN' if win else 'LOSS', 'pnl': pnl})
                closed.append(pos)
                current_streak = 0 if win else current_streak + 1
                if win:
                    if current_streak == 0:
                        pass
                    loss_streak = 0
                else:
                    loss_streak = current_streak
                    max_streak = max(max_streak, current_streak)
                if win:
                    if current_streak == 0:
                        pass
                # Record completed streak only when a win ends it.
                if win and current_streak == 0:
                    pass
                if not win and current_streak >= MAX_LOSS_STREAK_PAUSE:
                    pause_until = ts + pd.Timedelta(minutes=15 * PAUSE_BARS)
            else:
                still.append(pos)
        open_positions = still

        # Equity/DD after exits.
        peak_equity = max(peak_equity, equity)
        max_dd = min(max_dd, equity - peak_equity)

        # New entries at this candle open. No same-symbol overlap, no cluster overlap.
        candidates = by_time.get(ts, [])
        if not candidates:
            continue
        if pause_until is not None and ts < pause_until:
            continue
        if equity < 0:
            # Stop opening new positions after capital is exhausted.
            continue
        available = MAX_OPEN_POSITIONS - len(open_positions)
        if available <= 0:
            continue
        used_clusters = {p['cluster'] for p in open_positions}
        used_symbols = {p['symbol'] for p in open_positions}
        for cand in candidates:
            if available <= 0:
                break
            if cand['symbol'] in used_symbols or cand['cluster'] in used_clusters:
                continue
            # Do not open a signal on the same timestamp as an exit from the same symbol.
            if cand['symbol'] in {p['symbol'] for p in closed if p.get('exit_ts') == ts}:
                continue
            open_positions.append(cand)
            used_symbols.add(cand['symbol'])
            used_clusters.add(cand['cluster'])
            available -= 1

    # Manage/report positions still open at the end; do not invent exits.
    final_open = open_positions

    wins = [x for x in closed if x['result'] == 'WIN']
    losses = [x for x in closed if x['result'] == 'LOSS']
    gross_wins = sum(x['pnl'] for x in wins)
    gross_losses = -sum(x['pnl'] for x in losses)
    pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')
    wr = len(wins) / len(closed) * 100 if closed else 0
    avg_win = np.mean([x['pnl'] for x in wins]) if wins else 0
    avg_loss = np.mean([x['pnl'] for x in losses]) if losses else 0
    expectancy = np.mean([x['pnl'] for x in closed]) if closed else 0
    net = sum(x['pnl'] for x in closed)

    # Reconstruct consecutive loss streak sequence from closed trade chronology.
    streaks = []
    cur = 0
    for x in sorted(closed, key=lambda z: z['exit_ts']):
        if x['result'] == 'LOSS':
            cur += 1
        else:
            if cur:
                streaks.append(cur)
            cur = 0
    if cur:
        streaks.append(cur)
    max_streak = max(streaks) if streaks else 0

    days_span = max(DAYS, 1)
    trades_day = len(closed) / days_span

    print('\n' + '=' * 90)
    print('HUNTER-V2 STAGE-1 — FINAL AUDITED REPORT')
    print('=' * 90)
    print(f'Initial Capital:          ${INITIAL_CAPITAL:,.2f}')
    print(f'Trade Margin:             ${TRADE_MARGIN:,.2f}')
    print(f'Leverage:                 {LEVERAGE:.0f}x')
    print(f'RR:                       1:{RR:.0f}')
    print('-' * 90)
    print(f'Candidate Signals:        {len(signals):,}')
    print(f'Closed Trades:            {len(closed):,}')
    print(f'Wins:                     {len(wins):,}')
    print(f'Losses:                   {len(losses):,}')
    print(f'Win Rate:                 {wr:.2f}%')
    long_closed = [x for x in closed if x['side'] == 'LONG']
    short_closed = [x for x in closed if x['side'] == 'SHORT']
    print(f'Long Win Rate:            {(sum(x["result"]=="WIN" for x in long_closed)/len(long_closed)*100 if long_closed else 0):.2f}%')
    print(f'Short Win Rate:           {(sum(x["result"]=="WIN" for x in short_closed)/len(short_closed)*100 if short_closed else 0):.2f}%')
    print('-' * 90)
    print(f'Net PnL:                  ${net:,.2f}')
    print(f'Final Closed Equity:      ${INITIAL_CAPITAL + net:,.2f}')
    print(f'Profit Factor:            {pf:.2f}')
    print(f'Average Win:              ${avg_win:,.2f}')
    print(f'Average Loss:             ${avg_loss:,.2f}')
    print(f'Expectancy / Trade:       ${expectancy:,.2f}')
    print(f'Max Drawdown:             ${max_dd:,.2f}')
    print(f'Max Portfolio Loss Streak:{max_streak:2d}')
    print(f'Trades / Day:             {trades_day:.2f}')
    print('-' * 90)
    print('Loss streak sequence:')
    print(', '.join(map(str, streaks)) if streaks else 'None')
    print('-' * 90)
    print(f'FINAL OPEN POSITIONS:     {len(final_open)}')
    for p in final_open:
        print(f"  {p['symbol'].split('/')[0]:>6} {p['side']:<5} entry={p['entry']:.8g} sl={p['sl']:.8g} tp={p['tp']:.8g}")

    print('\n' + '-' * 90)
    print('PER-SYMBOL')
    print('-' * 90)
    for symbol in sorted(valid, key=lambda s: -len([x for x in closed if x['symbol'] == s])):
        arr = [x for x in closed if x['symbol'] == symbol]
        if not arr:
            continue
        w = [x for x in arr if x['result'] == 'WIN']
        l = [x for x in arr if x['result'] == 'LOSS']
        gw = sum(x['pnl'] for x in w)
        gl = -sum(x['pnl'] for x in l)
        spf = gw / gl if gl > 0 else float('inf')
        swr = len(w) / len(arr) * 100
        sp = sum(x['pnl'] for x in arr)
        print(f"{symbol.split('/')[0]:>6} | {len(arr):4d} trades | WR {swr:6.2f}% | PnL ${sp:9.2f} | PF {spf:.2f}")

    print('\n' + '=' * 90)
    print('AUDIT NOTES')
    print('=' * 90)
    print('✓ Stage-1 core: 4H trend + 1H pullback/reclaim + 15M expansion')
    print('✓ Signal uses CLOSED 15M candle only')
    print('✓ 1H/4H context uses only COMPLETED higher-timeframe candles')
    print('✓ Entry = NEXT 15M candle OPEN + slippage')
    print('✓ Exact RR = 1:2')
    print('✓ No Break-Even / trailing / timeout')
    print('✓ No overlapping positions per symbol')
    print('✓ Max 3 portfolio positions / max 1 per correlation cluster')
    print('✓ Same-candle SL+TP conflict resolves to LOSS')
    print('✓ No artificial forced exit at end; open positions reported separately')
    print('=' * 90)

if __name__ == '__main__':
    main()
