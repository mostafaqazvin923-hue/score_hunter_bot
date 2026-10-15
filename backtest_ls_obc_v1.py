#!/usr/bin/env python3
"""LS-OBC V1: liquidity sweep -> confirmed ChoCH -> order-block retest.

Research backtest only. Uses Binance USD-M Futures public monthly kline archives.
No live-trading functionality. Conservative OHLC execution: if stop and target are both
reachable in one candle, stop wins. A symbol cannot re-enter on the same candle it exits.
"""
from __future__ import annotations
import io, os, time, zipfile, warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

BASE = "https://data.binance.vision/data/futures/um"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "SUIUSDT", "AVAXUSDT", "NEARUSDT", "ADAUSDT", "BNBUSDT", "APTUSDT", "CRVUSDT", "ONDOUSDT", "PENDLEUSDT", "ICPUSDT", "WIFUSDT"]
TEST_YEARS = int(os.getenv("TEST_YEARS", "3"))
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "1000"))
MARGIN_PER_TRADE = float(os.getenv("MARGIN_PER_TRADE", "100"))
LEVERAGE = float(os.getenv("LEVERAGE", "50"))
NOTIONAL = MARGIN_PER_TRADE * LEVERAGE
RR = 2.0
FEE_RATE = 0.0007       # per side, charged on notional at entry and exit
SLIPPAGE = 0.0003       # adverse price slippage on entry/exit
MAX_TRADE_LOSS_USD = float(os.getenv("MAX_TRADE_LOSS_USD", "75"))
MAX_STOP_PCT = 0.015
MIN_STOP_PCT = 0.001
SWEEP_LOOKBACK = 20
STRUCTURE_LOOKBACK = 8
CHOCH_MAX_BARS = 6
OB_MAX_AGE_BARS = 12
ATR_PERIOD = 14
IMPULSE_ATR_MULT = 1.0
CACHE = Path("market_cache_ls_obc")
OUT = Path("ls_obc_results")
OUT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "LS-OBC-research-backtest/1.0"})

COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def month_starts(start: pd.Timestamp, end: pd.Timestamp):
    cur = start.normalize().replace(day=1)
    while cur < end:
        yield cur
        cur = cur + pd.offsets.MonthBegin(1)


def download_month(symbol: str, interval: str, month: pd.Timestamp):
    name = f"{symbol}-{interval}-{month.year}-{month.month:02d}.zip"
    cache_file = CACHE / name
    if cache_file.exists() and cache_file.stat().st_size > 100:
        raw = cache_file.read_bytes()
    else:
        url = f"{BASE}/monthly/klines/{symbol}/{interval}/{name}"
        try:
            r = SESSION.get(url, timeout=35)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            raw = r.content
            cache_file.write_bytes(raw)
        except Exception as e:
            warnings.warn(f"Download failed {symbol} {interval} {month:%Y-%m}: {e}")
            return None
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            csv_name = next((n for n in zf.namelist() if n.endswith(".csv")), None)
            if not csv_name:
                return None
            df = pd.read_csv(zf.open(csv_name), header=None)
        if df.empty:
            return None
        df = df.iloc[:, :len(COLS)]
        df.columns = COLS[:df.shape[1]]
        df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
        # Binance archives use milliseconds for UM klines.
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True, errors="coerce")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open_time", "open", "high", "low", "close"])
        return df[["open_time", "open", "high", "low", "close", "volume"]]
    except Exception as e:
        warnings.warn(f"Bad archive {name}: {e}")
        return None


def load_symbol(symbol: str, start: pd.Timestamp, end: pd.Timestamp, interval: str):
    frames = []
    months = list(month_starts(start, end))
    with ThreadPoolExecutor(max_workers=5) as pool:
        futs = [pool.submit(download_month, symbol, interval, m) for m in months]
        for fut in as_completed(futs):
            x = fut.result()
            if x is not None and not x.empty:
                frames.append(x)
    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.concat(frames, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df = df.set_index("open_time")
    df = df.loc[(df.index >= start) & (df.index < end)]
    # Exclude any not-yet-closed candle using close-time boundary inferred from interval.
    delta = pd.Timedelta(interval)
    now = pd.Timestamp.now(tz="UTC")
    df = df.loc[(df.index + delta) <= now]
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    prev_close = x.close.shift(1)
    tr = pd.concat([(x.high-x.low), (x.high-prev_close).abs(), (x.low-prev_close).abs()], axis=1).max(axis=1)
    x["atr"] = tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()
    # Higher-timeframe trend context uses only completed 4h candles; shifted one 4h bar
    # before forward-filling to the 1h index to avoid using a still-forming 4h candle.
    return x


def load_all():
    end = pd.Timestamp.now(tz="UTC").floor("h")
    start = (end - pd.DateOffset(years=TEST_YEARS)).floor("h")
    all_data = {}
    print(f"Loading {TEST_YEARS} years of Binance UM Futures 1h and 4h data: {start} -> {end}", flush=True)
    for i, sym in enumerate(SYMBOLS, 1):
        print(f"[{i}/{len(SYMBOLS)}] {sym}", flush=True)
        h1 = load_symbol(sym, start, end, "1h")
        h4 = load_symbol(sym, start - pd.Timedelta(days=10), end, "4h")
        if len(h1) < 500 or len(h4) < 100:
            warnings.warn(f"Insufficient data for {sym}: 1h={len(h1)}, 4h={len(h4)}")
            continue
        h1 = add_features(h1)
        h4 = add_features(h4)
        # Simple regime filter: only use prior completed 4h close and EMA50/EMA200.
        h4["ema50"] = h4.close.ewm(span=50, adjust=False, min_periods=50).mean()
        h4["ema200"] = h4.close.ewm(span=200, adjust=False, min_periods=200).mean()
        h4_ctx = h4[["close", "ema50", "ema200"]].shift(1).rename(columns={"close":"h4_close", "ema50":"h4_ema50", "ema200":"h4_ema200"})
        h1 = h1.join(h4_ctx.reindex(h1.index, method="ffill"))
        all_data[sym] = h1
    if not all_data:
        raise RuntimeError("No usable data downloaded. Check archive access or symbol availability.")
    return all_data, start, end


@dataclass
class Setup:
    side: int                  # +1 long, -1 short
    sweep_i: int
    sweep_extreme: float
    choch_i: int
    ob_low: float
    ob_high: float
    expires_i: int


def build_signals(df: pd.DataFrame):
    """Causal state machine. Each setup is known only at the close of its confirmation bar."""
    n = len(df)
    o, h, l, c = [df[k].to_numpy(float) for k in ("open", "high", "low", "close")]
    atr = df.atr.to_numpy(float)
    h4c = df.h4_close.to_numpy(float)
    e50 = df.h4_ema50.to_numpy(float)
    e200 = df.h4_ema200.to_numpy(float)
    setups: list[Setup] = []
    pending_sweep = None
    active_ob = None
    for i in range(max(SWEEP_LOOKBACK, STRUCTURE_LOOKBACK, ATR_PERIOD)+2, n):
        if not np.isfinite(atr[i]) or not np.isfinite(e50[i]) or not np.isfinite(e200[i]):
            continue
        # Once a sweep occurs, wait up to CHOCH_MAX_BARS for a close beyond structure.
        if pending_sweep is not None:
            side = pending_sweep["side"]
            si = pending_sweep["i"]
            if i - si > CHOCH_MAX_BARS:
                pending_sweep = None
            else:
                struct_hi = np.max(h[max(0, si-STRUCTURE_LOOKBACK):si])
                struct_lo = np.min(l[max(0, si-STRUCTURE_LOOKBACK):si])
                impulse = (c[i]-o[i]) if side == 1 else (o[i]-c[i])
                broke = c[i] > struct_hi if side == 1 else c[i] < struct_lo
                aligned = (h4c[i] > e50[i] and e50[i] > e200[i]) if side == 1 else (h4c[i] < e50[i] and e50[i] < e200[i])
                if broke and impulse >= IMPULSE_ATR_MULT * atr[i] and aligned:
                    # Last opposing candle between sweep and ChoCH. The OB becomes known now.
                    candidates = range(si, i)
                    ob_idx = None
                    for j in reversed(list(candidates)):
                        if (side == 1 and c[j] < o[j]) or (side == -1 and c[j] > o[j]):
                            ob_idx = j
                            break
                    if ob_idx is not None:
                        active_ob = Setup(side, si, pending_sweep["extreme"], i,
                                          float(l[ob_idx]), float(h[ob_idx]), i + OB_MAX_AGE_BARS)
                    pending_sweep = None
                elif i - si == CHOCH_MAX_BARS:
                    pending_sweep = None
        # Detect new sweep using prior candles only. A sweep must close back inside prior range.
        lo_ref = np.min(l[i-SWEEP_LOOKBACK:i])
        hi_ref = np.max(h[i-SWEEP_LOOKBACK:i])
        # Avoid replacing a live order block with a new setup until it expires/gets filled.
        if active_ob is None:
            if l[i] < lo_ref and c[i] > lo_ref:
                pending_sweep = {"side": 1, "i": i, "extreme": float(l[i])}
            elif h[i] > hi_ref and c[i] < hi_ref:
                pending_sweep = {"side": -1, "i": i, "extreme": float(h[i])}
        if active_ob is not None and i > active_ob.choch_i and i > active_ob.sweep_i + 0:
            if i > active_ob.expires_i:
                active_ob = None
            else:
                # Only retest after OB is known; entry level is proximal edge of the OB.
                entry = active_ob.ob_high if active_ob.side == 1 else active_ob.ob_low
                if l[i] <= entry <= h[i]:
                    stop = active_ob.sweep_extreme
                    risk = (entry - stop) if active_ob.side == 1 else (stop - entry)
                    risk_pct = risk / entry if entry else np.inf
                    est_cost = NOTIONAL * (2*FEE_RATE + 2*SLIPPAGE)
                    gross_risk = NOTIONAL * risk_pct
                    if risk > 0 and MIN_STOP_PCT <= risk_pct <= MAX_STOP_PCT and gross_risk + est_cost <= MAX_TRADE_LOSS_USD:
                        target = entry + active_ob.side * RR * risk
                        setups.append({"symbol": None, "signal_i": i, "side": active_ob.side,
                                       "entry": entry, "stop": stop, "target": target,
                                       "sweep_i": active_ob.sweep_i, "choch_i": active_ob.choch_i,
                                       "ob_low": active_ob.ob_low, "ob_high": active_ob.ob_high,
                                       "risk_pct": risk_pct})
                    active_ob = None
    return setups


def simulate(all_data, start, end):
    signals_by_sym = {}
    for sym, df in all_data.items():
        sigs = build_signals(df)
        for s in sigs:
            s["symbol"] = sym
        signals_by_sym[sym] = {int(s["signal_i"]): s for s in sigs}
        print(f"{sym}: bars={len(df):,}, valid_entries={len(sigs)}", flush=True)

    balance = INITIAL_CAPITAL
    positions = {}  # symbol -> position
    trades = []
    skipped_margin = skipped_risk = 0
    # Event timeline based on 1h timestamps. No same-candle re-entry after an exit.
    timeline = sorted(set().union(*(set(df.index) for df in all_data.values())))
    last_exit_index = {sym: -1 for sym in all_data}
    index_maps = {sym: {ts:i for i,ts in enumerate(df.index)} for sym,df in all_data.items()}
    for ts in timeline:
        # Manage existing positions first; OHLC ambiguity resolved stop-first.
        for sym in list(positions):
            df = all_data[sym]
            idx = index_maps[sym].get(ts)
            if idx is None:
                continue
            row = df.iloc[idx]
            p = positions[sym]
            exit_price = None
            reason = None
            if p["side"] == 1:
                if row.open <= p["stop"]:
                    exit_price, reason = row.open * (1-SLIPPAGE), "SL_gap"
                elif row.open >= p["target"]:
                    exit_price, reason = row.open * (1-SLIPPAGE), "TP_gap"
                else:
                    hit_sl = row.low <= p["stop"]
                    hit_tp = row.high >= p["target"]
                    if hit_sl:
                        exit_price, reason = p["stop"]*(1-SLIPPAGE), "SL"
                    elif hit_tp:
                        exit_price, reason = p["target"]*(1-SLIPPAGE), "TP"
            else:
                if row.open >= p["stop"]:
                    exit_price, reason = row.open * (1+SLIPPAGE), "SL_gap"
                elif row.open <= p["target"]:
                    exit_price, reason = row.open * (1+SLIPPAGE), "TP_gap"
                else:
                    hit_sl = row.high >= p["stop"]
                    hit_tp = row.low <= p["target"]
                    if hit_sl:
                        exit_price, reason = p["stop"]*(1+SLIPPAGE), "SL"
                    elif hit_tp:
                        exit_price, reason = p["target"]*(1+SLIPPAGE), "TP"
            if exit_price is not None:
                qty = p["qty"]
                gross = (exit_price-p["entry_exec"])*qty*p["side"]
                exit_fee = abs(exit_price*qty)*FEE_RATE
                net = gross - p["entry_fee"] - exit_fee
                balance += net
                trades.append({**p, "exit_time": ts, "exit_price": exit_price, "exit_reason": reason,
                               "gross_pnl": gross, "exit_fee": exit_fee, "net_pnl": net,
                               "balance_after": balance, "bars_held": idx-p["entry_i"]})
                del positions[sym]
                last_exit_index[sym] = idx
        # Open new trades only from a previously computed signal; same symbol must not have exited this candle.
        for sym in sorted(all_data):
            if sym in positions:
                continue
            df = all_data[sym]
            idx = index_maps[sym].get(ts)
            if idx is None or idx <= last_exit_index[sym]:
                continue
            sig = signals_by_sym[sym].get(idx)
            if sig is None:
                continue
            # Entry limit is touched during this candle. Entry is assumed at limit with adverse slippage.
            row = df.iloc[idx]
            level = sig["entry"]
            if not (row.low <= level <= row.high):
                continue
            side = sig["side"]
            entry_exec = level*(1+SLIPPAGE if side == 1 else 1-SLIPPAGE)
            stop, target = sig["stop"], sig["target"]
            stop_dist = abs(entry_exec-stop)
            risk_pct = stop_dist / entry_exec
            est_cost = NOTIONAL*(2*FEE_RATE+2*SLIPPAGE)
            if risk_pct < MIN_STOP_PCT or risk_pct > MAX_STOP_PCT or NOTIONAL*risk_pct+est_cost > MAX_TRADE_LOSS_USD:
                skipped_risk += 1
                continue
            # Fixed margin, fixed notional; reserve margin for each simultaneous position.
            if balance < MARGIN_PER_TRADE or (len(positions)+1)*MARGIN_PER_TRADE > max(balance, 0):
                skipped_margin += 1
                continue
            qty = NOTIONAL/entry_exec
            entry_fee = NOTIONAL*FEE_RATE
            # If entry and SL/TP are both reachable on this same candle, stop-first handling below
            # is applied by immediately simulating the candle range after the fill.
            p = {"symbol":sym,"side":side,"signal_time":ts,"entry_time":ts,"entry_i":idx,
                 "entry":level,"entry_exec":entry_exec,"stop":stop,"target":target,"qty":qty,
                 "entry_fee":entry_fee,"margin":MARGIN_PER_TRADE,"notional":NOTIONAL,
                 "sweep_i":sig["sweep_i"],"choch_i":sig["choch_i"],"ob_low":sig["ob_low"],"ob_high":sig["ob_high"],
                 "risk_pct":risk_pct}
            # Entry-candle execution: do not re-check open gap because the limit may fill intrabar.
            hit_sl = row.low <= stop if side == 1 else row.high >= stop
            hit_tp = row.high >= target if side == 1 else row.low <= target
            if hit_sl or hit_tp:
                # Ambiguous/instant outcomes use SL whenever stop can be reached.
                if hit_sl:
                    exit_price = stop*(1-SLIPPAGE if side == 1 else 1+SLIPPAGE)
                    reason = "SL_entry_candle"
                else:
                    exit_price = target*(1-SLIPPAGE if side == 1 else 1+SLIPPAGE)
                    reason = "TP_entry_candle"
                gross = (exit_price-entry_exec)*qty*side
                exit_fee = abs(exit_price*qty)*FEE_RATE
                net = gross-entry_fee-exit_fee
                balance += net
                trades.append({**p,"exit_time":ts,"exit_price":exit_price,"exit_reason":reason,
                               "gross_pnl":gross,"exit_fee":exit_fee,"net_pnl":net,"balance_after":balance,"bars_held":0})
                last_exit_index[sym] = idx
            else:
                positions[sym] = p
    # Mark open trades at final close with fees/slippage; disclose as data-end marks.
    data_end_marks = 0
    for sym, p in list(positions.items()):
        row = all_data[sym].iloc[-1]
        exit_price = float(row.close)*(1-SLIPPAGE if p["side"] == 1 else 1+SLIPPAGE)
        gross = (exit_price-p["entry_exec"])*p["qty"]*p["side"]
        exit_fee = abs(exit_price*p["qty"])*FEE_RATE
        net = gross-p["entry_fee"]-exit_fee
        balance += net
        trades.append({**p,"exit_time":all_data[sym].index[-1],"exit_price":exit_price,"exit_reason":"DATA_END",
                       "gross_pnl":gross,"exit_fee":exit_fee,"net_pnl":net,"balance_after":balance,
                       "bars_held":len(all_data[sym])-1-p["entry_i"]})
        data_end_marks += 1
    return trades, balance, skipped_margin, skipped_risk, data_end_marks


def summarize(trades, final_balance, skipped_margin, skipped_risk, data_end_marks, start, end):
    df = pd.DataFrame(trades)
    if df.empty:
        return {"timeframe":"1h+4h_filter","start":str(start),"end":str(end),"trades":0,"wins":0,"losses":0,
                "win_rate_pct":0.0,"net_pnl":0.0,"final_equity":INITIAL_CAPITAL,"profit_factor":0.0,
                "max_drawdown_pct":0.0,"max_loss_streak":0,"fees_total":0.0,"data_end_marks":data_end_marks,
                "skipped_margin":skipped_margin,"skipped_risk":skipped_risk}
    df = df.sort_values(["exit_time","symbol"]).reset_index(drop=True)
    wins = int((df.net_pnl > 0).sum()); losses = int((df.net_pnl <= 0).sum())
    gross_profit = float(df.loc[df.net_pnl > 0,"net_pnl"].sum())
    gross_loss = float(-df.loc[df.net_pnl < 0,"net_pnl"].sum())
    pf = gross_profit/gross_loss if gross_loss > 0 else float("inf")
    equity = INITIAL_CAPITAL + df.net_pnl.cumsum()
    peak = equity.cummax().clip(lower=INITIAL_CAPITAL)
    dd = (peak-equity)/peak*100
    streak = best = 0
    for pnl in df.net_pnl:
        if pnl <= 0:
            streak += 1; best=max(best,streak)
        else:
            streak=0
    fees = float(df.entry_fee.sum()+df.exit_fee.sum())
    df.to_csv(OUT/"trades.csv",index=False)
    bysym = df.groupby("symbol").agg(trades=("net_pnl","size"),wins=("net_pnl",lambda x:int((x>0).sum())),net_pnl=("net_pnl","sum"),fees=("entry_fee","sum")).reset_index()
    bysym.to_csv(OUT/"by_symbol.csv",index=False)
    return {"timeframe":"1h+4h_filter","start":str(start),"end":str(end),"trades":len(df),"wins":wins,"losses":losses,
            "win_rate_pct":wins/len(df)*100,"net_pnl":float(df.net_pnl.sum()),"final_equity":float(final_balance),
            "profit_factor":pf,"max_drawdown_pct":float(dd.max()),"max_loss_streak":best,"fees_total":fees,
            "data_end_marks":data_end_marks,"skipped_margin":skipped_margin,"skipped_risk":skipped_risk}


def main():
    all_data,start,end=load_all()
    trades,final_balance,skip_m,skip_r,end_marks=simulate(all_data,start,end)
    summary=summarize(trades,final_balance,skip_m,skip_r,end_marks,start,end)
    pd.DataFrame([summary]).to_csv(OUT/"summary.csv",index=False)
    print("\nLS-OBC V1 SUMMARY")
    for k,v in summary.items(): print(f"{k:24s} {v}")
    print(f"\nSaved results to {OUT.resolve()}")
    print("Research-only: funding and exchange-specific maintenance-margin/liquidation rules are not modeled.")

if __name__ == "__main__":
    main()
