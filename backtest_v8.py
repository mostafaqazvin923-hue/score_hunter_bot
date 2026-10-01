#!/usr/bin/env python3
"""
HUNTER-V9-EBP: 4H ENGULFING BAR PATTERN BACKTEST ENGINE
- Market: XT USDT-M Futures REST API (fapi.xt.com)
- Timeframe: 4H Primary Setup with Daily Trend Context & Monthly Breakdown
- Zero Lookahead, Zero Leakage, Strict Causal Pipeline
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
import time
requests = None
try:
    import requests
except ImportError:
    pass
import numpy as np
import pandas as pd

SYMBOLS = [
    "btc_usdt", "eth_usdt", "sol_usdt", "sui_usdt", "avax_usdt",
    "near_usdt", "ada_usdt", "bnb_usdt", "apt_usdt", "crv_usdt",
    "ondo_usdt", "pendle_usdt", "icp_usdt", "wif_usdt"
]

DATA_DIR = Path("data/xt_futures_ebp_4h")
TOTAL_DAYS = 365
WARMUP_DAYS = 60

INITIAL_CAPITAL = 1000.0
MARGIN_PER_TRADE = 100.0
LEVERAGE = 20.0
RR = 2.0

FEE_RATE = 0.0007
SLIPPAGE = 0.0003


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(DATA_DIR))
    return p.parse_args()


def fetch_xt_futures_data(symbol: str, data_dir: Path) -> pd.DataFrame:
    file_path = data_dir / f"{symbol.upper()}_15m.csv"
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    interval_ms = 15 * 60 * 1000
    target_start_ms = now_ms - int((TOTAL_DAYS + WARMUP_DAYS) * 86400 * 1000)
    final_end_ms = (now_ms // interval_ms) * interval_ms - 1

    if file_path.exists() and file_path.stat().st_size > 1000:
        try:
            df_cached = pd.read_csv(file_path)
            if "timestamp" in df_cached.columns and len(df_cached) > 0:
                df_cached["timestamp_dt"] = pd.to_datetime(df_cached["timestamp"], unit="ms", utc=True)
                print(f"[CACHE] Loaded cached dataset for {symbol.upper()}")
                return df_cached.drop(columns=["timestamp_dt"])
        except Exception:
            pass

    if requests is None:
        raise RuntimeError("requests library is not installed. Please add it to requirements.txt.")

    data_dir.mkdir(parents=True, exist_ok=True)
    url = "https://fapi.xt.com/future/market/v1/public/q/kline"
    limit = 1500
    current_start = target_start_ms
    all_klines = []
    
    while current_start < final_end_ms:
        window_end = min(final_end_ms, current_start + limit * interval_ms - 1)
        params = {"symbol": symbol, "interval": "15m", "startTime": current_start, "endTime": window_end, "limit": limit}
        
        success, data = False, None
        for attempt in range(3):
            try:
                res = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
                if res.status_code == 200:
                    res_json = res.json()
                    data = res_json.get("result", res_json.get("data", res_json))
                    if isinstance(data, list):
                        success = True
                        break
            except Exception:
                pass
            time.sleep(1 * (attempt + 1))
            
        if not success or not data:
            break
            
        parsed_batch = []
        for k in data:
            try:
                ts, o, h, l, c, v = int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])
                parsed_batch.append([ts, o, h, l, c, v])
            except Exception:
                continue
                
        if not parsed_batch:
            break
            
        all_klines.extend(parsed_batch)
        max_ts = max(item[0] for item in parsed_batch)
        next_cursor = max_ts + interval_ms
        if next_cursor <= current_start:
            break
        current_start = next_cursor
        time.sleep(0.05)
        
    if not all_klines:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_klines, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    df.to_csv(file_path, index=False)
    return df


def load_and_resample(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if df.empty:
        return {"4h": pd.DataFrame(), "1d": pd.DataFrame()}
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp_dt").sort_index()
    df = df[~df.index.duplicated(keep="last")].copy()

    df_4h = df.resample('4h', closed='right', label='right').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    df_1d = df.resample('1d', closed='right', label='right').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    return {"4h": df_4h, "1d": df_1d}


def calculate_ebp_features(dfs: dict[str, pd.DataFrame]):
    h4 = dfs["4h"]
    if h4.empty:
        return
    prev_open = h4["open"].shift(1)
    prev_close = h4["close"].shift(1)
    curr_open = h4["open"]
    curr_close = h4["close"]

    prev_is_red = prev_close < prev_open
    curr_is_green = curr_close > curr_open
    bull_engulf = prev_is_red & curr_is_green & (curr_close >= prev_open) & (curr_open <= prev_close)

    prev_is_green = prev_close > prev_open
    curr_is_red = curr_close < curr_open
    bear_engulf = prev_is_green & curr_is_red & (curr_close <= prev_open) & (curr_close >= prev_close)

    h4["bull_ebp"] = bull_engulf
    h4["bear_ebp"] = bear_engulf

    d1 = dfs["1d"]
    if not d1.empty:
        prev_close_1d = d1["close"].shift(1)
        d1["ema50"] = prev_close_1d.ewm(span=50, adjust=False, min_periods=50).mean()
        d1["ema200"] = prev_close_1d.ewm(span=200, adjust=False, min_periods=200).mean()
        d1["ema50_slope"] = d1["ema50"].diff()


def run_backtest_4h_ebp(all_symbol_data: dict[str, dict[str, pd.DataFrame]], start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> tuple[list[dict], dict | None]:
    all_4h_times = sorted(list(set().union(*(dfs["4h"].index for dfs in all_symbol_data.values() if not dfs["4h"].empty))))
    all_4h_times = [t for t in all_4h_times if start_dt <= t <= end_dt]

    active_position = None
    trades = []
    cooldowns = {sym: 0 for sym in all_symbol_data.keys()}

    symbol_arrays = {}
    for sym, dfs in all_symbol_data.items():
        symbol_arrays[sym] = {
            "h4_times": dfs["4h"].index if not dfs["4h"].empty else pd.DatetimeIndex([]),
            "h4_df": dfs["4h"],
            "d1_times": dfs["1d"].index if not dfs["1d"].empty else pd.DatetimeIndex([]),
            "d1_df": dfs["1d"],
        }

    for ts in all_4h_times:
        for sym in cooldowns:
            if cooldowns[sym] > 0:
                cooldowns[sym] -= 1

        position_closed_this_iteration = False

        if active_position is not None:
            sym = active_position["symbol"]
            h4_df = symbol_arrays[sym]["h4_df"]
            if not h4_df.empty and ts in h4_df.index:
                bar = h4_df.loc[ts]
                hi, lo = float(bar["high"]), float(bar["low"])
                side, sl, tp, entry = active_position["side"], active_position["sl"], active_position["tp"], active_position["entry"]
                notional = MARGIN_PER_TRADE * LEVERAGE

                hit_sl = lo <= sl if side == "LONG" else hi >= sl
                hit_tp = hi >= tp if side == "LONG" else lo <= tp

                if hit_sl or hit_tp:
                    outcome = "LOSS" if (hit_sl and hit_tp) or hit_sl else "WIN"
                    exit_price = sl if outcome == "LOSS" else tp

                    gross = (exit_price - entry) / entry * notional if side == "LONG" else (entry - exit_price) / entry * notional
                    fees = notional * FEE_RATE * 2.0
                    pnl = gross - fees

                    trades.append({
                        "symbol": sym, "side": side, "entry_ts": active_position["entry_ts"], "exit_ts": ts,
                        "outcome": outcome, "pnl": float(pnl), "entry": entry, "exit": exit_price,
                    })
                    cooldowns[sym] = 2
                    active_position = None
                    position_closed_this_iteration = True

        if position_closed_this_iteration:
            continue

        if active_position is None:
            candidates = []
            for symbol, data in symbol_arrays.items():
                if cooldowns[symbol] > 0:
                    continue

                h4_df, d1_df = data["h4_df"], data["d1_df"]
                if h4_df.empty or ts not in h4_df.index:
                    continue

                h4_bar = h4_df.loc[ts]
                d1_times = data["d1_times"]
                if len(d1_times) == 0:
                    continue
                d1_idx = d1_times.searchsorted(ts, side="right") - 1
                if d1_idx < 0:
                    continue
                d1_row = d1_df.iloc[d1_idx]

                ema50, ema200, slope = d1_row.get("ema50", np.nan), d1_row.get("ema200", np.nan), d1_row.get("ema50_slope", np.nan)
                if not all(np.isfinite([ema50, ema200, slope])):
                    continue

                daily_long = (d1_row["close"] > ema200) and (ema50 > ema200) and (slope > 0)
                daily_short = (d1_row["close"] < ema200) and (ema50 < ema200) and (slope < 0)

                bull_ebp = h4_bar.get("bull_ebp", False)
                bear_ebp = h4_bar.get("bear_ebp", False)

                if daily_long and bull_ebp:
                    h4_idx = h4_df.index.get_loc(ts)
                    if h4_idx >= 1:
                        prev_bar = h4_df.iloc[h4_idx - 1]
                        sl = min(h4_bar["low"], prev_bar["low"])
                        candidates.append((symbol, "LONG", ts, sl))

                elif daily_short and bear_ebp:
                    h4_idx = h4_df.index.get_loc(ts)
                    if h4_idx >= 1:
                        prev_bar = h4_df.iloc[h4_idx - 1]
                        sl = max(h4_bar["high"], prev_bar["high"])
                        candidates.append((symbol, "SHORT", ts, sl))

            if candidates:
                symbol, side, trigger_ts, sl = candidates[0]
                h4_df = symbol_arrays[symbol]["h4_df"]
                next_indices = h4_df.index[h4_df.index > trigger_ts]
                if len(next_indices) == 0:
                    continue

                entry_ts = next_indices[0]
                raw_open = float(h4_df.loc[entry_ts, "open"])
                entry = raw_open * (1.0 + SLIPPAGE) if side == "LONG" else raw_open * (1.0 - SLIPPAGE)
                risk = (entry - sl) if side == "LONG" else (sl - entry)

                if risk <= 0:
                    continue

                tp = entry + (RR * risk) if side == "LONG" else entry - (RR * risk)
                active_position = {
                    "symbol": symbol, "side": side,
                    "entry_ts": entry_ts, "entry": entry, "sl": sl, "tp": tp,
                }

    return trades, active_position


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HUNTER-V9-EBP: 4H ENGULFING BAR PATTERN BACKTEST ENGINE")
    print("=" * 70)

    all_symbol_data = {}
    for sym in SYMBOLS:
        try:
            df_raw = fetch_xt_futures_data(sym, data_dir)
            dfs = load_and_resample(df_raw)
            if dfs["4h"].empty:
                print(f"[WARNING] Skipping {sym}: 4H data is empty.")
                continue
            calculate_ebp_features(dfs)
            all_symbol_data[sym] = dfs
        except Exception as e:
            print(f"[ABORT] Error for symbol {sym}: {e}")
            sys.exit(1)

    if not all_symbol_data:
        print("[ABORT] No valid symbol data available. Exiting.")
        sys.exit(1)

    max_first_dt = max(dfs["4h"].index[0] for dfs in all_symbol_data.values() if not dfs["4h"].empty)
    max_last_dt = min(dfs["4h"].index[-1] for dfs in all_symbol_data.values() if not dfs["4h"].empty)
    
    total_span = max_last_dt - max_first_dt
    train_end = max_first_dt + total_span * 0.60
    val_end = train_end + total_span * 0.20

    print(f"[TIMELINE] OOS Test: {val_end} -> {max_last_dt}")

    trades, _ = run_backtest_4h_ebp(all_symbol_data, val_end, max_last_dt)

    total = len(trades)
    wins = sum(1 for t in trades if t["outcome"] == "WIN")
    losses = total - wins
    win_rate = (wins / total * 100.0) if total > 0 else 0.0
    net_pnl = sum(t["pnl"] for t in trades)

    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)

    consec, max_consec = 0, 0
    for t in sorted(trades, key=lambda x: pd.Timestamp(x["exit_ts"])):
        if t["outcome"] == "LOSS":
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0

    print("\n" + "=" * 40 + " OOS PERFORMANCE REPORT (4H EBP) " + "=" * 40)
    print(f"OOS Trades          : {total}")
    print(f"OOS Wins            : {wins}")
    print(f"OOS Losses          : {losses}")
    print(f"OOS Win Rate        : {win_rate:.2f}%")
    print(f"OOS Profit Factor   : {profit_factor:.2f}")
    print(f"OOS Net PnL         : ${net_pnl:,.2f}")
    print(f"OOS Max Loss Streak : {max_consec}")
    print("=" * 68)

    # Monthly Breakdown Report
    if trades:
        df_trades = pd.DataFrame(trades)
        df_trades["exit_month"] = pd.to_datetime(df_trades["exit_ts"]).dt.to_period("M")
        print("\n" + "=" * 35 + " MONTHLY PERFORMANCE BREAKDOWN " + "=" * 35)
        print(f"{'Month':<10} | {'Trades':<8} | {'Wins':<6} | {'Win Rate':<10} | {'PnL ($)':<10}")
        print("-" * 55)
        for month, group in df_trades.groupby("exit_month"):
            m_total = len(group)
            m_wins = sum(1 for x in group["outcome"] if x == "WIN")
            m_wr = (m_wins / m_total * 100.0) if m_total > 0 else 0.0
            m_pnl = group["pnl"].sum()
            print(f"{str(month):<10} | {m_total:<8} | {m_wins:<6} | {m_wr:>6.2f}%    | ${m_pnl:>9.2f}")
        print("=" * 55)

    oos_eligible = (total >= 100) and (win_rate > 45.0) and (profit_factor > 1.15) and (net_pnl > 0)
    print(f"FINAL ACCEPTANCE STATUS: {'ACCEPTED = TRUE' if oos_eligible else 'ACCEPTED = FALSE (REJECTED)'}")


if __name__ == "__main__":
    main()
