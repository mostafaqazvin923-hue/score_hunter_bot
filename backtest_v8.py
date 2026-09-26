#!/usr/bin/env python3
"""
HUNTER-V4: 1-YEAR FAILED AUCTION + VWAP RECLAIM BACKTEST ENGINE
- Market: XT USDT-M Futures (Perpetual Swap)
- Timeframe Architecture: 15m raw -> 1H, 4H, 1D causal construction
- Zero Lookahead, Zero Leakage, Zero Repainting
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time
import ccxt
import numpy as np
import pandas as pd

SYMBOLS = [
    "BTC_USDT", "ETH_USDT", "SOL_USDT", "SUI_USDT", "AVAX_USDT",
    "NEAR_USDT", "ADA_USDT", "BNB_USDT", "APT_USDT", "CRV_USDT",
    "ONDO_USDT", "PENDLE_USDT", "ICP_USDT", "WIF_USDT"
]

DATA_DIR = Path("data/xt_futures_v4")
WARMUP_DAYS = 60
TOTAL_DAYS = 365  # بازه دقیق یک‌ساله فیوچرز

INITIAL_CAPITAL = 1000.0
MARGIN_PER_TRADE = 100.0
LEVERAGE = 50.0
RR = 2.0

FEE_RATE = 0.0007
SLIPPAGE = 0.0003


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(DATA_DIR))
    return p.parse_args()


def fetch_xt_futures_data(symbol: str, data_dir: Path) -> pd.DataFrame:
    file_path = data_dir / f"{symbol}_15m.csv"
    
    if file_path.exists() and file_path.stat().st_size > 1000:
        df_cached = pd.read_csv(file_path)
        if len(df_cached) > 20000:
            df_cached["timestamp"] = pd.to_datetime(df_cached["timestamp"], unit="ms", utc=True)
            df_cached = df_cached.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
            span_days = (df_cached["timestamp"].iloc[-1] - df_cached["timestamp"].iloc[0]).days
            if span_days >= 300:
                print(f"Using valid cached futures data for {symbol} (Rows: {len(df_cached)})")
                return df_cached

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading 1-year XT Futures (Swap) data for {symbol} via CCXT...")

    # تنظیم دقیق صرافی روی حالت فیوچرز سواپ
    exchange = ccxt.xt({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',
        }
    })

    since = exchange.milliseconds() - int((TOTAL_DAYS + WARMUP_DAYS) * 86400 * 1000)
    all_ohlcv = []

    try:
        while True:
            # در CCXT برای بخش سواپ XT فرمت نماد ممکن است به صورت BTC/USDT:USDT باشد که خود ccxt مدیریت می‌کند
            market_symbol = symbol.replace("_", "/")
            ohlcv = exchange.fetch_ohlcv(market_symbol, timeframe='15m', since=since, limit=1500)
            if not ohlcv:
                break
            since = ohlcv[-1][0] + 1
            all_ohlcv.extend(ohlcv)
            if len(ohlcv) < 1500:
                break
            time.sleep(0.2)
    except Exception as e:
        print(f"CCXT Futures Error for {symbol}: {e}")

    if not all_ohlcv:
        raise RuntimeError(f"Failed to fetch futures data for {symbol} from XT.")

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    df.to_csv(file_path, index=False)
    return df


def validate_and_print_dataset(df: pd.DataFrame, symbol: str):
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    row_count = len(df)
    first_ts = df["timestamp_dt"].iloc[0]
    last_ts = df["timestamp_dt"].iloc[-1]
    days = (last_ts - first_ts).total_seconds() / 86400.0
    
    time_diffs = df["timestamp_dt"].diff().dt.total_seconds().dropna()
    gaps = int((time_diffs > 900).sum())

    print(f"FUTURES SYMBOL: {symbol} | Rows: {row_count} | Days: {days:.1f} | Gaps: {gaps}")

    if row_count < 20000 or days < 300:
        raise RuntimeError(f"Futures dataset for {symbol} is incomplete.")


def load_and_resample(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp_dt").sort_index()
    df = df[~df.index.duplicated(keep="last")].copy()

    df_1h = df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    df_4h = df.resample('4h', closed='right', label='right').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    df_1d = df.resample('1d', closed='right', label='right').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()

    return {"15m": df, "1h": df_1h, "4h": df_4h, "1d": df_1d}


def calculate_indicators(dfs: dict[str, pd.DataFrame]):
    d1 = dfs["1d"]
    prev_close_1d = d1["close"].shift(1)
    d1["ema50"] = prev_close_1d.ewm(span=50, adjust=False, min_periods=50).mean()
    d1["ema200"] = prev_close_1d.ewm(span=200, adjust=False, min_periods=200).mean()
    d1["ema50_slope"] = d1["ema50"].diff()

    h4 = dfs["4h"]
    highs = h4["high"].values
    lows = h4["low"].values
    n_h4 = len(h4)
    swing_highs = [np.nan] * n_h4
    swing_lows = [np.nan] * n_h4

    last_sh, last_sl = np.nan, np.nan
    for i in range(2, n_h4):
        if highs[i-2] >= highs[i-1] and highs[i-2] >= highs[i]:
            last_sh = highs[i-2]
        if lows[i-2] <= lows[i-1] and lows[i-2] <= lows[i]:
            last_sl = lows[i-2]
        swing_highs[i] = last_sh
        swing_lows[i] = last_sl

    h4["confirmed_sh"] = swing_highs
    h4["confirmed_sl"] = swing_lows

    h1 = dfs["1h"]
    prev_close_1h = h1["close"].shift(1)
    tr = pd.concat([
        h1["high"] - h1["low"],
        (h1["high"] - prev_close_1h).abs(),
        (h1["low"] - prev_close_1h).abs(),
    ], axis=1).max(axis=1)
    h1["atr14"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    h1["high24"] = h1["high"].shift(1).rolling(window=24).max()
    h1["low24"] = h1["low"].shift(1).rolling(window=24).min()
    h1["vol_sma20"] = h1["volume"].shift(1).rolling(window=20).mean()

    tp = (h1["high"] + h1["low"] + h1["close"]) / 3.0
    dates = h1.index.date
    h1["date"] = dates
    h1["cum_tp_vol"] = (tp * h1["volume"]).groupby(dates).cumsum()
    h1["cum_vol"] = h1["volume"].groupby(dates).cumsum()
    h1["vwap"] = h1["cum_tp_vol"] / h1["cum_vol"].replace(0, np.nan)


def run_backtest(all_symbol_data: dict[str, dict[str, pd.DataFrame]]) -> list[dict]:
    all_1h_times = sorted(list(set().union(*(dfs["1h"].index for dfs in all_symbol_data.values()))))

    active_position = None
    trades = []
    cooldowns = {sym: 0 for sym in all_symbol_data.keys()}

    symbol_arrays = {}
    for sym, dfs in all_symbol_data.items():
        h1 = dfs["1h"]
        h4 = dfs["4h"]
        d1 = dfs["1d"]
        symbol_arrays[sym] = {
            "h1_times": h1.index,
            "h1_df": h1,
            "h4_times": h4.index,
            "h4_df": h4,
            "d1_times": d1.index,
            "d1_df": d1,
        }

    for ts in all_1h_times:
        for sym in cooldowns:
            if cooldowns[sym] > 0:
                cooldowns[sym] -= 1

        if active_position is not None:
            sym = active_position["symbol"]
            h1_df = symbol_arrays[sym]["h1_df"]
            if ts in h1_df.index:
                bar = h1_df.loc[ts]
                hi, lo = float(bar["high"]), float(bar["low"])
                side, sl, tp, entry = active_position["side"], active_position["sl"], active_position["tp"], active_position["entry"]
                notional = MARGIN_PER_TRADE * LEVERAGE

                hit_sl = lo <= sl if side == "LONG" else hi >= sl
                hit_tp = hi >= tp if side == "LONG" else lo <= tp

                if hit_sl or hit_tp:
                    if hit_sl and hit_tp:
                        outcome = "LOSS"
                        exit_price = sl
                    elif hit_sl:
                        outcome = "LOSS"
                        exit_price = sl
                    else:
                        outcome = "WIN"
                        exit_price = tp

                    gross = (exit_price - entry) / entry * notional if side == "LONG" else (entry - exit_price) / entry * notional
                    fees = notional * FEE_RATE * 2.0
                    pnl = gross - fees

                    trades.append({
                        "symbol": sym, "side": side, "entry_ts": active_position["entry_ts"], "exit_ts": ts,
                        "outcome": outcome, "pnl": float(pnl), "entry": entry, "exit": exit_price,
                    })
                    cooldowns[sym] = 3
                    active_position = None

        if active_position is None:
            candidates = []
            for symbol, data in symbol_arrays.items():
                if cooldowns[symbol] > 0:
                    continue

                h1_df = data["h1_df"]
                h4_df = data["h4_df"]
                d1_df = data["d1_df"]

                d1_idx = data["d1_times"].searchsorted(ts, side="right") - 1
                h4_idx = data["h4_times"].searchsorted(ts, side="right") - 1
                h1_idx = data["h1_times"].searchsorted(ts, side="right") - 1

                if d1_idx < 0 or h4_idx < 0 or h1_idx < 25:
                    continue

                d1_row = d1_df.iloc[d1_idx]
                h4_sub = h4_df.iloc[:h4_idx]
                h1_sub = h1_df.iloc[:h1_idx]

                if len(h4_sub) == 0 or len(h1_sub) < 25:
                    continue

                h4_row = h4_sub.iloc[-1]

                d1_close = d1_row["close"]
                ema50, ema200, slope = d1_row.get("ema50", np.nan), d1_row.get("ema200", np.nan), d1_row.get("ema50_slope", np.nan)
                if not all(np.isfinite([ema50, ema200, slope])):
                    continue

                daily_long = (d1_close > ema200) and (ema50 > ema200) and (slope > 0)
                daily_short = (d1_close < ema200) and (ema50 < ema200) and (slope < 0)

                if not (daily_long or daily_short):
                    continue

                c_sh, c_sl, h4_close = h4_row.get("confirmed_sh", np.nan), h4_row.get("confirmed_sl", np.nan), h4_row["close"]
                if not all(np.isfinite([c_sh, c_sl])):
                    continue

                h4_completed = h4_sub
                if daily_long:
                    h4_ok = any(h4_completed["close"] > c_sh) and all(h4_completed["low"] >= c_sl) and (h4_close > c_sh)
                    side = "LONG"
                else:
                    h4_ok = any(h4_completed["close"] < c_sl) and all(h4_completed["high"] <= c_sh) and (h4_close < c_sl)
                    side = "SHORT"

                if not h4_ok:
                    continue

                prev_bar = h1_sub.iloc[-1]
                prev_prev_bar = h1_sub.iloc[-2]

                low24, high24 = prev_bar.get("low24", np.nan), prev_bar.get("high24", np.nan)
                atr = prev_bar.get("atr14", np.nan)
                vol_sma = prev_bar.get("vol_sma20", np.nan)
                vwap = prev_bar.get("vwap", np.nan)
                prev_vwap = prev_prev_bar.get("vwap", np.nan)

                if not all(np.isfinite([low24, high24, atr, vol_sma, vwap, prev_vwap])):
                    continue

                p_low, p_high, p_close, p_vol = prev_bar["low"], prev_bar["high"], prev_bar["close"], prev_bar["volume"]

                if side == "LONG":
                    sweep = (p_low < low24 - 0.10 * atr) and (p_low >= low24 - 1.00 * atr)
                    reclaim = p_close > low24
                    vol_ok = p_vol >= 1.10 * vol_sma
                    vwap_reclaim = (prev_prev_bar["close"] <= prev_vwap) and (p_close > vwap)

                    if not (sweep and reclaim and vol_ok and vwap_reclaim):
                        continue
                    sl = p_low - 0.20 * atr
                else:
                    sweep = (p_high > high24 + 0.10 * atr) and (p_high <= high24 + 1.00 * atr)
                    reclaim = p_close < high24
                    vol_ok = p_vol >= 1.10 * vol_sma
                    vwap_reclaim = (prev_prev_bar["close"] >= prev_vwap) and (p_close < vwap)

                    if not (sweep and reclaim and vol_ok and vwap_reclaim):
                        continue
                    sl = p_high + 0.20 * atr

                candidates.append((symbol, side, prev_bar.name, sl, atr))

            if candidates:
                symbol, side, trigger_ts, sl, atr_val = candidates[0]
                h1_df = symbol_arrays[symbol]["h1_df"]
                next_indices = h1_df.index[h1_df.index > trigger_ts]
                if len(next_indices) == 0:
                    continue

                entry_ts = next_indices[0]
                raw_open = float(h1_df.loc[entry_ts, "open"])
                entry = raw_open * (1.0 + SLIPPAGE) if side == "LONG" else raw_open * (1.0 - SLIPPAGE)
                risk = (entry - sl) if side == "LONG" else (sl - entry)

                if risk <= 0:
                    continue

                tp = entry + (RR * risk) if side == "LONG" else entry - (RR * risk)
                active_position = {
                    "symbol": symbol, "side": side,
                    "entry_ts": entry_ts, "entry": entry, "sl": sl, "tp": tp,
                }

    return trades


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HUNTER-V4: 1-YEAR FUTURES FAILED AUCTION + VWAP RECLAIM BACKTEST")
    print("=" * 70)

    all_symbol_data = {}
    for sym in SYMBOLS:
        try:
            df_raw = fetch_xt_futures_data(sym, data_dir)
            validate_and_print_dataset(df_raw, sym)
            dfs = load_and_resample(df_raw)
            calculate_indicators(dfs)
            all_symbol_data[sym] = dfs
        except Exception as e:
            print(f"Error loading futures for {sym}: {e}")

    if not all_symbol_data:
        print("No valid futures symbol data loaded. Aborting.")
        return

    trades = run_backtest(all_symbol_data)

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

    print("\n" + "=" * 40 + " 1-YEAR FUTURES RESULTS " + "=" * 40)
    print(f"Total Trades        : {total}")
    print(f"Wins                : {wins}")
    print(f"Losses              : {losses}")
    print(f"Win Rate            : {win_rate:.2f}%")
    print(f"Profit Factor       : {profit_factor:.2f}")
    print(f"Net PnL             : ${net_pnl:,.2f}")
    print(f"Max Consecutive Loss: {max_consec}")
    print("=" * 61)


if __name__ == "__main__":
    main()
