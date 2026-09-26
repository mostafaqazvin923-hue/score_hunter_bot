#!/usr/bin/env python3
"""
HUNTER-V4: FAILED AUCTION + VWAP RECLAIM ENGINE
- Zero Lookahead, Strict Causal Resampling, No Repainting.
- Focus: Daily Regime + 4H Structure + 1H Failed Auction Sweep + Daily VWAP Reclaim + Fixed RR 1:2.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

SYMBOLS = [
    "BTC", "ETH", "SOL", "SUI", "AVAX", "NEAR", "ADA",
    "BNB", "APT", "CRV", "ONDO", "PENDLE", "ICP", "WIF",
]

DATA_DIR = Path("data/xt_futures_v4")
OUT_DIR = DATA_DIR / "backtest_v4"

INITIAL_CAPITAL = 1000.0
MARGIN_PER_TRADE = 100.0
LEVERAGE = 50.0
RR = 2.0

FEE_RATE = 0.0007   # Configurable
SLIPPAGE = 0.0003  # Configurable


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    return p.parse_args()


def ensure_xt_data(data_dir: Path, symbols: list[str]):
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        import ccxt
    except ImportError:
        return

    exchange = ccxt.xt({'enableRateLimit': True})
    exchange.options['defaultType'] = 'swap'

    for symbol in symbols:
        file_path = data_dir / f"{symbol}_USDT_15m.csv"
        if file_path.exists() and file_path.stat().st_size > 200:
            continue
        try:
            ccxt_symbol = f"{symbol}/USDT:USDT"
            ohlcv = exchange.fetch_ohlcv(ccxt_symbol, timeframe='15m', limit=1500)
            if ohlcv:
                pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']).to_csv(file_path, index=False)
        except Exception:
            pass


def load_and_resample_data(data_dir: Path, symbol: str) -> dict[str, pd.DataFrame]:
    path = data_dir / f"{symbol}_USDT_15m.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing 15m file for {symbol}: {path}")

    df = pd.read_csv(path)
    if {"Date", "Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
        df = df.rename(columns={"Date": "timestamp", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})

    ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True) if np.issubdtype(df["timestamp"].dtype, np.number) else pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = ts
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")].copy()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    # Causal Resampling (closed bars)
    df_1h = df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    df_4h = df.resample('4h', closed='right', label='right').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    df_1d = df.resample('1d', closed='right', label='right').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()

    return {"15m": df, "1h": df_1h, "4h": df_4h, "1d": df_1d}


def calculate_indicators(dfs: dict[str, pd.DataFrame]):
    # --- 1D Indicators (Shifted for safety) ---
    d1 = dfs["1d"]
    prev_close_1d = d1["close"].shift(1)
    d1["ema200"] = prev_close_1d.ewm(span=200, adjust=False, min_periods=200).mean()
    d1["ema50"] = prev_close_1d.ewm(span=50, adjust=False, min_periods=50).mean()
    d1["ema50_slope"] = d1["ema50"].diff()

    # --- 4H Swings & Indicators ---
    h4 = dfs["4h"]
    prev_close_4h = h4["close"].shift(1)
    h4["ema50"] = prev_close_4h.ewm(span=50, adjust=False, min_periods=50).mean()
    h4["ema200"] = prev_close_4h.ewm(span=200, adjust=False, min_periods=200).mean()

    # Causal Swing Detection for 4H (using past 5 bars local extrema, confirmed with 1 bar lag)
    highs = h4["high"].values
    lows = h4["low"].values
    n_h4 = len(h4)
    swing_highs = [np.nan] * n_h4
    swing_lows = [np.nan] * n_h4

    for i in range(5, n_h4 - 1):
        # Local swing high
        if highs[i] == max(highs[i-5:i+1]):
            swing_highs[i] = highs[i]
        else:
            swing_highs[i] = swing_highs[i-1]

        # Local swing low
        if lows[i] == min(lows[i-5:i+1]):
            swing_lows[i] = lows[i]
        else:
            swing_lows[i] = swing_lows[i-1]

    h4["confirmed_swing_high"] = swing_highs
    h4["confirmed_swing_low"] = swing_lows

    # --- 1H Indicators ---
    h1 = dfs["1h"]
    prev_close_1h = h1["close"].shift(1)
    tr = pd.concat([
        h1["high"] - h1["low"],
        (h1["high"] - prev_close_1h).abs(),
        (h1["low"] - prev_close_1h).abs(),
    ], axis=1).max(axis=1)
    h1["atr14"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # 24-period rolling levels (excluding current candle via shift(1))
    h1["low24"] = h1["low"].shift(1).rolling(window=24).min()
    h1["high24"] = h1["high"].shift(1).rolling(window=24).max()
    
    h1["vol_sma20"] = h1["volume"].shift(1).rolling(window=20).mean()
    h1["candle_range"] = h1["high"] - h1["low"]
    h1["close_loc"] = np.where(h1["candle_range"] > 0, (h1["close"] - h1["low"]) / h1["candle_range"], 0.5)

    # Daily VWAP Calculation (Causal, resets at UTC day)
    typical_price = (h1["high"] + h1["low"] + h1["close"]) / 3.0
    tp_vol = typical_price * h1["volume"]
    
    # Group by UTC date and compute cumulative sum causally
    dates = h1.index.date
    h1["date"] = dates
    h1["cum_tp_vol"] = tp_vol.groupby(dates).cumsum()
    h1["cum_vol"] = h1["volume"].groupby(dates).cumsum()
    h1["daily_vwap"] = h1["cum_tp_vol"] / h1["cum_vol"].replace(0, np.nan)


def run_backtest(all_symbol_data: dict[str, dict[str, pd.DataFrame]]) -> list[dict]:
    all_1h_times = sorted(list(set().union(*(dfs["1h"].index for dfs in all_symbol_data.values()))))

    active_positions = {}
    trades = []
    symbol_cooldown = {sym: 0 for sym in all_symbol_data.keys()}

    for ts in all_1h_times:
        for sym in symbol_cooldown:
            if symbol_cooldown[sym] > 0:
                symbol_cooldown[sym] -= 1

        # 1. Manage active positions
        for symbol, pos in list(active_positions.items()):
            h1_df = all_symbol_data[symbol]["1h"]
            if ts not in h1_df.index:
                continue
            bar = h1_df.loc[ts]
            hi, lo = float(bar["high"]), float(bar["low"])
            side, sl, tp, entry = pos["side"], pos["sl"], pos["tp"], pos["entry"]
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
                    "symbol": symbol, "side": side, "entry_ts": pos["entry_ts"], "exit_ts": ts,
                    "outcome": outcome, "pnl": float(pnl), "entry": entry, "exit": exit_price,
                })
                symbol_cooldown[symbol] = 3  # Wait 3 completed 1H candles
                del active_positions[symbol]

        # 2. Look for new entries (< 3 open positions)
        if len(active_positions) < 3:
            candidates = []
            for symbol, dfs in all_symbol_data.items():
                if symbol in active_positions:
                    continue
                if symbol_cooldown[symbol] > 0:
                    continue

                h1_df, h4_df, d1_df = dfs["1h"], dfs["4h"], dfs["1d"]

                # Strict Lookahead Prevention: Use strictly completed higher timeframe bars (< ts)
                d1_subset = d1_df[d1_df.index < ts]
                h4_subset = h4_df[h4_df.index < ts]
                
                if d1_subset.empty or h4_subset.empty:
                    continue

                d1_row = d1_subset.iloc[-1]
                h4_row = h4_subset.iloc[-1]
                
                # Daily Regime Check
                d1_close = d1_row["close"]
                d1_ema200 = d1_row.get("ema200", np.nan)
                d1_ema50 = d1_row.get("ema50", np.nan)
                d1_slope = d1_row.get("ema50_slope", np.nan)

                if not all(np.isfinite([d1_ema200, d1_ema50, d1_slope])):
                    continue

                daily_long = (d1_close > d1_ema200) and (d1_ema50 > d1_ema200) and (d1_slope > 0)
                daily_short = (d1_close < d1_ema200) and (d1_ema50 < d1_ema200) and (d1_slope < 0)

                if not (daily_long or daily_short):
                    continue

                # 4H Structure Check
                confirmed_sh = h4_row.get("confirmed_swing_high", np.nan)
                confirmed_sl = h4_row.get("confirmed_swing_low", np.nan)
                h4_close = h4_row["close"]

                if not all(np.isfinite([confirmed_sh, confirmed_sl])):
                    continue

                # Check if latest confirmed swing high/low has been broken/intact
                # For long: latest confirmed swing high broken by at least one completed 4H candle closing above it
                h4_completed = h4_subset
                broken_sh = any(h4_completed["close"] > confirmed_sh)
                intact_sl = all(h4_completed["low"] >= confirmed_sl)

                broken_sl = any(h4_completed["close"] < confirmed_sl)
                intact_sh = all(h4_completed["high"] <= confirmed_sh)

                if daily_long:
                    h4_ok = broken_sh and intact_sl and (h4_close > confirmed_sh)
                    side = "LONG"
                else:
                    h4_ok = broken_sl and intact_sh and (h4_close < confirmed_sl)
                    side = "SHORT"

                if not h4_ok:
                    continue

                # 1H Failed Auction Setup on completed candles up to ts
                completed_1h = h1_df[h1_df.index < ts]
                if len(completed_1h) < 25:
                    continue

                prev_bar = completed_1h.iloc[-1] # Completed signal candle

                low24 = prev_bar.get("low24", np.nan)
                high24 = prev_bar.get("high24", np.nan)
                atr = prev_bar.get("atr14", np.nan)
                vol_sma = prev_bar.get("vol_sma20", np.nan)
                close_loc = prev_bar.get("close_loc", 0.5)
                vwap = prev_bar.get("daily_vwap", np.nan)

                if not all(np.isfinite([low24, high24, atr, vol_sma, vwap])):
                    continue

                p_low = prev_bar["low"]
                p_high = prev_bar["high"]
                p_close = prev_bar["close"]
                p_vol = prev_bar["volume"]

                if side == "LONG":
                    sweep_cond = (p_low < (low24 - 0.10 * atr)) and (p_low >= (low24 - 1.00 * atr))
                    reclaim_cond = p_close > low24
                    loc_cond = close_loc >= 0.65
                    vol_cond = p_vol >= 1.10 * vol_sma
                    vwap_cond = p_close > vwap

                    if not (sweep_cond and reclaim_cond and loc_cond and vol_cond and vwap_cond):
                        continue
                    sl = p_low - 0.10 * atr
                else:
                    sweep_cond = (p_high > (high24 + 0.10 * atr)) and (p_high <= (high24 + 1.00 * atr))
                    reclaim_cond = p_close < high24
                    loc_cond = close_loc <= 0.35
                    vol_cond = p_vol >= 1.10 * vol_sma
                    vwap_cond = p_close < vwap

                    if not (sweep_cond and reclaim_cond and loc_cond and vol_cond and vwap_cond):
                        continue
                    sl = p_high + 0.10 * atr

                candidates.append((symbol, side, prev_bar.name, sl, atr))

            # Open trades on next candle OPEN
            for symbol, side, trigger_ts, sl, atr_val in candidates:
                if len(active_positions) >= 3:
                    break
                if symbol in active_positions:
                    continue

                h1_df = all_symbol_data[symbol]["1h"]
                next_indices = h1_df.index[h1_df.index > trigger_ts]
                if len(next_indices) == 0:
                    continue

                entry_ts = next_indices[0]
                raw_open = float(h1_df.loc[entry_ts, "open"])
                entry = raw_open * (1.0 + SLIPPAGE) if side == "LONG" else raw_open * (1.0 - SLIPPAGE)

                risk = (entry - sl) if side == "LONG" else (sl - entry)

                # Risk validation: 0.4 * ATR14 <= risk <= 1.8 * ATR14
                if not (0.4 * atr_val <= risk <= 1.8 * atr_val):
                    continue

                tp = entry + (RR * risk) if side == "LONG" else entry - (RR * risk)

                active_positions[symbol] = {
                    "symbol": symbol, "side": side,
                    "entry_ts": entry_ts, "entry": entry, "sl": sl, "tp": tp,
                }

    return trades


def main():
    args = parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("HUNTER-V4: FAILED AUCTION + VWAP RECLAIM ENGINE")
    print("=" * 88)

    ensure_xt_data(data_dir, SYMBOLS)

    all_symbol_data = {}
    for sym in SYMBOLS:
        try:
            dfs = load_and_resample_data(data_dir, sym)
            calculate_indicators(dfs)
            all_symbol_data[sym] = dfs
        except Exception as e:
            print(f"Skipping {sym}: {e}")

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

    print("\n" + "=" * 40 + " OVERALL RESULTS " + "=" * 40)
    print(f"Total Trades        : {total}")
    print(f"Wins                : {wins}")
    print(f"Losses              : {losses}")
    print(f"Win Rate            : {win_rate:.2f}%")
    print(f"Profit Factor       : {profit_factor:.2f}")
    print(f"Net PnL             : ${net_pnl:,.2f}")
    print(f"Max Consecutive Loss: {max_consec}")
    print("=" * 97)


if __name__ == "__main__":
    main()
