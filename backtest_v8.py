#!/usr/bin/env python3
"""
HUNTER-V3: Trend Continuation After Compression Engine (AUDITED & SECURED)
- Zero Lookahead, Strict Causal Resampling, No Repainting.
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

DATA_DIR = Path("data/xt_futures_v3")
OUT_DIR = DATA_DIR / "backtest_v3"

INITIAL_CAPITAL = 1000.0
MARGIN_PER_TRADE = 100.0
LEVERAGE = 50.0
RR = 2.0

FEE_RATE = 0.0007
SLIPPAGE = 0.0003


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

    # --- 4H Indicators (Shifted for safety) ---
    h4 = dfs["4h"]
    prev_close_4h = h4["close"].shift(1)
    h4["ema50"] = prev_close_4h.ewm(span=50, adjust=False, min_periods=50).mean()
    h4["ema200"] = prev_close_4h.ewm(span=200, adjust=False, min_periods=200).mean()

    # --- 1H Indicators ---
    h1 = dfs["1h"]
    prev_close_1h = h1["close"].shift(1)
    tr = pd.concat([
        h1["high"] - h1["low"],
        (h1["high"] - prev_close_1h).abs(),
        (h1["low"] - prev_close_1h).abs(),
    ], axis=1).max(axis=1)
    h1["atr14"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    h1["atr_sma48"] = h1["atr14"].shift(1).rolling(window=48).mean()
    
    h1["candle_range"] = h1["high"] - h1["low"]
    h1["avg_range_20"] = h1["candle_range"].shift(1).rolling(window=20).mean()
    
    h1["ema20"] = prev_close_1h.ewm(span=20, adjust=False, min_periods=20).mean()
    h1["vol_sma20"] = h1["volume"].shift(1).rolling(window=20).mean()
    h1["body"] = (h1["close"] - h1["open"]).abs()


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
                symbol_cooldown[symbol] = 4
                del active_positions[symbol]

        # 2. Look for new entries
        if len(active_positions) < 3:
            candidates = []
            for symbol, dfs in all_symbol_data.items():
                if symbol in active_positions:
                    continue
                if symbol_cooldown[symbol] > 0:
                    continue

                h1_df, h4_df, d1_df = dfs["1h"], dfs["4h"], dfs["1d"]

                # STRICT LOOKAHEAD PREVENTION: Use strictly completed higher timeframe bars (< ts)
                d1_subset = d1_df[d1_df.index < ts]
                h4_subset = h4_df[h4_df.index < ts]
                
                if d1_subset.empty or h4_subset.empty:
                    continue

                d1_row = d1_subset.iloc[-1]
                h4_row = h4_subset.iloc[-1]
                
                # Daily Trend Check
                d1_close = d1_row["close"]
                d1_ema200 = d1_row.get("ema200", np.nan)
                d1_ema50 = d1_row.get("ema50", np.nan)
                d1_slope = d1_row.get("ema50_slope", np.nan)

                if not (np.isfinite(d1_ema200) and np.isfinite(d1_ema50) and np.isfinite(d1_slope)):
                    continue

                daily_long = (d1_close > d1_ema200) and (d1_ema50 > d1_ema200) and (d1_slope > 0)
                daily_short = (d1_close < d1_ema200) and (d1_ema50 < d1_ema200) and (d1_slope < 0)

                if not (daily_long or daily_short):
                    continue

                # 4H Confirmation Check
                h4_close = h4_row["close"]
                h4_ema50 = h4_row.get("ema50", np.nan)
                h4_ema200 = h4_row.get("ema200", np.nan)
                if not (np.isfinite(h4_ema50) and np.isfinite(h4_ema200)):
                    continue

                h4_last_3 = h4_subset.tail(3)
                if len(h4_last_3) < 3:
                    continue

                if daily_long:
                    h4_ok = (h4_ema50 > h4_ema200) and (h4_close > h4_ema50) and any(h4_last_3["close"] > h4_last_3["ema50"])
                    side = "LONG"
                else:
                    h4_ok = (h4_ema50 < h4_ema200) and (h4_close < h4_ema50) and any(h4_last_3["close"] < h4_last_3["ema50"])
                    side = "SHORT"

                if not h4_ok:
                    continue

                # 1H Compression & Trigger on completed candles up to ts
                completed_1h = h1_df[h1_df.index < ts]
                if len(completed_1h) < 25:
                    continue

                prev_bar = completed_1h.iloc[-1] # Last completed 1H candle (Trigger candidate)

                atr = prev_bar.get("atr14", np.nan)
                atr_sma = prev_bar.get("atr_sma48", np.nan)
                candle_rng = prev_bar.get("candle_range", np.nan)
                avg_rng = prev_bar.get("avg_range_20", np.nan)
                ema20 = prev_bar.get("ema20", np.nan)
                vol_sma = prev_bar.get("vol_sma20", np.nan)

                if not all(np.isfinite([atr, atr_sma, candle_rng, avg_rng, ema20, vol_sma])):
                    continue

                compression_ok = (atr < atr_sma) and (candle_rng < avg_rng)
                if not compression_ok:
                    continue

                p_close = prev_bar["close"]
                p_open = prev_bar["open"]
                p_vol = prev_bar["volume"]
                p_body = prev_bar["body"]

                prev_to_prev_high = completed_1h.iloc[-2]["high"] if len(completed_1h) >= 2 else prev_bar["high"]
                prev_to_prev_low = completed_1h.iloc[-2]["low"] if len(completed_1h) >= 2 else prev_bar["low"]

                if side == "LONG":
                    is_bullish = p_close > p_open
                    cond_trigger = is_bullish and (p_close > prev_to_prev_high) and (p_body >= 0.5 * candle_rng) and (p_vol >= 1.2 * vol_sma)
                    not_too_far = prev_bar["low"] >= (ema20 - 2.0 * atr)
                    if not (cond_trigger and not_too_far):
                        continue
                    sl = prev_bar["low"]
                else:
                    is_bearish = p_close < p_open
                    cond_trigger = is_bearish and (p_close < prev_to_prev_low) and (p_body >= 0.5 * candle_rng) and (p_vol >= 1.2 * vol_sma)
                    not_too_far = prev_bar["high"] <= (ema20 + 2.0 * atr)
                    if not (cond_trigger and not_too_far):
                        continue
                    sl = prev_bar["high"]

                candidates.append((symbol, side, prev_bar.name, sl))

            # Open trades on next candle OPEN
            for symbol, side, trigger_ts, sl in candidates:
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
                atr_val = h1_df.loc[trigger_ts].get("atr14", risk)

                if not (0.5 * atr_val <= risk <= 2.0 * atr_val):
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
    print("HUNTER-V3: Trend Continuation After Compression Engine (AUDITED & SECURED)")
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

    print("\n" + "=" * 40 + " AUDITED RESULTS " + "=" * 40)
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
