#!/usr/init/env python3
"""
HUNTER-XT-STYLE4-ULTRA-STABLE
Single-Position Breakout Momentum Engine with Fixed RR = 2.0 (1h Timeframe)
- Focus: Win Rate > 50%, Max Loss Streak <= 4, Fixed RR 1:2, Zero Correlation Risk
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

DATA_DIR = Path("data/xt_futures_style4_stable")
OUT_DIR = DATA_DIR / "backtest_style4_stable"

INITIAL_EQUITY = 1000.0
TRADE_MARGIN = 100.0
LEVERAGE = 50.0

FEE_RATE = 0.0007
SLIPPAGE = 0.0003
RR = 2.0                      # Fixed Risk-Reward exactly 1:2
MAX_OPEN_POSITIONS = 1        # CRITICAL: Single position eliminates correlated simultaneous streaks
CIRCUIT_BREAKER_COOLDOWN = 3  # Cooldown hours after a loss


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
        file_path = data_dir / f"{symbol}_USDT_1h.csv"
        if file_path.exists() and file_path.stat().st_size > 200:
            continue
        try:
            ccxt_symbol = f"{symbol}/USDT:USDT"
            ohlcv = exchange.fetch_ohlcv(ccxt_symbol, timeframe='1h', limit=1500)
            if ohlcv:
                pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']).to_csv(file_path, index=False)
        except Exception:
            pass


def load_xt_csv(data_dir: Path, asset: str) -> pd.DataFrame:
    path = data_dir / f"{asset}_USDT_1h.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    if {"Date", "Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
        df = df.rename(columns={"Date": "timestamp", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})

    required = ["timestamp", "open", "high", "low", "close", "volume"]
    ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True) if np.issubdtype(df["timestamp"].dtype, np.number) else pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = ts
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")].copy()

    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=required[1:])


def calculate_breakout_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates breakout levels, structural lows, and volume using shift(1)."""
    prev_close = df["close"].shift(1)
    
    # Structural swing low for stop loss (past 5 bars)
    df["swing_low"] = df["low"].shift(1).rolling(window=5).min()

    # Breakout filter: Price breaking above the highest high of the last 24 hours
    df["high_24h"] = df["high"].shift(1).rolling(window=24).max()
    df["breakout"] = prev_close >= df["high_24h"]

    # Trend filter
    df["ema20"] = prev_close.ewm(span=20, adjust=False).mean()
    df["ema50"] = prev_close.ewm(span=50, adjust=False).mean()
    df["trend_ok"] = (prev_close > df["ema20"]) & (df["ema20"] > df["ema50"])

    # Volume confirmation
    prev_volume = df["volume"].shift(1)
    df["avg_volume"] = prev_volume.rolling(window=20).mean()
    df["volume_ok"] = prev_volume > (1.2 * df["avg_volume"])

    df["setup_valid"] = df["trend_ok"] & df["breakout"] & df["volume_ok"]
    return df


def run_stable_backtest(all_data: dict[str, pd.DataFrame]) -> list[dict]:
    all_times = sorted(list(set().union(*(df.index for df in all_data.values()))))
    
    active_positions = {}
    trades = []
    equity = INITIAL_EQUITY
    peak_equity = INITIAL_EQUITY
    
    cooldown_counter = 0
    btc_df = all_data.get("BTC")

    for idx, ts in enumerate(all_times):
        if cooldown_counter > 0:
            cooldown_counter -= 1

        btc_bullish = True
        if btc_df is not None and ts in btc_df.index:
            btc_bullish = bool(btc_df.loc[ts, "trend_ok"])

        # 1. Manage active position (Only 1 active max)
        for symbol, pos in list(active_positions.items()):
            df = all_data[symbol]
            if ts not in df.index:
                continue
            bar = df.loc[ts]
            hi, lo = float(bar["high"]), float(bar["low"])
            side, sl, tp, entry = pos["side"], pos["sl"], pos["tp"], pos["entry"]
            notional = TRADE_MARGIN * LEVERAGE

            hit_sl = lo <= sl if side == "LONG" else hi >= sl
            hit_tp = hi >= tp if side == "LONG" else lo <= tp

            if hit_sl or hit_tp:
                exit_price, outcome = (sl, "LOSS") if (hit_sl and hit_tp or hit_sl) else (tp, "WIN")

                gross = (exit_price - entry) / entry * notional if side == "LONG" else (entry - exit_price) / entry * notional
                fees = notional * FEE_RATE * 2.0
                pnl = gross - fees

                equity += pnl
                peak_equity = max(peak_equity, equity)

                if outcome == "LOSS":
                    cooldown_counter = CIRCUIT_BREAKER_COOLDOWN

                trades.append({
                    "symbol": symbol, "side": side, "entry_ts": pos["entry_ts"], "exit_ts": ts,
                    "outcome": outcome, "pnl": float(pnl), "equity": float(equity),
                })
                del active_positions[symbol]

        # 2. Execution (Strictly 1 position max, Breakout + Fixed RR 2.0)
        if btc_bullish and cooldown_counter == 0 and len(active_positions) < MAX_OPEN_POSITIONS:
            candidates = []
            for symbol, df in all_data.items():
                if ts not in df.index:
                    continue
                row = df.loc[ts]
                if not row.get("setup_valid", False):
                    continue
                if symbol in active_positions:
                    continue
                
                # Use distance from EMA20 as quality ranking score
                close_val = row.get("close", 0.0)
                ema_val = row.get("ema20", 0.0)
                score = (close_val - ema_val) / ema_val if ema_val > 0 else 0.0
                candidates.append((symbol, score))

            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                symbol = candidates[0][0]
                
                df = all_data[symbol]
                next_indices = df.index[df.index > ts]
                if len(next_indices) > 0:
                    next_ts = next_indices[0]
                    entry = float(df.loc[next_ts, "open"]) * (1.0 + SLIPPAGE)
                    row = df.loc[ts]
                    
                    swing_low = row.get("swing_low", np.nan)
                    if not np.isfinite(swing_low) or swing_low >= entry:
                        sl = entry * 0.98
                    else:
                        sl = swing_low * 0.995

                    risk = entry - sl
                    if risk > 0:
                        tp = entry + (RR * risk)  # Fixed RR = 2.0

                        active_positions[symbol] = {
                            "symbol": symbol, "side": "LONG",
                            "entry_ts": next_ts, "entry": entry, "sl": sl, "tp": tp,
                        }

    return trades


def main():
    args = parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("HUNTER-XT-STYLE4-ULTRA-STABLE (Single Position + Breakout Momentum + Fixed RR 1:2)")
    print("=" * 88)

    ensure_xt_data(data_dir, SYMBOLS)

    all_data = {}
    for asset in SYMBOLS:
        try:
            df = load_xt_csv(data_dir, asset)
            all_data[asset] = calculate_breakout_features(df)
        except Exception:
            pass

    trades = run_stable_backtest(all_data)
    
    total = len(trades)
    wins = sum(1 for t in trades if t["outcome"] == "WIN")
    win_rate = (wins / total * 100.0) if total > 0 else 0.0
    net_pnl = sum(t["pnl"] for t in trades)

    consec, max_consec = 0, 0
    for t in sorted(trades, key=lambda x: pd.Timestamp(x["exit_ts"])):
        if t["outcome"] == "LOSS":
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0

    print("\n===== STYLE 4 ULTRA-STABLE RESULTS =====")
    print(f"Closed Trades : {total}")
    print(f"Win Rate      : {win_rate:.2f}% (Target: >50%)")
    print(f"Net PnL       : ${net_pnl:,.2f}")
    print(f"Max Loss Streak: {max_consec} (Target: <=4)")
    print(f"Final Equity  : ${INITIAL_EQUITY + net_pnl:,.2f}")
    print("=" * 88)


if __name__ == "__main__":
    main()
