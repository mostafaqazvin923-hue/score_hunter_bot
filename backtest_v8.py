#!/usr/bin/env python3
"""
HUNTER-XT-STYLE3-HIGH-FREQUENCY-SQUEEZE
High-Frequency Volatility Squeeze & Momentum Expansion Engine (1h Timeframe)
- Focus: High Trade Count, Win Rate > 50%, Strict Streak Control
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

CORRELATION_CLUSTERS = {
    "BTC": "MAJOR", "ETH": "MAJOR",
    "SOL": "L1", "SUI": "L1", "AVAX": "L1", "NEAR": "L1", "ADA": "L1", "BNB": "L1", "APT": "L1",
    "CRV": "DEFI", "ONDO": "DEFI", "PENDLE": "DEFI",
    "ICP": "OTHER",
    "WIF": "MEME",
}

DATA_DIR = Path("data/xt_futures_style3_hf")
OUT_DIR = DATA_DIR / "backtest_style3_hf"

INITIAL_EQUITY = 1000.0
TRADE_MARGIN = 100.0
LEVERAGE = 50.0

FEE_RATE = 0.0007
SLIPPAGE = 0.0003
RR = 2.0
ATR_N = 14
MAX_OPEN_POSITIONS = 3
MAX_ONE_PER_CLUSTER = True
CIRCUIT_BREAKER_COOLDOWN = 8   # Smart cooling to protect streaks


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


def calculate_hf_squeeze_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates High-Frequency Bandwidth Squeeze & Momentum Breakout using shift(1)."""
    prev_close = df["close"].shift(1)
    
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / ATR_N, adjust=False, min_periods=ATR_N).mean()

    # Bollinger Bands (20, 2.0)
    sma20 = prev_close.rolling(window=20).mean()
    std20 = prev_close.rolling(window=20).std()
    bb_width = (4.0 * std20) / (sma20 + 1e-8)

    # Squeeze: BB Width is in the lower 25 percentile of its recent 50-period range
    bb_width_min = bb_width.rolling(window=50).quantile(0.25)
    df["squeeze_cond"] = bb_width <= bb_width_min

    # Trend & Momentum filter
    df["ema_trend"] = prev_close.ewm(span=30, adjust=False).mean()
    df["trend_ok"] = prev_close > df["ema_trend"]

    # Breakout execution: Squeeze was active recently, now price breaks above 20 SMA with volume surge
    df["breakout_long"] = df["squeeze_cond"].shift(1) & (prev_close > sma20) & (prev_close > prev_close.shift(1))

    # Volume surge confirmation
    prev_volume = df["volume"].shift(1)
    df["avg_volume"] = prev_volume.rolling(window=20).mean()
    df["volume_surge"] = prev_volume > (1.1 * df["avg_volume"])

    df["setup_valid"] = df["trend_ok"] & df["breakout_long"] & df["volume_surge"]
    return df


def run_hf_squeeze_backtest(all_data: dict[str, pd.DataFrame]) -> list[dict]:
    all_times = sorted(list(set().union(*(df.index for df in all_data.values()))))
    
    active_positions = {}
    trades = []
    equity = INITIAL_EQUITY
    peak_equity = INITIAL_EQUITY
    
    consecutive_losses = 0
    cooldown_counter = 0

    def cluster_active(cluster_name):
        return any(CORRELATION_CLUSTERS.get(p["symbol"], "OTHER") == cluster_name for p in active_positions.values())

    for idx, ts in enumerate(all_times):
        if cooldown_counter > 0:
            cooldown_counter -= 1

        # 1. Manage active positions
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
                    consecutive_losses += 1
                    if consecutive_losses >= 2:
                        cooldown_counter = CIRCUIT_BREAKER_COOLDOWN
                else:
                    consecutive_losses = 0

                trades.append({
                    "symbol": symbol, "side": side, "entry_ts": pos["entry_ts"], "exit_ts": ts,
                    "outcome": outcome, "pnl": float(pnl), "equity": float(equity),
                })
                del active_positions[symbol]

        # 2. Entry Execution
        if cooldown_counter == 0 and len(active_positions) < MAX_OPEN_POSITIONS:
            for symbol, df in all_data.items():
                if ts not in df.index:
                    continue
                row = df.loc[ts]
                if not row.get("setup_valid", False):
                    continue

                cluster = CORRELATION_CLUSTERS.get(symbol, "OTHER")
                if symbol in active_positions or (MAX_ONE_PER_CLUSTER and cluster_active(cluster)):
                    continue
                if len(active_positions) >= MAX_OPEN_POSITIONS:
                    break

                next_indices = df.index[df.index > ts]
                if len(next_indices) == 0:
                    continue
                
                next_ts = next_indices[0]
                entry = float(df.loc[next_ts, "open"]) * (1.0 + SLIPPAGE)
                atr = row["atr"]
                if not np.isfinite(atr) or atr <= 0:
                    continue

                sl = entry - (1.5 * atr)
                risk = entry - sl
                if risk <= 0:
                    continue
                
                tp = entry + (RR * risk)

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
    print("HUNTER-XT-STYLE3-HIGH-FREQUENCY-SQUEEZE")
    print("=" * 88)

    ensure_xt_data(data_dir, SYMBOLS)

    all_data = {}
    for asset in SYMBOLS:
        try:
            df = load_xt_csv(data_dir, asset)
            all_data[asset] = calculate_hf_squeeze_features(df)
        except Exception:
            pass

    trades = run_hf_squeeze_backtest(all_data)
    
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

    print("\n===== STYLE 3 HF SQUEEZE RESULTS =====")
    print(f"Closed Trades : {total}")
    print(f"Win Rate      : {win_rate:.2f}% (Target: >50%)")
    print(f"Net PnL       : ${net_pnl:,.2f}")
    print(f"Max Loss Streak: {max_consec} (Target: <=4)")
    print(f"Final Equity  : ${INITIAL_EQUITY + net_pnl:,.2f}")
    print("=" * 88)


if __name__ == "__main__":
    main()
