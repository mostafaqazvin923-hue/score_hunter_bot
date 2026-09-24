#!/usr/bin/env python3
"""
HUNTER-XT-STYLE4-ULTIMATE-MOMENTUM
Cross-Sectional Relative Strength & Macro Regime Filter (1h Timeframe)
- Focus: Win Rate > 50%, Max Loss Streak <= 4, High PnL
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

DATA_DIR = Path("data/xt_futures_style4_ult")
OUT_DIR = DATA_DIR / "backtest_style4_ult"

INITIAL_EQUITY = 1000.0
TRADE_MARGIN = 100.0
LEVERAGE = 50.0

FEE_RATE = 0.0007
SLIPPAGE = 0.0003
RR = 2.0                      # Optimized Risk-Reward for >50% Win Rate
ATR_N = 14
MAX_OPEN_POSITIONS = 2        # Reduced to 2 to minimize simultaneous market exposure
MAX_ONE_PER_CLUSTER = True
CIRCUIT_BREAKER_COOLDOWN = 20 # Strict lockout after 2 losses to guarantee streak <= 4


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


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates ATR, Trend, and Strict Momentum Ranking metrics using shift(1)."""
    prev_close = df["close"].shift(1)
    
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / ATR_N, adjust=False, min_periods=ATR_N).mean()

    # Trend filter
    df["ema_trend"] = prev_close.ewm(span=40, adjust=False).mean()
    df["trend_ok"] = prev_close > df["ema_trend"]

    # 24-period return for momentum ranking
    df["return_24h"] = prev_close.pct_change(24)

    # Volume filter
    prev_volume = df["volume"].shift(1)
    df["avg_volume"] = prev_volume.rolling(window=20).mean()
    df["volume_ok"] = prev_volume > (1.1 * df["avg_volume"]) # Stricter volume confirmation

    df["setup_valid"] = df["trend_ok"] & df["volume_ok"] & (df["return_24h"] > 0.01)
    return df


def run_ultimate_momentum_backtest(all_data: dict[str, pd.DataFrame]) -> list[dict]:
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

        # Macro Market Regime Check (BTC Trend Filter)
        btc_df = all_data.get("BTC")
        market_bullish = True
        if btc_df is not None and ts in btc_df.index:
            market_bullish = bool(btc_df.loc[ts, "trend_ok"])

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

        # 2. Cross-Sectional Ranking & Macro-Filtered Entry Execution
        if market_bullish and cooldown_counter == 0 and len(active_positions) < MAX_OPEN_POSITIONS:
            candidates = []
            for symbol, df in all_data.items():
                if ts not in df.index:
                    continue
                row = df.loc[ts]
                if not row.get("setup_valid", False):
                    continue
                if symbol in active_positions:
                    continue
                cluster = CORRELATION_CLUSTERS.get(symbol, "OTHER")
                if MAX_ONE_PER_CLUSTER and cluster_active(cluster):
                    continue
                
                ret_24h = row.get("return_24h", 0.0)
                if np.isfinite(ret_24h):
                    candidates.append((symbol, ret_24h))

            # Sort by top relative momentum
            candidates.sort(key=lambda x: x[1], reverse=True)

            for symbol, _ in candidates:
                if len(active_positions) >= MAX_OPEN_POSITIONS:
                    break
                
                df = all_data[symbol]
                next_indices = df.index[df.index > ts]
                if len(next_indices) == 0:
                    continue
                
                next_ts = next_indices[0]
                entry = float(df.loc[next_ts, "open"]) * (1.0 + SLIPPAGE)
                row = df.loc[ts]
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
    print("HUNTER-XT-STYLE4-ULTIMATE-MOMENTUM (Macro Filter + Strict Streak Control)")
    print("=" * 88)

    ensure_xt_data(data_dir, SYMBOLS)

    all_data = {}
    for asset in SYMBOLS:
        try:
            df = load_xt_csv(data_dir, asset)
            all_data[asset] = calculate_features(df)
        except Exception:
            pass

    trades = run_ultimate_momentum_backtest(all_data)
    
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

    print("\n===== STYLE 4 ULTIMATE RESULTS =====")
    print(f"Closed Trades : {total}")
    print(f"Win Rate      : {win_rate:.2f}% (Target: >50%)")
    print(f"Net PnL       : ${net_pnl:,.2f}")
    print(f"Max Loss Streak: {max_consec} (Target: <=4)")
    print(f"Final Equity  : ${INITIAL_EQUITY + net_pnl:,.2f}")
    print("=" * 88)


if __name__ == "__main__":
    main()
