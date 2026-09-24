#!/usr/bin/env python3
"""
HUNTER-XT-STRICT-TARGETS-ENGINE
Strict Causal / Price-Action Confirmed Momentum / XT USDT-M Futures
- Target Win Rate: > 50%
- Risk-to-Reward: 1:2 (Fixed)
- Target Trades: ~4 per day across universe
- Max Consecutive Losses: <= 4 (Controlled via volatility & confirmation filters)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

DATA_DIR = Path("data/xt_futures_15m")
OUT_DIR = DATA_DIR / "backtest_strict_targets"

INITIAL_EQUITY = 1000.0
TRADE_MARGIN = 100.0
LEVERAGE = 50.0

FEE_RATE = 0.0007
SLIPPAGE = 0.0003
RR = 2.0
ATR_N = 14
STOP_BUFFER_ATR = 0.15
MOMENTUM_LOOKBACK = 48  # Shorter lookback for higher frequency & responsiveness
MAX_OPEN_POSITIONS = 3
MAX_ONE_PER_CLUSTER = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    return p.parse_args()


def ensure_xt_data(data_dir: Path, symbols: list[str]):
    """Ensures XT Futures 15m data exists, downloading via CCXT if necessary."""
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        import ccxt
    except ImportError:
        print("CCXT not installed. Please ensure ccxt is available.")
        return

    exchange = ccxt.xt({'enableRateLimit': True})
    exchange.options['defaultType'] = 'swap'  # Explicitly XT Futures USDT-M

    for symbol in symbols:
        file_path = data_dir / f"{symbol}_USDT_15m.csv"
        if file_path.exists() and file_path.stat().st_size > 200:
            continue
        
        print(f"Downloading XT Futures 15m swap data for {symbol}...")
        try:
            ccxt_symbol = f"{symbol}/USDT:USDT"
            ohlcv = exchange.fetch_ohlcv(ccxt_symbol, timeframe='15m', limit=1500)
            if ohlcv:
                df_dl = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df_dl.to_csv(file_path, index=False)
                print(f"Saved -> {file_path} ({len(df_dl)} rows)")
        except Exception as e:
            print(f"Warning: Could not fetch {symbol}: {e}")


def load_xt_csv(data_dir: Path, asset: str) -> pd.DataFrame:
    path = data_dir / f"{asset}_USDT_15m.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing XT Futures data file: {path}")

    df = pd.read_csv(path)
    if {"Date", "Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
        df = df.rename(columns={
            "Date": "timestamp", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })

    required = ["timestamp", "open", "high", "low", "close", "volume"]
    ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True) if np.issubdtype(df["timestamp"].dtype, np.number) else pd.to_datetime(df["timestamp"], utc=True)

    df["timestamp"] = ts
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")].copy()

    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=required[1:])


def calculate_confirmed_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates indicators with strict causal shift and price-action filters."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)

    df["atr"] = tr.ewm(alpha=1 / ATR_N, adjust=False, min_periods=ATR_N).mean()
    df["momentum"] = df["close"].shift(1).pct_change(MOMENTUM_LOOKBACK)
    df["ema_trend"] = df["close"].shift(1).ewm(span=100, adjust=False).mean()
    
    # Price Action Confirmation: Body strength relative to ATR
    df["body_size"] = (df["close"].shift(1) - df["open"].shift(1)).abs()
    df["is_bullish_confirmation"] = (df["close"].shift(1) > df["open"].shift(1)) & (df["body_size"] >= 0.6 * df["atr"])
    return df


def run_target_optimized_backtest(all_data: dict[str, pd.DataFrame]) -> tuple[list[dict], list[dict]]:
    all_times = sorted(list(set().union(*(df.index for df in all_data.values()))))
    
    active_positions = {}
    trades = []
    equity = INITIAL_EQUITY
    peak_equity = INITIAL_EQUITY
    max_dd = 0.0
    rebalance_interval = 8  # Check more frequently for higher frequency (~2 hours)

    def cluster_active(cluster_name):
        return any(CORRELATION_CLUSTERS.get(p["symbol"], "OTHER") == cluster_name for p in active_positions.values())

    for idx, ts in enumerate(all_times):
        # 1. Manage active positions strictly on current candle high/low
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
                max_dd = min(max_dd, equity - peak_equity)

                trades.append({
                    "symbol": symbol, "cluster": CORRELATION_CLUSTERS.get(symbol, "OTHER"),
                    "side": side, "entry_ts": pos["entry_ts"], "exit_ts": ts,
                    "entry_price": entry, "exit_price": exit_price,
                    "outcome": outcome, "pnl": float(pnl), "equity": float(equity),
                })
                del active_positions[symbol]

        # 2. Rebalancing with Strict Confirmation Filters
        if idx % rebalance_interval == 0 and len(active_positions) < MAX_OPEN_POSITIONS:
            scores = []
            for symbol, df in all_data.items():
                if ts not in df.index:
                    continue
                row = df.loc[ts]
                mom = row.get("momentum")
                close = row.get("close")
                ema = row.get("ema_trend")
                atr = row.get("atr")
                confirmed = row.get("is_bullish_confirmation", False)

                if all(np.isfinite([mom, close, ema, atr])) and atr > 0:
                    # Strict trend + Price Action Confirmation filter to boost Win Rate
                    if close > ema and confirmed:
                        scores.append({"symbol": symbol, "score": mom, "atr": atr, "cluster": CORRELATION_CLUSTERS.get(symbol, "OTHER")})

            scores.sort(key=lambda x: x["score"], reverse=True)

            for item in scores:
                symbol = item["symbol"]
                cluster = item["cluster"]

                if symbol in active_positions:
                    continue
                if len(active_positions) >= MAX_OPEN_POSITIONS:
                    break
                if MAX_ONE_PER_CLUSTER and cluster_active(cluster):
                    continue

                df = all_data[symbol]
                next_indices = df.index[df.index > ts]
                if len(next_indices) == 0:
                    continue
                
                next_ts = next_indices[0]
                raw_open = float(df.loc[next_ts, "open"])
                entry = raw_open * (1.0 + SLIPPAGE)
                atr = item["atr"]
                sl = entry - (1.5 * atr)
                risk = entry - sl
                
                if risk <= 0:
                    continue
                tp = entry + (RR * risk)

                active_positions[symbol] = {
                    "symbol": symbol, "side": "LONG",
                    "entry_ts": next_ts, "entry": entry, "sl": sl, "tp": tp,
                }

    open_positions = []
    for symbol, pos in active_positions.items():
        df = all_data[symbol]
        last_close = float(df.iloc[-1]["close"])
        unrealized = (last_close - pos["entry"]) / pos["entry"] * TRADE_MARGIN * LEVERAGE
        open_positions.append({"symbol": symbol, "side": pos["side"], "entry_ts": pos["entry_ts"], "entry_price": pos["entry"], "last_price": last_close, "unrealized_pnl": unrealized})

    return trades, open_positions


def main():
    args = parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("HUNTER-XT-STRICT-TARGETS-ENGINE — Optimized for WinRate & Controlled Streaks")
    print("=" * 88)

    ensure_xt_data(data_dir, SYMBOLS)

    all_data = {}
    for asset in SYMBOLS:
        try:
            df = load_xt_csv(data_dir, asset)
            all_data[asset] = calculate_confirmed_indicators(df)
        except Exception as e:
            print(f"Skipping {asset}: {e}")

    trades, open_positions = run_target_optimized_backtest(all_data)
    
    total = len(trades)
    wins = sum(1 for t in trades if t["outcome"] == "WIN")
    win_rate = (wins / total * 100.0) if total > 0 else 0.0
    net_pnl = sum(t["pnl"] for t in trades)

    # Calculate max consecutive losses
    consec, max_consec = 0, 0
    for t in sorted(trades, key=lambda x: pd.Timestamp(x["exit_ts"])):
        if t["outcome"] == "LOSS":
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0

    print("\n===== OPTIMIZED TARGET RESULTS =====")
    print(f"Closed Trades : {total}")
    print(f"Win Rate      : {win_rate:.2f}% (Target: >50%)")
    print(f"Net PnL       : ${net_pnl:,.2f}")
    print(f"Max Loss Streak: {max_consec} (Target: <=4)")
    print(f"Final Equity  : ${INITIAL_EQUITY + net_pnl:,.2f}")
    print(f"Open Positions: {len(open_positions)}")
    print("=" * 88)


if __name__ == "__main__":
    main()
