
#!/usr/bin/env python3
"""
HUNTER-V3 XT FUTURES BACKTEST ENGINE
------------------------------------
Runs HUNTER-V3 Stage-1 on REAL XT USDT-M Futures 15m CSVs produced by
xt_futures_15m_collector_v4.py.

Strategy:
  1) Completed 1H compression.
  2) 15m breakout of the completed compression box.
  3) Relative-volume + range expansion confirmation.
  4) Entry on NEXT 15m candle OPEN.
  5) Fixed nominal RR = 1:2.
  6) No BE / trailing / timeout.
  7) No overlapping positions.
  8) Max 3 portfolio positions.
  9) Max 1 position per correlation cluster.
 10) No new entry on the same candle on which any accepted position closes.
 11) If SL and TP are both touched on one candle, SL wins.
 12) Positions still open at dataset end are NOT force-closed and are
     reported separately.

No exchange API call is made by this engine. It consumes the already
downloaded XT Futures CSV files, so strategy results are reproducible.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SYMBOLS = [
    "BTC", "ETH", "SOL", "SUI", "AVAX", "NEAR", "ADA",
    "BNB", "APT", "CRV", "ONDO", "PENDLE", "ICP", "WIF",
]

CLUSTERS = {
    "BTC": "MAJOR", "ETH": "MAJOR",
    "SOL": "L1", "SUI": "L1", "AVAX": "L1", "NEAR": "L1",
    "ADA": "L1", "BNB": "L1", "APT": "L1",
    "CRV": "DEFI", "ONDO": "DEFI", "PENDLE": "DEFI",
    "ICP": "OTHER", "WIF": "MEME",
}

TIMEFRAME_MINUTES = 15
TIMEFRAME_MS = TIMEFRAME_MINUTES * 60 * 1000

INITIAL_CAPITAL = 1000.0
TRADE_MARGIN = 100.0
LEVERAGE = 50.0
NOTIONAL = TRADE_MARGIN * LEVERAGE

FEE_RATE = 0.0007
SLIPPAGE = 0.0003
RR = 2.0

MAX_POSITIONS = 3
MAX_ONE_PER_CLUSTER = True

# HUNTER-V3 Stage-1 signal parameters.
HOUR_ATR_PERIOD = 14
HOUR_RANGE_PERIOD = 20
BOX_PERIOD = 4
COMP_RECENT_PERIOD = 2
RVOL_PERIOD = 20
RANGE_MEAN_PERIOD = 20
MIN_RVOL = 1.20
MIN_EXPANSION = 1.10

# Data-quality policy:
# The collector already reports real gaps. We do not pretend missing candles
# are present. Indicators are calculated separately inside contiguous 15m
# segments, so a gap cannot contaminate rolling windows across the gap.
MAX_ALLOWED_GAP_BARS = 0

OUTPUT_DEFAULT = Path("data/xt_futures_15m/backtest_v3")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir",
        default="data/xt_futures_15m",
        help="Directory containing XT *_15m.csv files.",
    )
    p.add_argument(
        "--output-dir",
        default=str(OUTPUT_DEFAULT),
        help="Directory for trades/results.",
    )
    return p.parse_args()


def asset_to_filename(asset: str) -> Path:
    return Path(f"{asset}_USDT_15m.csv")


def load_csv(data_dir: Path, asset: str) -> pd.DataFrame:
    path = data_dir / asset_to_filename(asset)
    if not path.exists():
        raise FileNotFoundError(f"Missing XT Futures data file: {path}")

    df = pd.read_csv(path)

    required = {"Timestamp", "Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"{path}: missing columns: {sorted(missing)}")

    for col in ["Timestamp", "Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"])
    df = df[(df["Open"] > 0) & (df["Close"] > 0) & (df["High"] >= df["Low"])]

    # Strong OHLC sanity checks.
    bad = (
        (df["High"] < df[["Open", "Close"]].max(axis=1))
        | (df["Low"] > df[["Open", "Close"]].min(axis=1))
        | (df["Volume"] < 0)
    )
    if bad.any():
        raise RuntimeError(f"{asset}: {int(bad.sum())} invalid OHLCV rows")

    df = (
        df.drop_duplicates(subset=["Timestamp"], keep="last")
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )

    if len(df) < 1000:
        raise RuntimeError(f"{asset}: too few rows ({len(df)})")

    delta = df["Timestamp"].diff().dropna()
    gap_mask = delta > TIMEFRAME_MS
    gaps = int(gap_mask.sum())
    max_gap_ms = int(delta.max()) if not delta.empty else 0

    # Gaps are allowed in the collected market history, but strategy
    # calculations must not roll across them.
    df["_segment"] = gap_mask.cumsum().astype(int)

    df.attrs["gaps"] = gaps
    df.attrs["max_gap_bars"] = max(0, math.ceil(max_gap_ms / TIMEFRAME_MS) - 1)
    df.attrs["path"] = str(path)
    return df


def prepare_segment(seg: pd.DataFrame) -> pd.DataFrame:
    """Prepare one contiguous 15m segment with causal HTF context."""
    x = seg.copy().set_index("Date")

    # Resample contiguous 15m candles to completed 1H candles.
    h = (
        x[["Open", "High", "Low", "Close", "Volume"]]
        .resample("1h", label="left", closed="left")
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        })
        .dropna()
    )

    if len(h) < 25:
        return pd.DataFrame()

    h["bar_range"] = h["High"] - h["Low"]
    h["atr14"] = h["bar_range"].rolling(HOUR_ATR_PERIOD).mean()
    h["range20"] = (
        h["High"].rolling(HOUR_RANGE_PERIOD).max()
        - h["Low"].rolling(HOUR_RANGE_PERIOD).min()
    )
    h["atr20"] = h["bar_range"].rolling(HOUR_RANGE_PERIOD).mean()

    h["compressed"] = (
        (h["bar_range"] < 0.75 * h["atr20"])
        & (h["range20"] < 4.0 * h["atr14"])
    )

    # Box built from the four completed hours BEFORE the reference hour.
    h["box_high"] = h["High"].rolling(BOX_PERIOD).max().shift(1)
    h["box_low"] = h["Low"].rolling(BOX_PERIOD).min().shift(1)

    # Compression must have occurred in one of the two completed hours
    # before the 15m signal context.
    h["comp_recent"] = (
        h["compressed"]
        .rolling(COMP_RECENT_PERIOD)
        .max()
        .shift(1)
        .fillna(0)
        .astype(bool)
    )

    # Map hourly context to 15m. A 15m candle inside hour H must only see
    # the last COMPLETED hour H-1, never the currently-forming H.
    x["hour_key"] = x.index.floor("1h")
    hctx = h[["box_high", "box_low", "comp_recent"]].copy()
    hctx.index.name = "hour_key"
    x = x.join(hctx, on="hour_key")

    # For every 15m candle in H, the direct H row is not complete.
    # Shift by one hour in the context mapping.
    x["context_hour"] = x["hour_key"] - pd.Timedelta(hours=1)
    x = x.drop(columns=["box_high", "box_low", "comp_recent"])
    x = x.join(
        hctx.rename(columns={
            "box_high": "box_high",
            "box_low": "box_low",
            "comp_recent": "comp_recent",
        }),
        on="context_hour",
    )

    x["range"] = x["High"] - x["Low"]
    x["body"] = (x["Close"] - x["Open"]).abs()
    x["range20_mean"] = x["range"].rolling(RANGE_MEAN_PERIOD).mean()
    vol_mean = x["Volume"].rolling(RVOL_PERIOD).mean().replace(0, np.nan)
    x["rvol20"] = x["Volume"] / vol_mean

    x = x.drop(columns=["hour_key", "context_hour"])
    return x.reset_index()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, seg in df.groupby("_segment", sort=True):
        p = prepare_segment(seg)
        if not p.empty:
            pieces.append(p)

    if not pieces:
        return pd.DataFrame()

    out = pd.concat(pieces, ignore_index=True)
    out = out.sort_values("Timestamp").reset_index(drop=True)
    return out


def adverse_entry_price(raw_open: float, side: str) -> float:
    if side == "LONG":
        return raw_open * (1.0 + SLIPPAGE)
    return raw_open * (1.0 - SLIPPAGE)


def adverse_exit_price(raw_exit: float, side: str) -> float:
    if side == "LONG":
        return raw_exit * (1.0 - SLIPPAGE)
    return raw_exit * (1.0 + SLIPPAGE)


def build_trade(
    asset: str,
    side: str,
    signal_idx: int,
    entry_idx: int,
    df: pd.DataFrame,
    box_high: float,
    box_low: float,
):
    raw_entry = float(df.iloc[entry_idx]["Open"])
    entry = adverse_entry_price(raw_entry, side)

    if side == "LONG":
        sl = float(box_low)
        risk = entry - sl
        if risk <= 0:
            return None
        tp = entry + RR * risk
    else:
        sl = float(box_high)
        risk = sl - entry
        if risk <= 0:
            return None
        tp = entry - RR * risk

    exit_idx = None
    outcome = None
    raw_exit = None

    k = entry_idx
    while k < len(df):
        c = df.iloc[k]
        low = float(c["Low"])
        high = float(c["High"])

        hit_sl = low <= sl if side == "LONG" else high >= sl
        hit_tp = high >= tp if side == "LONG" else low <= tp

        if hit_sl and hit_tp:
            outcome = "LOSS"
            raw_exit = sl
            exit_idx = k
            break
        if hit_sl:
            outcome = "LOSS"
            raw_exit = sl
            exit_idx = k
            break
        if hit_tp:
            outcome = "WIN"
            raw_exit = tp
            exit_idx = k
            break

        k += 1

    if exit_idx is None:
        return {
            "closed": False,
            "asset": asset,
            "side": side,
            "signal_idx": signal_idx,
            "entry_idx": entry_idx,
            "entry_time": df.iloc[entry_idx]["Date"],
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "risk": risk,
            "exit_idx": None,
            "exit_time": pd.NaT,
            "exit": np.nan,
            "outcome": "OPEN_END",
            "pnl": 0.0,
            "cluster": CLUSTERS[asset],
        }

    exit_price = adverse_exit_price(float(raw_exit), side)
    qty = NOTIONAL / entry

    if side == "LONG":
        gross = qty * (exit_price - entry)
    else:
        gross = qty * (entry - exit_price)

    entry_fee = NOTIONAL * FEE_RATE
    exit_notional = qty * exit_price
    exit_fee = exit_notional * FEE_RATE
    pnl = gross - entry_fee - exit_fee

    return {
        "closed": True,
        "asset": asset,
        "side": side,
        "signal_idx": signal_idx,
        "entry_idx": entry_idx,
        "entry_time": df.iloc[entry_idx]["Date"],
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "risk": risk,
        "exit_idx": exit_idx,
        "exit_time": df.iloc[exit_idx]["Date"],
        "exit": exit_price,
        "outcome": outcome,
        "pnl": float(pnl),
        "cluster": CLUSTERS[asset],
    }


def run_symbol(asset: str, df: pd.DataFrame):
    candidates = []
    open_end = []

    # Use an explicit while loop so that, after a position opens, the scanner
    # jumps past the candle on which its result becomes known. This is a true
    # no-overlap implementation, not merely a post-hoc filter.
    i = 1
    n = len(df)

    while i < n - 1:
        r = df.iloc[i]

        if not bool(r["comp_recent"]):
            i += 1
            continue

        bh = r["box_high"]
        bl = r["box_low"]
        rm = r["range20_mean"]
        rv = r["rvol20"]

        if any(pd.isna(v) for v in [bh, bl, rm, rv]):
            i += 1
            continue
        if bh <= bl:
            i += 1
            continue

        expansion = float(r["range"]) > MIN_EXPANSION * float(rm)
        volume_ok = float(rv) >= MIN_RVOL
        if not expansion or not volume_ok:
            i += 1
            continue

        side = None
        if float(r["Close"]) > float(bh) and float(r["Open"]) <= float(bh):
            side = "LONG"
        elif float(r["Close"]) < float(bl) and float(r["Open"]) >= float(bl):
            side = "SHORT"

        if side is None:
            i += 1
            continue

        entry_idx = i + 1
        trade = build_trade(
            asset, side, i, entry_idx, df, float(bh), float(bl)
        )

        if trade is None:
            i += 1
            continue

        if trade["closed"]:
            candidates.append(trade)
            # Result becomes known on exit_idx. The next signal must be on a
            # strictly later candle, never the same candle.
            i = int(trade["exit_idx"]) + 1
        else:
            open_end.append(trade)
            # No forced exit and no future data: no further signal can be
            # evaluated safely after this open position.
            break

    return candidates, open_end


def portfolio_filter(candidates):
    """Chronological portfolio simulation with strict no-same-close-candle."""
    candidates = sorted(
        candidates,
        key=lambda t: (t["entry_time"], t["asset"], t["side"])
    )

    accepted = []
    active = []
    close_times = set()

    for t in candidates:
        et = t["entry_time"]

        # Remove positions that have already closed strictly BEFORE entry.
        active = [p for p in active if p["exit_time"] >= et]

        # Never enter on a candle where any accepted trade closes.
        if et in close_times:
            continue

        if len(active) >= MAX_POSITIONS:
            continue

        if MAX_ONE_PER_CLUSTER and any(
            p["cluster"] == t["cluster"] for p in active
        ):
            continue

        accepted.append(t)
        active.append(t)
        close_times.add(t["exit_time"])

    return accepted


def max_loss_streak(outcomes):
    best = cur = 0
    for x in outcomes:
        if x == "LOSS":
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def loss_streaks(outcomes):
    out = []
    cur = 0
    for x in outcomes:
        if x == "LOSS":
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def summarize(trades: pd.DataFrame, raw_candidates: int, open_positions: int):
    print("\n" + "=" * 88)
    print("HUNTER-V3 — XT USDT-M FUTURES / 15m / 365-DAY BACKTEST")
    print("=" * 88)

    if trades.empty:
        print("Closed trades: 0")
        return {}

    trades = trades.sort_values(["exit_time", "entry_time", "asset"]).reset_index(drop=True)

    wins = int((trades["outcome"] == "WIN").sum())
    losses = int((trades["outcome"] == "LOSS").sum())
    total = len(trades)
    wr = 100.0 * wins / total

    gross_profit = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
    gross_loss = float(-trades.loc[trades["pnl"] < 0, "pnl"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    net = float(trades["pnl"].sum())

    equity = INITIAL_CAPITAL
    peak = equity
    max_dd = 0.0
    for pnl in trades["pnl"]:
        equity += float(pnl)
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)

    streak_list = loss_streaks(trades["outcome"].tolist())
    max_streak = max(streak_list, default=0)

    days = max(
        1.0,
        (trades["entry_time"].max() - trades["entry_time"].min()).total_seconds()
        / 86400.0,
    )

    print(f"Raw symbol candidates : {raw_candidates:,}")
    print(f"Accepted closed trades: {total:,}")
    print(f"Wins / Losses         : {wins:,} / {losses:,}")
    print(f"Win Rate              : {wr:.2f}%")
    print(f"Profit Factor         : {pf:.3f}")
    print(f"Net PnL               : ${net:,.2f}")
    print(f"Final equity          : ${equity:,.2f}")
    print(f"Max Drawdown          : ${max_dd:,.2f}")
    print(f"Max loss streak       : {max_streak}")
    print(f"Loss streak list      : {streak_list}")
    print(f"Trades / day          : {total / days:.2f}")
    print(f"Open-at-end positions : {open_positions}")

    print("\nPer-symbol:")
    rows = []
    for asset in SYMBOLS:
        g = trades[trades["asset"] == asset]
        if g.empty:
            rows.append([asset, 0, 0.0, 0.0, 0.0])
            continue
        w = int((g["outcome"] == "WIN").sum())
        swr = 100.0 * w / len(g)
        spnl = float(g["pnl"].sum())
        rows.append([asset, len(g), swr, spnl, float(g["pnl"].mean())])
        print(
            f"  {asset:<7} {len(g):>4} trades | "
            f"WR {swr:>6.2f}% | PnL ${spnl:>10.2f} | "
            f"Exp ${float(g['pnl'].mean()):>7.2f}"
        )

    return {
        "raw_candidates": raw_candidates,
        "closed_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": wr,
        "profit_factor": pf,
        "net_pnl": net,
        "final_equity": equity,
        "max_drawdown": max_dd,
        "max_consecutive_losses": max_streak,
        "loss_streaks": streak_list,
        "trades_per_day": total / days,
        "open_at_end": open_positions,
        "per_symbol": rows,
    }


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("HUNTER-V3 STAGE-1 — XT REAL FUTURES DATA ENGINE")
    print("=" * 88)
    print(f"Data directory: {data_dir.resolve()}")
    print(f"Margin=${TRADE_MARGIN:.2f} | Leverage={LEVERAGE:.1f}x | RR=1:{RR:.1f}")
    print(f"Fee={FEE_RATE:.4f} | Slippage={SLIPPAGE:.4f}")
    print(f"Max positions={MAX_POSITIONS} | Max one/cluster={MAX_ONE_PER_CLUSTER}")

    prepared = {}
    data_audit = []

    for asset in SYMBOLS:
        raw = load_csv(data_dir, asset)
        p = prepare(raw)
        if p.empty:
            raise RuntimeError(f"{asset}: no usable prepared data")

        prepared[asset] = p
        data_audit.append({
            "asset": asset,
            "rows": len(raw),
            "prepared_rows": len(p),
            "gaps": int(raw.attrs.get("gaps", 0)),
            "max_gap_bars": int(raw.attrs.get("max_gap_bars", 0)),
        })
        print(
            f"{asset:<7} rows={len(raw):,} prepared={len(p):,} "
            f"gaps={raw.attrs.get('gaps', 0)}"
        )

    candidates = []
    open_end = []

    for asset, df in prepared.items():
        c, o = run_symbol(asset, df)
        candidates.extend(c)
        open_end.extend(o)
        print(f"Signals {asset:<7}: closed={len(c):>5} open_end={len(o):>3}")

    accepted = portfolio_filter(candidates)
    trades = pd.DataFrame(accepted)

    if not trades.empty:
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)

    # Open-end positions can only be retained if they are portfolio-valid.
    # They are informational only and never included in PnL.
    accepted_ids = {
        (t["asset"], t["entry_time"]) for t in accepted
    }
    open_end_accepted = [
        t for t in open_end if (t["asset"], t["entry_time"]) in accepted_ids
    ]

    summary = summarize(
        trades,
        raw_candidates=len(candidates),
        open_positions=len(open_end_accepted),
    )

    if not trades.empty:
        trades.to_csv(out_dir / "TRADES.csv", index=False)

    report = {
        "engine": "HUNTER-V3-XT-FUTURES",
        "data_source": "XT USDT-M Futures",
        "timeframe": "15m",
        "strategy": "Compression -> Breakout -> Relative Volume",
        "rr": RR,
        "initial_capital": INITIAL_CAPITAL,
        "trade_margin": TRADE_MARGIN,
        "leverage": LEVERAGE,
        "fee_rate": FEE_RATE,
        "slippage": SLIPPAGE,
        "max_positions": MAX_POSITIONS,
        "max_one_per_cluster": MAX_ONE_PER_CLUSTER,
        "no_timeout": True,
        "same_candle_sl_tp": "LOSS",
        "same_candle_close_reentry": False,
        "data_audit": data_audit,
        "summary": summary,
        "open_end_positions": [
            {
                k: (v.isoformat() if isinstance(v, pd.Timestamp) else v)
                for k, v in t.items()
                if k not in {"closed"}
            }
            for t in open_end_accepted
        ],
    }

    (out_dir / "BACKTEST_REPORT.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"\nReports written to: {out_dir}")
    print(f"  {out_dir / 'BACKTEST_REPORT.json'}")
    if not trades.empty:
        print(f"  {out_dir / 'TRADES.csv'}")


if __name__ == "__main__":
    main()
