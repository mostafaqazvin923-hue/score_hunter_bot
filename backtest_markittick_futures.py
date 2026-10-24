#!/usr/bin/env python3
"""One-year backtest for Dynamic Market Metrics [MarkitTick] on Binance USD-M Futures.

Uses public Binance Futures klines only; no API key is required. Signals are evaluated
on closed 1h candles and entries are filled at the next candle open. The strategy
uses TP2 (2R) and AlphaTrend as the stop reference. See README for assumptions.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://fapi.binance.com/fapi/v1/klines"
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "OPUSDT",
    "DOGEUSDT", "LINKUSDT", "AVAXUSDT", "ADAUSDT", "NEARUSDT",
]
INTERVAL = "1h"
DAYS = 365
# User-selected settings
ATR_MULT = 1.5
ATR_MFI_LEN = 14
BO_OVERRIDE = True
BO_SENS = 2.5
CLOUD_LEN = 10
HTF_TF = "1d"  # retained as setting; Show HTF is disabled, so no HTF filter is applied
# Existing backtest conventions from the user's project
INITIAL_CAPITAL = 1000.0
MARGIN_PER_TRADE = 100.0
LEVERAGE = 50.0
FEE_RATE = 0.0007
SLIPPAGE = 0.0003
RR = 2.0

DATA_DIR = Path("data")
OUT_DIR = Path("backtest_output")

@dataclass
class Trade:
    symbol: str
    side: str
    signal_time: str
    entry_time: str
    exit_time: str
    entry: float
    stop: float
    target: float
    exit_price: float
    exit_reason: str
    risk_pct: float
    gross_pnl: float
    entry_fee: float
    exit_fee: float
    net_pnl: float
    r_multiple_net: float
    bars_held: int


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def fetch_klines(symbol: str, start: datetime, end: datetime, session: requests.Session) -> pd.DataFrame:
    """Fetch 1h USD-M Futures candles, paginating by open time and rejecting bad data."""
    rows: list[list[Any]] = []
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    cursor = start_ms
    last_cursor = None
    while cursor < end_ms:
        params = {"symbol": symbol, "interval": INTERVAL, "startTime": cursor, "endTime": end_ms, "limit": 1500}
        response = session.get(BASE_URL, params=params, timeout=30)
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(2)
            response.raise_for_status()
        response.raise_for_status()
        batch = response.json()
        if not isinstance(batch, list):
            raise RuntimeError(f"Unexpected Binance response for {symbol}: {batch}")
        if not batch:
            break
        rows.extend(batch)
        newest = int(batch[-1][0])
        if newest <= cursor or newest == last_cursor:
            raise RuntimeError(f"Pagination made no progress for {symbol}; stopping rather than fabricating data")
        last_cursor = newest
        cursor = newest + 1
        if len(batch) < 1500:
            break
        time.sleep(0.12)

    columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        raise RuntimeError(f"No Binance Futures candles returned for {symbol}")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(pd.to_numeric(df["open_time"]), unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(pd.to_numeric(df["close_time"]), unit="ms", utc=True)
    df = df.dropna(subset=["open_time", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    # Do not use the currently forming candle.
    now = pd.Timestamp.now(tz="UTC")
    df = df[df["close_time"] < now].copy().reset_index(drop=True)
    if len(df) < ATR_MFI_LEN + 20:
        raise RuntimeError(f"Too few closed candles for {symbol}: {len(df)}")
    # A one-hour dataset with missing candles can distort all indicators. Fail loudly.
    deltas = df["open_time"].diff().dropna()
    bad = deltas[deltas != pd.Timedelta(hours=1)]
    if not bad.empty:
        first_bad_idx = int(bad.index[0])
        raise RuntimeError(f"Data gap/duplicate interval for {symbol} near {df.loc[first_bad_idx, 'open_time']}; refusing to silently fill candles")
    return df


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    parts = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1)
    # Pine ta.tr uses high-low on the first bar if previous close is unavailable.
    return parts.max(axis=1)


def mfi(hlc3: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    raw_flow = hlc3 * volume
    direction = hlc3.diff()
    positive = raw_flow.where(direction > 0, 0.0)
    negative = raw_flow.where(direction < 0, 0.0)
    pos_sum = positive.rolling(length, min_periods=length).sum()
    neg_sum = negative.rolling(length, min_periods=length).sum()
    ratio = pos_sum / neg_sum.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + ratio))
    out = out.where(~((pos_sum > 0) & (neg_sum == 0)), 100.0)
    out = out.where(~((pos_sum == 0) & (neg_sum > 0)), 0.0)
    return out


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Translate the supplied Pine logic with bar-indexed, causal series."""
    d = df.copy()
    tr = true_range(d)
    # Pine: ATR = sma(tr[1], AP), meaning this bar's ATR uses prior-bar TR values.
    d["atr"] = tr.shift(1).rolling(ATR_MFI_LEN, min_periods=ATR_MFI_LEN).mean()
    hlc3 = (d["high"] + d["low"] + d["close"]) / 3.0
    # Pine: ta.mfi(hlc3, AP)[1]
    d["mfi"] = mfi(hlc3, d["volume"], ATR_MFI_LEN).shift(1)
    d["up_t"] = d["low"].shift(1) - d["atr"] * ATR_MULT
    d["down_t"] = d["high"].shift(1) + d["atr"] * ATR_MULT

    alpha = np.full(len(d), np.nan, dtype=float)
    for i in range(len(d)):
        if i < 1 or not np.isfinite(d.at[i, "mfi"]) or not np.isfinite(d.at[i, "up_t"]) or not np.isfinite(d.at[i, "down_t"]):
            continue
        prev = alpha[i - 1] if i > 0 and np.isfinite(alpha[i - 1]) else 0.0
        if d.at[i, "mfi"] >= 50.0:
            alpha[i] = prev if d.at[i, "up_t"] < prev else d.at[i, "up_t"]
        else:
            alpha[i] = prev if d.at[i, "down_t"] > prev else d.at[i, "down_t"]
    d["alpha"] = alpha

    n = len(d)
    buy = np.zeros(n, dtype=bool)
    sell = np.zeros(n, dtype=bool)
    valid_long = np.zeros(n, dtype=bool)
    valid_short = np.zeros(n, dtype=bool)
    direction = np.ones(n, dtype=int)
    # Pine var direction starts at 1 and is updated from AlphaTrend[1] vs [3].
    for i in range(n):
        if i > 0:
            direction[i] = direction[i - 1]
        if i >= 3 and np.isfinite(alpha[i-1]) and np.isfinite(alpha[i-3]):
            if alpha[i-1] > alpha[i-3]:
                direction[i] = 1
            elif alpha[i-1] < alpha[i-3]:
                direction[i] = -1

        if i >= 4 and all(np.isfinite(alpha[j]) for j in (i-1, i-2, i-3, i-4)):
            is_cross_up = d.at[i-1, "close"] > alpha[i-1] and d.at[i-2, "close"] <= (alpha[i-2] if np.isfinite(alpha[i-2]) else 0.0)
            is_cross_dn = d.at[i-1, "close"] < alpha[i-1] and d.at[i-2, "close"] >= (alpha[i-2] if np.isfinite(alpha[i-2]) else 0.0)
            atr_prev = d.at[i-1, "atr"]
            bo_momentum = BO_OVERRIDE and np.isfinite(atr_prev) and abs(d.at[i-1, "close"] - d.at[i-2, "close"]) > atr_prev * BO_SENS
            is_trend_up = alpha[i-1] > alpha[i-3] and alpha[i-2] <= alpha[i-4]
            is_trend_dn = alpha[i-1] < alpha[i-3] and alpha[i-2] >= alpha[i-4]
            buy[i] = bool(is_trend_up or (bo_momentum and is_cross_up))
            sell[i] = bool(is_trend_dn or (bo_momentum and is_cross_dn))

        # Pine counters: O1/K1 reset after previous buy; O2/K2 after previous sell.
        # At bar i, current counter values are updated using buy/sell at i-1.
        # Store counts at each bar for subsequent valid signal evaluation.
    o1 = np.zeros(n, dtype=int); o2 = np.zeros(n, dtype=int)
    k1 = np.zeros(n, dtype=int); k2 = np.zeros(n, dtype=int)
    for i in range(n):
        if i == 0:
            o1[i] = o2[i] = k1[i] = k2[i] = 0
        else:
            o1[i] = 0 if buy[i-1] else o1[i-1] + 1
            k1[i] = 0 if buy[i-1] else k1[i-1] + 1
            o2[i] = 0 if sell[i-1] else o2[i-1] + 1
            k2[i] = 0 if sell[i-1] else k2[i-1] + 1
        valid_long[i] = buy[i] and o1[i] > k2[i]
        valid_short[i] = sell[i] and o2[i] > k1[i]

    d["direction"] = direction
    d["buy_signal"] = buy
    d["sell_signal"] = sell
    d["valid_long"] = valid_long
    d["valid_short"] = valid_short
    d["rising_long"] = valid_long & ~np.r_[False, valid_long[:-1]]
    d["rising_short"] = valid_short & ~np.r_[False, valid_short[:-1]]
    # Cloud is calculated for parity/config tracking, but it is visual-only and not a filter.
    d["cloud_ref"] = hlc3.shift(1).rolling(CLOUD_LEN, min_periods=CLOUD_LEN).mean()
    d["cloud_mid"] = (d["alpha"] + d["cloud_ref"]) / 2.0
    return d


def fill_price(raw: float, side: str, is_entry: bool) -> float:
    # Adverse slippage: pay up to buy, receive less when selling.
    if is_entry:
        return raw * (1.0 + SLIPPAGE) if side == "LONG" else raw * (1.0 - SLIPPAGE)
    return raw * (1.0 - SLIPPAGE) if side == "LONG" else raw * (1.0 + SLIPPAGE)


def simulate_symbol(symbol: str, d: pd.DataFrame) -> list[Trade]:
    trades: list[Trade] = []
    position: dict[str, Any] | None = None
    # Entry only on next bar open after a confirmed signal; no same-bar re-entry.
    for i in range(1, len(d)):
        bar = d.iloc[i]
        prev = d.iloc[i - 1]
        if position is None:
            sig_i = i - 1
            sig = d.iloc[sig_i]
            is_long = bool(sig["rising_long"])
            is_short = bool(sig["rising_short"])
            if not (is_long or is_short):
                continue
            side = "LONG" if is_long else "SHORT"
            # Pine's displayed levels use close[1] and AlphaTrend[1] on the signal bar.
            signal_entry_ref = float(sig["close"])
            stop = float(sig["alpha"])
            if not np.isfinite(stop) or signal_entry_ref <= 0:
                continue
            risk_ref = max(abs(signal_entry_ref - stop), signal_entry_ref * 1e-8)
            entry = fill_price(float(bar["open"]), side, True)
            # Keep target based on actual execution price and the signal's AlphaTrend stop,
            # so the realized order levels correspond to a real next-open fill.
            risk = abs(entry - stop)
            if risk <= 0 or risk / entry > 0.20:
                continue
            target = entry + RR * risk if side == "LONG" else entry - RR * risk
            if side == "LONG" and stop >= entry:
                continue
            if side == "SHORT" and stop <= entry:
                continue
            notional = MARGIN_PER_TRADE * LEVERAGE
            entry_fee = notional * FEE_RATE
            position = {
                "symbol": symbol, "side": side, "signal_time": sig["open_time"],
                "entry_time": bar["open_time"], "entry_i": i, "entry": entry,
                "stop": stop, "target": target, "notional": notional,
                "entry_fee": entry_fee, "risk_pct": risk / entry,
            }
            # Fall through and evaluate the entry candle too: its high/low occur after
            # the next-open fill and may legitimately hit SL or TP2.

        side = position["side"]
        # Evaluate exits on each completed bar after entry. If both touched within a bar,
        # assume SL first (conservative; 1h OHLC cannot reveal intrabar ordering).
        high = float(bar["high"]); low = float(bar["low"])
        if side == "LONG":
            sl_hit = low <= position["stop"]
            tp_hit = high >= position["target"]
        else:
            sl_hit = high >= position["stop"]
            tp_hit = low <= position["target"]
        exit_reason = None; raw_exit = None
        if sl_hit and tp_hit:
            exit_reason = "SL_AND_TP_SAME_CANDLE_SL_FIRST"
            raw_exit = position["stop"]
        elif sl_hit:
            exit_reason = "SL"; raw_exit = position["stop"]
        elif tp_hit:
            exit_reason = "TP2"; raw_exit = position["target"]
        if exit_reason:
            exit_price = fill_price(float(raw_exit), side, False)
            qty = position["notional"] / position["entry"]
            gross = (exit_price - position["entry"]) * qty if side == "LONG" else (position["entry"] - exit_price) * qty
            exit_fee = abs(exit_price * qty) * FEE_RATE
            net = gross - position["entry_fee"] - exit_fee
            initial_risk_cash = position["notional"] * position["risk_pct"]
            trades.append(Trade(
                symbol=symbol, side=side,
                signal_time=pd.Timestamp(position["signal_time"]).isoformat(),
                entry_time=pd.Timestamp(position["entry_time"]).isoformat(),
                exit_time=pd.Timestamp(bar["open_time"]).isoformat(),
                entry=float(position["entry"]), stop=float(position["stop"]), target=float(position["target"]),
                exit_price=float(exit_price), exit_reason=exit_reason,
                risk_pct=float(position["risk_pct"]), gross_pnl=float(gross),
                entry_fee=float(position["entry_fee"]), exit_fee=float(exit_fee), net_pnl=float(net),
                r_multiple_net=float(net / initial_risk_cash) if initial_risk_cash else 0.0,
                bars_held=int(i - position["entry_i"] + 1),
            ))
            position = None
    # Do not force-close open trades at the end of the test; report them separately as censored.
    return trades


def summarize(trades_df: pd.DataFrame, symbols: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for symbol in symbols:
        sub = trades_df[trades_df["symbol"] == symbol] if not trades_df.empty else pd.DataFrame()
        if sub.empty:
            rows.append({"symbol": symbol, "trades": 0, "wins": 0, "losses": 0, "win_rate_pct": 0.0,
                         "net_pnl_usd": 0.0, "profit_factor": None, "avg_net_pnl_usd": 0.0,
                         "max_consecutive_losses": 0})
            continue
        wins = int((sub["net_pnl"] > 0).sum()); losses = int((sub["net_pnl"] <= 0).sum())
        gross_win = float(sub.loc[sub["net_pnl"] > 0, "net_pnl"].sum())
        gross_loss = float(-sub.loc[sub["net_pnl"] < 0, "net_pnl"].sum())
        streak = max_streak = 0
        for pnl in sub.sort_values("exit_time")["net_pnl"]:
            if pnl <= 0:
                streak += 1; max_streak = max(max_streak, streak)
            else:
                streak = 0
        rows.append({"symbol": symbol, "trades": len(sub), "wins": wins, "losses": losses,
                     "win_rate_pct": 100.0 * wins / len(sub), "net_pnl_usd": float(sub["net_pnl"].sum()),
                     "profit_factor": (gross_win / gross_loss if gross_loss > 0 else (math.inf if gross_win > 0 else None)),
                     "avg_net_pnl_usd": float(sub["net_pnl"].mean()), "max_consecutive_losses": max_streak})
    per_symbol = pd.DataFrame(rows)
    if trades_df.empty:
        overall = {"trades": 0, "wins": 0, "losses": 0, "win_rate_pct": 0.0, "net_pnl_usd": 0.0,
                   "profit_factor": None, "final_equity_usd": INITIAL_CAPITAL, "max_drawdown_usd": 0.0,
                   "max_drawdown_pct": 0.0, "max_consecutive_losses_global": 0}
        return per_symbol, overall
    ordered = trades_df.sort_values(["exit_time", "symbol"]).reset_index(drop=True)
    ordered["equity"] = INITIAL_CAPITAL + ordered["net_pnl"].cumsum()
    ordered["equity_peak"] = ordered["equity"].cummax().clip(lower=INITIAL_CAPITAL)
    ordered["drawdown_usd"] = ordered["equity_peak"] - ordered["equity"]
    ordered["drawdown_pct"] = np.where(ordered["equity_peak"] != 0, ordered["drawdown_usd"] / ordered["equity_peak"] * 100.0, 0.0)
    wins = int((ordered["net_pnl"] > 0).sum()); losses = int((ordered["net_pnl"] <= 0).sum())
    gw = float(ordered.loc[ordered["net_pnl"] > 0, "net_pnl"].sum())
    gl = float(-ordered.loc[ordered["net_pnl"] < 0, "net_pnl"].sum())
    streak = max_streak = 0
    for pnl in ordered["net_pnl"]:
        if pnl <= 0:
            streak += 1; max_streak = max(max_streak, streak)
        else:
            streak = 0
    overall = {"trades": len(ordered), "wins": wins, "losses": losses,
               "win_rate_pct": 100.0 * wins / len(ordered), "net_pnl_usd": float(ordered["net_pnl"].sum()),
               "profit_factor": (gw / gl if gl > 0 else (math.inf if gw > 0 else None)),
               "final_equity_usd": float(INITIAL_CAPITAL + ordered["net_pnl"].sum()),
               "max_drawdown_usd": float(ordered["drawdown_usd"].max()),
               "max_drawdown_pct": float(ordered["drawdown_pct"].max()),
               "max_consecutive_losses_global": max_streak}
    return per_symbol, overall


def main() -> None:
    parser = argparse.ArgumentParser(description="MarkitTick one-year Binance USD-M Futures backtest")
    parser.add_argument("--days", type=int, default=DAYS)
    parser.add_argument("--symbols", nargs="*", default=SYMBOLS)
    parser.add_argument("--start", help="UTC start date, e.g. 2025-10-24; overrides --days")
    parser.add_argument("--end", help="UTC end date, e.g. 2026-10-24; defaults to now")
    args = parser.parse_args()
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc) if args.end else utc_now()
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc) if args.start else end - timedelta(days=args.days)
    if start >= end:
        raise SystemExit("start must be before end")
    DATA_DIR.mkdir(parents=True, exist_ok=True); OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session(); session.headers.update({"User-Agent": "MarkitTick-Backtest/1.0"})
    all_trades: list[Trade] = []; data_report = []
    print(f"Backtest window UTC: {start.isoformat()} -> {end.isoformat()}")
    print(f"Settings: ATR_MULT={ATR_MULT}, LEN={ATR_MFI_LEN}, BO_OVERRIDE={BO_OVERRIDE}, BO_SENS={BO_SENS}, CLOUD_LEN={CLOUD_LEN}, HTF={HTF_TF} (filter disabled)")
    print(f"Costs: fee={FEE_RATE:.5%} each side, slippage={SLIPPAGE:.5%} each side; margin=${MARGIN_PER_TRADE:.2f}, leverage={LEVERAGE:g}x")
    for symbol in args.symbols:
        print(f"\n[{symbol}] Downloading Binance USD-M Futures 1h klines...")
        try:
            df = fetch_klines(symbol, start, end, session)
            # Save exact downloaded source for reproducibility.
            df.to_csv(DATA_DIR / f"{symbol}_1h.csv", index=False)
            signals = build_signals(df)
            trades = simulate_symbol(symbol, signals)
            all_trades.extend(trades)
            data_report.append({"symbol": symbol, "candles": len(df), "first_open_utc": df["open_time"].iloc[0].isoformat(),
                                "last_open_utc": df["open_time"].iloc[-1].isoformat(), "trades": len(trades), "status": "OK"})
            print(f"  candles={len(df):,}; trades={len(trades)}")
        except Exception as exc:
            data_report.append({"symbol": symbol, "candles": 0, "first_open_utc": None, "last_open_utc": None, "trades": 0,
                                "status": f"ERROR: {type(exc).__name__}: {exc}"})
            print(f"  ERROR: {type(exc).__name__}: {exc}")
    if not all(r["status"] == "OK" for r in data_report):
        print("\nWARNING: At least one symbol failed data validation. Aggregate results are partial; do not treat them as a complete 10-symbol backtest.")
    trades_df = pd.DataFrame([asdict(t) for t in all_trades])
    trades_path = OUT_DIR / "trades.csv"
    trades_df.to_csv(trades_path, index=False)
    per_symbol, overall = summarize(trades_df, args.symbols)
    per_symbol.to_csv(OUT_DIR / "summary_by_symbol.csv", index=False)
    (OUT_DIR / "summary_overall.json").write_text(json.dumps(overall, indent=2, allow_nan=False, default=str), encoding="utf-8")
    pd.DataFrame(data_report).to_csv(OUT_DIR / "data_validation.csv", index=False)
    report = {
        "window_start_utc": start.isoformat(), "window_end_utc": end.isoformat(),
        "market": "Binance USD-M Futures", "interval": INTERVAL, "symbols": args.symbols,
        "settings": {"atr_mult": ATR_MULT, "atr_mfi_len": ATR_MFI_LEN, "bo_override": BO_OVERRIDE,
                     "bo_sens": BO_SENS, "cloud_len": CLOUD_LEN, "htf_tf": HTF_TF, "htf_filter_enabled": False,
                     "rr": RR, "initial_capital_usd": INITIAL_CAPITAL, "margin_per_trade_usd": MARGIN_PER_TRADE,
                     "leverage": LEVERAGE, "fee_rate_per_side": FEE_RATE, "slippage_per_side": SLIPPAGE},
        "data_validation": data_report, "overall": overall,
        "notes": ["Entry is next candle open after confirmed signal.",
                  "If SL and TP2 are both touched in the same 1h candle, SL is assumed first.",
                  "Trades still open at the end are not force-closed and are omitted from closed-trade statistics.",
                  "Fixed notional is used per trade; portfolio margin contention and liquidation engine are not modeled.",
                  "Pine source's O/K counter logic is preserved as written; MFI edge cases can differ from TradingView's internal implementation."]
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False, default=str), encoding="utf-8")
    print("\n=== OVERALL CLOSED-TRADE RESULTS ===")
    print(json.dumps(overall, indent=2, allow_nan=False, default=str))
    print("\n=== PER SYMBOL ===")
    print(per_symbol.to_string(index=False))
    print(f"\nFiles written to {OUT_DIR.resolve()} and candle data to {DATA_DIR.resolve()}")

if __name__ == "__main__":
    main()
