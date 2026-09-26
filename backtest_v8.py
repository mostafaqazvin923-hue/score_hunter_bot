#!/usr/bin/env python3
"""
HUNTER-V4: FAILED AUCTION + VWAP RECLAIM BACKTEST ENGINE
- Market: XT USDT-M Futures REST API (fapi.xt.com)
- Zero Lookahead, Zero Leakage, Zero Repainting
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import requests
import numpy as np
import pandas as pd

SYMBOLS = [
    "btc_usdt", "eth_usdt", "sol_usdt", "sui_usdt", "avax_usdt",
    "near_usdt", "ada_usdt", "bnb_usdt", "apt_usdt", "crv_usdt",
    "ondo_usdt", "pendle_usdt", "icp_usdt", "wif_usdt"
]

DATA_DIR = Path("data/xt_futures_v4")
TOTAL_DAYS = 365
WARMUP_DAYS = 60

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
    file_path = data_dir / f"{symbol.upper()}_15m.csv"

    if file_path.exists() and file_path.stat().st_size > 1000:
        try:
            df_cached = pd.read_csv(file_path)
            if "timestamp" in df_cached.columns:
                df_cached["timestamp_dt"] = pd.to_datetime(
                    df_cached["timestamp"], unit="ms", utc=True
                )
                df_cached = df_cached.sort_values("timestamp").drop_duplicates(
                    subset=["timestamp"]
                ).reset_index(drop=True)

                row_count = len(df_cached)
                span_days = (
                    df_cached["timestamp_dt"].iloc[-1]
                    - df_cached["timestamp_dt"].iloc[0]
                ).total_seconds() / 86400.0
                time_diffs = (
                    df_cached["timestamp_dt"].diff().dt.total_seconds().dropna()
                )
                gaps = int((time_diffs > 900).sum())

                if row_count >= 38000 and span_days >= 400 and gaps <= 100:
                    print(
                        f"[CACHE] Validated cached dataset for {symbol.upper()} "
                        f"(Rows: {row_count}, Span: {span_days:.1f}d, Gaps: {gaps})"
                    )
                    return df_cached.drop(columns=["timestamp_dt"])

        except Exception:
            pass

    data_dir.mkdir(parents=True, exist_ok=True)

    url = "https://fapi.xt.com/future/market/v1/public/q/kline"
    interval_ms = 15 * 60 * 1000
    limit = 1500

    final_end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = final_end_ms - int(
        (TOTAL_DAYS + WARMUP_DAYS) * 86400 * 1000
    )

    current_start = start_ms
    all_klines = []
    page = 0

    while current_start < final_end_ms:
        page += 1
        window_end = min(
            final_end_ms,
            current_start + limit * interval_ms - 1
        )

        params = {
            "symbol": symbol,
            "interval": "15m",
            "startTime": current_start,
            "endTime": window_end,
            "limit": limit
        }

        success = False
        data = None

        for attempt in range(3):
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=15
                )

                if response.status_code == 200:
                    res_json = response.json()
                    data = res_json.get(
                        "result",
                        res_json.get("data", res_json)
                    )
                    if isinstance(data, list):
                        success = True
                        break

            except Exception:
                pass

            time.sleep(1 * (attempt + 1))

        if not success or data is None:
            raise RuntimeError(
                f"API request failed after 3 retries for {symbol} at page {page}"
            )

        if not data:
            raise RuntimeError(
                f"Empty API page before reaching final range for "
                f"{symbol} at page {page}"
            )

        parsed_batch = []

        for k in data:
            try:
                ts = int(
                    k[0] if isinstance(k, list)
                    else k.get("time", k.get("timestamp"))
                )
                o = float(
                    k[1] if isinstance(k, list) else k.get("open")
                )
                h = float(
                    k[2] if isinstance(k, list) else k.get("high")
                )
                l = float(
                    k[3] if isinstance(k, list) else k.get("low")
                )
                c = float(
                    k[4] if isinstance(k, list) else k.get("close")
                )
                v = float(
                    k[5] if isinstance(k, list) else k.get("volume")
                )

                parsed_batch.append([ts, o, h, l, c, v])

            except Exception:
                continue

        if not parsed_batch:
            raise RuntimeError(
                f"Parsed batch is empty for {symbol} at page {page}"
            )

        row_count_page = len(parsed_batch)
        print(f"{symbol.upper()} page={page} rows={row_count_page}")

        all_klines.extend(parsed_batch)

        max_timestamp = max(item[0] for item in parsed_batch)
        next_start = max_timestamp + 1

        if next_start <= current_start:
            raise RuntimeError(
                f"Pagination stalled for {symbol}: "
                f"next_start ({next_start}) <= current_start ({current_start})"
            )

        current_start = next_start
        time.sleep(0.1)

    if not all_klines:
        raise RuntimeError(f"No klines fetched for {symbol}")

    df = pd.DataFrame(
        all_klines,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    df = (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"])
        .reset_index(drop=True)
    )

    current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    current_15m_floor = (
        current_time_ms - current_time_ms % interval_ms
    )

    df = df[df["timestamp"] < current_15m_floor].reset_index(drop=True)

    if df.empty:
        raise RuntimeError(f"No completed 15m candles remain for {symbol}")

    df["timestamp_dt"] = pd.to_datetime(
        df["timestamp"], unit="ms", utc=True
    )

    row_count = len(df)
    first_ts = df["timestamp_dt"].iloc[0]
    last_ts = df["timestamp_dt"].iloc[-1]
    span_days = (
        last_ts - first_ts
    ).total_seconds() / 86400.0

    time_diffs = (
        df["timestamp_dt"].diff().dt.total_seconds().dropna()
    )
    gaps = int((time_diffs > 900).sum())

    print(
        f"Validation {symbol.upper()}: "
        f"rows={row_count}, first={first_ts}, last={last_ts}, "
        f"span={span_days:.1f}d, gaps={gaps}"
    )

    if row_count < 38000 or span_days < 400 or gaps > 100:
        raise RuntimeError(
            f"Insufficient coverage or excessive gaps for {symbol}: "
            f"span={span_days:.1f}d, rows={row_count}, gaps={gaps}"
        )

    df_to_save = df.drop(columns=["timestamp_dt"])
    df_to_save.to_csv(file_path, index=False)

    return df_to_save


def load_and_resample(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = df.copy()
    df["timestamp_dt"] = pd.to_datetime(
        df["timestamp"], unit="ms", utc=True
    )
    df = df.set_index("timestamp_dt").sort_index()
    df = df[~df.index.duplicated(keep="last")].copy()

    df_1h = (
        df.resample("1h")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })
        .dropna()
    )

    df_4h = (
        df.resample("4h", closed="right", label="right")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })
        .dropna()
    )

    df_1d = (
        df.resample("1d", closed="right", label="right")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })
        .dropna()
    )

    return {
        "15m": df,
        "1h": df_1h,
        "4h": df_4h,
        "1d": df_1d
    }


def calculate_indicators(
    dfs: dict[str, pd.DataFrame]
) -> None:
    d1 = dfs["1d"]

    prev_close_1d = d1["close"].shift(1)
    d1["ema50"] = prev_close_1d.ewm(
        span=50, adjust=False, min_periods=50
    ).mean()
    d1["ema200"] = prev_close_1d.ewm(
        span=200, adjust=False, min_periods=200
    ).mean()
    d1["ema50_slope"] = d1["ema50"].diff()

    h4 = dfs["4h"]

    highs = h4["high"].values
    lows = h4["low"].values
    n_h4 = len(h4)

    swing_highs = [np.nan] * n_h4
    swing_lows = [np.nan] * n_h4

    last_sh = np.nan
    last_sl = np.nan

    for i in range(2, n_h4):
        if highs[i - 2] >= highs[i - 1] and highs[i - 2] >= highs[i]:
            last_sh = highs[i - 2]

        if lows[i - 2] <= lows[i - 1] and lows[i - 2] <= lows[i]:
            last_sl = lows[i - 2]

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

    h1["atr14"] = tr.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    h1["high24"] = h1["high"].shift(1).rolling(
        window=24
    ).max()

    h1["low24"] = h1["low"].shift(1).rolling(
        window=24
    ).min()

    h1["vol_sma20"] = h1["volume"].shift(1).rolling(
        window=20
    ).mean()

    tp = (h1["high"] + h1["low"] + h1["close"]) / 3.0
    dates = h1.index.date

    h1["date"] = dates
    h1["cum_tp_vol"] = (
        tp * h1["volume"]
    ).groupby(dates).cumsum()

    h1["cum_vol"] = (
        h1["volume"]
        .groupby(dates)
        .cumsum()
    )

    h1["vwap"] = (
        h1["cum_tp_vol"]
        / h1["cum_vol"].replace(0, np.nan)
    )


def run_backtest(
    all_symbol_data: dict[str, dict[str, pd.DataFrame]]
) -> tuple[list[dict], dict | None]:

    all_1h_times = sorted(
        list(
            set().union(
                *(dfs["1h"].index for dfs in all_symbol_data.values())
            )
        )
    )

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

        position_closed_this_iteration = False

        if active_position is not None:
            sym = active_position["symbol"]
            h1_df = symbol_arrays[sym]["h1_df"]

            if ts == active_position["entry_ts"]:
                # Position becomes active exactly at the next 1H candle open.
                pass

            if ts >= active_position["entry_ts"] and ts in h1_df.index:
                bar = h1_df.loc[ts]

                hi = float(bar["high"])
                lo = float(bar["low"])

                side = active_position["side"]
                sl = active_position["sl"]
                tp = active_position["tp"]
                entry = active_position["entry"]

                notional = MARGIN_PER_TRADE * LEVERAGE

                hit_sl = (
                    lo <= sl if side == "LONG"
                    else hi >= sl
                )
                hit_tp = (
                    hi >= tp if side == "LONG"
                    else lo <= tp
                )

                if hit_sl or hit_tp:
                    outcome = (
                        "LOSS"
                        if (hit_sl and hit_tp) or hit_sl
                        else "WIN"
                    )

                    exit_price = (
                        sl if outcome == "LOSS"
                        else tp
                    )

                    gross = (
                        (exit_price - entry) / entry * notional
                        if side == "LONG"
                        else
                        (entry - exit_price) / entry * notional
                    )

                    fees = notional * FEE_RATE * 2.0
                    pnl = gross - fees

                    trades.append({
                        "symbol": sym,
                        "side": side,
                        "entry_ts": active_position["entry_ts"],
                        "exit_ts": ts,
                        "outcome": outcome,
                        "pnl": float(pnl),
                        "entry": entry,
                        "exit": exit_price,
                    })

                    cooldowns[sym] = 3
                    active_position = None
                    position_closed_this_iteration = True

        if position_closed_this_iteration:
            continue

        if active_position is None:
            candidates = []

            for symbol, data in symbol_arrays.items():
                if cooldowns[symbol] > 0:
                    continue

                h1_df = data["h1_df"]
                h4_df = data["h4_df"]
                d1_df = data["d1_df"]

                d1_idx = (
                    data["d1_times"]
                    .searchsorted(ts, side="right") - 1
                )
                h4_idx = (
                    data["h4_times"]
                    .searchsorted(ts, side="right") - 1
                )
                h1_idx = (
                    data["h1_times"]
                    .searchsorted(ts, side="right") - 1
                )

                if d1_idx < 0 or h4_idx < 0 or h1_idx < 25:
                    continue

                d1_row = d1_df.iloc[d1_idx]

                h4_sub = h4_df.iloc[:h4_idx]
                h1_sub = h1_df.iloc[:h1_idx]

                if len(h4_sub) == 0 or len(h1_sub) < 25:
                    continue

                h4_row = h4_sub.iloc[-1]

                d1_close = d1_row["close"]

                ema50 = d1_row.get("ema50", np.nan)
                ema200 = d1_row.get("ema200", np.nan)
                slope = d1_row.get(
                    "ema50_slope",
                    np.nan
                )

                if not all(
                    np.isfinite([ema50, ema200, slope])
                ):
                    continue

                daily_long = (
                    d1_close > ema200
                    and ema50 > ema200
                    and slope > 0
                )

                daily_short = (
                    d1_close < ema200
                    and ema50 < ema200
                    and slope < 0
                )

                if not (daily_long or daily_short):
                    continue

                c_sh = h4_row.get(
                    "confirmed_sh",
                    np.nan
                )
                c_sl = h4_row.get(
                    "confirmed_sl",
                    np.nan
                )
                h4_close = h4_row["close"]

                if not all(
                    np.isfinite([c_sh, c_sl])
                ):
                    continue

                if daily_long:
                    h4_ok = h4_close > c_sh
                    side = "LONG"
                else:
                    h4_ok = h4_close < c_sl
                    side = "SHORT"

                if not h4_ok:
                    continue

                prev_bar = h1_sub.iloc[-1]
                prev_prev_bar = h1_sub.iloc[-2]

                low24 = prev_bar.get(
                    "low24", np.nan
                )
                high24 = prev_bar.get(
                    "high24", np.nan
                )
                atr = prev_bar.get(
                    "atr14", np.nan
                )
                vol_sma = prev_bar.get(
                    "vol_sma20", np.nan
                )
                vwap = prev_bar.get(
                    "vwap", np.nan
                )
                prev_vwap = prev_prev_bar.get(
                    "vwap", np.nan
                )

                if not all(
                    np.isfinite([
                        low24,
                        high24,
                        atr,
                        vol_sma,
                        vwap,
                        prev_vwap
                    ])
                ):
                    continue

                p_low = prev_bar["low"]
                p_high = prev_bar["high"]
                p_close = prev_bar["close"]
                p_vol = prev_bar["volume"]

                if side == "LONG":
                    sweep = (
                        p_low < low24 - 0.10 * atr
                        and p_low >= low24 - 1.00 * atr
                    )
                    reclaim = p_close > low24
                    vol_ok = p_vol >= 1.10 * vol_sma
                    vwap_reclaim = (
                        prev_prev_bar["close"] <= prev_vwap
                        and p_close > vwap
                    )

                    if not (
                        sweep
                        and reclaim
                        and vol_ok
                        and vwap_reclaim
                    ):
                        continue

                    sl = p_low - 0.20 * atr

                else:
                    sweep = (
                        p_high > high24 + 0.10 * atr
                        and p_high <= high24 + 1.00 * atr
                    )
                    reclaim = p_close < high24
                    vol_ok = p_vol >= 1.10 * vol_sma
                    vwap_reclaim = (
                        prev_prev_bar["close"] >= prev_vwap
                        and p_close < vwap
                    )

                    if not (
                        sweep
                        and reclaim
                        and vol_ok
                        and vwap_reclaim
                    ):
                        continue

                    sl = p_high + 0.20 * atr

                candidates.append(
                    (
                        symbol,
                        side,
                        prev_bar.name,
                        sl
                    )
                )

            if candidates:
                symbol, side, trigger_ts, sl = candidates[0]

                h1_df = symbol_arrays[symbol]["h1_df"]

                next_indices = h1_df.index[
                    h1_df.index > trigger_ts
                ]

                if len(next_indices) == 0:
                    continue

                entry_ts = next_indices[0]
                raw_open = float(
                    h1_df.loc[entry_ts, "open"]
                )

                entry = (
                    raw_open * (1.0 + SLIPPAGE)
                    if side == "LONG"
                    else raw_open * (1.0 - SLIPPAGE)
                )

                risk = (
                    entry - sl
                    if side == "LONG"
                    else sl - entry
                )

                if risk <= 0:
                    continue

                tp = (
                    entry + RR * risk
                    if side == "LONG"
                    else entry - RR * risk
                )

                active_position = {
                    "symbol": symbol,
                    "side": side,
                    "entry_ts": entry_ts,
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                }

    return trades, active_position


def calculate_max_drawdown(
    trades: list[dict]
) -> tuple[float, float]:
    equity = INITIAL_CAPITAL
    peak = equity
    max_dd_dollar = 0.0
    max_dd_pct = 0.0

    ordered_trades = sorted(
        trades,
        key=lambda x: pd.Timestamp(x["exit_ts"])
    )

    for trade in ordered_trades:
        equity += float(trade["pnl"])
        peak = max(peak, equity)

        dd_dollar = peak - equity
        dd_pct = (
            dd_dollar / peak * 100.0
            if peak > 0
            else 0.0
        )

        max_dd_dollar = max(
            max_dd_dollar,
            dd_dollar
        )
        max_dd_pct = max(
            max_dd_pct,
            dd_pct
        )

    return max_dd_dollar, max_dd_pct


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print(
        "HUNTER-V4: 1-YEAR FUTURES REST API "
        "PIPELINE & BACKTEST"
    )
    print("=" * 70)

    all_symbol_data = {}

    for sym in SYMBOLS:
        try:
            df_raw = fetch_xt_futures_data(
                sym,
                data_dir
            )
            dfs = load_and_resample(df_raw)
            calculate_indicators(dfs)
            all_symbol_data[sym] = dfs

        except Exception as e:
            print(
                f"\n[ABORT] Critical error for mandatory "
                f"symbol {sym.upper()}: {e}"
            )
            print(
                "[ABORT] Complete universe required. "
                "Aborting entire backtest."
            )
            sys.exit(1)

    trades, open_pos = run_backtest(
        all_symbol_data
    )

    total = len(trades)
    wins = sum(
        1 for t in trades
        if t["outcome"] == "WIN"
    )
    losses = total - wins

    win_rate = (
        wins / total * 100.0
        if total > 0
        else 0.0
    )

    net_pnl = sum(
        t["pnl"] for t in trades
    )

    gross_profit = sum(
        t["pnl"]
        for t in trades
        if t["pnl"] > 0
    )

    gross_loss = abs(
        sum(
            t["pnl"]
            for t in trades
            if t["pnl"] < 0
        )
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else (
            float("inf")
            if gross_profit > 0
            else 0.0
        )
    )

    max_dd_dollar, max_dd_pct = (
        calculate_max_drawdown(trades)
    )

    consec = 0
    max_consec = 0

    for t in sorted(
        trades,
        key=lambda x: pd.Timestamp(
            x["exit_ts"]
        )
    ):
        if t["outcome"] == "LOSS":
            consec += 1
            max_consec = max(
                max_consec,
                consec
            )
        else:
            consec = 0

    all_1h_times_flat = sorted(
        list(
            set().union(
                *(
                    dfs["1h"].index
                    for dfs in all_symbol_data.values()
                )
            )
        )
    )

    total_span_days = (
        (
            all_1h_times_flat[-1]
            - all_1h_times_flat[0]
        ).total_seconds() / 86400.0
        if all_1h_times_flat
        else 1.0
    )

    trades_per_day = (
        total / total_span_days
        if total_span_days > 0
        else 0.0
    )

    open_count = (
        1 if open_pos is not None
        else 0
    )

    print(
        "\n"
        + "=" * 30
        + " PER-SYMBOL BREAKDOWN "
        + "=" * 30
    )

    print(
        f"{'Symbol':<12} | "
        f"{'Trades':<8} | "
        f"{'Wins':<6} | "
        f"{'Losses':<6} | "
        f"{'Win Rate':<10} | "
        f"{'PnL ($)':<10}"
    )

    print("-" * 65)

    for sym in SYMBOLS:
        sym_trades = [
            t for t in trades
            if t["symbol"] == sym
        ]

        s_total = len(sym_trades)

        s_wins = sum(
            1 for t in sym_trades
            if t["outcome"] == "WIN"
        )

        s_losses = (
            s_total - s_wins
        )

        s_wr = (
            s_wins / s_total * 100.0
            if s_total > 0
            else 0.0
        )

        s_pnl = sum(
            t["pnl"]
            for t in sym_trades
        )

        print(
            f"{sym.upper():<12} | "
            f"{s_total:<8} | "
            f"{s_wins:<6} | "
            f"{s_losses:<6} | "
            f"{s_wr:>6.2f}%    | "
            f"${s_pnl:>9.2f}"
        )

    print(
        "\n"
        + "=" * 40
        + " PORTFOLIO BACKTEST RESULTS "
        + "=" * 40
    )

    print(f"Total Trades         : {total}")
    print(f"Wins                 : {wins}")
    print(f"Losses               : {losses}")
    print(f"Win Rate             : {win_rate:.2f}%")
    print(f"Gross Profit         : ${gross_profit:,.2f}")
    print(f"Gross Loss           : ${gross_loss:,.2f}")
    print(f"Profit Factor        : {profit_factor:.2f}")
    print(f"Net PnL              : ${net_pnl:,.2f}")
    print(f"Max Drawdown ($)     : ${max_dd_dollar:,.2f}")
    print(f"Max Drawdown (%)     : {max_dd_pct:.2f}%")
    print(f"Max Consecutive Loss : {max_consec}")
    print(f"Trades Per Day       : {trades_per_day:.4f}")
    print(f"Open Positions At End: {open_count}")
    print("=" * 68)


if __name__ == "__main__":
    main()
