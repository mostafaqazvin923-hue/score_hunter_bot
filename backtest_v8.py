#!/usr/bin/env python3
"""
HUNTER-V7.1 — AUDITED / CAUSAL BACKTEST ENGINE
XT USDT-M Futures | 15m raw -> 1H / 4H / 1D completed bars

Core rules:
- Fixed RR 1:2
- One portfolio position at a time
- No timeout / no break-even / no trailing
- No lookahead / leakage / repainting
- Entry only on the next completed-bar boundary after a signal bar closes
- Same-candle SL+TP => LOSS
- Position still open at dataset end => OPEN, not forced to LOSS
- 365-day test window with prior history used only as indicator warmup
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

DATA_DIR = Path("data/xt_futures_v7")
TOTAL_DAYS = 365
WARMUP_DAYS = 60

INITIAL_CAPITAL = 1000.0
MARGIN_PER_TRADE = 100.0
LEVERAGE = 50.0
RR = 2.0

FEE_RATE = 0.0007
SLIPPAGE = 0.0003
INTERVAL_MS = 15 * 60 * 1000
LIMIT = 1500


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(DATA_DIR))
    return p.parse_args()


def _extract_rows(payload):
    """Accept XT list payloads and common nested wrappers."""
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return None
    for key in ("result", "data", "rows", "list", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            nested = _extract_rows(value)
            if nested is not None:
                return nested
    return None


def _parse_kline(k):
    """Parse XT compact dict, verbose dict, or list kline formats."""
    if isinstance(k, (list, tuple)):
        if len(k) < 6:
            return None
        ts, o, h, l, c, v = k[:6]
    elif isinstance(k, dict):
        def first_present(*keys):
            for key in keys:
                if key in k and k[key] is not None:
                    return k[key]
            raise KeyError(keys)

        ts = first_present("t", "time", "timestamp")
        o = first_present("o", "open")
        h = first_present("h", "high")
        l = first_present("l", "low")
        c = first_present("c", "close")
        # XT compact kline exposes both `a` and `v`. `v` is used as volume;
        # `a` is kept only as a fallback for verbose/alternate payloads.
        v = first_present("v", "volume", "a")
    else:
        return None

    vals = [int(ts), float(o), float(h), float(l), float(c), float(v)]
    if not all(np.isfinite(x) for x in vals):
        return None
    ts, o, h, l, c, v = vals
    if o <= 0 or h <= 0 or l <= 0 or c <= 0 or h < l:
        return None
    return [int(ts), o, h, l, c, v]


def _validate_dataset(df: pd.DataFrame, target_start_ms: int, final_end_ms: int, symbol: str):
    if df.empty:
        raise RuntimeError(f"No klines fetched for {symbol}")

    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    current_floor = (current_time_ms // INTERVAL_MS) * INTERVAL_MS
    df = df[df["timestamp"] < current_floor].copy()
    df = df[(df["timestamp"] >= target_start_ms) & (df["timestamp"] <= final_end_ms)].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        raise RuntimeError(f"Dataset empty after timestamp filtering for {symbol}")

    ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    row_count = len(df)
    first_ts, last_ts = ts.iloc[0], ts.iloc[-1]
    span_days = (last_ts - first_ts).total_seconds() / 86400.0
    diffs = df["timestamp"].diff().dropna()
    gaps = int((diffs > INTERVAL_MS).sum())
    duplicates = int(df["timestamp"].duplicated().sum())
    monotonic = bool(df["timestamp"].is_monotonic_increasing)

    # Do not accept a merely long dataset that is missing the beginning/end.
    # Tolerance is 2 hours because the first/last API candle can sit on a boundary.
    start_tolerance = 2 * 60 * 60 * 1000
    end_tolerance = 2 * 60 * 60 * 1000
    start_ok = int(df["timestamp"].iloc[0]) <= target_start_ms + start_tolerance
    end_ok = int(df["timestamp"].iloc[-1]) >= final_end_ms - end_tolerance

    print(
        f"Validation {symbol.upper()}: rows={row_count}, first={first_ts}, last={last_ts}, "
        f"span={span_days:.1f}d, gaps={gaps}, duplicates={duplicates}, monotonic={monotonic}, "
        f"start_ok={start_ok}, end_ok={end_ok}"
    )

    min_rows = int((TOTAL_DAYS + WARMUP_DAYS) * 24 * 4 * 0.97)
    min_span = TOTAL_DAYS + WARMUP_DAYS - 5
    if (
        row_count < min_rows
        or span_days < min_span
        or gaps > 50
        or duplicates > 0
        or not monotonic
        or not start_ok
        or not end_ok
    ):
        raise RuntimeError(
            f"Dataset validation failed for {symbol}: rows={row_count}, span={span_days:.1f}d, "
            f"gaps={gaps}, start_ok={start_ok}, end_ok={end_ok}"
        )

    return df


def fetch_xt_futures_data(symbol: str, data_dir: Path) -> pd.DataFrame:
    file_path = data_dir / f"{symbol.upper()}_15m.csv"

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    target_start_ms = now_ms - int((TOTAL_DAYS + WARMUP_DAYS) * 86400 * 1000)
    final_end_ms = (now_ms // INTERVAL_MS) * INTERVAL_MS - 1

    if file_path.exists() and file_path.stat().st_size > 1000:
        try:
            cached = pd.read_csv(file_path)
            required = {"timestamp", "open", "high", "low", "close", "volume"}
            if required.issubset(cached.columns):
                cached["timestamp"] = pd.to_numeric(cached["timestamp"], errors="raise").astype("int64")
                validated = _validate_dataset(cached[list(required)], target_start_ms, final_end_ms, symbol)
                print(f"[CACHE] Using validated dataset for {symbol.upper()}")
                return validated.reset_index(drop=True)
        except Exception as exc:
            print(f"[CACHE] Rejected cached {symbol.upper()} dataset: {exc}")

    data_dir.mkdir(parents=True, exist_ok=True)
    url = "https://fapi.xt.com/future/market/v1/public/q/kline"

    current_start = target_start_ms
    all_klines = []
    page = 0

    while current_start <= final_end_ms:
        page += 1
        window_end = min(final_end_ms, current_start + LIMIT * INTERVAL_MS - 1)
        params = {
            "symbol": symbol,
            "interval": "15m",
            "startTime": current_start,
            "endTime": window_end,
            "limit": LIMIT,
        }

        success = False
        data = None
        last_error = None
        for attempt in range(3):
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=20,
                )
                response.raise_for_status()
                rows = _extract_rows(response.json())
                if rows is None:
                    raise RuntimeError("XT response did not contain a kline list")
                data = rows
                success = True
                break
            except Exception as exc:
                last_error = exc
                time.sleep(attempt + 1)

        if not success:
            raise RuntimeError(f"API request failed for {symbol} page {page}: {last_error}")

        if not data:
            # Empty page is only accepted when the final requested window is reached;
            # otherwise it is a hard data failure.
            if window_end >= final_end_ms:
                break
            raise RuntimeError(f"Unexpected empty XT page for {symbol} page {page}")

        raw_rows = []
        for k in data:
            try:
                parsed = _parse_kline(k)
                if parsed is not None:
                    raw_rows.append(parsed)
            except Exception:
                continue

        if not raw_rows:
            raise RuntimeError(f"No parseable klines for {symbol} page {page}")

        # XT may return an overlapping/out-of-window row near the end of history.
        # Filter to the exact requested page before advancing the cursor.
        in_window = [r for r in raw_rows if current_start <= r[0] <= window_end]
        raw_max = max(r[0] for r in raw_rows)
        raw_min = min(r[0] for r in raw_rows)

        if not in_window:
            # Clean end-of-history condition: API returned only rows older than
            # the requested cursor. Otherwise fail instead of silently truncating.
            if raw_max < current_start:
                break
            raise RuntimeError(
                f"XT returned timestamps but none were inside requested window for {symbol} page {page}: "
                f"requested=[{current_start},{window_end}], returned=[{raw_min},{raw_max}]"
            )

        all_klines.extend(in_window)
        max_timestamp = max(r[0] for r in in_window)

        if max_timestamp < current_start:
            raise RuntimeError(f"Pagination made no progress for {symbol} page {page}")

        next_cursor = max_timestamp + INTERVAL_MS
        if next_cursor <= current_start:
            raise RuntimeError(f"Pagination stalled for {symbol} page {page}")
        current_start = next_cursor
        time.sleep(0.10)

    df = pd.DataFrame(all_klines, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = _validate_dataset(df, target_start_ms, final_end_ms, symbol)
    df.to_csv(file_path, index=False)
    return df.reset_index(drop=True)


def _resample_complete(df: pd.DataFrame, rule: str, expected_count: int) -> pd.DataFrame:
    """Build only fully populated UTC bars from 15m candles."""
    ohlcv = df[["open", "high", "low", "close", "volume"]]
    grouped = ohlcv.resample(rule, closed="left", label="right").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        count=("close", "count"),
    )
    grouped = grouped[grouped["count"] == expected_count].drop(columns=["count"]).dropna()
    return grouped


def load_and_resample(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    work = df.copy()
    work["timestamp_dt"] = pd.to_datetime(work["timestamp"], unit="ms", utc=True)
    work = work.set_index("timestamp_dt").sort_index()
    work = work[~work.index.duplicated(keep="last")].copy()

    df_15m = work[["open", "high", "low", "close", "volume"]].copy()
    df_1h = _resample_complete(work, "1h", 4)
    df_4h = _resample_complete(work, "4h", 16)
    df_1d = _resample_complete(work, "1d", 96)
    return {"15m": df_15m, "1h": df_1h, "4h": df_4h, "1d": df_1d}


def calculate_features(dfs: dict[str, pd.DataFrame]):
    h1 = dfs["1h"].copy()
    prev_close_1h = h1["close"].shift(1)

    tr = pd.concat(
        [
            h1["high"] - h1["low"],
            (h1["high"] - prev_close_1h).abs(),
            (h1["low"] - prev_close_1h).abs(),
        ],
        axis=1,
    ).max(axis=1)
    h1["tr"] = tr
    h1["atr14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    ret_24h = h1["close"] - h1["close"].shift(24)
    vol_24h = h1["atr14"].rolling(24, min_periods=24).mean().replace(0, np.nan)
    h1["norm_ret_24h"] = ret_24h / vol_24h

    # Compression uses only CLOSED 1H bars. The signal bar is therefore fully known.
    rolling_std = prev_close_1h.rolling(20, min_periods=20).std()
    rolling_sma = prev_close_1h.rolling(20, min_periods=20).mean()
    bb_width = (4 * rolling_std) / rolling_sma.replace(0, np.nan)
    h1["bb_width_percentile"] = bb_width.rolling(96, min_periods=96).rank(pct=True)
    h1["vol_expansion"] = tr / h1["atr14"].replace(0, np.nan)

    # Confirmed 4H swing: pivot at t-2 is confirmed at t.
    h4 = dfs["4h"].copy()
    ph = h4["high"].shift(2)
    pl = h4["low"].shift(2)
    h4["swing_high"] = ph.where(
        (ph > h4["high"].shift(3))
        & (ph > h4["high"].shift(4))
        & (ph > h4["high"].shift(1))
        & (ph > h4["high"])
    ).ffill()
    h4["swing_low"] = pl.where(
        (pl < h4["low"].shift(3))
        & (pl < h4["low"].shift(4))
        & (pl < h4["low"].shift(1))
        & (pl < h4["low"])
    ).ffill()

    # Daily bars are RIGHT-LABELED at the instant they finish. At a 1H timestamp,
    # searchsorted(..., right)-1 therefore selects only a completed daily candle.
    d1 = dfs["1d"].copy()
    prior_close = d1["close"].shift(1)
    d1["ema50"] = prior_close.ewm(span=50, adjust=False, min_periods=50).mean()
    d1["ema200"] = prior_close.ewm(span=200, adjust=False, min_periods=200).mean()
    d1["ema50_slope"] = d1["ema50"].diff()

    dfs["1h"] = h1
    dfs["4h"] = h4
    dfs["1d"] = d1


def run_backtest(all_symbol_data: dict[str, dict[str, pd.DataFrame]], test_start_dt: pd.Timestamp):
    all_1h_times = sorted(set().union(*(dfs["1h"].index for dfs in all_symbol_data.values())))

    active_position = None
    trades = []
    cooldown_until = {sym: None for sym in all_symbol_data}

    symbol_arrays = {
        sym: {
            "h1_df": dfs["1h"],
            "h4_df": dfs["4h"],
            "d1_df": dfs["1d"],
        }
        for sym, dfs in all_symbol_data.items()
    }

    for ts in all_1h_times:
        # 1) Manage the existing position ONLY after its actual entry timestamp.
        if active_position is not None and ts >= active_position["entry_ts"]:
            sym = active_position["symbol"]
            h1_df = symbol_arrays[sym]["h1_df"]
            if ts in h1_df.index:
                bar = h1_df.loc[ts]
                hi, lo = float(bar["high"]), float(bar["low"])
                side = active_position["side"]
                sl = active_position["sl"]
                tp = active_position["tp"]
                entry = active_position["entry"]
                notional = MARGIN_PER_TRADE * LEVERAGE

                hit_sl = lo <= sl if side == "LONG" else hi >= sl
                hit_tp = hi >= tp if side == "LONG" else lo <= tp

                if hit_sl or hit_tp:
                    outcome = "LOSS" if (hit_sl and hit_tp) or hit_sl else "WIN"
                    exit_price = sl if outcome == "LOSS" else tp
                    gross = (
                        (exit_price - entry) / entry * notional
                        if side == "LONG"
                        else (entry - exit_price) / entry * notional
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
                    cooldown_until[sym] = ts + pd.Timedelta(hours=3)
                    active_position = None
                    # Critical: no re-entry on the same candle that revealed the exit.
                    continue

        # No signal can be created from the same candle on which a previous trade closed.
        if active_position is not None or ts < test_start_dt:
            continue

        candidates = []
        current_rs = {}
        for symbol, data in symbol_arrays.items():
            h1_df = data["h1_df"]
            if ts in h1_df.index:
                val = h1_df.at[ts, "norm_ret_24h"]
                if np.isfinite(val):
                    current_rs[symbol] = float(val)

        if len(current_rs) < max(5, int(len(SYMBOLS) * 0.7)):
            continue

        sorted_rs = sorted(current_rs.items(), key=lambda x: (x[1], x[0]), reverse=True)
        top_symbols = {s for s, _ in sorted_rs[:3]}
        bot_symbols = {s for s, _ in sorted_rs[-3:]}

        for symbol, data in symbol_arrays.items():
            until = cooldown_until[symbol]
            if until is not None and ts < until:
                continue

            h1_df = data["h1_df"]
            h4_df = data["h4_df"]
            d1_df = data["d1_df"]
            if ts not in h1_df.index:
                continue

            # RIGHT-labeled completed 4H/1D bars only.
            h4_idx = h4_df.index.searchsorted(ts, side="right") - 1
            d1_idx = d1_df.index.searchsorted(ts, side="right") - 1
            h1_idx = h1_df.index.get_loc(ts)
            if h4_idx < 10 or d1_idx < 0 or h1_idx < 48:
                continue

            h4_row = h4_df.iloc[h4_idx]
            d1_row = d1_df.iloc[d1_idx]
            h1_bar = h1_df.iloc[h1_idx]

            ema50 = d1_row.get("ema50", np.nan)
            ema200 = d1_row.get("ema200", np.nan)
            slope = d1_row.get("ema50_slope", np.nan)
            if not all(np.isfinite([ema50, ema200, slope])):
                continue

            daily_long = (d1_row["close"] > ema200) and (ema50 > ema200) and (slope > 0)
            daily_short = (d1_row["close"] < ema200) and (ema50 < ema200) and (slope < 0)
            if not (daily_long or daily_short):
                continue

            bb_pct = h1_bar.get("bb_width_percentile", np.nan)
            vol_exp = h1_bar.get("vol_expansion", np.nan)
            if not np.isfinite(bb_pct) or not np.isfinite(vol_exp):
                continue
            compression_ok = (bb_pct <= 0.30) and (vol_exp >= 1.05)
            if not compression_ok:
                continue

            swing_high = h4_row.get("swing_high", np.nan)
            swing_low = h4_row.get("swing_low", np.nan)
            atr = h1_bar.get("atr14", np.nan)
            p_close = float(h1_bar["close"])
            p_low = float(h1_bar["low"])
            p_high = float(h1_bar["high"])
            if not np.isfinite(atr) or atr <= 0:
                continue

            if daily_long and symbol in top_symbols and np.isfinite(swing_high):
                breakout = p_close > swing_high + 0.10 * atr
                retest = (
                    p_low <= swing_high + 0.20 * atr
                    and p_close >= swing_high - 0.20 * atr
                )
                if breakout or retest:
                    sl = swing_low - 0.20 * atr if np.isfinite(swing_low) else p_low - 1.5 * atr
                    score = float(current_rs[symbol] + vol_exp)
                    candidates.append((score, symbol, "LONG", ts, float(sl)))

            elif daily_short and symbol in bot_symbols and np.isfinite(swing_low):
                breakout = p_close < swing_low - 0.10 * atr
                retest = (
                    p_high >= swing_low - 0.20 * atr
                    and p_close <= swing_low + 0.20 * atr
                )
                if breakout or retest:
                    sl = swing_high + 0.20 * atr if np.isfinite(swing_high) else p_high + 1.5 * atr
                    score = float(-current_rs[symbol] + vol_exp)
                    candidates.append((score, symbol, "SHORT", ts, float(sl)))

        if not candidates:
            continue

        # Deterministic selection without candidates[0] symbol-order bias.
        candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
        _, symbol, side, trigger_ts, sl = candidates[0]

        h1_df = symbol_arrays[symbol]["h1_df"]
        future_indices = h1_df.index[h1_df.index > trigger_ts]
        if len(future_indices) == 0:
            continue
        entry_ts = future_indices[0]
        raw_open = float(h1_df.loc[entry_ts, "open"])
        entry = raw_open * (1.0 + SLIPPAGE) if side == "LONG" else raw_open * (1.0 - SLIPPAGE)
        risk = (entry - sl) if side == "LONG" else (sl - entry)
        if not np.isfinite(risk) or risk <= 0:
            continue

        tp = entry + RR * risk if side == "LONG" else entry - RR * risk
        active_position = {
            "symbol": symbol,
            "side": side,
            "entry_ts": entry_ts,
            "entry": float(entry),
            "sl": float(sl),
            "tp": float(tp),
        }

    return trades, active_position


def _stats(trades):
    total = len(trades)
    wins = sum(t["outcome"] == "WIN" for t in trades)
    losses = total - wins
    wr = wins / total * 100.0 if total else 0.0
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    ordered = sorted(trades, key=lambda t: pd.Timestamp(t["exit_ts"]))
    equity = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    max_dd_dollar = 0.0
    max_dd_pct = 0.0
    streak = 0
    max_streak = 0
    pnls = []
    for t in ordered:
        pnl = float(t["pnl"])
        pnls.append(pnl)
        equity += pnl
        peak = max(peak, equity)
        dd_dollar = peak - equity
        dd_pct = dd_dollar / peak * 100.0 if peak > 0 else 0.0
        max_dd_dollar = max(max_dd_dollar, dd_dollar)
        max_dd_pct = max(max_dd_pct, dd_pct)
        if t["outcome"] == "LOSS":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    avg_win = float(np.mean([x for x in pnls if x > 0])) if any(x > 0 for x in pnls) else 0.0
    avg_loss = float(np.mean([x for x in pnls if x < 0])) if any(x < 0 for x in pnls) else 0.0
    expectancy = float(np.mean(pnls)) if pnls else 0.0
    median = float(np.median(pnls)) if pnls else 0.0
    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "wr": wr,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "pf": pf,
        "net_pnl": float(sum(pnls)),
        "max_dd_dollar": max_dd_dollar,
        "max_dd_pct": max_dd_pct,
        "max_streak": max_streak,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "median": median,
    }


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("HUNTER-V7.1 — AUDITED REGIME / RS / COMPRESSION / STRUCTURE BACKTEST")
    print("=" * 78)

    all_symbol_data = {}
    for sym in SYMBOLS:
        try:
            raw = fetch_xt_futures_data(sym, data_dir)
            dfs = load_and_resample(raw)
            calculate_features(dfs)
            all_symbol_data[sym] = dfs
        except Exception as exc:
            print(f"\n[ABORT] Critical error for mandatory symbol {sym.upper()}: {exc}")
            sys.exit(1)

    # Exactly 365 days of test data; earlier rows are indicator warmup only.
    common_end = min(dfs["1h"].index.max() for dfs in all_symbol_data.values())
    test_start_dt = common_end - pd.Timedelta(days=TOTAL_DAYS)
    print(f"[TIMELINE] common_1h_end={common_end}; test_start={test_start_dt}; warmup=prior data only")

    trades, open_pos = run_backtest(all_symbol_data, test_start_dt)
    s = _stats(trades)
    trades_per_day = s["total"] / TOTAL_DAYS

    print("\n" + "=" * 26 + " PER-SYMBOL " + "=" * 26)
    print(f"{'Symbol':<12} | {'Trades':<7} | {'WR':>7} | {'PF':>7} | {'PnL':>10}")
    print("-" * 58)
    for sym in SYMBOLS:
        st = [t for t in trades if t["symbol"] == sym]
        ss = _stats(st)
        print(f"{sym.upper():<12} | {ss['total']:<7} | {ss['wr']:>6.2f}% | {ss['pf']:>7.2f} | ${ss['net_pnl']:>9.2f}")

    print("\n" + "=" * 31 + " PORTFOLIO " + "=" * 31)
    print(f"Total Trades         : {s['total']}")
    print(f"Wins                 : {s['wins']}")
    print(f"Losses               : {s['losses']}")
    print(f"Win Rate             : {s['wr']:.2f}%")
    print(f"Gross Profit         : ${s['gross_profit']:,.2f}")
    print(f"Gross Loss           : ${s['gross_loss']:,.2f}")
    print(f"Profit Factor        : {s['pf']:.2f}")
    print(f"Net PnL              : ${s['net_pnl']:,.2f}")
    print(f"Max Drawdown ($)     : ${s['max_dd_dollar']:,.2f}")
    print(f"Max Drawdown (%)     : {s['max_dd_pct']:.2f}%")
    print(f"Max Consecutive Loss : {s['max_streak']}")
    print(f"Trades Per Day       : {trades_per_day:.4f}")
    print(f"Average Win          : ${s['avg_win']:,.2f}")
    print(f"Average Loss         : ${s['avg_loss']:,.2f}")
    print(f"Expectancy / Trade   : ${s['expectancy']:,.2f}")
    print(f"Median Trade PnL     : ${s['median']:,.2f}")
    print(f"Open Positions At End: {1 if open_pos is not None else 0}")
    print("=" * 74)

    accepted = (
        s["wr"] > 50.0
        and s["pf"] > 1.20
        and s["net_pnl"] > 0
        and s["max_streak"] <= 4
        and s["total"] >= 150
    )
    print("\n" + "=" * 31 + " ACCEPTANCE " + "=" * 31)
    print("NOTE: This is an in-sample acceptance gate, not an OOS claim.")
    print(f"Data validity     : PASSED ({TOTAL_DAYS}d test + prior warmup)")
    print(f"Win Rate > 50%    : {'PASS' if s['wr'] > 50 else 'FAIL'} ({s['wr']:.2f}%)")
    print(f"Profit Factor >1.2: {'PASS' if s['pf'] > 1.20 else 'FAIL'} ({s['pf']:.2f})")
    print(f"Net PnL > 0       : {'PASS' if s['net_pnl'] > 0 else 'FAIL'} (${s['net_pnl']:,.2f})")
    print(f"Max Loss Streak<=4: {'PASS' if s['max_streak'] <= 4 else 'FAIL'} ({s['max_streak']})")
    print(f"Sample >=150      : {'PASS' if s['total'] >= 150 else 'FAIL'} ({s['total']})")
    print(f"ACCEPTED           : {accepted}")
    print("=" * 74)


if __name__ == "__main__":
    main()
