#!/usr/bin/env python3
"""
HUNTER-V22-STAGE0
CROSS-SECTIONAL RELATIVE-VALUE ROTATION

Hypothesis
----------
At each completed 1H candle, compare each asset's trailing return to BTC's
trailing return. Assets with unusually strong/weak relative performance may
continue to mean-revert/rotate toward the cross-sectional median over the
next short horizon. The signal is built only from information available at
the completed signal candle; entry is at the next 1H open.

This is intentionally different from V17-V21:
- no breakout/continuation trigger,
- no failed-auction pattern,
- no candle-wick/capitulation trigger,
- no volume/order-flow trigger.
The primary information source is cross-sectional relative value.

Integrity protocol
------------------
- XT USDT-M perpetual futures, direct 1H OHLCV.
- Fixed universe; no symbol selection from future results.
- Latest incomplete candle removed.
- All ranks/returns use only completed candles at or before signal time.
- Entry is next completed candle OPEN.
- One open position per symbol; different symbols may be simultaneous.
- Maximum 10 simultaneous $100-margin positions.
- No same-symbol re-entry on the exit candle.
- Same-candle SL+TP = LOSS.
- RR fixed at 1:2.
- No trailing stop, BE, timeout, pyramiding, or parameter optimization.
- Stop values 1.00/1.25/1.50 ATR are sensitivity outputs only; none is selected.
- No future-data filters or symbol exclusions based on results.

Important limitation
--------------------
Relative return is a price-based proxy for relative value/rotation. It is not
true order-book flow, funding, open interest, or institutional positioning.
"""

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


SYMBOLS = [
    "btc_usdt", "eth_usdt", "sol_usdt", "sui_usdt", "avax_usdt",
    "near_usdt", "ada_usdt", "bnb_usdt", "apt_usdt", "crv_usdt",
    "ondo_usdt", "pendle_usdt", "icp_usdt", "wif_usdt",
]

FUTURES_URL = "https://fapi.xt.com/future/market/v1/public/q/kline"

LOOKBACK_DAYS = 455
WARMUP_DAYS = 90
INTERVAL_MS = 60 * 60 * 1000
LIMIT = 1500

INITIAL_CAPITAL = 1000.0
MARGIN = 100.0
LEVERAGE = 50.0
NOTIONAL = MARGIN * LEVERAGE
RR = 2.0
FEE_RATE = 0.0007
SLIPPAGE = 0.0003
MAX_OPEN_POSITIONS = int(INITIAL_CAPITAL // MARGIN)

ATR_PERIOD = 14
RELATIVE_LOOKBACK = 24          # 24 completed 1H bars
RELATIVE_Z_LOOKBACK = 72        # causal cross-sectional history
ENTRY_QUANTILE = 0.20           # bottom/top 20% only
MIN_RELATIVE_Z = 0.75
MIN_CROSS_SECTION = 8
STOP_ATR_VALUES = (1.00, 1.25, 1.50)

CACHE_DIR = Path("data/xt_v22_stage0")
REPORT_DIR = Path("reports/xt_v22_stage0")


def payload_rows(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("result", "data", "rows", "list"):
            if isinstance(obj.get(key), list):
                return obj[key]
    return None


def parse_row(row):
    try:
        if isinstance(row, dict):
            ts = row.get("t", row.get("timestamp"))
            o = row.get("o", row.get("open"))
            h = row.get("h", row.get("high"))
            l = row.get("l", row.get("low"))
            c = row.get("c", row.get("close"))
            v = row.get("a", row.get("volume"))
            if None in (ts, o, h, l, c, v):
                return None
            return int(ts), float(o), float(h), float(l), float(c), float(v)
        if isinstance(row, (list, tuple)) and len(row) >= 6:
            return int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])
    except Exception:
        return None
    return None


def validate(df, symbol, start_ms, end_ms):
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    x = x.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    x = x[
        (x["timestamp"] >= pd.to_datetime(start_ms, unit="ms", utc=True))
        & (x["timestamp"] <= pd.to_datetime(end_ms, unit="ms", utc=True))
    ].copy()

    cols = ["open", "high", "low", "close", "volume"]
    for col in cols:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    if x[cols].isna().any().any():
        raise RuntimeError(f"{symbol}: NaN in OHLCV")
    if (x[["open", "high", "low", "close"]] <= 0).any().any():
        raise RuntimeError(f"{symbol}: non-positive price")
    if (x["volume"] < 0).any():
        raise RuntimeError(f"{symbol}: negative volume")

    gaps = x["timestamp"].diff().dropna().dt.total_seconds().div(3600.0)
    if (gaps > 1.0 + 1e-9).any():
        raise RuntimeError(f"{symbol}: 1H data gap detected; no OHLCV fabrication allowed")

    minimum_rows = int((LOOKBACK_DAYS + WARMUP_DAYS) * 24 * 0.95)
    if len(x) < minimum_rows:
        raise RuntimeError(f"{symbol}: only {len(x)} rows; expected near full history")

    if len(x) >= 2:
        deltas = x["timestamp"].diff().dropna().dt.total_seconds().div(3600.0)
        if (deltas <= 0).any():
            raise RuntimeError(f"{symbol}: non-monotonic timestamp sequence")

    return x.reset_index(drop=True)


def fetch_xt(symbol, refresh=False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{symbol}.csv"

    now = int(time.time() * 1000)
    start_raw = now - int((LOOKBACK_DAYS + WARMUP_DAYS) * 86400 * 1000)
    start_ms = ((start_raw + INTERVAL_MS - 1) // INTERVAL_MS) * INTERVAL_MS
    end_ms = (now // INTERVAL_MS) * INTERVAL_MS - 1

    if cache.exists() and not refresh:
        return validate(pd.read_csv(cache), symbol, start_ms, end_ms)

    session = requests.Session()
    cursor = start_ms
    all_rows = []
    page = 0

    while cursor <= end_ms:
        page += 1
        if page > 1000:
            raise RuntimeError(f"{symbol}: pagination guard tripped")

        window_end = min(end_ms, cursor + LIMIT * INTERVAL_MS - 1)
        params = {
            "symbol": symbol,
            "interval": "1h",
            "startTime": cursor,
            "endTime": window_end,
            "limit": LIMIT,
        }

        payload = None
        last_error = None
        for attempt in range(4):
            try:
                response = session.get(FUTURES_URL, params=params, timeout=30)
                response.raise_for_status()
                payload = payload_rows(response.json())
                if payload is None:
                    raise RuntimeError("XT response contains no kline list")
                break
            except Exception as exc:
                last_error = exc
                time.sleep(1.0 + attempt)

        if payload is None:
            raise RuntimeError(f"{symbol}: page {page} failed: {last_error}")

        parsed = [p for p in (parse_row(r) for r in payload) if p is not None]
        inside = [p for p in parsed if cursor <= p[0] <= window_end]

        if not inside:
            # XT can occasionally return an empty/stale page for a large
            # requested window even though later candles exist. Do NOT
            # fabricate candles and do NOT silently skip the interval.
            # Retry the exact cursor with progressively smaller windows.
            recovered = None
            for divisor in (2, 4, 8):
                retry_end = min(
                    end_ms,
                    cursor + max(1, LIMIT // divisor) * INTERVAL_MS - 1,
                )
                if retry_end < cursor:
                    continue

                retry_params = dict(params)
                retry_params["endTime"] = retry_end

                for attempt in range(3):
                    try:
                        response = session.get(
                            FUTURES_URL,
                            params=retry_params,
                            timeout=30,
                        )
                        response.raise_for_status()
                        retry_payload = payload_rows(response.json())
                        if retry_payload is None:
                            raise RuntimeError("XT retry response contains no kline list")

                        retry_parsed = [
                            p for p in (parse_row(r) for r in retry_payload)
                            if p is not None
                        ]
                        retry_inside = [
                            p for p in retry_parsed
                            if cursor <= p[0] <= retry_end
                        ]

                        if retry_inside:
                            recovered = (retry_inside, retry_end)
                            break
                    except Exception:
                        pass
                    time.sleep(1.0 + attempt)

                if recovered is not None:
                    break

            if recovered is None:
                raise RuntimeError(
                    f"{symbol}: page {page} returned no valid rows after "
                    "pagination retries; refusing to skip/fabricate data"
                )

            inside, window_end = recovered

        all_rows.extend(inside)
        max_ts = max(p[0] for p in inside)
        next_cursor = max_ts + INTERVAL_MS

        if next_cursor <= cursor:
            raise RuntimeError(f"{symbol}: pagination stalled")
        cursor = next_cursor

        if max_ts >= end_ms:
            break
        time.sleep(0.05)

    x = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    x["timestamp"] = pd.to_datetime(x.pop("ts"), unit="ms", utc=True)

    # Never use the currently forming 1H candle.
    cutoff = pd.Timestamp.now(tz="UTC").floor("1h")
    x = x[x["timestamp"] < cutoff]

    x = validate(x, symbol, start_ms, end_ms)
    x.to_csv(cache, index=False)
    return x


def true_range(x):
    prev_close = x["close"].shift(1)
    return pd.concat(
        [
            x["high"] - x["low"],
            (x["high"] - prev_close).abs(),
            (x["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def build_features(df):
    x = df.set_index("timestamp").sort_index().copy()
    x["tr"] = true_range(x)
    x["atr"] = x["tr"].rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()

    # Return ending at t uses close[t] and close[t-L].
    # It is only consumed after candle t is fully closed.
    x["ret_24h"] = x["close"].pct_change(RELATIVE_LOOKBACK, fill_method=None)

    return x.replace([np.inf, -np.inf], np.nan)


def prepare_panel(raw):
    streams = {s: build_features(df) for s, df in raw.items()}
    closes = pd.concat(
        {s: x["close"] for s, x in streams.items()},
        axis=1,
        join="inner",
    )
    closes.columns = list(closes.columns)

    rel = closes.pct_change(RELATIVE_LOOKBACK, fill_method=None)
    btc_rel = rel["btc_usdt"]
    relative = rel.sub(btc_rel, axis=0)

    # Cross-sectional median and dispersion are computed at each completed t.
    # Historical z-score is deliberately causal: it uses the current relative
    # score plus only scores from earlier completed timestamps.
    cs_median = relative.median(axis=1, skipna=True)
    demeaned = relative.sub(cs_median, axis=0)

    hist_mean = demeaned.rolling(
        RELATIVE_Z_LOOKBACK,
        min_periods=RELATIVE_Z_LOOKBACK,
    ).mean().shift(1)
    hist_std = demeaned.rolling(
        RELATIVE_Z_LOOKBACK,
        min_periods=RELATIVE_Z_LOOKBACK,
    ).std(ddof=0).shift(1)

    z = (demeaned - hist_mean) / hist_std.replace(0, np.nan)

    return streams, relative, z


def candidates_at(relative, z, ts):
    row = relative.loc[ts]
    zrow = z.loc[ts]
    valid = row.notna() & zrow.notna()
    if int(valid.sum()) < MIN_CROSS_SECTION:
        return []

    values = row[valid]
    lower = values.quantile(ENTRY_QUANTILE)
    upper = values.quantile(1.0 - ENTRY_QUANTILE)

    out = []
    for symbol in values.index:
        score = float(values[symbol])
        zscore = float(zrow[symbol])
        if not np.isfinite(score) or not np.isfinite(zscore):
            continue

        # Relative weakness -> LONG (reversion toward the cross-sectional
        # median); relative strength -> SHORT.
        if score <= lower and zscore <= -MIN_RELATIVE_Z:
            out.append((symbol, 1, zscore))
        elif score >= upper and zscore >= MIN_RELATIVE_Z:
            out.append((symbol, -1, zscore))

    # Deterministic ordering; this is not a performance-based selection.
    out.sort(key=lambda item: (abs(item[2]), item[0]), reverse=True)
    return out


def compute_exit(x, entry_idx, side, entry, sl, tp):
    for j in range(entry_idx, len(x)):
        bar = x.iloc[j]
        hit_sl = float(bar["low"]) <= sl if side == 1 else float(bar["high"]) >= sl
        hit_tp = float(bar["high"]) >= tp if side == 1 else float(bar["low"]) <= tp

        if not (hit_sl or hit_tp):
            continue

        # If both are touched within one candle, classify as LOSS.
        win = bool(hit_tp and not hit_sl)
        exit_px = tp if win else sl

        gross = (
            NOTIONAL * (exit_px - entry) / entry
            if side == 1
            else NOTIONAL * (entry - exit_px) / entry
        )
        fees = NOTIONAL * FEE_RATE * 2.0
        return x.index[j], float(gross - fees), win

    return None


def simulate(streams, relative, z, stop_mult):
    events = []
    raw_candidates = 0

    for ts in relative.index:
        cands = candidates_at(relative, z, ts)
        raw_candidates += len(cands)
        for symbol, side, zscore in cands:
            x = streams[symbol]
            if ts not in x.index:
                continue
            idx = x.index.get_loc(ts)
            if isinstance(idx, slice) or idx + 1 >= len(x):
                continue
            events.append((ts, symbol, idx, side, zscore))

    events.sort(key=lambda e: (e[0], e[1]))

    equity = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    max_dd = 0.0
    open_positions = {}
    last_exit_by_symbol = {}
    trades = []

    for signal_ts, symbol, i, side, zscore in events:
        x = streams[symbol]
        entry_idx = i + 1
        entry_ts = x.index[entry_idx]

        # Keep a position alive through its exit timestamp. It becomes
        # removable only once the event stream is strictly later.
        stale = [
            s for s, p in open_positions.items()
            if p["exit_ts"] < entry_ts
        ]
        for s in stale:
            del open_positions[s]

        if symbol in open_positions:
            continue

        prev_exit = last_exit_by_symbol.get(symbol)
        if prev_exit is not None and entry_ts <= prev_exit:
            continue

        if len(open_positions) >= MAX_OPEN_POSITIONS:
            continue

        if equity < (len(open_positions) + 1) * MARGIN:
            continue

        atr = float(x.iloc[i]["atr"])
        if not np.isfinite(atr) or atr <= 0:
            continue

        entry_raw = float(x.iloc[entry_idx]["open"])
        entry = entry_raw * (1.0 + SLIPPAGE) if side == 1 else entry_raw * (1.0 - SLIPPAGE)

        stop_dist = stop_mult * atr
        if side == 1:
            sl = entry - stop_dist
            tp = entry + RR * stop_dist
        else:
            sl = entry + stop_dist
            tp = entry - RR * stop_dist

        result = compute_exit(x, entry_idx, side, entry, sl, tp)
        if result is None:
            continue

        exit_ts, pnl, win = result
        open_positions[symbol] = {"entry_ts": entry_ts, "exit_ts": exit_ts}

        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

        trades.append(
            {
                "signal_ts": signal_ts,
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "symbol": symbol,
                "side": "LONG" if side == 1 else "SHORT",
                "relative_score": float(relative.loc[signal_ts, symbol]),
                "relative_z": zscore,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "win": int(win),
                "pnl": pnl,
            }
        )
        last_exit_by_symbol[symbol] = exit_ts

    return pd.DataFrame(trades), raw_candidates


def summarize(trades):
    if trades.empty:
        return dict(trades=0, wins=0, losses=0, wr=0.0, pf=0.0, pnl=0.0,
                    dd=0.0, dd_pct=0.0, streak=0, expectancy=0.0, final=INITIAL_CAPITAL)

    pnl = trades["pnl"].astype(float)
    wins = int((pnl > 0).sum())
    losses = int((pnl <= 0).sum())
    gw = float(pnl[pnl > 0].sum())
    gl = float(-pnl[pnl <= 0].sum())
    pf = gw / gl if gl > 0 else float("inf")

    eq = INITIAL_CAPITAL + pnl.cumsum()
    peak = eq.cummax()
    dd = peak - eq
    max_dd = float(dd.max())
    dd_pct = float((dd / peak.replace(0, np.nan)).max() * 100.0)

    streak = current = 0
    for v in pnl:
        if v <= 0:
            current += 1
            streak = max(streak, current)
        else:
            current = 0

    return dict(
        trades=len(trades),
        wins=wins,
        losses=losses,
        wr=100.0 * wins / len(trades),
        pf=pf,
        pnl=float(pnl.sum()),
        dd=max_dd,
        dd_pct=dd_pct,
        streak=streak,
        expectancy=float(pnl.mean()),
        final=float(eq.iloc[-1]),
    )


def print_report(stop_mult, trades, raw_candidates):
    s = summarize(trades)
    print("\n" + "=" * 78)
    print(f"STAGE-0 STOP = {stop_mult:.2f} ATR | RR = 1:2")
    print("=" * 78)
    print(f"Raw candidate signals   : {raw_candidates}")
    print(f"Closed trades           : {s['trades']}")
    print(f"Wins                    : {s['wins']}")
    print(f"Losses                  : {s['losses']}")
    print(f"Win Rate                : {s['wr']:.2f}%")
    print(f"Profit Factor           : {s['pf']:.4f}")
    print(f"Net PnL                 : ${s['pnl']:,.2f}")
    print(f"Max Drawdown            : ${s['dd']:,.2f}")
    print(f"Max Drawdown %          : {s['dd_pct']:.2f}%")
    print(f"Max Loss Streak         : {s['streak']}")
    print(f"Expectancy / Trade      : ${s['expectancy']:.2f}")
    print(f"Final Equity            : ${s['final']:,.2f}")

    if trades.empty:
        return

    for key, g in trades.groupby("side", sort=True):
        wr = 100.0 * g["win"].sum() / len(g)
        print(f"SIDE {key:5s} trades={len(g):4d} WR={wr:6.2f}% PnL=${g['pnl'].sum():,.2f}")

    print("\nBY SYMBOL")
    for symbol, g in trades.groupby("symbol", sort=True):
        wr = 100.0 * g["win"].sum() / len(g)
        print(f"{symbol:10s} trades={len(g):4d} WR={wr:6.2f}% PnL=${g['pnl'].sum():,.2f}")


def main():
    print("HUNTER-V22-STAGE0 — CROSS-SECTIONAL RELATIVE-VALUE ROTATION")
    print(f"Lookback={LOOKBACK_DAYS}d + warmup={WARMUP_DAYS}d | direct XT 1H futures")
    print(f"RR=1:2 | margin=${MARGIN:.0f} | notional=${NOTIONAL:.0f} | fee={FEE_RATE} | slippage={SLIPPAGE}")
    print(f"Relative lookback={RELATIVE_LOOKBACK}h | causal history={RELATIVE_Z_LOOKBACK}h")
    print(f"Top/bottom quantile={ENTRY_QUANTILE:.2f} | min relative z={MIN_RELATIVE_Z:.2f}")
    print("No parameter optimization; stop values are sensitivity outputs only.")

    raw = {}
    for n, symbol in enumerate(SYMBOLS, 1):
        print(f"\n[{n}/{len(SYMBOLS)}] Fetching {symbol}")
        raw[symbol] = fetch_xt(symbol)

    streams, relative, z = prepare_panel(raw)

    # Require a stable common timeline. Candidate ranks are calculated only
    # where the fixed universe has enough valid observations.
    print("\nData preparation complete.")
    print(f"Common completed timestamps: {len(relative)}")

    for stop_mult in STOP_ATR_VALUES:
        trades, raw_candidates = simulate(streams, relative, z, stop_mult)
        print_report(stop_mult, trades, raw_candidates)

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        if not trades.empty:
            trades.to_csv(
                REPORT_DIR / f"trades_stop_{stop_mult:.2f}.csv",
                index=False,
            )

    print("\nSTAGE-0 DECISION PROTOCOL")
    print("No stop is selected here. Apply the pre-registered activity/edge/DD gates externally.")
    print("If the family fails the gates, close V22 without tuning symbols, sides, quantiles, or stops.")
    print("If it passes, proceed to clean walk-forward validation with a locked hypothesis.")


if __name__ == "__main__":
    main()
