#!/usr/bin/env python3
"""HUNTER-V23.4-STAGE0 — FUTURES BASIS DISLOCATION / REPRICING

LOCKED RESEARCH HYPOTHESIS
--------------------------
Perpetual-vs-spot basis dislocation/repricing contains short-horizon
information in crypto futures. Signal logic is fixed before Stage-0 results:
  LONG : high basis tail + basis falling + spot 24h return >= 0
  SHORT: low basis tail  + basis rising  + spot 24h return <= 0
with causal cross-sectional historical basis z-score.

IMPORTANT
---------
This version is rebuilt as a clean data/backtest engine. Funding is NOT a
signal input and is not fetched. No missing candle is fabricated. Spot gaps
up to 6h are tolerated only as real missing observations; timestamp-dependent
features become NaN at affected timestamps. Futures gaps are fatal.

Backtest rules
--------------
- 1H completed candles only
- 455 days research window + 7 days warmup
- initial equity $1,000; $100 margin; $5,000 notional (50x)
- RR 1:2
- fee 0.07% per side; slippage 0.03% per side
- one position per symbol; max 10 simultaneous positions
- next-bar-open entry
- no timeout / BE / trailing / pyramiding
- same candle SL+TP => LOSS
- no re-entry on the exit candle
- no carrying an open trade past the available data
- fixed universe; no future symbol selection
- Stage-0 tests all three pre-registered ATR stops; no selection here
"""

import hashlib
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SYMBOLS = [
    "btc_usdt", "eth_usdt", "sol_usdt", "sui_usdt", "avax_usdt",
    "near_usdt", "ada_usdt", "bnb_usdt", "apt_usdt", "crv_usdt",
    "ondo_usdt", "pendle_usdt", "icp_usdt", "wif_usdt",
]

FUT_URL = "https://fapi.xt.com/future/market/v1/public/q/kline"
SPOT_URL = "https://sapi.xt.com/v4/public/kline"

LOOKBACK_DAYS = 455
WARMUP_DAYS = 7
REQUIRED_DAYS = LOOKBACK_DAYS + WARMUP_DAYS
H = 3_600_000
LIMIT = 1000

INITIAL = 1000.0
MARGIN = 100.0
NOTIONAL = 5000.0
RR = 2.0
FEE = 0.0007
SLIP = 0.0003
MAX_POS = 10

ATR_N = 14
BASIS_Z_N = 120
BASIS_CHANGE_N = 24
QUANT = 0.20
MIN_Z = 1.0
MIN_CS = 8
STOPS = (1.0, 1.25, 1.5)

CACHE = Path("data/xt_v23_4")
REPORT = Path("reports/xt_v23_4")

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "hunter-v23.4-stage0/1.0"})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now_hour_ms() -> int:
    return (int(time.time() * 1000) // H) * H


def rows(payload):
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return None
    for key in ("result", "data", "rows", "list"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    result = payload.get("result")
    if isinstance(result, dict):
        for key in ("items", "list", "rows", "data"):
            value = result.get(key)
            if isinstance(value, list):
                return value
    return None


def parse_kline_row(row):
    try:
        if isinstance(row, dict):
            values = [
                row.get("t", row.get("timestamp")),
                row.get("o", row.get("open")),
                row.get("h", row.get("high")),
                row.get("l", row.get("low")),
                row.get("c", row.get("close")),
                row.get("a", row.get("v", row.get("volume", row.get("amount")))),
            ]
        elif isinstance(row, (list, tuple)) and len(row) >= 6:
            values = list(row[:6])
        else:
            return None
        if any(v is None for v in values):
            return None
        ts = int(float(values[0]))
        if ts < 10_000_000_000:
            ts *= 1000
        return (ts,) + tuple(float(v) for v in values[1:])
    except Exception:
        return None


def validate_ohlcv(df, name, start_ms, end_ms, allow_spot_gaps):
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name}: missing columns {missing}")

    x = df[required].copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    if x.isna().any().any():
        bad = int(x.isna().sum().sum())
        raise RuntimeError(f"{name}: invalid/NaN OHLCV values={bad}")

    x = x.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    start = pd.to_datetime(start_ms, unit="ms", utc=True)
    end = pd.to_datetime(end_ms, unit="ms", utc=True)
    x = x[(x.timestamp >= start) & (x.timestamp <= end)].copy()

    if x.empty:
        raise RuntimeError(f"{name}: zero candles after date filtering")
    if (x[["open", "high", "low", "close"]] <= 0).any().any():
        raise RuntimeError(f"{name}: non-positive OHLC")
    if (x.volume < 0).any():
        raise RuntimeError(f"{name}: negative volume")

    gaps = x.timestamp.diff().dropna().dt.total_seconds() / 3600.0
    bad = gaps > 1.0 + 1e-9
    if bad.any():
        max_gap = float(gaps.max())
        count = int(bad.sum())
        if not allow_spot_gaps:
            raise RuntimeError(
                f"{name}: FUTURES data gap detected; max_gap={max_gap:.2f}h; refusing fabrication"
            )
        if max_gap > 6.0:
            raise RuntimeError(
                f"{name}: SPOT gap too large; max_gap={max_gap:.2f}h; refusing fabrication"
            )
        print(f"[DATA] {name}: {count} spot gap(s), max_gap={max_gap:.2f}h; no filling")

    expected = REQUIRED_DAYS * 24
    min_required = int(expected * 0.95)
    if len(x) < min_required:
        span_days = (x.timestamp.iloc[-1] - x.timestamp.iloc[0]).total_seconds() / 86400.0
        raise RuntimeError(
            f"{name}: insufficient history: rows={len(x)}, need>={min_required}, span={span_days:.1f}d"
        )

    return x.reset_index(drop=True)


def request_json(url, params, label, attempts=4):
    last = None
    for attempt in range(1, attempts + 1):
        try:
            response = SESSION.get(url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            return payload
        except Exception as exc:
            last = exc
            if attempt < attempts:
                time.sleep(attempt)
    raise RuntimeError(f"{label}: request failed after {attempts} attempts: {last}")


def fetch_kline(url, symbol, kind):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{kind}_{symbol}.csv"

    end_ms = now_hour_ms() - H  # last fully closed 1H candle
    start_ms = end_ms - REQUIRED_DAYS * 86_400_000
    allow_spot = kind == "spot"

    if path.exists():
        try:
            cached = pd.read_csv(path)
            return validate_ohlcv(cached, f"{kind}_{symbol}[cache]", start_ms, end_ms, allow_spot)
        except Exception as exc:
            print(f"[CACHE] {kind}_{symbol}: invalid/stale cache ({exc}); refetching")

    cur = start_ms
    all_rows = []
    page = 0

    while cur <= end_ms:
        page += 1
        if page > 1000:
            raise RuntimeError(f"{kind}_{symbol}: pagination guard exceeded")

        window_end = min(end_ms, cur + LIMIT * H - H)
        params = {
            "symbol": symbol,
            "interval": "1h",
            "startTime": cur,
            "endTime": window_end,
            "limit": LIMIT,
        }
        payload = request_json(url, params, f"{kind}_{symbol} page={page}")
        raw = rows(payload)
        if raw is None:
            raise RuntimeError(f"{kind}_{symbol} page={page}: API returned no candle list")

        parsed = [parse_kline_row(r) for r in raw]
        inside = sorted({r for r in parsed if r is not None and cur <= r[0] <= window_end})

        if not inside:
            raise RuntimeError(
                f"{kind}_{symbol}: empty interior page at {pd.to_datetime(cur, unit='ms', utc=True)}; refusing skip/fabricate"
            )

        all_rows.extend(inside)
        max_ts = max(r[0] for r in inside)
        next_cur = max_ts + H
        if next_cur <= cur:
            raise RuntimeError(f"{kind}_{symbol}: pagination stalled at page={page}")
        cur = next_cur

        if page % 4 == 0:
            print(f"[FETCH] {kind}_{symbol} page={page} rows={len(all_rows)}")
        if max_ts >= end_ms:
            break
        time.sleep(0.05)

    frame = pd.DataFrame(
        all_rows, columns=["ts", "open", "high", "low", "close", "volume"]
    )
    frame["timestamp"] = pd.to_datetime(frame.pop("ts"), unit="ms", utc=True)
    frame = frame[frame.timestamp <= pd.to_datetime(end_ms, unit="ms", utc=True)]
    frame = validate_ohlcv(frame, f"{kind}_{symbol}", start_ms, end_ms, allow_spot)
    frame.to_csv(path, index=False)
    return frame


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


def build_stream(futures, spot):
    f = futures.set_index("timestamp").sort_index()
    s = spot.set_index("timestamp").sort_index()

    x = f.join(
        s[["close", "open", "high", "low", "volume"]].add_suffix("_spot"),
        how="inner",
    )
    if x.empty:
        raise RuntimeError("merged futures/spot series is empty")

    x["basis"] = (x["close"] - x["close_spot"]) / x["close_spot"]
    x["atr"] = true_range(x).rolling(ATR_N, min_periods=ATR_N).mean()

    # Exact elapsed-time lookup. A missing spot candle can never be converted
    # into a fake 24-row/24-hour observation.
    x["spot_ret24"] = np.nan
    x["basis_chg24"] = np.nan
    prior_ts = x.index - pd.Timedelta(hours=BASIS_CHANGE_N)
    exact_prior = prior_ts.isin(x.index)

    spot_lookup = pd.Series(x["close_spot"].to_numpy(), index=x.index)
    basis_lookup = pd.Series(x["basis"].to_numpy(), index=x.index)
    spot_prev = spot_lookup.reindex(prior_ts)
    basis_prev = basis_lookup.reindex(prior_ts)
    spot_prev.index = x.index
    basis_prev.index = x.index

    x.loc[exact_prior, "spot_ret24"] = (
        x.loc[exact_prior, "close_spot"] / spot_prev.loc[exact_prior] - 1.0
    )
    x.loc[exact_prior, "basis_chg24"] = (
        x.loc[exact_prior, "basis"] - basis_prev.loc[exact_prior]
    )

    x = x.replace([np.inf, -np.inf], np.nan)
    return x


def build_panel(streams):
    # OUTER join is intentional. Missing data for one symbol must not delete
    # valid observations from every other symbol. No values are filled.
    basis = pd.concat({s: x["basis"] for s, x in streams.items()}, axis=1, join="outer")
    change = pd.concat(
        {s: x["basis_chg24"] for s, x in streams.items()}, axis=1, join="outer"
    )

    median = basis.median(axis=1, skipna=True)
    deviation = basis.sub(median, axis=0)

    # Historical cross-sectional factor statistics are shifted one bar so the
    # current signal cannot influence its own z-score.
    mu = deviation.rolling(BASIS_Z_N, min_periods=BASIS_Z_N).mean().shift(1)
    sd = deviation.rolling(BASIS_Z_N, min_periods=BASIS_Z_N).std(ddof=0).shift(1)
    z = (deviation - mu) / sd.replace(0.0, np.nan)
    return basis, change, z


def candidates_at(ts, basis, change, z, streams):
    r = basis.loc[ts]
    zr = z.loc[ts]
    cr = change.loc[ts]

    valid = r.notna() & zr.notna() & cr.notna()
    if int(valid.sum()) < MIN_CS:
        return []

    v = r[valid]
    low = float(v.quantile(QUANT))
    high = float(v.quantile(1.0 - QUANT))
    result = []

    for symbol in v.index:
        stream = streams[symbol]
        if ts not in stream.index:
            continue
        bz = float(zr[symbol])
        bc = float(cr[symbol])
        spot_ret = float(stream.loc[ts, "spot_ret24"])
        if not np.isfinite(bz) or not np.isfinite(bc) or not np.isfinite(spot_ret):
            continue

        basis_value = float(v[symbol])
        if basis_value >= high and bz >= MIN_Z and bc <= 0.0 and spot_ret >= 0.0:
            side = 1
        elif basis_value <= low and bz <= -MIN_Z and bc >= 0.0 and spot_ret <= 0.0:
            side = -1
        else:
            continue
        result.append((symbol, side, bz))

    result.sort(key=lambda q: (abs(q[2]), q[0]), reverse=True)
    return result


def resolve_exit(x, entry_idx, side, entry, stop, target):
    for j in range(entry_idx, len(x)):
        bar = x.iloc[j]
        hit_stop = float(bar["low"]) <= stop if side == 1 else float(bar["high"]) >= stop
        hit_target = float(bar["high"]) >= target if side == 1 else float(bar["low"]) <= target
        if not (hit_stop or hit_target):
            continue

        # Conservative deterministic rule: if both touched in one candle,
        # classify as loss because intrabar order is unknown.
        win = bool(hit_target and not hit_stop)
        exit_price = target if win else stop
        if side == 1:
            gross = NOTIONAL * (exit_price - entry) / entry
        else:
            gross = NOTIONAL * (entry - exit_price) / entry
        net = gross - NOTIONAL * FEE * 2.0
        return x.index[j], net, win
    return None


def simulate(streams, basis, change, z, stop_mult):
    events = []
    raw_candidates = 0

    for ts in basis.index:
        for symbol, side, bz in candidates_at(ts, basis, change, z, streams):
            raw_candidates += 1
            stream = streams[symbol]
            if ts not in stream.index:
                continue
            idx = stream.index.get_loc(ts)
            if isinstance(idx, slice) or idx + 1 >= len(stream):
                continue
            events.append((ts, symbol, idx, side, bz))

    events.sort(key=lambda e: (e[0], e[1]))
    open_until = {}
    last_exit = {}
    equity = INITIAL
    trades = []

    for signal_ts, symbol, signal_idx, side, bz in events:
        x = streams[symbol]
        entry_idx = signal_idx + 1
        entry_ts = x.index[entry_idx]

        # Release positions whose exit happened strictly before this entry.
        for s in list(open_until):
            if open_until[s] < entry_ts:
                del open_until[s]

        # Per-symbol overlap lock + no same-exit-candle re-entry.
        if symbol in open_until:
            continue
        if symbol in last_exit and entry_ts <= last_exit[symbol]:
            continue
        if len(open_until) >= MAX_POS:
            continue
        if equity < (len(open_until) + 1) * MARGIN:
            continue

        atr = float(x.iloc[signal_idx]["atr"])
        if not np.isfinite(atr) or atr <= 0:
            continue

        raw_entry = float(x.iloc[entry_idx]["open"])
        entry = raw_entry * (1.0 + SLIP) if side == 1 else raw_entry * (1.0 - SLIP)
        distance = stop_mult * atr
        if side == 1:
            stop = entry - distance
            target = entry + RR * distance
        else:
            stop = entry + distance
            target = entry - RR * distance

        result = resolve_exit(x, entry_idx, side, entry, stop, target)
        if result is None:
            # Do not invent an exit at the dataset boundary.
            continue

        exit_ts, pnl, win = result
        open_until[symbol] = exit_ts
        last_exit[symbol] = exit_ts
        equity += pnl

        trades.append(
            {
                "signal_ts": signal_ts,
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "symbol": symbol,
                "side": "LONG" if side == 1 else "SHORT",
                "basis": float(basis.loc[signal_ts, symbol]),
                "basis_z": bz,
                "basis_change_24h": float(change.loc[signal_ts, symbol]),
                "entry": entry,
                "sl": stop,
                "tp": target,
                "win": int(win),
                "pnl": float(pnl),
            }
        )

    return pd.DataFrame(trades), raw_candidates


def report(stop_mult, trades, raw_candidates):
    print("\n" + "=" * 80)
    print(f"STAGE-0 STOP = {stop_mult:.2f} ATR | RR = 1:2")
    print("=" * 80)

    n = len(trades)
    print(f"Raw candidate signals   : {raw_candidates}")
    print(f"Closed trades            : {n}")
    if n == 0:
        return

    pnl = trades["pnl"]
    wins = int((pnl > 0).sum())
    losses = n - wins
    gross_win = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl <= 0].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    equity = INITIAL + pnl.cumsum()
    peak = equity.cummax()
    dd = peak - equity
    dd_pct = float((dd / peak.replace(0, np.nan)).max() * 100.0)

    streak = 0
    max_streak = 0
    for value in pnl:
        streak = streak + 1 if value <= 0 else 0
        max_streak = max(max_streak, streak)

    print(f"Wins                    : {wins}")
    print(f"Losses                  : {losses}")
    print(f"Win Rate                : {100.0 * wins / n:.2f}%")
    print(f"Profit Factor           : {pf:.4f}")
    print(f"Net PnL                 : ${pnl.sum():,.2f}")
    print(f"Max Drawdown            : ${dd.max():,.2f}")
    print(f"Max Drawdown %          : {dd_pct:.2f}%")
    print(f"Max Loss Streak         : {max_streak}")
    print(f"Expectancy / Trade      : ${pnl.mean():,.2f}")
    print(f"Final Equity            : ${equity.iloc[-1]:,.2f}")

    for side, group in trades.groupby("side"):
        print(
            f"SIDE {side:5s} trades={len(group):4d} "
            f"WR={100.0 * group.win.mean():6.2f}% PnL=${group.pnl.sum():,.2f}"
        )

    print("\nBY SYMBOL")
    for symbol, group in trades.groupby("symbol"):
        print(
            f"{symbol:10s} trades={len(group):4d} "
            f"WR={100.0 * group.win.mean():6.2f}% PnL=${group.pnl.sum():,.2f}"
        )


def main():
    print("HUNTER-V23.4-STAGE0 — FUTURES BASIS DISLOCATION / REPRICING")
    print(
        "XT 1H spot + perpetual | 455d research + 7d warmup | RR 1:2 | "
        "fixed universe | no funding dependency | no spot-gap filling"
    )
    print(f"Code SHA256: {sha256_file(Path(__file__))}")

    streams = {}
    for number, symbol in enumerate(SYMBOLS, 1):
        print(f"\n[{number}/{len(SYMBOLS)}] {symbol}: futures")
        futures = fetch_kline(FUT_URL, symbol, "futures")

        print(f"[{number}/{len(SYMBOLS)}] {symbol}: spot")
        spot = fetch_kline(SPOT_URL, symbol, "spot")

        streams[symbol] = build_stream(futures, spot)
        merged_days = (
            streams[symbol].index[-1] - streams[symbol].index[0]
        ).total_seconds() / 86400.0
        print(
            f"[DATA] {symbol}: merged_rows={len(streams[symbol])}, "
            f"merged_span={merged_days:.1f}d"
        )
        if merged_days < LOOKBACK_DAYS:
            raise RuntimeError(
                f"{symbol}: merged futures/spot span {merged_days:.1f}d < required {LOOKBACK_DAYS}d"
            )

    basis, change, z = build_panel(streams)
    print("\nData preparation complete.")
    print(f"Panel timestamps: {len(basis)}")
    print(f"Panel start     : {basis.index.min()}")
    print(f"Panel end       : {basis.index.max()}")

    REPORT.mkdir(parents=True, exist_ok=True)
    for stop_mult in STOPS:
        trades, raw_candidates = simulate(streams, basis, change, z, stop_mult)
        report(stop_mult, trades, raw_candidates)
        if not trades.empty:
            trades.to_csv(
                REPORT / f"trades_stop_{stop_mult:.2f}.csv", index=False
            )

    print("\nSTAGE-0 DECISION PROTOCOL")
    print("All three pre-registered stops are reported; no stop is selected here.")
    print("If the family fails the pre-registered gates, close V23 without tuning.")
    print("If it passes, proceed to clean walk-forward validation with the hypothesis locked.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nFATAL BACKTEST ERROR")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
