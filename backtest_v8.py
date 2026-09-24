
#!/usr/bin/env python3
"""
LBank Futures / TradingView CSV -> clean 15m dataset
HUNTER-V3 data loader.

IMPORTANT:
- This loader expects CSV exported from TradingView for LBank perpetual futures,
  e.g. LBANK:BTCUSDT.P.
- It does NOT use CCXT fetch_ohlcv(), because LBank Futures historical OHLCV
  is not reliably exposed through CCXT's unified endpoint.
- Put one exported CSV per symbol in INPUT_DIR.
- The script validates, deduplicates, checks 15-minute spacing, removes an
  incomplete final candle when requested, and writes normalized CSVs.

Expected TradingView columns can be:
time, open, high, low, close, Volume
or:
Time, Open, High, Low, Close, Volume
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
import pandas as pd

INPUT_DIR = Path("data/tv_lbank_futures_raw")
OUTPUT_DIR = Path("data/lbank_futures_15m")
DAYS = 365
INTERVAL_MINUTES = 15
REMOVE_INCOMPLETE_LAST = True

SYMBOLS = {
    "BTCUSDT.P": "BTC",
    "ETHUSDT.P": "ETH",
    "SOLUSDT.P": "SOL",
    "SUIUSDT.P": "SUI",
    "AVAXUSDT.P": "AVAX",
    "NEARUSDT.P": "NEAR",
    "ADAUSDT.P": "ADA",
    "BNBUSDT.P": "BNB",
    "APTUSDT.P": "APT",
    "CRVUSDT.P": "CRV",
    "ONDOUSDT.P": "ONDO",
    "PENDLEUSDT.P": "PENDLE",
    "ICPUSDT.P": "ICP",
    "WIFUSDT.P": "WIF",
}

REQUIRED = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]


def normalize_col(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    aliases = {
        "time": "Timestamp",
        "timestamp": "Timestamp",
        "date": "Timestamp",
        "datetime": "Timestamp",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "vol": "Volume",
    }
    return aliases.get(s, str(name).strip())


def read_tv_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [normalize_col(c) for c in df.columns]

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")

    df = df[REQUIRED].copy()

    # TradingView exports normally use epoch seconds for time. Also accept
    # milliseconds and ISO-like timestamps.
    raw_time = df["Timestamp"]
    if pd.api.types.is_numeric_dtype(raw_time):
        vals = pd.to_numeric(raw_time, errors="coerce")
        unit = "ms" if vals.dropna().median() > 10_000_000_000 else "s"
        df["Timestamp"] = pd.to_datetime(vals, unit=unit, utc=True, errors="coerce")
    else:
        df["Timestamp"] = pd.to_datetime(raw_time, utc=True, errors="coerce")

    for c in REQUIRED[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=REQUIRED)
    df = df.drop_duplicates(subset=["Timestamp"], keep="last")
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # OHLC integrity.
    bad = (
        (df["High"] < df[["Open", "Close", "Low"]].max(axis=1))
        | (df["Low"] > df[["Open", "Close", "High"]].min(axis=1))
        | (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
        | (df["Volume"] < 0)
    )
    if bad.any():
        raise ValueError(f"{path.name}: {int(bad.sum())} invalid OHLCV rows")

    if REMOVE_INCOMPLETE_LAST and not df.empty:
        now = pd.Timestamp.now(tz="UTC")
        last = df["Timestamp"].iloc[-1]
        # A 15m candle is complete only after its interval has elapsed.
        if last + pd.Timedelta(minutes=INTERVAL_MINUTES) > now:
            df = df.iloc[:-1].copy()

    return df


def validate_15m(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        raise ValueError(f"{label}: dataset is empty")

    diffs = df["Timestamp"].diff().dropna()
    gaps = diffs[diffs > pd.Timedelta(minutes=INTERVAL_MINUTES)]
    duplicates = int(df["Timestamp"].duplicated().sum())

    # We do not silently fill missing candles. HUNTER-V3 must see the real data.
    return {
        "rows": len(df),
        "start": str(df["Timestamp"].iloc[0]),
        "end": str(df["Timestamp"].iloc[-1]),
        "duplicates": duplicates,
        "gaps": len(gaps),
        "max_gap_minutes": (
            int(gaps.max().total_seconds() / 60) if len(gaps) else 15
        ),
    }


def locate_symbol_file(tv_symbol: str) -> Path | None:
    candidates = [
        INPUT_DIR / f"{tv_symbol}.csv",
        INPUT_DIR / f"{tv_symbol.replace('.', '_')}.csv",
        INPUT_DIR / f"{tv_symbol.lower()}.csv",
        INPUT_DIR / f"{tv_symbol.replace('.', '_').lower()}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    # Fallback: find a CSV containing the symbol name.
    for p in INPUT_DIR.glob("*.csv"):
        if tv_symbol.lower() in p.stem.lower():
            return p
    return None


def process_symbol(tv_symbol: str, short_name: str) -> dict:
    src = locate_symbol_file(tv_symbol)
    if src is None:
        raise FileNotFoundError(
            f"Missing CSV for {tv_symbol}. Expected {INPUT_DIR / (tv_symbol + '.csv')}"
        )

    df = read_tv_csv(src)

    # Keep approximately the latest 365 days from the available export.
    end = df["Timestamp"].iloc[-1]
    start = end - pd.Timedelta(days=DAYS)
    df = df[(df["Timestamp"] >= start) & (df["Timestamp"] <= end)].copy()

    report = validate_15m(df, tv_symbol)

    out = OUTPUT_DIR / f"{short_name}_USDT_FUTURES_15m.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    report.update({
        "tv_symbol": tv_symbol,
        "symbol": short_name,
        "output": str(out),
    })
    return report


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reports = []
    failures = []

    for tv_symbol, short_name in SYMBOLS.items():
        try:
            reports.append(process_symbol(tv_symbol, short_name))
        except Exception as exc:
            failures.append({"symbol": tv_symbol, "error": str(exc)})

    print("\n=== LBank Futures / TradingView 15m DATA AUDIT ===")
    print(f"Symbols requested : {len(SYMBOLS)}")
    print(f"Symbols loaded    : {len(reports)}")
    print(f"Symbols failed    : {len(failures)}")

    for r in reports:
        print(
            f"OK {r['symbol']:7s} rows={r['rows']:6d} "
            f"{r['start']} -> {r['end']} gaps={r['gaps']} "
            f"max_gap={r['max_gap_minutes']}m"
        )

    for f in failures:
        print(f"FAIL {f['symbol']}: {f['error']}")

    if failures:
        print("\nNo strategy/backtest should be run until all required symbols load.")
        return 2

    # 365 days of 15m bars is approximately 35,040 rows/symbol.
    expected = DAYS * 24 * 60 // INTERVAL_MINUTES
    print(f"\nExpected rows for a complete {DAYS}d export: ~{expected:,}/symbol")
    print(f"Normalized files: {OUTPUT_DIR.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
