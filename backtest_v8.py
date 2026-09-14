# CLTS v1 — Causal Liquidity/Structure Trend System
# LBank USDT-M Futures / Direct REST
#
# FIXED:
# - Resolves real LBank SwapU futures contracts directly from LBank
# - Does not use CCXT fetch_ohlcv() for Futures candles
# - Tests Futures OHLCV support before downloading a full year
# - Uses only closed 1H/4H candles
# - Entry = next 1H candle open
# - No timeout / max-bars exit
# - No overlapping position per symbol
# - Portfolio max positions + correlation-group lock
# - Same-candle re-entry blocked
# - Conservative SL-first ambiguity
# - Structural setup invalidation is used; NO time-based setup expiry

import sys
import subprocess
from datetime import datetime, timedelta, timezone

try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "requests"])
    import requests

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "ccxt"])
    import ccxt

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DAYS = 365

BASES = ["BTC", "ETH", "SOL", "XRP", "SUI", "NEAR", "ADA", "LINK", "AVAX", "DOT"]

CORRELATION_GROUP = {
    "BTC": "MAJOR",
    "ETH": "MAJOR",
    "SOL": "L1_HIGH_BETA",
    "SUI": "L1_HIGH_BETA",
    "AVAX": "L1_HIGH_BETA",
    "XRP": "PAYMENTS",
    "ADA": "PAYMENTS",
    "DOT": "L1_OTHER",
    "LINK": "INFRA",
    "NEAR": "INFRA",
}

MAX_OPEN_POSITIONS = 3

PIVOT_LEFT = 2
PIVOT_RIGHT = 2

EMA_FAST = 50
EMA_SLOW = 200
ADX_LEN = 14
ATR_LEN = 14
RSI_LEN = 14
RVOL_LEN = 20

ADX_MIN = 20.0
BOS_ATR_BUFFER = 0.10
RETEST_ATR_BUFFER = 0.15
SL_ATR_BUFFER = 0.20
MAX_RISK_PCT_OF_ENTRY = 0.025
TP_R = 2.0

TAKER_FEE = 0.0006
SLIPPAGE = 0.0002

TIMEOUT_ENABLED = False
CONSERVATIVE_INTRABAR = True


# ============================================================
# LBank FUTURES DATA CONNECTION
# ============================================================

# IMPORTANT:
# We do NOT use ccxt.fetch_ohlcv() here. LBank's documented Contract API
# exposes instrument/market/order endpoints, while historical futures K-lines
# are not documented there. The legacy LBank Futures REST K-line endpoint is
# used directly because this is the same futures data path, not Spot data.
#
# The endpoint is kept configurable and the response is strictly validated.
# If LBank changes/removes it, the script FAILS rather than silently falling
# back to Spot candles.

LBANK_FUTURES_BASE = "https://api.lbkex.com"
LBANK_CONTRACT_BASE = "https://lbkperp.lbank.com"
PRODUCT_GROUP = "SwapU"
REQUEST_TIMEOUT = 20
PAGE_LIMIT = 1000

# Legacy Futures REST K-line endpoint. This is not the documented Contract v1
# endpoint, so we try only known futures variants and NEVER use spot kline.
FUTURES_KLINE_ENDPOINTS = [
    "/v2/future/kline.do",
]

session = requests.Session()
session.headers.update({
    "User-Agent": "CLTS-LBank-Futures-Backtest/3.0",
    "Accept": "application/json",
})


def utc_now():
    return datetime.now(timezone.utc)


def timeframe_ms(timeframe):
    return {"1h": 3600000, "4h": 14400000}[timeframe]


def floor_timestamp_ms(ts_ms, tf_ms):
    return (ts_ms // tf_ms) * tf_ms


def _empty_ohlcv():
    return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])


def _json_from_response(resp):
    try:
        return resp.json()
    except Exception:
        return None


def _extract_kline_rows(payload):
    """Accept common LBank futures kline response shapes only."""
    if payload is None:
        return []

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("data", "result", "rows", "klines", "candles"):
            value = payload.get(key)
            if isinstance(value, list):
                return value

    return []


def _normalize_kline_row(row):
    # LBank legacy futures responses commonly return:
    # [timestamp, open, high, low, close, volume, ...]
    if isinstance(row, (list, tuple)) and len(row) >= 6:
        try:
            return [
                int(float(row[0])),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
            ]
        except Exception:
            return None

    # Also accept a dict-shaped candle if LBank returns one.
    if isinstance(row, dict):
        try:
            ts = row.get("timestamp", row.get("time", row.get("ts", row.get("open_time"))))
            op = row.get("open", row.get("o"))
            hi = row.get("high", row.get("h"))
            lo = row.get("low", row.get("l"))
            cl = row.get("close", row.get("c"))
            vol = row.get("volume", row.get("vol", row.get("v", 0)))
            if ts is None or op is None or hi is None or lo is None or cl is None:
                return None
            return [int(float(ts)), float(op), float(hi), float(lo), float(cl), float(vol)]
        except Exception:
            return None

    return None


def get_futures_contracts():
    """Resolve actual LBank SwapU contract IDs from LBank itself."""
    url = LBANK_CONTRACT_BASE + "/cfd/openApi/v1/pub/instrument"
    resp = session.get(url, params={"productGroup": PRODUCT_GROUP}, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()

    data = payload.get("data", payload) if isinstance(payload, dict) else payload
    if isinstance(data, dict):
        data = data.get("data", data.get("list", []))
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected LBank instrument response: {payload}")

    contracts = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).upper()
        base = str(item.get("baseCurrency", "")).upper()
        quote = str(item.get("clearCurrency", item.get("priceCurrency", ""))).upper()
        if not symbol or not base:
            continue
        if quote not in ("USDT", ""):
            continue
        contracts[base] = symbol

    return contracts


def _fetch_kline_page(contract_symbol, timeframe, start_ms, end_ms):
    minutes = {"1h": 60, "4h": 240}[timeframe]
    # The legacy endpoint has appeared with both interval spellings in LBank
    # integrations. We try these futures-only forms, never spot endpoints.
    variants = [
        {"symbol": contract_symbol, "type": f"{minutes}min", "size": PAGE_LIMIT},
        {"symbol": contract_symbol, "type": str(minutes), "size": PAGE_LIMIT},
        {"symbol": contract_symbol, "period": f"{minutes}min", "size": PAGE_LIMIT},
        {"symbol": contract_symbol, "period": str(minutes), "size": PAGE_LIMIT},
    ]

    last_error = None
    for path in FUTURES_KLINE_ENDPOINTS:
        for params in variants:
            # Add time bounds when the endpoint accepts/ignores them. If ignored,
            # the result is still filtered locally and pagination advances safely.
            params = dict(params)
            params["start"] = start_ms
            params["end"] = end_ms
            try:
                resp = session.get(
                    LBANK_FUTURES_BASE + path,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )
                if resp.status_code != 200:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    continue
                payload = _json_from_response(resp)
                rows = _extract_kline_rows(payload)
                normalized = []
                for row in rows:
                    x = _normalize_kline_row(row)
                    if x is not None:
                        normalized.append(x)
                if normalized:
                    return normalized
                last_error = f"No candle rows in response: {str(payload)[:300]}"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

    raise RuntimeError(
        f"LBank Futures historical K-line endpoint unavailable for {contract_symbol} "
        f"{timeframe}. Last error: {last_error}. "
        "No Spot fallback is permitted."
    )


def fetch_ohlcv_paginated(contract_symbol, timeframe, since_ms, until_ms):
    """Download futures candles only, validate, deduplicate and remove open candle."""
    tf_ms = timeframe_ms(timeframe)
    all_rows = []
    cursor = since_ms
    seen_min = None
    safety = 0

    while cursor < until_ms and safety < 500:
        safety += 1
        batch = _fetch_kline_page(contract_symbol, timeframe, cursor, until_ms)
        if not batch:
            break

        all_rows.extend(batch)
        timestamps = [int(x[0]) for x in batch]
        newest = max(timestamps)
        oldest = min(timestamps)

        if seen_min is not None and oldest <= seen_min and newest <= cursor:
            break
        seen_min = oldest

        # Move strictly forward. If the server ignores bounds and returns a
        # fixed recent window, stop rather than looping forever.
        next_cursor = newest + tf_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor

        # If the server returned fewer than a full page and the newest candle
        # reached the requested interval, there is nothing else to request.
        if len(batch) < PAGE_LIMIT and newest >= until_ms - tf_ms:
            break

        # Hard stop if endpoint is clearly not paginatable.
        if safety >= 3 and newest < since_ms + tf_ms:
            break

    if not all_rows:
        return _empty_ohlcv()

    df = pd.DataFrame(all_rows, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).copy()
    df["Timestamp"] = df["Timestamp"].astype(np.int64)

    # Normalize seconds -> milliseconds if needed.
    if len(df) and int(df["Timestamp"].median()) < 10_000_000_000:
        df["Timestamp"] *= 1000

    df = df[(df["Timestamp"] >= since_ms) & (df["Timestamp"] < until_ms)].copy()
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df = df.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    bad = (
        (df["High"] < df[["Open", "Close"]].max(axis=1)) |
        (df["Low"] > df[["Open", "Close"]].min(axis=1)) |
        (df["High"] < df["Low"]) |
        (df["Volume"] < 0)
    )
    if bad.any():
        print(f"  ⚠️ removing {int(bad.sum())} invalid futures candles for {contract_symbol} {timeframe}")
        df = df.loc[~bad].reset_index(drop=True)

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    last_complete_open = floor_timestamp_ms(now_ms, tf_ms) - tf_ms
    df = df[df["Date"].astype("int64") // 10**6 <= last_complete_open].reset_index(drop=True)

    return df


def resolve_markets():
    """Resolve actual LBank contract symbols directly from the Contract API."""
    contracts = get_futures_contracts()
    resolved = {}
    print("\n🔎 Resolving real LBank SwapU Futures contracts directly from LBank...")
    for base in BASES:
        symbol = contracts.get(base)
        if symbol:
            resolved[base] = symbol
            print(f"  ✅ {base}: LBank contract={symbol}")
        else:
            print(f"  ⚠️ {base}: no SwapU USDT contract returned by LBank")
    if not resolved:
        raise RuntimeError("LBank returned no requested SwapU USDT contracts.")
    return resolved


# ============================================================
# INDICATORS
# ============================================================

def true_range(df):
    prev_close = df["Close"].shift(1)
    return pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr_wilder(df, length=14):
    return true_range(df).ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length,
    ).mean()


def rsi_wilder(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def adx_wilder(df, length=14):
    high = df["High"]
    low = df["Low"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0,
        ),
        index=df.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0,
        ),
        index=df.index,
    )

    tr = true_range(df)

    atr = tr.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length,
    ).mean()

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / length,
            adjust=False,
            min_periods=length,
        ).mean()
        / atr
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / length,
            adjust=False,
            min_periods=length,
        ).mean()
        / atr
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    return dx.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length,
    ).mean()


def add_indicators(df):
    df = df.copy()

    df["EMA50"] = df["Close"].ewm(
        span=EMA_FAST,
        adjust=False,
        min_periods=EMA_FAST,
    ).mean()

    df["EMA200"] = df["Close"].ewm(
        span=EMA_SLOW,
        adjust=False,
        min_periods=EMA_SLOW,
    ).mean()

    df["ATR"] = atr_wilder(df, ATR_LEN)
    df["ADX"] = adx_wilder(df, ADX_LEN)
    df["RSI"] = rsi_wilder(df["Close"], RSI_LEN)

    volume_mean = df["Volume"].rolling(
        RVOL_LEN,
        min_periods=RVOL_LEN,
    ).mean()

    df["RVOL"] = df["Volume"] / volume_mean.replace(0, np.nan)

    return df


# ============================================================
# CONFIRMED PIVOTS
# ============================================================

def build_confirmed_pivots(df):
    n = len(df)

    swing_high_confirmed = [None] * n
    swing_low_confirmed = [None] * n

    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()

    for i in range(PIVOT_LEFT, n - PIVOT_RIGHT):
        left_highs = highs[i - PIVOT_LEFT:i]
        right_highs = highs[i + 1:i + 1 + PIVOT_RIGHT]

        left_lows = lows[i - PIVOT_LEFT:i]
        right_lows = lows[i + 1:i + 1 + PIVOT_RIGHT]

        is_high = (
            highs[i] > left_highs.max()
            and highs[i] >= right_highs.max()
        )

        is_low = (
            lows[i] < left_lows.min()
            and lows[i] <= right_lows.min()
        )

        confirmation_index = i + PIVOT_RIGHT

        if is_high:
            swing_high_confirmed[confirmation_index] = {
                "pivot_index": i,
                "confirmed_index": confirmation_index,
                "price": float(highs[i]),
            }

        if is_low:
            swing_low_confirmed[confirmation_index] = {
                "pivot_index": i,
                "confirmed_index": confirmation_index,
                "price": float(lows[i]),
            }

    df["NewSwingHigh"] = swing_high_confirmed
    df["NewSwingLow"] = swing_low_confirmed

    return df


# ============================================================
# 4H REGIME
# ============================================================

def regime_from_4h(row, previous_row):
    vals = [row["EMA50"], row["EMA200"], row["ADX"], row["Close"]]

    if any(pd.isna(x) for x in vals):
        return "NEUTRAL"

    if pd.isna(previous_row["EMA50"]):
        return "NEUTRAL"

    bullish = (
        row["EMA50"] > row["EMA200"]
        and row["Close"] > row["EMA50"]
        and row["EMA50"] > previous_row["EMA50"]
        and row["ADX"] >= ADX_MIN
    )

    bearish = (
        row["EMA50"] < row["EMA200"]
        and row["Close"] < row["EMA50"]
        and row["EMA50"] < previous_row["EMA50"]
        and row["ADX"] >= ADX_MIN
    )

    if bullish:
        return "BULL"
    if bearish:
        return "BEAR"
    return "NEUTRAL"


def build_4h_regime(df4h):
    df = df4h.copy()
    regimes = ["NEUTRAL"] * len(df)

    for i in range(1, len(df)):
        regimes[i] = regime_from_4h(df.iloc[i], df.iloc[i - 1])

    df["Regime"] = regimes
    return df


def attach_4h_regime(df1h, df4h):
    h4 = df4h.copy()

    h4["RegimeAvailableAt"] = h4["Date"] + pd.Timedelta(hours=4)

    regime_map = h4[["RegimeAvailableAt", "Regime"]].sort_values(
        "RegimeAvailableAt"
    )

    return pd.merge_asof(
        df1h.sort_values("Date"),
        regime_map,
        left_on="Date",
        right_on="RegimeAvailableAt",
        direction="backward",
        allow_exact_matches=True,
    ).assign(
        Regime=lambda x: x["Regime"].fillna("NEUTRAL")
    )


# ============================================================
# PIVOT LOOKUP / SWEEP
# ============================================================

def find_recent_confirmed_pivot(pivot_column, upto_index):
    for j in range(upto_index, -1, -1):
        x = pivot_column.iloc[j]
        if isinstance(x, dict):
            return x
    return None


def generate_sweep(df, i):
    if i < 1:
        return None

    row = df.iloc[i]

    if row["Regime"] not in ("BULL", "BEAR"):
        return None

    if any(pd.isna(row[c]) for c in ["ATR", "RSI", "RVOL"]):
        return None

    atr = float(row["ATR"])
    if atr <= 0:
        return None

    prior_high = find_recent_confirmed_pivot(df["NewSwingHigh"], i - 1)
    prior_low = find_recent_confirmed_pivot(df["NewSwingLow"], i - 1)

    if prior_high is None or prior_low is None:
        return None

    if row["Regime"] == "BULL":
        sweep_level = prior_low["price"]

        if (
            float(row["Low"]) < sweep_level
            and float(row["Close"]) > sweep_level
        ):
            pre_sweep_high = find_recent_confirmed_pivot(
                df["NewSwingHigh"], i - 1
            )

            if pre_sweep_high is not None:
                return {
                    "side": "LONG",
                    "sweep_index": i,
                    "sweep_low": sweep_level,
                    "bos_level": float(pre_sweep_high["price"]),
                }

    else:
        sweep_level = prior_high["price"]

        if (
            float(row["High"]) > sweep_level
            and float(row["Close"]) < sweep_level
        ):
            pre_sweep_low = find_recent_confirmed_pivot(
                df["NewSwingLow"], i - 1
            )

            if pre_sweep_low is not None:
                return {
                    "side": "SHORT",
                    "sweep_index": i,
                    "sweep_high": sweep_level,
                    "bos_level": float(pre_sweep_low["price"]),
                }

    return None


# ============================================================
# STATE MACHINE
# ============================================================

def build_signals(df):
    df = df.copy()
    pending = None
    signals = []

    warmup = max(EMA_SLOW, RVOL_LEN, ATR_LEN, ADX_LEN) + 5

    for i in range(len(df)):
        if i < warmup:
            continue

        row = df.iloc[i]

        if pending is not None:
            side = pending["side"]

            # Regime invalidation is structural/contextual, not time-based.
            if (
                (side == "LONG" and row["Regime"] != "BULL")
                or
                (side == "SHORT" and row["Regime"] != "BEAR")
            ):
                pending = None
                continue

            if pd.isna(row["ATR"]):
                continue

            atr = float(row["ATR"])
            if atr <= 0:
                continue

            # ---------------- LONG ----------------
            if side == "LONG":
                sweep_low = pending["sweep_low"]

                # Invalidate if price closes back below the swept low.
                if float(row["Close"]) < sweep_low:
                    pending = None
                    continue

                if pending["stage"] == "SWEEP":
                    if float(row["Close"]) > pending["bos_level"] + BOS_ATR_BUFFER * atr:
                        pending["stage"] = "BOS"
                        pending["bos_index"] = i
                        continue

                elif pending["stage"] == "BOS":
                    bos = pending["bos_level"]

                    # Structural invalidation before retest.
                    if float(row["Close"]) < sweep_low:
                        pending = None
                        continue

                    retest = (
                        float(row["Low"]) <= bos + RETEST_ATR_BUFFER * atr
                        and float(row["Close"]) > bos
                    )

                    if retest and i + 1 < len(df):
                        structure_low = float(
                            df["Low"].iloc[pending["sweep_index"]:i + 1].min()
                        )

                        signals.append({
                            "signal_index": i,
                            "entry_index": i + 1,
                            "side": "LONG",
                            "bos_level": bos,
                            "structure_low": structure_low,
                        })

                        pending = None
                        continue

            # ---------------- SHORT ----------------
            else:
                sweep_high = pending["sweep_high"]

                if float(row["Close"]) > sweep_high:
                    pending = None
                    continue

                if pending["stage"] == "SWEEP":
                    if float(row["Close"]) < pending["bos_level"] - BOS_ATR_BUFFER * atr:
                        pending["stage"] = "BOS"
                        pending["bos_index"] = i
                        continue

                elif pending["stage"] == "BOS":
                    bos = pending["bos_level"]

                    if float(row["Close"]) > sweep_high:
                        pending = None
                        continue

                    retest = (
                        float(row["High"]) >= bos - RETEST_ATR_BUFFER * atr
                        and float(row["Close"]) < bos
                    )

                    if retest and i + 1 < len(df):
                        structure_high = float(
                            df["High"].iloc[pending["sweep_index"]:i + 1].max()
                        )

                        signals.append({
                            "signal_index": i,
                            "entry_index": i + 1,
                            "side": "SHORT",
                            "bos_level": bos,
                            "structure_high": structure_high,
                        })

                        pending = None
                        continue

        # New sweep only when no setup is pending.
        if pending is None:
            event = generate_sweep(df, i)

            if event is not None:
                event["stage"] = "SWEEP"
                pending = event

    return signals



def validate_ohlcv_support(contract_symbol):
    """Preflight the FUTURES-only historical K-line adapter.

    This deliberately does not use CCXT and does not fall back to Spot.
    It requests a small recent closed-candle window and validates that the
    response contains real OHLCV rows for the requested futures contract.
    """
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    tf_ms = timeframe_ms("1h")
    end_ms = floor_timestamp_ms(now_ms, tf_ms) - tf_ms
    start_ms = end_ms - 3 * tf_ms

    try:
        rows = _fetch_kline_page(
            contract_symbol,
            "1h",
            start_ms,
            end_ms + tf_ms,
        )
    except Exception as exc:
        print(
            f"  ⚠️ OHLCV preflight failed for {contract_symbol}: "
            f"{type(exc).__name__}: {exc}"
        )
        return False

    valid = []
    for row in rows:
        x = _normalize_kline_row(row)
        if x is None:
            continue
        ts, op, hi, lo, cl, vol = x
        if hi >= max(op, cl) and lo <= min(op, cl) and hi >= lo and vol >= 0:
            valid.append(x)

    if not valid:
        print(f"  ⚠️ OHLCV preflight returned no valid Futures candles for {contract_symbol}")
        return False

    print(f"  ✓ Futures OHLCV preflight passed for {contract_symbol} ({len(valid)} candles)")
    return True

# ============================================================
# PREPARE
# ============================================================

def prepare_symbol(name, unified_symbol, since_ms, until_ms):
    print(f"\n📥 {name}: {unified_symbol} | downloading 1H + 4H")

    if not validate_ohlcv_support(unified_symbol):
        print(f"  ⚠️ {name}: skipped because OHLCV preflight failed.")
        return None

    df1h = fetch_ohlcv_paginated(
        unified_symbol, "1h", since_ms, until_ms
    )

    df4h = fetch_ohlcv_paginated(
        unified_symbol,
        "4h",
        since_ms - 400 * 4 * 3600000,
        until_ms,
    )

    if len(df1h) < 500 or len(df4h) < 250:
        print(f"  ⚠️ insufficient data: 1H={len(df1h)}, 4H={len(df4h)}")
        return None

    df4h = build_4h_regime(add_indicators(df4h))
    df1h = attach_4h_regime(
        build_confirmed_pivots(add_indicators(df1h)),
        df4h,
    )

    signals = build_signals(df1h)

    print(
        f"  ✓ 1H={len(df1h)}, 4H={len(df4h)}, "
        f"signals={len(signals)}"
    )

    return {"df": df1h, "signals": signals, "futures_contract": unified_symbol}


# ============================================================
# EXECUTION
# ============================================================

def apply_entry_slippage(side, price):
    return price * (1 + SLIPPAGE) if side == "LONG" else price * (1 - SLIPPAGE)


def apply_exit_slippage(side, price):
    return price * (1 - SLIPPAGE) if side == "LONG" else price * (1 + SLIPPAGE)


def calculate_net_r(pos, exit_price):
    entry = pos["entry_price"]
    risk = abs(entry - pos["stop_loss"])

    if risk <= 0:
        return 0.0

    if pos["side"] == "LONG":
        gross_r = (exit_price - entry) / risk
    else:
        gross_r = (entry - exit_price) / risk

    fee_r = (entry * TAKER_FEE + exit_price * TAKER_FEE) / risk
    return gross_r - fee_r


def score_signal(df, signal):
    row = df.iloc[signal["signal_index"]]
    score = 0.0

    adx = float(row["ADX"])
    rvol = float(row["RVOL"])
    rsi = float(row["RSI"])
    atr = float(row["ATR"])

    if adx >= 30:
        score += 2
    elif adx >= 20:
        score += 1

    if rvol >= 1.50:
        score += 2
    elif rvol >= 1.20:
        score += 1

    if signal["side"] == "LONG":
        if rsi >= 60:
            score += 2
        elif rsi >= 50:
            score += 1
        displacement = float(row["Close"]) - signal["bos_level"]
    else:
        if rsi <= 40:
            score += 2
        elif rsi <= 50:
            score += 1
        displacement = signal["bos_level"] - float(row["Close"])

    if displacement >= 0.25 * atr:
        score += 2
    elif displacement >= 0.10 * atr:
        score += 1

    return score


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(processed):
    active_positions = {}
    all_trades = []

    timestamps = sorted({
        ts
        for item in processed.values()
        for ts in item["df"]["Date"]
    })

    signals_by_entry_ts = {}

    for symbol, item in processed.items():
        df = item["df"]

        for sig in item["signals"]:
            entry_i = sig["entry_index"]
            if entry_i >= len(df):
                continue

            sig2 = dict(sig)
            sig2["symbol"] = symbol
            sig2["entry_ts"] = df["Date"].iloc[entry_i]
            sig2["score"] = score_signal(df, sig)

            signals_by_entry_ts.setdefault(
                sig2["entry_ts"], []
            ).append(sig2)

    for ts in timestamps:
        closed_symbols = set()

        # 1) exits first
        for symbol in list(active_positions):
            pos = active_positions[symbol]
            df = processed[symbol]["df"]

            matches = df.index[df["Date"] == ts]
            if len(matches) == 0:
                continue

            i = int(matches[0])
            candle = df.iloc[i]

            if pos["side"] == "LONG":
                hit_sl = float(candle["Low"]) <= pos["stop_loss"]
                hit_tp = float(candle["High"]) >= pos["take_profit"]
            else:
                hit_sl = float(candle["High"]) >= pos["stop_loss"]
                hit_tp = float(candle["Low"]) <= pos["take_profit"]

            if not (hit_sl or hit_tp):
                continue

            if hit_sl and hit_tp and CONSERVATIVE_INTRABAR:
                outcome = "LOSS"
                reason = "SL_AMBIGUOUS"
                raw_exit = pos["stop_loss"]
            elif hit_sl:
                outcome = "LOSS"
                reason = "SL"
                raw_exit = pos["stop_loss"]
            else:
                outcome = "WIN"
                reason = "TP"
                raw_exit = pos["take_profit"]

            exit_price = apply_exit_slippage(pos["side"], raw_exit)
            net_r = calculate_net_r(pos, exit_price)

            all_trades.append({
                "Timestamp": ts,
                "Symbol": symbol,
                "Side": pos["side"],
                "SignalTimestamp": pos["signal_timestamp"],
                "EntryTimestamp": pos["entry_timestamp"],
                "EntryPrice": pos["entry_price"],
                "StopLoss": pos["stop_loss"],
                "TakeProfit": pos["take_profit"],
                "ExitPrice": exit_price,
                "ExitReason": reason,
                "Outcome": outcome,
                "Gross_R": -1.0 if outcome == "LOSS" else TP_R,
                "Net_R": net_r,
                "BarsHeld": i - pos["entry_index"],
                "HoldingHours": float(i - pos["entry_index"]),
                "Score": pos["score"],
            })

            closed_symbols.add(symbol)

        for symbol in closed_symbols:
            del active_positions[symbol]

        # 2) same-candle lock
        blocked = closed_symbols

        # 3) entries
        candidates = sorted(
            signals_by_entry_ts.get(ts, []),
            key=lambda x: (-x["score"], x["symbol"]),
        )

        used_groups = {
            CORRELATION_GROUP[s]
            for s in active_positions
        }

        for sig in candidates:
            symbol = sig["symbol"]

            if symbol in blocked:
                continue
            if symbol in active_positions:
                continue
            if len(active_positions) >= MAX_OPEN_POSITIONS:
                break

            group = CORRELATION_GROUP[symbol]
            if group in used_groups:
                continue

            df = processed[symbol]["df"]
            entry_i = sig["entry_index"]
            signal_i = sig["signal_index"]

            if entry_i >= len(df):
                continue

            entry_raw = float(df.iloc[entry_i]["Open"])
            entry_price = apply_entry_slippage(sig["side"], entry_raw)
            atr = float(df.iloc[signal_i]["ATR"])

            if sig["side"] == "LONG":
                stop_loss = sig["structure_low"] - SL_ATR_BUFFER * atr
                risk = entry_price - stop_loss

                if risk <= 0 or risk / entry_price > MAX_RISK_PCT_OF_ENTRY:
                    continue

                take_profit = entry_price + TP_R * risk

            else:
                stop_loss = sig["structure_high"] + SL_ATR_BUFFER * atr
                risk = stop_loss - entry_price

                if risk <= 0 or risk / entry_price > MAX_RISK_PCT_OF_ENTRY:
                    continue

                take_profit = entry_price - TP_R * risk

            active_positions[symbol] = {
                "side": sig["side"],
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "entry_index": entry_i,
                "entry_timestamp": ts,
                "signal_timestamp": df["Date"].iloc[signal_i],
                "score": sig["score"],
            }

            used_groups.add(group)

    return pd.DataFrame(all_trades), active_positions


# ============================================================
# REPORT
# ============================================================

def loss_streaks(outcomes):
    result = []
    cur = 0

    for x in outcomes:
        if x == "LOSS":
            cur += 1
        elif cur:
            result.append(cur)
            cur = 0

    if cur:
        result.append(cur)

    return result


def max_streak(outcomes, target):
    best = cur = 0
    for x in outcomes:
        if x == target:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def max_drawdown_from_r(trades):
    if trades.empty:
        return 0.0

    equity = trades["Net_R"].cumsum()
    return float((equity - equity.cummax()).min())


def report(trades, processed):
    print("\n" + "=" * 72)
    print("📊 CLTS v1 — LBank Futures")
    print("=" * 72)

    if trades.empty:
        print("⚠️ No completed trades.")
        return

    trades = trades.sort_values(
        ["Timestamp", "Symbol"]
    ).reset_index(drop=True)

    total = len(trades)
    wins = int((trades["Outcome"] == "WIN").sum())
    losses = int((trades["Outcome"] == "LOSS").sum())

    win_rate = 100 * wins / total
    gross_r = trades["Gross_R"].sum()
    net_r = trades["Net_R"].sum()

    gp = trades.loc[trades["Net_R"] > 0, "Net_R"].sum()
    gl = -trades.loc[trades["Net_R"] < 0, "Net_R"].sum()
    pf = gp / gl if gl > 0 else np.inf

    avg_win = trades.loc[
        trades["Outcome"] == "WIN", "Net_R"
    ].mean() if wins else 0.0

    avg_loss = trades.loc[
        trades["Outcome"] == "LOSS", "Net_R"
    ].mean() if losses else 0.0

    outcomes = trades["Outcome"].tolist()
    streaks = loss_streaks(outcomes)

    print(f"Total Trades:             {total}")
    print(f"Wins:                     {wins}")
    print(f"Losses:                   {losses}")
    print(f"Win Rate:                 {win_rate:.2f}%")
    print(f"Gross R:                  {gross_r:.4f}R")
    print(f"Net R:                    {net_r:.4f}R")
    print(f"Profit Factor:            {pf:.4f}")
    print(f"Average Win:              {avg_win:.4f}R")
    print(f"Average Loss:             {avg_loss:.4f}R")
    print(f"Expectancy:               {trades['Net_R'].mean():.4f}R")
    print(f"Max Drawdown:             {max_drawdown_from_r(trades):.4f}R")
    print(f"Max Win Streak:           {max_streak(outcomes, 'WIN')}")
    print(f"Max Loss Streak:          {max_streak(outcomes, 'LOSS')}")
    print(f"Avg Trades / Day:         {total / DAYS:.4f}")
    print(f"Avg Holding Hours:        {trades['HoldingHours'].mean():.2f}")
    print(f"Long Trades:              {int((trades['Side'] == 'LONG').sum())}")
    print(f"Short Trades:             {int((trades['Side'] == 'SHORT').sum())}")

    print("\n" + "-" * 72)
    print("📉 ALL CONSECUTIVE LOSS STREAKS")
    print("-" * 72)
    print(", ".join(map(str, streaks)) if streaks else "None")
    print(f"Number of loss streaks: {len(streaks)}")

    print("\n" + "-" * 72)
    print("📈 SYMBOL-BY-SYMBOL")
    print("-" * 72)

    rows = []

    for sym in BASES:
        x = trades[trades["Symbol"] == sym]

        if x.empty:
            rows.append({
                "Symbol": sym, "Trades": 0, "Wins": 0, "Losses": 0,
                "WinRate%": 0.0, "GrossR": 0.0, "NetR": 0.0,
                "PF": 0.0, "MaxLossStreak": 0, "AvgHoldH": 0.0,
            })
            continue

        w = int((x["Outcome"] == "WIN").sum())
        l = int((x["Outcome"] == "LOSS").sum())

        gp = x.loc[x["Net_R"] > 0, "Net_R"].sum()
        gl = -x.loc[x["Net_R"] < 0, "Net_R"].sum()

        rows.append({
            "Symbol": sym,
            "Trades": len(x),
            "Wins": w,
            "Losses": l,
            "WinRate%": round(100 * w / len(x), 2),
            "GrossR": round(x["Gross_R"].sum(), 3),
            "NetR": round(x["Net_R"].sum(), 3),
            "PF": round(gp / gl, 3) if gl > 0 else np.inf,
            "MaxLossStreak": max_streak(x["Outcome"].tolist(), "LOSS"),
            "AvgHoldH": round(x["HoldingHours"].mean(), 2),
        })

    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "-" * 72)
    print("🔐 AUDIT")
    print("-" * 72)
    print("✓ Real LBank SwapU contract resolved directly from LBank Contract API")
    print("✓ Futures K-line REST adapter; NO Spot fallback")
    print("✓ Closed 1H + 4H candles only")
    print("✓ Entry = next 1H OPEN")
    print("✓ Confirmed pivots only")
    print("✓ No lookahead / no negative shift / no centered rolling")
    print("✓ No timeout / no max-bars exit")
    print("✓ Structural setup invalidation only")
    print("✓ Max portfolio positions:", MAX_OPEN_POSITIONS)
    print("✓ One position per correlation group")
    print("✓ Same-candle re-entry blocked")
    print("✓ SL first when SL+TP both touched")
    print("✓ End-of-data open positions are not forcibly closed")
    print("✓ Funding is NOT fabricated; historical funding is excluded")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("🚀 CLTS v1 — LBank USDT-M Futures")
    print("   4H Regime → 1H Sweep → BOS → Retest → Entry → SL → 2R")
    print("=" * 72)

    resolved = resolve_markets()

    end_dt = utc_now()
    start_dt = end_dt - timedelta(days=DAYS)
    warmup_dt = start_dt - timedelta(days=100)

    since_ms = int(warmup_dt.timestamp() * 1000)
    until_ms = int(end_dt.timestamp() * 1000)

    print(
        f"\nTest period: {start_dt.isoformat()} → {end_dt.isoformat()}"
    )
    print("Warm-up: 100 days before test period")

    processed = {}

    for name, symbol in resolved.items():
        try:
            item = prepare_symbol(
                name,
                symbol,
                since_ms,
                until_ms,
            )

            if item is None:
                continue

            original_df = item["df"]

            # Keep the warmup in the dataframe while preserving the already
            # generated causal state. Then keep only test-period signals.
            test_start = pd.Timestamp(start_dt)
            test_end = pd.Timestamp(end_dt)

            kept = []

            for s in item["signals"]:
                sig_ts = original_df["Date"].iloc[s["signal_index"]]
                entry_ts = original_df["Date"].iloc[s["entry_index"]]

                if test_start <= sig_ts < test_end and test_start <= entry_ts < test_end:
                    kept.append(s)

            # Keep full dataframe because signal indices reference it.
            item["signals"] = kept
            processed[name] = item

        except Exception as exc:
            print(f"❌ {name} failed: {type(exc).__name__}: {exc}")

    if not processed:
        raise RuntimeError(
            "No symbol data was successfully prepared. "
            "Check LBank Futures historical K-line endpoint availability and the contract-resolution output."
        )

    print("\n⚙️ Starting chronological backtest...")
    trades, open_positions = run_backtest(processed)

    report(trades, processed)

    if open_positions:
        print("\n" + "-" * 72)
        print("🔓 OPEN AT END OF DATA — NOT FORCE-CLOSED")
        print("-" * 72)

        for symbol, pos in open_positions.items():
            print(
                symbol,
                pos["side"],
                "entry=", pos["entry_price"],
                "SL=", pos["stop_loss"],
                "TP=", pos["take_profit"],
                "entry_time=", pos["entry_timestamp"],
            )


if __name__ == "__main__":
    main()
