
import os
import subprocess
import sys
from datetime import datetime, timedelta
from collections import defaultdict, Counter

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import numpy as np
import pandas as pd


# ============================================================
# HUNTER-V74-LSP
#
# Golden Core = EXACT V74 signal / execution logic
# ONLY ADDITION = Loss-Streak Protection in candidate priority
#
# IMPORTANT:
# - No new indicator filter
# - No new coin
# - No change to leverage / margin
# - No change to SL / trailing / timeout
# - No hard "stop trading" after losses
# - Valid signal count is preserved; LSP only changes priority
#   when the 5-position portfolio is full.
# ============================================================

exchange = ccxt.lbank({"enableRateLimit": True})

SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT",
    "LINK": "LINK/USDT",
    "UNI": "UNI/USDT",
    "ICP": "ICP/USDT",
    "INJ": "INJ/USDT",
    "ATOM": "ATOM/USDT",
    "RENDER": "RENDER/USDT",
    "XLM": "XLM/USDT",
    "AAVE": "AAVE/USDT",
    "WIF": "WIF/USDT",
    "ONDO": "ONDO/USDT",
    "DOGE": "DOGE/USDT",
    "BNB": "BNB/USDT",
    "ADA": "ADA/USDT",
}

REMOVED_COINS = {
    "NEAR", "OP", "HYPE", "HBAR", "AVAX", "SUI", "PENDLE", "TIA",
    "FET", "SEI", "ARB", "DOT", "ETC", "SHIB", "STX", "RUNE",
    "MKR", "APT", "LTC", "AR", "IMX", "PEPE", "BONK",
}

SYMBOLS = {
    k: v for k, v in SYMBOLS.items()
    if k not in REMOVED_COINS
}

LOOKBACK_DAYS = 365
TIMEFRAME = "4h"

MAX_POSITIONS = 5

SLIPPAGE = 0.0003
FEE_RATE = 0.0007

ATR_PERIOD = 14
TRAILING_ATR_MULTIPLIER = 2.0
INITIAL_ATR_MULTIPLIER = 1.8
TIMEOUT_CANDLES = 45
EMA_WARMUP = 200

INITIAL_CAPITAL = 1000.0
TRADE_MARGIN = 100.0
LEVERAGE = 80.0

# ------------------------------------------------------------
# LSP SETTINGS
#
# The shield is deliberately SOFT.
# It never rejects an otherwise valid V74 signal.
# It only changes priority while portfolio slots are limited.
# ------------------------------------------------------------

# Global loss streak before the shield becomes active.
LSP_ACTIVATE_AT = 2

# Penalty by consecutive losses of the same symbol + side.
# This is used only for ranking candidates.
LSP_PENALTY_1 = 0.75
LSP_PENALTY_2 = 1.50
LSP_PENALTY_3_PLUS = 2.25

# Extra penalty when the WHOLE portfolio is in a loss streak.
# Kept deliberately small so V74's signal stream is preserved.
GLOBAL_STREAK_PENALTY = 0.35

OUTPUT_DIR = "hunter_v74_lsp_output"

# ============================================================
# LSP3 — HARD LOSS-STREAK CIRCUIT BREAKER
# ============================================================
# The Golden Core signal logic is untouched. LSP3 only blocks NEW
# entries in a direction after repeated losses in that same direction.
LSP3_TRIGGER_STREAK = 2
LSP3_COOLDOWN_CANDLES = 4   # 4 x 4h = 16 hours

start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
since_timestamp = int(start_date.timestamp() * 1000)

print("=" * 68)
print("HUNTER-V74-LSP")
print("Golden Core + Soft Loss-Streak Protection")
print("=" * 68)


# ============================================================
# DATA
# ============================================================

processed_data = {}


def fetch_symbol_data(lbank_symbol):
    all_ohlcv = []
    current_since = since_timestamp
    last_seen = None

    while current_since < exchange.milliseconds():
        batch = None

        for attempt in range(3):
            try:
                batch = exchange.fetch_ohlcv(
                    lbank_symbol,
                    timeframe=TIMEFRAME,
                    since=current_since,
                    limit=1000,
                )
                break
            except Exception:
                if attempt == 2:
                    return None

        if not batch:
            break

        last_ts = batch[-1][0]

        if last_seen is not None and last_ts <= last_seen:
            return None

        all_ohlcv.extend(batch)
        last_seen = last_ts
        current_since = last_ts + 1

        if len(batch) < 1000:
            break

    if not all_ohlcv:
        return None

    df = pd.DataFrame(
        all_ohlcv,
        columns=[
            "Timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ],
    )

    df["Date"] = pd.to_datetime(
        df["Timestamp"],
        unit="ms",
    )

    df = df[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    ]

    df.dropna(inplace=True)
    df.drop_duplicates(
        subset=["Date"],
        keep="last",
        inplace=True,
    )
    df.sort_values(
        "Date",
        inplace=True,
    )
    df.reset_index(
        drop=True,
        inplace=True,
    )

    # Remove incomplete final candle.
    if len(df) >= 2:
        now_ms = exchange.milliseconds()
        last_ms = int(
            df.iloc[-1]["Date"].timestamp() * 1000
        )

        if (
            last_ms
            + 4 * 60 * 60 * 1000
            > now_ms
        ):
            df = df.iloc[:-1].copy()

    if len(df) < EMA_WARMUP + 50:
        return None

    deltas = df["Date"].diff().dropna()

    if (
        not deltas.empty
        and deltas.max()
        > pd.Timedelta(
            hours=4,
            minutes=10,
        )
    ):
        return None

    tr1 = (
        df["High"]
        - df["Low"]
    )

    tr2 = np.abs(
        df["High"]
        - df["Close"].shift(1)
    )

    tr3 = np.abs(
        df["Low"]
        - df["Close"].shift(1)
    )

    df["ATR"] = (
        pd.concat(
            [tr1, tr2, tr3],
            axis=1,
        )
        .max(axis=1)
        .rolling(ATR_PERIOD)
        .mean()
    )

    df["EMA20"] = df[
        "Close"
    ].ewm(
        span=20,
        adjust=False,
    ).mean()

    df["EMA50"] = df[
        "Close"
    ].ewm(
        span=50,
        adjust=False,
    ).mean()

    df["EMA200"] = df[
        "Close"
    ].ewm(
        span=200,
        adjust=False,
    ).mean()

    df["Mom_Short"] = (
        df["Close"]
        - df["Close"].shift(10)
    ) / df["Close"].shift(10)

    df["Mom_Long"] = (
        df["Close"]
        - df["Close"].shift(30)
    ) / df["Close"].shift(30)

    df.set_index(
        "Date",
        inplace=True,
    )

    return df


for symbol, lbank_symbol in SYMBOLS.items():
    df4h = fetch_symbol_data(
        lbank_symbol
    )

    if df4h is not None:
        processed_data[
            symbol
        ] = df4h

print(
    f"Valid symbols: "
    f"{len(processed_data)} / "
    f"{len(SYMBOLS)}"
)


# ============================================================
# COMMON HELPERS
# ============================================================

def get_all_timestamps(data):
    return sorted(
        {
            ts
            for df in data.values()
            for ts in df.index
        }
    )


def build_valid_candidates(
    processed_data,
    ts,
    active_positions,
    market_bull,
    allow_longs,
    allow_shorts,
):
    """
    EXACT V74 signal construction.

    This function intentionally keeps the original V74 rules.
    """

    current_scores = {}

    for symbol, df in processed_data.items():
        if ts not in df.index:
            continue

        val = df.loc[
            ts,
            "Mom_Long",
        ]

        if not np.isnan(val):
            current_scores[
                symbol
            ] = val

    if not current_scores:
        return []

    # EXACT V74 ranking.
    ranked_symbols = sorted(
        current_scores.keys(),
        key=lambda x: float(
            current_scores[x]
        ),
        reverse=market_bull,
    )

    candidates = []

    for symbol in ranked_symbols:
        if symbol in active_positions:
            continue

        df = processed_data[
            symbol
        ]

        if ts not in df.index:
            continue

        i = df.index.get_loc(ts)

        if i < EMA_WARMUP + 1:
            continue

        c4h = df.iloc[i]
        prev_c = df.iloc[i - 1]

        if market_bull:
            if not allow_longs:
                continue

            regime_ok = (
                c4h["Close"]
                > c4h["EMA20"]
                and c4h["EMA20"]
                > c4h["EMA50"]
                and c4h["Close"]
                > c4h["EMA200"]
            )

            pullback_ok = (
                prev_c["Low"]
                <= prev_c["EMA20"]
                * 1.015
            )

            valid_signal = (
                regime_ok
                and pullback_ok
                and (
                    c4h["Mom_Short"]
                    > 0.012
                )
                and (
                    c4h["Mom_Long"]
                    > 0.035
                )
            )

            side = "LONG"

        else:
            if not allow_shorts:
                continue

            regime_ok = (
                c4h["Close"]
                < c4h["EMA20"]
                and c4h["EMA20"]
                < c4h["EMA50"]
                and c4h["Close"]
                < c4h["EMA200"]
            )

            pullback_ok = (
                prev_c["High"]
                >= prev_c["EMA20"]
                * 0.985
            )

            valid_signal = (
                regime_ok
                and pullback_ok
                and (
                    c4h["Mom_Short"]
                    < -0.012
                )
                and (
                    c4h["Mom_Long"]
                    < -0.035
                )
            )

            side = "SHORT"

        if not valid_signal:
            continue

        entry_price = (
            c4h["Open"]
            * (
                1 + SLIPPAGE
            )
            if side == "LONG"
            else
            c4h["Open"]
            * (
                1 - SLIPPAGE
            )
        )

        initial_sl = (
            entry_price
            - INITIAL_ATR_MULTIPLIER
            * c4h["ATR"]
            if side == "LONG"
            else
            entry_price
            + INITIAL_ATR_MULTIPLIER
            * c4h["ATR"]
        )

        initial_risk = abs(
            entry_price
            - initial_sl
        )

        sl_dist_pct = (
            initial_risk
            / entry_price
        )

        if not (
            0.01
            <= sl_dist_pct
            <= 0.04
        ):
            continue

        candidates.append(
            {
                "symbol": symbol,
                "side": side,
                "entry_price": float(
                    entry_price
                ),
                "initial_sl": float(
                    initial_sl
                ),
                "initial_risk": float(
                    initial_risk
                ),
                "entry_index": int(i),
                "mom_long": float(
                    c4h["Mom_Long"]
                ),
            }
        )

    return candidates


# ============================================================
# LOSS-STREAK SHIELD
# ============================================================

def lsp_penalty(
    symbol,
    side,
    symbol_side_losses,
    global_loss_streak,
):
    """
    Soft penalty only.

    The signal remains valid.
    No signal is rejected here.
    """

    streak = symbol_side_losses.get(
        (symbol, side),
        0,
    )

    if streak <= 0:
        penalty = 0.0

    elif streak == 1:
        penalty = LSP_PENALTY_1

    elif streak == 2:
        penalty = LSP_PENALTY_2

    else:
        penalty = LSP_PENALTY_3_PLUS

    if (
        global_loss_streak
        >= LSP_ACTIVATE_AT
    ):
        penalty += GLOBAL_STREAK_PENALTY

    return penalty


def select_candidates_lsp(
    candidates,
    slots,
    symbol_side_losses,
    global_loss_streak,
):
    """
    LSP selection.

    Crucial behavior:
    - Does NOT remove valid candidates.
    - Does NOT impose a direction cap.
    - Does NOT reduce MAX_POSITIONS.
    - Only changes which valid candidate gets a scarce slot.
    """

    if slots <= 0:
        return []

    if not candidates:
        return []

    ranked = []

    for original_rank, candidate in enumerate(
        candidates
    ):
        penalty = lsp_penalty(
            candidate["symbol"],
            candidate["side"],
            symbol_side_losses,
            global_loss_streak,
        )

        # Primary criterion remains the ORIGINAL V74
        # Momentum ranking. The penalty is only a small
        # secondary adjustment when candidates compete.
        #
        # Convert original rank to a stable quality number:
        # earlier V74 candidates receive a higher base score.
        base_score = (
            len(candidates)
            - original_rank
        )

        adjusted_score = (
            float(base_score)
            - penalty
        )

        ranked.append(
            (
                adjusted_score,
                original_rank,
                candidate,
            )
        )

    ranked.sort(
        key=lambda x: (
            x[0],
            -x[1],
        ),
        reverse=True,
    )

    selected = [
        item[2]
        for item in ranked[:slots]
    ]

    return selected


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    processed_data,
    use_lsp=False,
    use_lsp3=False,
    gate_mode=None,
    use_firewall=False,
):
    all_timestamps = get_all_timestamps(
        processed_data
    )

    active_positions = {}
    all_trades = []

    # Consecutive losses for exact symbol + side.
    symbol_side_losses = defaultdict(int)

    # Consecutive losses across portfolio exit order.
    global_loss_streak = 0

    # LSP3 state: directional loss-event streak + temporary entry cooldown.
    # IMPORTANT: losses on the same 4h timestamp count as ONE directional event,
    # so simultaneous exits cannot artificially accelerate the breaker.
    direction_loss_streak = {"LONG": 0, "SHORT": 0}
    direction_cooldown = {"LONG": 0, "SHORT": 0}
    lsp3_trigger_log = []

    # LOSS FIREWALL:
    # After 2 consecutive loss EVENTS in one direction, stop NEW entries
    # in that direction while any position in that direction remains open.
    # Existing positions are never touched. No signal/SL/trailing/sizing change.
    firewall_loss_events = {"LONG": 0, "SHORT": 0}
    firewall_locked = {"LONG": False, "SHORT": False}

    diagnostics = Counter()

    equity_curve = []

    for ts in all_timestamps:
        symbols_to_close = []

        # --------------------------------------------------------
        # 1. EXACT V74 POSITION MANAGEMENT
        # --------------------------------------------------------
        for symbol, pos in list(
            active_positions.items()
        ):
            df = processed_data[
                symbol
            ]

            if ts not in df.index:
                continue

            c4h = df.loc[ts]

            if pos["side"] == "LONG":
                if (
                    c4h["High"]
                    > pos["highest_price"]
                ):
                    pos[
                        "highest_price"
                    ] = c4h["High"]

                    new_trailing_sl = (
                        pos[
                            "highest_price"
                        ]
                        - TRAILING_ATR_MULTIPLIER
                        * c4h["ATR"]
                    )

                    if (
                        new_trailing_sl
                        > pos["stop_loss"]
                    ):
                        pos[
                            "stop_loss"
                        ] = new_trailing_sl

                hit_sl = (
                    c4h["Low"]
                    <= pos["stop_loss"]
                )

            else:
                if (
                    c4h["Low"]
                    < pos["lowest_price"]
                ):
                    pos[
                        "lowest_price"
                    ] = c4h["Low"]

                    new_trailing_sl = (
                        pos[
                            "lowest_price"
                        ]
                        + TRAILING_ATR_MULTIPLIER
                        * c4h["ATR"]
                    )

                    if (
                        new_trailing_sl
                        < pos["stop_loss"]
                    ):
                        pos[
                            "stop_loss"
                        ] = new_trailing_sl

                hit_sl = (
                    c4h["High"]
                    >= pos["stop_loss"]
                )

            curr_i = df.index.get_loc(
                ts
            )

            candles_held = (
                curr_i
                - pos["entry_index"]
            )

            is_timeout = (
                candles_held
                >= TIMEOUT_CANDLES
            )

            if not (
                hit_sl
                or is_timeout
            ):
                continue

            initial_risk = (
                pos["initial_risk"]
            )

            if pos["side"] == "LONG":
                exit_p = (
                    min(
                        pos["stop_loss"],
                        c4h["Open"],
                    )
                    if hit_sl
                    else c4h["Close"]
                )

                r_real = (
                    (
                        exit_p
                        - pos[
                            "entry_price"
                        ]
                    )
                    / initial_risk
                ) - (
                    FEE_RATE * 2
                )

                price_return_pct = (
                    exit_p
                    - pos[
                        "entry_price"
                    ]
                ) / pos[
                    "entry_price"
                ]

            else:
                exit_p = (
                    max(
                        pos["stop_loss"],
                        c4h["Open"],
                    )
                    if hit_sl
                    else c4h["Close"]
                )

                r_real = (
                    (
                        pos[
                            "entry_price"
                        ]
                        - exit_p
                    )
                    / initial_risk
                ) - (
                    FEE_RATE * 2
                )

                price_return_pct = (
                    pos[
                        "entry_price"
                    ]
                    - exit_p
                ) / pos[
                    "entry_price"
                ]

            outcome = (
                "WIN"
                if r_real > 0
                else "LOSS"
            )

            position_notional = (
                TRADE_MARGIN
                * LEVERAGE
            )

            dollar_pnl = (
                position_notional
                * price_return_pct
            ) - (
                position_notional
                * FEE_RATE
                * 2
            )

            if outcome == "LOSS":
                global_loss_streak += 1

                symbol_side_losses[
                    (
                        symbol,
                        pos["side"],
                    )
                ] += 1

                # LSP3 is updated AFTER all exits at this timestamp are
                # aggregated below. Do not update it per individual trade.

            else:
                global_loss_streak = 0

                symbol_side_losses[
                    (
                        symbol,
                        pos["side"],
                    )
                ] = 0

                # LSP3 win/reset is handled after all exits at this timestamp.

            all_trades.append(
                {
                    "Timestamp": ts,
                    "Symbol": symbol,
                    "Side": pos["side"],
                    "Outcome": outcome,
                    "Return": r_real,
                    "Dollar_PnL": dollar_pnl,
                    "ExitOrder": len(
                        all_trades
                    ),
                    # Diagnostic fields only; they do NOT affect execution.
                    "EntryTimestamp": pos.get("entry_timestamp", pd.NaT),
                    "EntryPrice": pos["entry_price"],
                    "InitialStop": pos["initial_stop"],
                    "ExitPrice": exit_p,
                    "LSP_Active": int(
                        use_lsp
                    ),
                    "LSP3_Active": int(
                        use_lsp3
                    ),
                }
            )

            symbols_to_close.append(
                symbol
            )

        for sym in symbols_to_close:
            del active_positions[
                sym
            ]

        # --------------------------------------------------------
        # LSP3 — EVENT-BASED DIRECTIONAL CIRCUIT BREAKER
        # --------------------------------------------------------
        # All exits at the same 4h timestamp are one event per direction.
        # A WIN in a direction resets that direction, even if another position
        # in the same direction also lost on the same timestamp.
        if use_lsp3:
            ts_outcomes = [
                t["Outcome"]
                for t in all_trades
                if t["Timestamp"] == ts
            ]

            for side in ("LONG", "SHORT"):
                side_outcomes = [
                    t["Outcome"]
                    for t in all_trades
                    if t["Timestamp"] == ts and t["Side"] == side
                ]

                if not side_outcomes:
                    continue

                diagnostics["lsp3_event_loss_updates"] += int("LOSS" in side_outcomes)

                if "WIN" in side_outcomes:
                    direction_loss_streak[side] = 0
                    direction_cooldown[side] = 0
                    diagnostics["lsp3_win_reset_events"] += 1
                elif "LOSS" in side_outcomes:
                    direction_loss_streak[side] += 1
                    diagnostics[f"lsp3_{side.lower()}_loss_events"] += 1

                    if direction_loss_streak[side] >= LSP3_TRIGGER_STREAK:
                        direction_cooldown[side] = LSP3_COOLDOWN_CANDLES
                        diagnostics["lsp3_trigger_events"] += 1
                        diagnostics[f"lsp3_{side.lower()}_trigger_events"] += 1
                        diagnostics["lsp3_blocked_candles_scheduled"] += LSP3_COOLDOWN_CANDLES
                        lsp3_trigger_log.append({
                            "Timestamp": ts,
                            "Side": side,
                            "LossEventStreak": direction_loss_streak[side],
                            "CooldownCandles": LSP3_COOLDOWN_CANDLES,
                        })

        # --------------------------------------------------------
        # LOSS FIREWALL — UPDATE ONLY FROM COMPLETED EXIT EVENTS
        # --------------------------------------------------------
        if use_firewall:
            for side in ("LONG", "SHORT"):
                side_outcomes = [
                    tr["Outcome"] for tr in all_trades
                    if tr["Timestamp"] == ts and tr["Side"] == side
                ]
                if side_outcomes:
                    # One timestamp = one loss event. A WIN resets the side.
                    if "WIN" in side_outcomes:
                        firewall_loss_events[side] = 0
                        firewall_locked[side] = False
                    elif "LOSS" in side_outcomes:
                        firewall_loss_events[side] += 1
                        if firewall_loss_events[side] >= 2:
                            firewall_locked[side] = True

                # Unlock only after the side has become flat.
                if firewall_locked[side]:
                    still_open = any(
                        p["side"] == side for p in active_positions.values()
                    )
                    if not still_open:
                        firewall_locked[side] = False
                        firewall_loss_events[side] = 0

        # --------------------------------------------------------
        # 2. EXACT V74 MARKET REGIME
        # --------------------------------------------------------
        market_bull = True

        if (
            "BTC"
            in processed_data
            and ts
            in processed_data[
                "BTC"
            ].index
        ):
            btc_c = processed_data[
                "BTC"
            ].loc[ts]

            market_bull = (
                btc_c["Close"]
                > btc_c["EMA200"]
            )

        bullish_count = 0
        total_active_syms = 0

        for symbol, df in processed_data.items():
            if ts not in df.index:
                continue

            total_active_syms += 1

            if (
                df.loc[
                    ts,
                    "Close",
                ]
                > df.loc[
                    ts,
                    "EMA200",
                ]
            ):
                bullish_count += 1

        market_breadth_ratio = (
            bullish_count
            / total_active_syms
            if total_active_syms > 0
            else 0.5
        )

        allow_longs = (
            market_breadth_ratio
            >= 0.35
        )

        allow_shorts = (
            market_breadth_ratio
            <= 0.65
        )

        # --------------------------------------------------------
        # 3. EXACT V74 VALID CANDIDATES
        # --------------------------------------------------------
        candidates = build_valid_candidates(
            processed_data,
            ts,
            active_positions,
            market_bull,
            allow_longs,
            allow_shorts,
        )

        diagnostics[
            "valid_candidates"
        ] += len(candidates)

        # --------------------------------------------------------
        # CAUSAL LATE-SHORT / OVEREXTENSION TEST
        # Only after the exact V74 candidate is valid.
        # No signal/ranking/SL/trailing parameter is changed.
        # --------------------------------------------------------
        if gate_mode and candidates:
            btc = processed_data.get("BTC")
            btc_b2_atr = np.nan
            if btc is not None and ts in btc.index:
                bi = btc.index.get_loc(ts)
                if bi >= 2:
                    bc = btc.iloc[bi]
                    b2 = float(bc["Close"] / btc.iloc[bi-2]["Close"] - 1.0)
                    b_atr_pct = float(bc["ATR"] / bc["Close"])
                    btc_b2_atr = b2 / b_atr_pct if b_atr_pct else np.nan

            if gate_mode == "A":
                breadth_limit, ema50_limit, min_open = 0.10, -1.80, 3
            elif gate_mode == "B":
                breadth_limit, ema50_limit, min_open = 0.15, -1.80, 3
            elif gate_mode == "C":
                breadth_limit, ema50_limit, min_open = 0.20, -1.50, 3
            else:
                breadth_limit, ema50_limit, min_open = 0.15, -1.80, 3

            kept=[]
            for cand in candidates:
                block=False
                if cand["side"] == "SHORT":
                    d=processed_data[cand["symbol"]]
                    ci=d.index.get_loc(ts)
                    cc=d.iloc[ci]
                    ema50_atr=float((cc["Close"]-cc["EMA50"])/cc["ATR"]) if cc["ATR"] else np.nan
                    open_short=sum(v["side"]=="SHORT" for v in active_positions.values())
                    # Evidence-driven: extreme bearish breadth + late/extended short + crowded existing shorts.
                    if (market_breadth_ratio <= breadth_limit and
                        ema50_atr <= ema50_limit and
                        open_short >= min_open):
                        block=True
                if block:
                    diagnostics["gate_blocked"] += 1
                    diagnostics[f"gate_blocked_{gate_mode}"] += 1
                else:
                    kept.append(cand)
            candidates=kept

        if not candidates:
            if use_lsp3:
                for _side in ("LONG", "SHORT"):
                    if direction_cooldown[_side] > 0:
                        direction_cooldown[_side] -= 1
            continue

        # --------------------------------------------------------
        # LSP3: HARD ENTRY CIRCUIT BREAKER ONLY
        # --------------------------------------------------------
        # V74 candidates are built exactly as before. We only suppress
        # NEW entries in a direction whose own loss streak is active.
        if use_lsp3:
            original_candidate_count = len(candidates)
            blocked_sides = {
                side for side in ("LONG", "SHORT")
                if direction_cooldown[side] > 0
            }

            if blocked_sides:
                had_blocked_long = any(c["side"] == "LONG" for c in candidates) and "LONG" in blocked_sides
                had_blocked_short = any(c["side"] == "SHORT" for c in candidates) and "SHORT" in blocked_sides
                kept_candidates = []
                blocked_count = 0
                for c in candidates:
                    if c["side"] in blocked_sides:
                        blocked_count += 1
                        diagnostics["lsp3_blocked_candidates"] += 1
                        diagnostics[f"lsp3_blocked_{c['side'].lower()}_candidates"] += 1
                    else:
                        kept_candidates.append(c)
                candidates = kept_candidates
                if blocked_count > 0:
                    diagnostics["lsp3_active_events"] += 1
                    diagnostics["lsp3_blocked_entry_events"] += 1
                    if had_blocked_long:
                        diagnostics["lsp3_blocked_long_entry_events"] += 1
                    if had_blocked_short:
                        diagnostics["lsp3_blocked_short_entry_events"] += 1

            if not candidates:
                diagnostics["lsp3_no_entry_events"] += 1
                # This timestamp is still one cooldown candle.
                if use_lsp3:
                    for _side in ("LONG", "SHORT"):
                        if direction_cooldown[_side] > 0:
                            direction_cooldown[_side] -= 1
                continue

        # --------------------------------------------------------
        # LOSS FIREWALL — NEW ENTRIES ONLY
        # --------------------------------------------------------
        if use_firewall and candidates:
            kept = []
            for c in candidates:
                if firewall_locked.get(c["side"], False):
                    diagnostics["firewall_blocked"] += 1
                    diagnostics[f"firewall_blocked_{c['side'].lower()}"] += 1
                else:
                    kept.append(c)
            candidates = kept

        slots = (
            MAX_POSITIONS
            - len(active_positions)
        )

        if slots <= 0:
            diagnostics[
                "portfolio_full"
            ] += 1
            if use_lsp3:
                for _side in ("LONG", "SHORT"):
                    if direction_cooldown[_side] > 0:
                        direction_cooldown[_side] -= 1
            continue

        # --------------------------------------------------------
        # 4. SELECT
        #
        # BASELINE:
        #   first candidates exactly as V74 ranking gives them.
        #
        # LSP:
        #   same candidates, but soft penalty can reorder them.
        # --------------------------------------------------------
        if use_lsp:
            selected = select_candidates_lsp(
                candidates,
                slots,
                symbol_side_losses,
                global_loss_streak,
            )

            diagnostics[
                "lsp_selection_events"
            ] += 1

            if (
                global_loss_streak
                >= LSP_ACTIVATE_AT
            ):
                diagnostics[
                    "lsp_active_events"
                ] += 1

        else:
            selected = candidates[
                :slots
            ]

        diagnostics[
            "selected_entries"
        ] += len(selected)

        # --------------------------------------------------------
        # 5. EXACT V74 ENTRY
        # --------------------------------------------------------
        for candidate in selected:
            symbol = candidate[
                "symbol"
            ]

            side = candidate[
                "side"
            ]

            active_positions[
                symbol
            ] = {
                "side": side,
                "entry_price": candidate[
                    "entry_price"
                ],
                # Diagnostic-only fields; do not affect V74 execution.
                "initial_stop": candidate[
                    "initial_sl"
                ],
                "entry_timestamp": ts,
                "stop_loss": candidate[
                    "initial_sl"
                ],
                "highest_price": candidate[
                    "entry_price"
                ],
                "lowest_price": candidate[
                    "entry_price"
                ],
                "initial_risk": candidate[
                    "initial_risk"
                ],
                "entry_index": candidate[
                    "entry_index"
                ],
            }

        # --------------------------------------------------------
        # 6. Simple mark-to-market curve.
        # Does not affect trading logic.
        # --------------------------------------------------------
        closed_pnl = sum(
            trade["Dollar_PnL"]
            for trade in all_trades
        )

        unrealized = 0.0

        for pos_symbol, pos in (
            active_positions.items()
        ):
            df = processed_data[
                pos_symbol
            ]

            if ts not in df.index:
                continue

            close_price = df.loc[
                ts,
                "Close",
            ]

            if pos["side"] == "LONG":
                unrealized += (
                    close_price
                    - pos["entry_price"]
                ) * (
                    TRADE_MARGIN
                    * LEVERAGE
                    / pos["entry_price"]
                )

            else:
                unrealized += (
                    pos["entry_price"]
                    - close_price
                ) * (
                    TRADE_MARGIN
                    * LEVERAGE
                    / pos["entry_price"]
                )

        equity_curve.append(
            {
                "Timestamp": ts,
                "Equity": (
                    INITIAL_CAPITAL
                    + closed_pnl
                    + unrealized
                ),
            }
        )

    if use_lsp3:
        diagnostics["lsp3_trigger_log"] = lsp3_trigger_log

    return (
        pd.DataFrame(all_trades),
        pd.DataFrame(equity_curve),
        diagnostics,
    )



# ============================================================
# LOSS-STREAK FORENSICS
# ============================================================

def loss_streak_forensics(trades_df, top_n=20):
    """
    Finds every consecutive-loss sequence in exit order and prints
    the exact trades inside the largest sequences.

    This is diagnostic only. It does NOT change the backtest.
    """

    if trades_df.empty:
        print("\nNo trades available for loss-streak forensics.")
        return pd.DataFrame()

    df = trades_df.sort_values(
        ["Timestamp", "ExitOrder"],
        kind="stable",
    ).reset_index(drop=True).copy()

    sequences = []
    current = []

    for _, row in df.iterrows():
        if row["Outcome"] == "LOSS":
            current.append(row)
        else:
            if current:
                sequences.append(current)
                current = []

    if current:
        sequences.append(current)

    records = []

    for seq_id, seq in enumerate(
        sequences,
        start=1,
    ):
        if not seq:
            continue

        start = seq[0]
        end = seq[-1]

        records.append(
            {
                "Sequence_ID": seq_id,
                "Length": len(seq),
                "Start": start["Timestamp"],
                "End": end["Timestamp"],
                "Total_PnL": sum(
                    float(x["Dollar_PnL"])
                    for x in seq
                ),
                "Symbols": ",".join(
                    str(x["Symbol"])
                    for x in seq
                ),
                "Sides": ",".join(
                    str(x["Side"])
                    for x in seq
                ),
            }
        )

    seq_df = pd.DataFrame(records)

    if seq_df.empty:
        print("\nNo loss streaks found.")
        return seq_df

    seq_df = seq_df.sort_values(
        ["Length", "Start"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)

    print("\n" + "=" * 68)
    print("LOSS-STREAK FORENSICS — BASELINE")
    print("=" * 68)

    print(
        f"Total loss sequences: {len(seq_df)}"
    )
    print(
        f"Maximum loss streak: "
        f"{int(seq_df.iloc[0]['Length'])}"
    )

    print("\nTOP LOSS SEQUENCES")

    for _, r in seq_df.head(top_n).iterrows():
        print(
            f"#{int(r['Sequence_ID']):3d} | "
            f"Length={int(r['Length']):2d} | "
            f"PnL=${float(r['Total_PnL']):,.2f} | "
            f"{r['Start']} -> {r['End']}"
        )
        print(
            f"    Symbols: {r['Symbols']}"
        )
        print(
            f"    Sides:   {r['Sides']}"
        )

    # Detailed trades for the single largest sequence.
    longest_id = int(
        seq_df.iloc[0]["Sequence_ID"]
    )

    current = []
    sequence_counter = 0

    for _, row in df.iterrows():
        if row["Outcome"] == "LOSS":
            current.append(row)
        else:
            if current:
                sequence_counter += 1

                if (
                    sequence_counter
                    == longest_id
                ):
                    break

                current = []

    if not current:
        current = [
            row
            for _, row in df.iterrows()
            if row["Outcome"] == "LOSS"
        ]

    detail_rows = []

    for n, row in enumerate(
        current,
        start=1,
    ):
        detail_rows.append(
            {
                "Streak_Position": n,
                "Timestamp": row["Timestamp"],
                "Symbol": row["Symbol"],
                "Side": row["Side"],
                "Return_R": float(
                    row["Return"]
                ),
                "Dollar_PnL": float(
                    row["Dollar_PnL"]
                ),
            }
        )

    detail_df = pd.DataFrame(
        detail_rows
    )

    print("\nLONGEST STREAK — TRADE BY TRADE")

    if detail_df.empty:
        print("No detail available.")
    else:
        for _, r in detail_df.iterrows():
            print(
                f"{int(r['Streak_Position']):2d}. "
                f"{r['Timestamp']} | "
                f"{str(r['Symbol']):8s} | "
                f"{str(r['Side']):5s} | "
                f"R={float(r['Return_R']):7.3f} | "
                f"PnL=${float(r['Dollar_PnL']):9,.2f}"
            )

    # Useful cluster statistics.
    symbol_counter = Counter(
        x["Symbol"]
        for x in current
    )

    side_counter = Counter(
        x["Side"]
        for x in current
    )

    print("\nLONGEST STREAK COMPOSITION")

    print(
        "Symbols: "
        + ", ".join(
            f"{k}={v}"
            for k, v in symbol_counter.most_common()
        )
    )

    print(
        "Sides: "
        + ", ".join(
            f"{k}={v}"
            for k, v in side_counter.most_common()
        )
    )

    # Save forensic CSV.
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    seq_path = os.path.join(
        OUTPUT_DIR,
        "baseline_loss_streak_sequences.csv",
    )

    detail_path = os.path.join(
        OUTPUT_DIR,
        "baseline_longest_loss_streak.csv",
    )

    seq_df.to_csv(
        seq_path,
        index=False,
    )

    detail_df.to_csv(
        detail_path,
        index=False,
    )

    print(
        f"\nSaved: {seq_path}"
    )
    print(
        f"Saved: {detail_path}"
    )

    return seq_df


# ============================================================
# V74-LSP2 DESIGN
# ============================================================

def build_lsp2_priority(
    candidates,
    symbol_side_losses,
    global_loss_streak,
):
    """
    LSP2 is intentionally diagnostic-first.

    Unlike V74-LSP1, it does NOT use a numeric penalty that can
    accidentally overpower V74's Momentum ranking.

    It performs a very narrow priority swap ONLY when:
      1) the portfolio has a live global loss streak >= 2, and
      2) two candidates are competing for the same scarce slot, and
      3) one candidate has a recent same-symbol+side loss streak.

    The original V74 ordering remains the default.

    No signal is rejected.
    """

    if not candidates:
        return []

    ranked = list(candidates)

    if global_loss_streak < 2:
        return ranked

    # Partition into candidates that are currently carrying a
    # same-symbol+side loss streak and those that are not.
    clean = []
    clustered = []

    for idx, candidate in enumerate(ranked):
        streak = symbol_side_losses.get(
            (
                candidate["symbol"],
                candidate["side"],
            ),
            0,
        )

        item = (
            idx,
            candidate,
            streak,
        )

        if streak >= 2:
            clustered.append(item)
        else:
            clean.append(item)

    # Only move a clustered candidate behind a clean candidate
    # if the clean candidate was already very close in original
    # V74 rank. This prevents LSP2 from rewriting the whole
    # ranking and protects the Golden Core.
    #
    # "Near" means one adjacent rank only.
    #
    # This is deliberately conservative.
    result = list(ranked)

    for idx, candidate, streak in clustered:
        if idx <= 0:
            continue

        prev_candidate = result[idx - 1]

        prev_streak = symbol_side_losses.get(
            (
                prev_candidate["symbol"],
                prev_candidate["side"],
            ),
            0,
        )

        if prev_streak < 2:
            result[idx - 1], result[idx] = (
                result[idx],
                result[idx - 1],
            )

    return result


def run_lsp2(
    processed_data,
):
    """
    Runs V74 with the conservative LSP2 priority rule.

    This function is intentionally separate from run_backtest()
    so the original baseline and V74-LSP1 remain untouched.
    """

    all_timestamps = get_all_timestamps(
        processed_data
    )

    active_positions = {}
    all_trades = []

    symbol_side_losses = defaultdict(int)
    global_loss_streak = 0

    diagnostics = Counter()

    equity_curve = []

    for ts in all_timestamps:

        symbols_to_close = []

        # EXACT V74 position management.
        for symbol, pos in list(
            active_positions.items()
        ):
            df = processed_data[symbol]

            if ts not in df.index:
                continue

            c4h = df.loc[ts]

            if pos["side"] == "LONG":
                if c4h["High"] > pos["highest_price"]:
                    pos["highest_price"] = c4h["High"]

                    new_trailing_sl = (
                        pos["highest_price"]
                        - TRAILING_ATR_MULTIPLIER
                        * c4h["ATR"]
                    )

                    if new_trailing_sl > pos["stop_loss"]:
                        pos["stop_loss"] = new_trailing_sl

                hit_sl = (
                    c4h["Low"]
                    <= pos["stop_loss"]
                )

            else:
                if c4h["Low"] < pos["lowest_price"]:
                    pos["lowest_price"] = c4h["Low"]

                    new_trailing_sl = (
                        pos["lowest_price"]
                        + TRAILING_ATR_MULTIPLIER
                        * c4h["ATR"]
                    )

                    if new_trailing_sl < pos["stop_loss"]:
                        pos["stop_loss"] = new_trailing_sl

                hit_sl = (
                    c4h["High"]
                    >= pos["stop_loss"]
                )

            curr_i = df.index.get_loc(ts)
            candles_held = (
                curr_i
                - pos["entry_index"]
            )

            is_timeout = (
                candles_held
                >= TIMEOUT_CANDLES
            )

            if not (
                hit_sl
                or is_timeout
            ):
                continue

            initial_risk = pos["initial_risk"]

            if pos["side"] == "LONG":
                exit_p = (
                    min(
                        pos["stop_loss"],
                        c4h["Open"],
                    )
                    if hit_sl
                    else c4h["Close"]
                )

                r_real = (
                    (
                        exit_p
                        - pos["entry_price"]
                    )
                    / initial_risk
                ) - (
                    FEE_RATE * 2
                )

                price_return_pct = (
                    exit_p
                    - pos["entry_price"]
                ) / pos["entry_price"]

            else:
                exit_p = (
                    max(
                        pos["stop_loss"],
                        c4h["Open"],
                    )
                    if hit_sl
                    else c4h["Close"]
                )

                r_real = (
                    (
                        pos["entry_price"]
                        - exit_p
                    )
                    / initial_risk
                ) - (
                    FEE_RATE * 2
                )

                price_return_pct = (
                    pos["entry_price"]
                    - exit_p
                ) / pos["entry_price"]

            outcome = (
                "WIN"
                if r_real > 0
                else "LOSS"
            )

            position_notional = (
                TRADE_MARGIN
                * LEVERAGE
            )

            dollar_pnl = (
                position_notional
                * price_return_pct
            ) - (
                position_notional
                * FEE_RATE
                * 2
            )

            if outcome == "LOSS":
                global_loss_streak += 1

                symbol_side_losses[
                    (
                        symbol,
                        pos["side"],
                    )
                ] += 1

            else:
                global_loss_streak = 0

                symbol_side_losses[
                    (
                        symbol,
                        pos["side"],
                    )
                ] = 0

            all_trades.append(
                {
                    "Timestamp": ts,
                    "Symbol": symbol,
                    "Side": pos["side"],
                    "Outcome": outcome,
                    "Return": r_real,
                    "Dollar_PnL": dollar_pnl,
                    "ExitOrder": len(all_trades),
                    "LSP_Active": 1,
                    "LSP2_Global_Streak": global_loss_streak,
                }
            )

            symbols_to_close.append(symbol)

        for sym in symbols_to_close:
            del active_positions[sym]

        # EXACT V74 regime/breadth.
        market_bull = True

        if (
            "BTC" in processed_data
            and ts in processed_data["BTC"].index
        ):
            btc_c = processed_data["BTC"].loc[ts]
            market_bull = (
                btc_c["Close"]
                > btc_c["EMA200"]
            )

        bullish_count = 0
        total_active_syms = 0

        for symbol, df in processed_data.items():
            if ts not in df.index:
                continue

            total_active_syms += 1

            if (
                df.loc[ts, "Close"]
                > df.loc[ts, "EMA200"]
            ):
                bullish_count += 1

        market_breadth_ratio = (
            bullish_count / total_active_syms
            if total_active_syms > 0
            else 0.5
        )

        allow_longs = (
            market_breadth_ratio >= 0.35
        )

        allow_shorts = (
            market_breadth_ratio <= 0.65
        )

        candidates = build_valid_candidates(
            processed_data,
            ts,
            active_positions,
            market_bull,
            allow_longs,
            allow_shorts,
        )

        diagnostics[
            "valid_candidates"
        ] += len(candidates)

        if not candidates:
            continue

        slots = (
            MAX_POSITIONS
            - len(active_positions)
        )

        if slots <= 0:
            diagnostics[
                "portfolio_full"
            ] += 1
            if use_lsp3:
                for _side in ("LONG", "SHORT"):
                    if direction_cooldown[_side] > 0:
                        direction_cooldown[_side] -= 1
            continue

        ranked = build_lsp2_priority(
            candidates,
            symbol_side_losses,
            global_loss_streak,
        )

        selected = ranked[:slots]

        diagnostics[
            "selected_entries"
        ] += len(selected)

        if global_loss_streak >= 2:
            diagnostics[
                "lsp2_active_events"
            ] += 1

        for candidate in selected:
            symbol = candidate["symbol"]
            side = candidate["side"]

            active_positions[symbol] = {
                "side": side,
                "entry_price": candidate[
                    "entry_price"
                ],
                "initial_stop": candidate[
                    "initial_sl"
                ],
                "entry_timestamp": ts,
                "stop_loss": candidate[
                    "initial_sl"
                ],
                "highest_price": candidate[
                    "entry_price"
                ],
                "lowest_price": candidate[
                    "entry_price"
                ],
                "initial_risk": candidate[
                    "initial_risk"
                ],
                "entry_index": candidate[
                    "entry_index"
                ],
            }

        closed_pnl = sum(
            trade["Dollar_PnL"]
            for trade in all_trades
        )

        unrealized = 0.0

        for pos_symbol, pos in active_positions.items():
            df = processed_data[pos_symbol]

            if ts not in df.index:
                continue

            close_price = df.loc[
                ts,
                "Close",
            ]

            if pos["side"] == "LONG":
                unrealized += (
                    close_price
                    - pos["entry_price"]
                ) * (
                    TRADE_MARGIN
                    * LEVERAGE
                    / pos["entry_price"]
                )
            else:
                unrealized += (
                    pos["entry_price"]
                    - close_price
                ) * (
                    TRADE_MARGIN
                    * LEVERAGE
                    / pos["entry_price"]
                )

        equity_curve.append(
            {
                "Timestamp": ts,
                "Equity": (
                    INITIAL_CAPITAL
                    + closed_pnl
                    + unrealized
                ),
            }
        )

    return (
        pd.DataFrame(all_trades),
        pd.DataFrame(equity_curve),
        diagnostics,
    )


# ============================================================
# REPORTING
# ============================================================

def calculate_loss_streaks(
    trades_df
):
    if trades_df.empty:
        return 0, []

    ordered = trades_df.sort_values(
        [
            "Timestamp",
            "ExitOrder",
        ],
        kind="stable",
    )

    current = 0
    maximum = 0
    sequences = []

    for outcome in ordered[
        "Outcome"
    ]:
        if outcome == "LOSS":
            current += 1
            maximum = max(
                maximum,
                current,
            )
        else:
            if current > 0:
                sequences.append(
                    current
                )
            current = 0

    if current > 0:
        sequences.append(
            current
        )

    return (
        maximum,
        sequences,
    )


def calculate_drawdown(
    equity_df
):
    if equity_df.empty:
        return 0.0, 0.0

    equity = equity_df[
        "Equity"
    ].astype(float)

    peak = equity.cummax()
    dd = equity - peak

    max_dd = float(
        dd.min()
    )

    if max_dd >= 0:
        return 0.0, 0.0

    peak_at_trough = float(
        peak.loc[
            dd.idxmin()
        ]
    )

    max_dd_pct = (
        max_dd
        / peak_at_trough
        * 100.0
        if peak_at_trough > 0
        else 0.0
    )

    return (
        max_dd,
        max_dd_pct,
    )


def report(
    name,
    trades_df,
    equity_df,
    diagnostics,
):
    print("\n" + "=" * 68)
    print(name)
    print("=" * 68)

    if trades_df.empty:
        print("No trades.")
        return {}

    trades_df = trades_df.sort_values(
        [
            "Timestamp",
            "ExitOrder",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    total = len(
        trades_df
    )

    wins = int(
        (
            trades_df["Outcome"]
            == "WIN"
        ).sum()
    )

    losses = total - wins

    wr = (
        wins
        / total
        * 100.0
    )

    net_r = float(
        trades_df[
            "Return"
        ].sum()
    )

    pnl = float(
        trades_df[
            "Dollar_PnL"
        ].sum()
    )

    final_capital = (
        INITIAL_CAPITAL
        + pnl
    )

    return_pct = (
        final_capital
        / INITIAL_CAPITAL
        - 1.0
    ) * 100.0

    max_ls, sequences = (
        calculate_loss_streaks(
            trades_df
        )
    )

    dd_dollar, dd_pct = (
        calculate_drawdown(
            equity_df
        )
    )

    print(
        f"Trades        : {total}"
    )
    print(
        f"Wins          : {wins}"
    )
    print(
        f"Losses        : {losses}"
    )
    print(
        f"Win Rate      : {wr:.2f}%"
    )
    print(
        f"Net R         : {net_r:.2f}R"
    )
    print(
        f"Net PnL       : ${pnl:,.2f}"
    )
    print(
        f"Final Capital : ${final_capital:,.2f}"
    )
    print(
        f"Return        : {return_pct:.2f}%"
    )
    print(
        f"Max Drawdown  : "
        f"${dd_dollar:,.2f} "
        f"({dd_pct:.2f}%)"
    )
    print(
        f"Max Loss Streak: {max_ls}"
    )

    print("\nDIRECTION")

    for side in [
        "LONG",
        "SHORT",
    ]:
        sdf = trades_df[
            trades_df["Side"]
            == side
        ]

        side_total = len(sdf)

        side_wins = int(
            (
                sdf["Outcome"]
                == "WIN"
            ).sum()
        )

        side_wr = (
            side_wins
            / side_total
            * 100.0
            if side_total
            else 0.0
        )

        side_pnl = float(
            sdf[
                "Dollar_PnL"
            ].sum()
        )

        print(
            f"{side:5s} | "
            f"Trades={side_total:4d} | "
            f"WR={side_wr:6.2f}% | "
            f"PnL=${side_pnl:10,.2f}"
        )

    print("\nLOSS STREAK DISTRIBUTION")

    distribution = Counter(
        sequences
    )

    if distribution:
        for length in sorted(
            distribution
        ):
            print(
                f"{length:2d} loss: "
                f"{distribution[length]} times"
            )
    else:
        print(
            "No loss streaks."
        )

    print("\nPER SYMBOL")

    for symbol in SYMBOLS:
        sdf = trades_df[
            trades_df["Symbol"]
            == symbol
        ]

        if sdf.empty:
            continue

        symbol_total = len(sdf)

        symbol_wins = int(
            (
                sdf["Outcome"]
                == "WIN"
            ).sum()
        )

        symbol_wr = (
            symbol_wins
            / symbol_total
            * 100.0
        )

        symbol_pnl = float(
            sdf[
                "Dollar_PnL"
            ].sum()
        )

        symbol_max_ls, _ = (
            calculate_loss_streaks(
                sdf
            )
        )

        print(
            f"{symbol:8s} | "
            f"Trades={symbol_total:3d} | "
            f"WR={symbol_wr:6.2f}% | "
            f"PnL=${symbol_pnl:10,.2f} | "
            f"MaxLS={symbol_max_ls}"
        )

    print("\nDIAGNOSTICS")

    for key in sorted(
        diagnostics
    ):
        print(
            f"{key:28s}: "
            f"{diagnostics[key]}"
        )

    return {
        "trades": total,
        "wins": wins,
        "losses": losses,
        "wr": wr,
        "net_r": net_r,
        "pnl": pnl,
        "final_capital": final_capital,
        "return_pct": return_pct,
        "max_dd": dd_dollar,
        "max_dd_pct": dd_pct,
        "max_loss_streak": max_ls,
    }


def save_csvs(
    baseline_trades,
    baseline_equity,
    lsp_trades,
    lsp_equity,
):
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    if not baseline_trades.empty:
        baseline_trades.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "baseline_v74_trades.csv",
            ),
            index=False,
        )

    if not lsp_trades.empty:
        lsp_trades.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "v74_lsp_trades.csv",
            ),
            index=False,
        )

    if not baseline_equity.empty:
        baseline_equity.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "baseline_v74_equity.csv",
            ),
            index=False,
        )

    if not lsp_equity.empty:
        lsp_equity.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "v74_lsp_equity.csv",
            ),
            index=False,
        )

    print(
        f"\nCSV output: "
        f"{OUTPUT_DIR}/"
    )


def print_comparison(
    baseline,
    lsp,
):
    print("\n" + "=" * 68)
    print("V74 BASELINE vs V74-LSP")
    print("=" * 68)

    if not baseline or not lsp:
        return

    print(
        f"{'Metric':24s}"
        f"{'V74':>16s}"
        f"{'V74-LSP':>16s}"
        f"{'Delta':>16s}"
    )

    print("-" * 72)

    rows = [
        (
            "Trades",
            baseline["trades"],
            lsp["trades"],
        ),
        (
            "Win Rate %",
            baseline["wr"],
            lsp["wr"],
        ),
        (
            "Net PnL $",
            baseline["pnl"],
            lsp["pnl"],
        ),
        (
            "Net R",
            baseline["net_r"],
            lsp["net_r"],
        ),
        (
            "Max DD $",
            baseline["max_dd"],
            lsp["max_dd"],
        ),
        (
            "Max DD %",
            baseline["max_dd_pct"],
            lsp["max_dd_pct"],
        ),
        (
            "Max Loss Streak",
            baseline["max_loss_streak"],
            lsp["max_loss_streak"],
        ),
    ]

    for metric, b, v in rows:
        delta = v - b

        if (
            "Rate" in metric
            or "DD %" in metric
        ):
            print(
                f"{metric:24s}"
                f"{b:16.2f}"
                f"{v:16.2f}"
                f"{delta:16.2f}"
            )

        elif (
            "PnL" in metric
            or "DD $" in metric
        ):
            print(
                f"{metric:24s}"
                f"${b:15,.2f}"
                f"${v:15,.2f}"
                f"${delta:15,.2f}"
            )

        else:
            print(
                f"{metric:24s}"
                f"{b:16.2f}"
                f"{v:16.2f}"
                f"{delta:16.2f}"
            )

    print("\nIMPORTANT:")
    print(
        "V74-LSP never rejects a valid V74 signal."
    )
    print(
        "It only changes priority when MAX_POSITIONS slots are scarce."
    )


# ============================================================
# ENHANCED LOSS-STREAK FORENSICS (DIAGNOSTIC ONLY)
# ============================================================

def enhanced_loss_streak_forensics(trades_df):
    """
    Diagnostic only. The V74 trading logic is NOT modified.

    Produces two different streak measurements:
      1) RAW: every losing trade in ExitOrder counts separately.
      2) EVENT: all losses sharing the exact same exit timestamp count as
         one loss event. This detects artificial streak inflation caused by
         several positions closing on the same 4h candle.

    Also prints the exact longest RAW streak trade-by-trade and the timestamp
    clusters around it.
    """
    if trades_df.empty:
        print("\nNo trades available for forensics.")
        return

    df = trades_df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    if "EntryTimestamp" in df.columns:
        df["EntryTimestamp"] = pd.to_datetime(df["EntryTimestamp"])
    df = df.sort_values(["Timestamp", "ExitOrder"], kind="stable").reset_index(drop=True)

    # ---------------- RAW streaks ----------------
    sequences = []
    cur = []
    for _, row in df.iterrows():
        if row["Outcome"] == "LOSS":
            cur.append(row)
        elif cur:
            sequences.append(cur); cur = []
    if cur:
        sequences.append(cur)

    max_raw = max((len(x) for x in sequences), default=0)
    longest = next((x for x in sequences if len(x) == max_raw), [])

    # ---------------- SAME-TIMESTAMP clusters ----------------
    loss_df = df[df["Outcome"] == "LOSS"].copy()
    clusters = (
        loss_df.groupby("Timestamp", sort=True)
        .agg(
            Losses=("Outcome", "size"),
            PnL=("Dollar_PnL", "sum"),
            Symbols=("Symbol", lambda x: ",".join(map(str, x))),
            Sides=("Side", lambda x: ",".join(map(str, x))),
        )
        .reset_index()
    )

    # ---------------- EVENT streaks ----------------
    event_loss_flags = []
    for _, g in df.groupby("Timestamp", sort=True):
        event_loss_flags.append((g["Timestamp"].iloc[0], bool((g["Outcome"] == "LOSS").all())))

    # Important: a timestamp is considered a LOSS EVENT only when ALL trades
    # exiting at that timestamp are losses. A mixed timestamp breaks the raw
    # sequence but is not itself a pure-loss event.
    event_streaks = []
    cur_events = []
    for ts, all_loss in event_loss_flags:
        if all_loss:
            cur_events.append(ts)
        else:
            if cur_events:
                event_streaks.append(cur_events); cur_events = []
    if cur_events:
        event_streaks.append(cur_events)
    max_event = max((len(x) for x in event_streaks), default=0)

    print("\n" + "=" * 76)
    print("HUNTER-V74 — LOSS-STREAK FORENSICS (V74 CORE UNCHANGED)")
    print("=" * 76)
    print(f"Total trades                 : {len(df)}")
    print(f"Raw maximum loss streak     : {max_raw}")
    print(f"Pure-loss event max streak  : {max_event}")
    print(f"Max losses on one 4h candle : {int(clusters['Losses'].max()) if not clusters.empty else 0}")

    if not clusters.empty:
        print("\nLOSS TIMESTAMP CLUSTERS — TOP 20")
        top = clusters.sort_values(["Losses", "Timestamp"], ascending=[False, True]).head(20)
        for _, r in top.iterrows():
            print(f"{r['Timestamp']} | losses={int(r['Losses']):2d} | PnL=${float(r['PnL']):9,.2f} | {r['Symbols']} | {r['Sides']}")

    print("\nLONGEST RAW STREAK — TRADE BY TRADE")
    if not longest:
        print("No losing streak found.")
    else:
        start_ts = longest[0]["Timestamp"]
        end_ts = longest[-1]["Timestamp"]
        print(f"Length={len(longest)} | {start_ts} -> {end_ts}")
        for n, row in enumerate(longest, 1):
            entry = row.get("EntryTimestamp", pd.NaT)
            held = "?"
            if pd.notna(entry):
                held = str(row["Timestamp"] - entry)
            print(
                f"{n:2d}. EXIT={row['Timestamp']} | ENTRY={entry} | "
                f"{str(row['Symbol']):8s} | {str(row['Side']):5s} | "
                f"R={float(row['Return']):7.3f} | PnL=${float(row['Dollar_PnL']):9,.2f} | Held={held}"
            )

        streak_ts = pd.to_datetime([x["Timestamp"] for x in longest])
        print("\nLONGEST STREAK — SAME-CANDLE ANALYSIS")
        for ts in sorted(set(streak_ts)):
            g = df[(df["Timestamp"] == ts) & (df["Outcome"] == "LOSS")]
            print(f"{ts} | {len(g)} simultaneous loss(es) | {', '.join(g['Symbol'].astype(str))}")

        print("\nLONGEST STREAK COMPOSITION")
        print("Symbols:", ", ".join(f"{k}={v}" for k, v in Counter(x["Symbol"] for x in longest).most_common()))
        print("Sides  :", ", ".join(f"{k}={v}" for k, v in Counter(x["Side"] for x in longest).most_common()))

        if "EntryTimestamp" in df.columns:
            entries = pd.to_datetime([x.get("EntryTimestamp") for x in longest])
            valid_entries = [x for x in entries if pd.notna(x)]
            if valid_entries:
                print(f"Entry window: {min(valid_entries)} -> {max(valid_entries)}")

    # Save machine-readable outputs for GitHub Actions artifacts.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, "v74_baseline_trades_forensics.csv"), index=False)
    clusters.to_csv(os.path.join(OUTPUT_DIR, "v74_loss_timestamp_clusters.csv"), index=False)
    pd.DataFrame([{
        "raw_max_loss_streak": max_raw,
        "pure_loss_event_max_streak": max_event,
        "max_losses_same_timestamp": int(clusters["Losses"].max()) if not clusters.empty else 0,
        "total_trades": len(df),
    }]).to_csv(os.path.join(OUTPUT_DIR, "v74_loss_streak_diagnostic_summary.csv"), index=False)

    print("\nSaved forensic CSVs under:", OUTPUT_DIR)
    print("IMPORTANT: No V74 entry/exit/filter/ranking parameter was changed by this diagnostic.")


# ============================================================
# MAIN — BASELINE ONLY + FORENSICS
# ============================================================


# ============================================================
# FORENSICS ONLY — GOLDEN V74 UNCHANGED
# ============================================================
FORENSIC_OUT = "hunter_v74_forensics"


def entry_features(ts, symbol, side, trade_index, baseline_trades):
    """Reconstruct only pre-entry information; never changes trading."""
    d = processed_data[symbol]
    i = d.index.get_loc(ts)
    c = d.iloc[i]

    # Reconstruct positions that were open immediately before this entry.
    active = {}
    for j, r in baseline_trades.iterrows():
        et = pd.to_datetime(r["EntryTimestamp"])
        xt = pd.to_datetime(r["Timestamp"])
        if pd.isna(et) or pd.isna(xt):
            continue
        if et <= ts and xt > ts:
            active[str(r["Symbol"])] = {"side": r["Side"]}

    # Exact V74 market regime/breadth.
    market_bull = True
    if "BTC" in processed_data and ts in processed_data["BTC"].index:
        b = processed_data["BTC"].loc[ts]
        market_bull = bool(b["Close"] > b["EMA200"])

    bullish = 0
    total = 0
    for s, df in processed_data.items():
        if ts in df.index:
            total += 1
            bullish += int(df.loc[ts, "Close"] > df.loc[ts, "EMA200"])
    breadth = bullish / total if total else 0.5
    allow_longs = breadth >= 0.35
    allow_shorts = breadth <= 0.65

    cs = build_valid_candidates(
        processed_data, ts, active, market_bull,
        allow_longs, allow_shorts
    )
    same_side = [x for x in cs if x["side"] == side]
    rank = next((n + 1 for n, x in enumerate(cs)
                 if x["symbol"] == symbol and x["side"] == side), np.nan)

    def ret(df, n):
        if i < n:
            return np.nan
        return float(df.iloc[i]["Close"] / df.iloc[i-n]["Close"] - 1.0)

    btc = processed_data.get("BTC")
    if btc is not None and ts in btc.index:
        bi = btc.index.get_loc(ts)
        bc = btc.iloc[bi]
        b1 = float(bc["Close"] / btc.iloc[bi-1]["Close"] - 1) if bi >= 1 else np.nan
        b2 = float(bc["Close"] / btc.iloc[bi-2]["Close"] - 1) if bi >= 2 else np.nan
        b3 = float(bc["Close"] / btc.iloc[bi-3]["Close"] - 1) if bi >= 3 else np.nan
        b_atr_pct = float(bc["ATR"] / bc["Close"]) if bc["Close"] else np.nan
        b1atr = b1 / b_atr_pct if b_atr_pct else np.nan
        b2atr = b2 / b_atr_pct if b_atr_pct else np.nan
        b3atr = b3 / b_atr_pct if b_atr_pct else np.nan
        bd20 = float(bc["Close"] / bc["EMA20"] - 1)
        bd50 = float(bc["Close"] / bc["EMA50"] - 1)
        bd200 = float(bc["Close"] / bc["EMA200"] - 1)
    else:
        b1=b2=b3=b1atr=b2atr=b3atr=bd20=bd50=bd200=np.nan

    atr_pct = float(c["ATR"] / c["Close"]) if c["Close"] else np.nan
    prev_atr_pct = float(d.iloc[i-1]["ATR"] / d.iloc[i-1]["Close"]) if i >= 1 else np.nan
    atr_change = atr_pct / prev_atr_pct - 1 if prev_atr_pct else np.nan

    return {
        "EntryTimestamp": ts,
        "Symbol": symbol,
        "Side": side,
        "Rank": rank,
        "CandidateCountSameSide": len(same_side),
        "OpenPositionsTotal": len(active),
        "OpenPositionsSameSide": sum(v["side"] == side for v in active.values()),
        "BTC_Return_1Bar": b1, "BTC_Return_2Bar": b2, "BTC_Return_3Bar": b3,
        "BTC_Return_1Bar_ATR": b1atr, "BTC_Return_2Bar_ATR": b2atr, "BTC_Return_3Bar_ATR": b3atr,
        "BTC_Distance_EMA20": bd20, "BTC_Distance_EMA50": bd50, "BTC_Distance_EMA200": bd200,
        "Breadth": breadth,
        "Symbol_Return_1Bar": ret(d,1), "Symbol_Return_2Bar": ret(d,2), "Symbol_Return_3Bar": ret(d,3),
        "Symbol_ATR_Percent": atr_pct, "Symbol_ATR_Change": atr_change,
        "Symbol_Distance_EMA20_ATR": float((c["Close"]-c["EMA20"])/c["ATR"]),
        "Symbol_Distance_EMA50_ATR": float((c["Close"]-c["EMA50"])/c["ATR"]),
        "Symbol_Mom_Short": float(c["Mom_Short"]),
        "Symbol_Mom_Long": float(c["Mom_Long"]),
    }


def run_forensics(trades):
    os.makedirs(FORENSIC_OUT, exist_ok=True)
    t = trades.copy()
    t["EntryTimestamp"] = pd.to_datetime(t["EntryTimestamp"])
    t["Timestamp"] = pd.to_datetime(t["Timestamp"])
    t = t.sort_values(["Timestamp", "ExitOrder"], kind="stable").reset_index(drop=True)

    rows=[]
    for n, r in t.iterrows():
        f=entry_features(r["EntryTimestamp"], r["Symbol"], r["Side"], n, t)
        f.update({
            "Outcome": r["Outcome"],
            "Return_R": float(r["Return"]),
            "Dollar_PnL": float(r["Dollar_PnL"]),
            "ExitTimestamp": r["Timestamp"],
        })
        rows.append(f)
    e=pd.DataFrame(rows)
    e.to_csv(f"{FORENSIC_OUT}/v74_entry_diagnostics.csv", index=False)
    t.to_csv(f"{FORENSIC_OUT}/v74_trades.csv", index=False)

    # Exact raw longest losing sequence.
    seq=[]; cur=[]
    for _,r in t.iterrows():
        if r["Outcome"] == "LOSS": cur.append(r)
        elif cur: seq.append(cur); cur=[]
    if cur: seq.append(cur)
    longest=max(seq,key=len) if seq else []
    target_keys={(str(r["EntryTimestamp"]),r["Symbol"],r["Side"]) for r in longest}
    target=e[e.apply(lambda r:(str(r["EntryTimestamp"]),r["Symbol"],r["Side"]) in target_keys,axis=1)]
    losses=e[e["Outcome"]=="LOSS"]
    wins=e[e["Outcome"]=="WIN"]

    features=[
        "Rank","CandidateCountSameSide","OpenPositionsTotal","OpenPositionsSameSide",
        "BTC_Return_1Bar","BTC_Return_2Bar","BTC_Return_3Bar",
        "BTC_Return_1Bar_ATR","BTC_Return_2Bar_ATR","BTC_Return_3Bar_ATR",
        "BTC_Distance_EMA20","BTC_Distance_EMA50","BTC_Distance_EMA200",
        "Breadth","Symbol_Return_1Bar","Symbol_Return_2Bar","Symbol_Return_3Bar",
        "Symbol_ATR_Percent","Symbol_ATR_Change","Symbol_Distance_EMA20_ATR",
        "Symbol_Distance_EMA50_ATR","Symbol_Mom_Short","Symbol_Mom_Long"
    ]
    comp=[]
    for f in features:
        a=pd.to_numeric(target[f],errors="coerce")
        b=pd.to_numeric(losses[f],errors="coerce")
        c=pd.to_numeric(wins[f],errors="coerce")
        comp.append({"Feature":f,"13_Loss_Avg":a.mean(),"All_Loss_Avg":b.mean(),"All_Win_Avg":c.mean()})
    comp=pd.DataFrame(comp)
    comp["AbsGap_13L_vs_Win"]=(comp["13_Loss_Avg"]-comp["All_Win_Avg"]).abs()
    comp=comp.sort_values("AbsGap_13L_vs_Win",ascending=False)
    comp.to_csv(f"{FORENSIC_OUT}/v74_feature_comparison.csv",index=False)

    print(f"V74 | Trades={len(t)} | WR={(t.Outcome.eq('WIN').mean()*100):.2f}% | PnL=${t.Dollar_PnL.sum():,.2f} | MaxLS={len(longest)}")
    print(f"FORENSIC | 13-loss={len(longest)} | CSV={FORENSIC_OUT}/")
    print("TOP FEATURES")
    for _,r in comp.head(10).iterrows():
        print(f"{r['Feature']} | 13L={r['13_Loss_Avg']:.6f} | Loss={r['All_Loss_Avg']:.6f} | Win={r['All_Win_Avg']:.6f}")


if __name__ == "__main__":
    base, _, _ = run_backtest(
        processed_data, use_lsp=False, use_lsp3=False, gate_mode=None, use_firewall=False
    )
    fw, _, fd = run_backtest(
        processed_data, use_lsp=False, use_lsp3=False, gate_mode=None, use_firewall=True
    )

    def stats(df):
        n = len(df)
        wr = (df["Outcome"].eq("WIN").mean() * 100) if n else 0
        pnl = float(df["Dollar_PnL"].sum()) if n else 0
        cur = mx = 0
        for x in df["Outcome"]:
            if x == "LOSS":
                cur += 1
                mx = max(mx, cur)
            else:
                cur = 0
        return n, wr, pnl, mx

    a = stats(base)
    b = stats(fw)

    print("=" * 68)
    print("HUNTER-V74 — LOSS FIREWALL")
    print("=" * 68)
    print(f"V74 | Trades={a[0]} | WR={a[1]:.2f}% | PnL=${a[2]:,.2f} | MaxLS={a[3]}")
    print(f"FW  | Trades={b[0]} | WR={b[1]:.2f}% | PnL=${b[2]:,.2f} | MaxLS={b[3]} | Blocked={int(fd.get('firewall_blocked',0))}")
