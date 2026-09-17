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

# ============================================================
# NEW (this session) — GLOBAL RAW-STREAK BREAKER + SIDE CONCENTRATION CAP
# ============================================================
# Both are OFF by default (backward compatible). They touch ONLY
# portfolio-construction (which/how-many candidates become new
# entries) — never the signal, SL, trailing, or timeout logic.
GLOBAL_BREAKER_TRIGGER = 3      # raw consecutive portfolio-wide losses
GLOBAL_BREAKER_COOLDOWN = 6     # candles (6 x 4h = 24h) fallback cap
MAX_PER_SIDE_DEFAULT = 3        # of the 5 MAX_POSITIONS slots, cap per direction

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
    use_single_loss_crowd=False,
    use_global_breaker=False,
    global_breaker_trigger=GLOBAL_BREAKER_TRIGGER,
    global_breaker_cooldown=GLOBAL_BREAKER_COOLDOWN,
    max_per_side=None,
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

    # NEW: cooldown counter for the global raw-streak breaker.
    # Blocks ALL new entries (both directions) while > 0.
    global_breaker_cooldown_left = 0

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

    # FW4: very narrow preventive portfolio-crowding control.
    # It activates only AFTER one completed loss event in a direction,
    # when >=3 positions of that same direction are already open.
    # It blocks only NEW entries in that direction.
    # Default is False, so the FW223 reference path is unchanged.
    SINGLE_LOSS_CROWD_MIN_OPEN = 3

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

                # NEW: arm/refresh the global breaker as soon as the
                # raw portfolio-wide streak reaches the trigger.
                if (
                    use_global_breaker
                    and global_loss_streak
                    >= global_breaker_trigger
                ):
                    global_breaker_cooldown_left = (
                        global_breaker_cooldown
                    )

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

                # NEW: a win clears the global breaker immediately —
                # no need to wait out the fixed cooldown once the
                # streak is actually broken.
                if use_global_breaker:
                    global_breaker_cooldown_left = 0

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
            if use_global_breaker and global_breaker_cooldown_left > 0:
                global_breaker_cooldown_left -= 1
            continue

        # --------------------------------------------------------
        # NEW: GLOBAL RAW-STREAK BREAKER — blocks ALL new entries
        # (both directions) while active. Existing positions are
        # never touched; only new-entry candidates are suppressed.
        # --------------------------------------------------------
        if use_global_breaker and global_breaker_cooldown_left > 0:
            diagnostics["global_breaker_blocked_candidates"] += len(candidates)
            diagnostics["global_breaker_active_events"] += 1
            candidates = []

        if not candidates:
            if use_lsp3:
                for _side in ("LONG", "SHORT"):
                    if direction_cooldown[_side] > 0:
                        direction_cooldown[_side] -= 1
            if use_global_breaker and global_breaker_cooldown_left > 0:
                global_breaker_cooldown_left -= 1
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
                if use_global_breaker and global_breaker_cooldown_left > 0:
                    global_breaker_cooldown_left -= 1
                continue

        # --------------------------------------------------------
        # FW4 — SINGLE-LOSS CROWD CONTROL (NEW ENTRIES ONLY)
        # --------------------------------------------------------
        # Narrow preventive layer:
        #   1) at least one completed loss event in this direction
        #   2) at least 3 positions already open in this direction
        #   3) only NEW entries are blocked
        # Existing positions and all V74 execution logic remain untouched.
        if use_single_loss_crowd and candidates:
            kept = []
            for c in candidates:
                side = c["side"]
                open_same_side = sum(
                    p["side"] == side for p in active_positions.values()
                )
                crowd_block = (
                    firewall_loss_events.get(side, 0) >= 1
                    and open_same_side >= SINGLE_LOSS_CROWD_MIN_OPEN
                )
                if crowd_block:
                    diagnostics["single_loss_crowd_blocked"] += 1
                    diagnostics[f"single_loss_crowd_blocked_{side.lower()}"] += 1
                else:
                    kept.append(c)
            candidates = kept

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

        # --------------------------------------------------------
        # NEW: MAX-PER-SIDE CONCENTRATION CAP (NEW ENTRIES ONLY)
        # --------------------------------------------------------
        # Caps how many of the MAX_POSITIONS slots can be occupied by
        # one direction at a time (existing + about-to-be-added).
        # Preserves V74 candidate RANK ORDER — it only skips a
        # candidate whose side is already at the cap, letting a
        # lower-ranked opposite-side candidate take the slot instead.
        if max_per_side is not None and candidates:
            running_count = {
                "LONG": sum(p["side"] == "LONG" for p in active_positions.values()),
                "SHORT": sum(p["side"] == "SHORT" for p in active_positions.values()),
            }
            kept = []
            for c in candidates:
                if running_count[c["side"]] < max_per_side:
                    kept.append(c)
                    running_count[c["side"]] += 1
                else:
                    diagnostics["side_cap_blocked"] += 1
                    diagnostics[f"side_cap_blocked_{c['side'].lower()}"] += 1
            candidates = kept

        if not candidates:
            if use_lsp3:
                for _side in ("LONG", "SHORT"):
                    if direction_cooldown[_side] > 0:
                        direction_cooldown[_side] -= 1
            if use_global_breaker and global_breaker_cooldown_left > 0:
                global_breaker_cooldown_left -= 1
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
            if use_global_breaker and global_breaker_cooldown_left > 0:
                global_breaker_cooldown_left -= 1
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

        # decrement cooldowns for the timestamp that just traded
        if use_lsp3:
            for _side in ("LONG", "SHORT"):
                if direction_cooldown[_side] > 0:
                    direction_cooldown[_side] -= 1
        if use_global_breaker and global_breaker_cooldown_left > 0:
            global_breaker_cooldown_left -= 1

    if use_lsp3:
        diagnostics["lsp3_trigger_log"] = lsp3_trigger_log

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


def stats(df):
    n = len(df)
    wr = (df["Outcome"].eq("WIN").mean() * 100) if n else 0
    pnl = float(df["Dollar_PnL"].sum()) if n else 0
    net_r = float(df["Return"].sum()) if n else 0
    cur = mx = 0
    for x in df["Outcome"]:
        if x == "LOSS":
            cur += 1
            mx = max(mx, cur)
        else:
            cur = 0
    return n, wr, pnl, mx, net_r


if __name__ == "__main__":
    base, base_eq, _ = run_backtest(
        processed_data,
        use_lsp=False, use_lsp3=False, gate_mode=None,
        use_firewall=False, use_single_loss_crowd=False,
    )
    fw223, fw223_eq, fd223 = run_backtest(
        processed_data,
        use_lsp=False, use_lsp3=False, gate_mode=None,
        use_firewall=True, use_single_loss_crowd=False,
    )
    # NEW: FW223 + side cap only (no global breaker yet)
    fw223_cap, fw223_cap_eq, fd223_cap = run_backtest(
        processed_data,
        use_lsp=False, use_lsp3=False, gate_mode=None,
        use_firewall=True, use_single_loss_crowd=False,
        max_per_side=MAX_PER_SIDE_DEFAULT,
    )
    # NEW: FW223 + global breaker only (no side cap)
    fw223_gb, fw223_gb_eq, fd223_gb = run_backtest(
        processed_data,
        use_lsp=False, use_lsp3=False, gate_mode=None,
        use_firewall=True, use_single_loss_crowd=False,
        use_global_breaker=True,
    )
    # NEW: FW223 + side cap + global breaker together
    fw223_full, fw223_full_eq, fd223_full = run_backtest(
        processed_data,
        use_lsp=False, use_lsp3=False, gate_mode=None,
        use_firewall=True, use_single_loss_crowd=False,
        max_per_side=MAX_PER_SIDE_DEFAULT,
        use_global_breaker=True,
    )

    a = stats(base)
    b = stats(fw223)
    c = stats(fw223_cap)
    d = stats(fw223_gb)
    e = stats(fw223_full)

    print("=" * 84)
    print("HUNTER-V74 — GLOBAL BREAKER + SIDE-CAP TEST (on top of FW223)")
    print("=" * 84)
    print(f"{'Variant':22s}{'Trades':>8s}{'WR%':>8s}{'PnL$':>14s}{'NetR':>10s}{'MaxLS':>8s}")
    print("-" * 84)
    for name, s in [
        ("V74 (baseline)", a), ("FW223", b), ("FW223+SideCap", c),
        ("FW223+GlobalBreak", d), ("FW223+Both", e),
    ]:
        n, wr, pnl, mx, net_r = s
        print(f"{name:22s}{n:8d}{wr:8.2f}{pnl:14,.2f}{net_r:10.2f}{mx:8d}")

    print("\nBLOCK COUNTS")
    print(f"FW223+SideCap  | side_cap_blocked_long={fd223_cap.get('side_cap_blocked_long',0)} "
          f"side_cap_blocked_short={fd223_cap.get('side_cap_blocked_short',0)}")
    print(f"FW223+GlobalBreak | global_breaker_active_events={fd223_gb.get('global_breaker_active_events',0)} "
          f"global_breaker_blocked_candidates={fd223_gb.get('global_breaker_blocked_candidates',0)}")
    print(f"FW223+Both | side_cap_blocked={fd223_full.get('side_cap_blocked',0)} "
          f"global_breaker_active_events={fd223_full.get('global_breaker_active_events',0)}")

    print("\nREFERENCE CHECK")
    print("Expected FW223 reference: 491 trades | 63.34% WR | $34,733.29 PnL | MaxLS=8")
    print("Goal: MaxLS(FW223+Both) should be clearly lower than 8 while Trades/WR/PnL stay close to FW223.")
