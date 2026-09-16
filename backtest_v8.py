
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
# HUNTER-V74-DUAL-LOSS-REGIME-SHIELD
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

# ============================================================
# SHOCK / ACCELERATION SHIELD
# ============================================================
# This layer is intentionally conservative and only filters NEW entries.
# V74 signal, ranking, SL, trailing, timeout and PnL mechanics remain intact.
# It detects an unusually fast move AGAINST the proposed trade direction
# using only information available at the current 4h candle.
SHOCK_LOOKBACK_CANDLES = 2
SHOCK_1BAR_ATR_MULT = 1.25
SHOCK_2BAR_ATR_MULT = 2.00
SHOCK_RANGE_ATR_MULT = 1.75
SHOCK_BODY_FRACTION = 0.55
SHOCK_REQUIRE_SCORE = 2
SHOCK_BTC_CONFIRM_ATR_MULT = 1.50

# ============================================================
# LOSS-STREAK REGIME SHIELD (NEW TEST)
# ============================================================
# Preventive, direction-aware protection. It activates only after
# 2 same-direction loss events AND evidence that the market is
# accelerating against that direction. It does NOT alter V74 signals,
# ranking, SL, trailing, timeout, leverage, or sizing.
LOSS_REGIME_TRIGGER = 2
LOSS_REGIME_BTC_ADVERSE_ATR = 1.20
LOSS_REGIME_SYMBOL_ADVERSE_ATR = 1.50
LOSS_REGIME_BEAR_BREADTH = 0.35
LOSS_REGIME_BULL_BREADTH = 0.65

# Correlated-loss cluster control: only engages after repeated losses in
# one direction AND when several fresh candidates want the same direction.
# It keeps the best-ranked candidate(s) instead of deleting the V74 signal rules.
CLUSTER_LOSS_TRIGGER = 2
CLUSTER_MIN_SAME_SIDE = 3
CLUSTER_KEEP_AFTER_TRIGGER = 1

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


def detect_shock_against_entry(processed_data, ts, candidate):
    """
    Conservative, causal shock detector.

    Returns (blocked, reason, details). It never looks beyond `ts`.
    A candidate is blocked only when at least SHOCK_REQUIRE_SCORE independent
    signs indicate a fast move against the proposed direction.
    """
    symbol = candidate["symbol"]
    side = candidate["side"]
    df = processed_data[symbol]

    if ts not in df.index:
        return False, "", {}

    i = df.index.get_loc(ts)
    if i < 2:
        return False, "", {}

    c = df.iloc[i]
    p1 = df.iloc[i - 1]
    p2 = df.iloc[i - 2]

    close = float(c["Close"])
    if close <= 0 or not np.isfinite(close):
        return False, "", {}

    atr = float(c["ATR"])
    atr_pct = atr / close if atr > 0 else 0.0
    if atr_pct <= 0 or not np.isfinite(atr_pct):
        return False, "", {}

    r1 = float(c["Close"]) / float(p1["Close"]) - 1.0
    r2 = float(c["Close"]) / float(p2["Close"]) - 1.0
    candle_range = float(c["High"]) - float(c["Low"])
    body = abs(float(c["Close"]) - float(c["Open"]))
    body_fraction = body / candle_range if candle_range > 0 else 0.0

    adverse = (
        (side == "SHORT" and r1 > 0) or
        (side == "LONG" and r1 < 0)
    )
    adverse2 = (
        (side == "SHORT" and r2 > 0) or
        (side == "LONG" and r2 < 0)
    )

    score = 0
    reasons = []

    if adverse and abs(r1) >= SHOCK_1BAR_ATR_MULT * atr_pct:
        score += 1
        reasons.append("1bar_impulse")

    if adverse2 and abs(r2) >= SHOCK_2BAR_ATR_MULT * atr_pct:
        score += 1
        reasons.append("2bar_acceleration")

    if (
        adverse
        and candle_range >= SHOCK_RANGE_ATR_MULT * atr
        and body_fraction >= SHOCK_BODY_FRACTION
    ):
        score += 1
        reasons.append("impulse_candle")

    # BTC confirmation is deliberately normalized by BTC's own ATR so this
    # remains scale-independent across market regimes.
    btc_confirm = False
    btc_r2 = np.nan
    if "BTC" in processed_data and ts in processed_data["BTC"].index:
        bdf = processed_data["BTC"]
        bi = bdf.index.get_loc(ts)
        if bi >= 2:
            bc = bdf.iloc[bi]
            bp2 = bdf.iloc[bi - 2]
            btc_close = float(bc["Close"])
            btc_atr = float(bc["ATR"])
            btc_r2 = btc_close / float(bp2["Close"]) - 1.0
            btc_atr_pct = btc_atr / btc_close if btc_atr > 0 else 0.0
            if btc_atr_pct > 0:
                btc_confirm = (
                    (side == "SHORT" and btc_r2 >= SHOCK_BTC_CONFIRM_ATR_MULT * btc_atr_pct) or
                    (side == "LONG" and btc_r2 <= -SHOCK_BTC_CONFIRM_ATR_MULT * btc_atr_pct)
                )
                if btc_confirm:
                    score += 1
                    reasons.append("btc_confirmation")

    blocked = adverse and score >= SHOCK_REQUIRE_SCORE
    details = {
        "score": score,
        "r1": r1,
        "r2": r2,
        "atr_pct": atr_pct,
        "range_atr": candle_range / atr if atr > 0 else np.nan,
        "body_fraction": body_fraction,
        "btc_r2": btc_r2,
        "reasons": reasons,
    }
    return blocked, "+".join(reasons), details


# ============================================================
# LOSS-STREAK REGIME SHIELD HELPER
# ============================================================
def detect_loss_regime_against_entry(processed_data, ts, candidate,
                                     direction_loss_streak,
                                     market_breadth_ratio):
    """
    Causal entry protection: only uses candles available at `ts`.
    Returns (blocked, reason, details).

    A candidate is blocked only when:
      1) its direction has already suffered >= LOSS_REGIME_TRIGGER
         consecutive loss EVENTS, and
      2) there is fresh acceleration against that direction in BTC
         or the candidate symbol, and
      3) market breadth is simultaneously at the extreme boundary
         that contradicts the proposed direction.

    This is deliberately much narrower than a generic loss cooldown.
    """
    side = candidate["side"]
    if direction_loss_streak.get(side, 0) < LOSS_REGIME_TRIGGER:
        return False, "", {"streak": direction_loss_streak.get(side, 0)}

    symbol = candidate["symbol"]
    df = processed_data.get(symbol)
    if df is None or ts not in df.index:
        return False, "", {"streak": direction_loss_streak.get(side, 0)}

    idx = df.index.get_loc(ts)
    if idx < 2:
        return False, "", {"streak": direction_loss_streak.get(side, 0)}

    row = df.iloc[idx]
    prev2 = df.iloc[idx-2]
    atr = float(row.get("ATR", np.nan))
    if not np.isfinite(atr) or atr <= 0:
        return False, "", {"streak": direction_loss_streak.get(side, 0)}

    sym_r2_abs_atr = abs(float(row["Close"]) - float(prev2["Close"])) / atr
    sym_r2 = (float(row["Close"]) / float(prev2["Close"]) - 1.0) if float(prev2["Close"]) else 0.0

    btc_r2_atr = 0.0
    btc_r2 = 0.0
    btc = processed_data.get("BTC")
    if btc is not None and ts in btc.index:
        bi = btc.index.get_loc(ts)
        if bi >= 2:
            br = btc.iloc[bi]
            bp = btc.iloc[bi-2]
            b_atr = float(br.get("ATR", np.nan))
            if np.isfinite(b_atr) and b_atr > 0 and float(bp["Close"]) != 0:
                btc_r2 = float(br["Close"]) / float(bp["Close"]) - 1.0
                btc_r2_atr = abs(float(br["Close"]) - float(bp["Close"])) / b_atr

    if side == "LONG":
        breadth_contradiction = market_breadth_ratio <= LOSS_REGIME_BEAR_BREADTH
        btc_adverse = btc_r2 <= 0 and btc_r2_atr >= LOSS_REGIME_BTC_ADVERSE_ATR
        symbol_adverse = sym_r2 <= 0 and sym_r2_abs_atr >= LOSS_REGIME_SYMBOL_ADVERSE_ATR
    else:
        breadth_contradiction = market_breadth_ratio >= LOSS_REGIME_BULL_BREADTH
        btc_adverse = btc_r2 >= 0 and btc_r2_atr >= LOSS_REGIME_BTC_ADVERSE_ATR
        symbol_adverse = sym_r2 >= 0 and sym_r2_abs_atr >= LOSS_REGIME_SYMBOL_ADVERSE_ATR

    # Require the broader regime evidence plus either BTC or symbol acceleration.
    blocked = breadth_contradiction and (btc_adverse or symbol_adverse)
    reasons = []
    if breadth_contradiction:
        reasons.append("breadth_contradiction")
    if btc_adverse:
        reasons.append("btc_adverse_acceleration")
    if symbol_adverse:
        reasons.append("symbol_adverse_acceleration")

    return blocked, "+".join(reasons), {
        "streak": direction_loss_streak.get(side, 0),
        "breadth": market_breadth_ratio,
        "symbol_r2": sym_r2,
        "symbol_r2_atr": sym_r2_abs_atr,
        "btc_r2": btc_r2,
        "btc_r2_atr": btc_r2_atr,
    }


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    processed_data,
    use_lsp=False,
    use_lsp3=False,
    use_shock_shield=False,
    long_only=False,
    use_loss_regime_shield=False,
    use_cluster_control=False,
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
    # Separate event-based streak used by the new preventive shield.
    loss_regime_streak = {"LONG": 0, "SHORT": 0}
    loss_regime_block_log = []
    lsp3_trigger_log = []
    shock_block_log = []

    diagnostics = Counter()
    diagnostics["cluster_control_events"] = 0
    diagnostics["cluster_control_blocked_long"] = 0
    diagnostics["cluster_control_blocked_short"] = 0
    diagnostics["cluster_control_block_log"] = []

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
        # NEW SHIELD STATE — DIRECTIONAL LOSS EVENTS
        # --------------------------------------------------------
        # Same timestamp = one event. Any WIN in a direction resets it.
        for side in ("LONG", "SHORT"):
            side_outcomes = [
                t["Outcome"] for t in all_trades
                if t["Timestamp"] == ts and t["Side"] == side
            ]
            if not side_outcomes:
                continue
            if "WIN" in side_outcomes:
                loss_regime_streak[side] = 0
            elif "LOSS" in side_outcomes:
                loss_regime_streak[side] += 1

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

        # LONG-ONLY TEST: keep the V74 candidate generation completely
        # unchanged, then remove SHORT candidates before ranking/entry.
        # No EMA, Momentum, ATR, SL, trailing, timeout, sizing, or LONG
        # signal rule is changed by this switch.
        if long_only:
            short_count = sum(1 for c in candidates if c["side"] == "SHORT")
            diagnostics["long_only_removed_short_candidates"] += short_count
            candidates = [c for c in candidates if c["side"] == "LONG"]

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
        # SHOCK / ACCELERATION SHIELD — NEW ENTRIES ONLY
        # --------------------------------------------------------
        if use_shock_shield and candidates:
            kept_candidates = []
            for c in candidates:
                blocked, reason, details = detect_shock_against_entry(
                    processed_data, ts, c
                )
                if blocked:
                    diagnostics["shock_blocked_candidates"] += 1
                    diagnostics[f"shock_blocked_{c['side'].lower()}_candidates"] += 1
                    diagnostics["shock_block_events"] += 1
                    shock_block_log.append({
                        "Timestamp": ts,
                        "Symbol": c["symbol"],
                        "Side": c["side"],
                        "Score": details.get("score", 0),
                        "Reason": reason,
                        "R1": details.get("r1"),
                        "R2": details.get("r2"),
                        "ATR_Pct": details.get("atr_pct"),
                        "Range_ATR": details.get("range_atr"),
                        "BodyFraction": details.get("body_fraction"),
                        "BTC_R2": details.get("btc_r2"),
                    })
                else:
                    kept_candidates.append(c)
            if len(kept_candidates) != len(candidates):
                diagnostics["shock_active_events"] += 1
            candidates = kept_candidates

        # --------------------------------------------------------
        # LOSS-STREAK REGIME SHIELD — NEW ENTRIES ONLY
        # --------------------------------------------------------
        # This is intentionally preventive and narrow: after repeated
        # losses in one direction, suppress only candidates when current
        # breadth + acceleration show that direction is being contradicted.
        if use_loss_regime_shield and candidates:
            kept_candidates = []
            for c in candidates:
                blocked, reason, details = detect_loss_regime_against_entry(
                    processed_data, ts, c, loss_regime_streak, market_breadth_ratio
                )
                if blocked:
                    diagnostics["loss_regime_blocked_candidates"] += 1
                    diagnostics[f"loss_regime_blocked_{c['side'].lower()}_candidates"] += 1
                    loss_regime_block_log.append({
                        "Timestamp": ts,
                        "Symbol": c["symbol"],
                        "Side": c["side"],
                        "LossEventStreak": details.get("streak", 0),
                        "Reason": reason,
                        "Breadth": details.get("breadth"),
                        "Symbol_R2": details.get("symbol_r2"),
                        "Symbol_R2_ATR": details.get("symbol_r2_atr"),
                        "BTC_R2": details.get("btc_r2"),
                        "BTC_R2_ATR": details.get("btc_r2_atr"),
                    })
                else:
                    kept_candidates.append(c)
            if len(kept_candidates) != len(candidates):
                diagnostics["loss_regime_active_events"] += 1
                diagnostics["loss_regime_blocked_entry_events"] += 1
            candidates = kept_candidates

        # --------------------------------------------------------
        # CORRELATED LOSS CLUSTER CONTROL — NEW ENTRIES ONLY
        # --------------------------------------------------------
        if use_cluster_control and candidates:
            candidates = apply_correlated_loss_cluster_control(
                candidates, loss_regime_streak, active_positions, diagnostics, ts
            )

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
    if use_shock_shield:
        diagnostics["shock_block_log"] = shock_block_log
    if use_loss_regime_shield:
        diagnostics["loss_regime_block_log"] = loss_regime_block_log

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
# CORRELATED LOSS CLUSTER CONTROL
# ============================================================
def apply_correlated_loss_cluster_control(candidates, direction_loss_streak,
                                           active_positions, diagnostics, ts):
    """
    Prevent repeated directional clustering without changing V74 signals.

    This is deliberately narrower than a direction cooldown:
      * only activates after >= 2 completed loss events in that direction;
      * only activates when >= 3 NEW candidates want that same direction;
      * keeps the highest-ranked candidate for that direction;
      * never touches existing positions or exit logic.

    Ranking is already established by V74 before this function is called, so
    keeping candidates[:N] preserves the original V74 preference ordering.
    """
    if not candidates:
        return candidates

    out = []
    by_side = {"LONG": [], "SHORT": []}
    for c in candidates:
        by_side.get(c.get("side"), []).append(c)

    blocked = []
    for side in ("LONG", "SHORT"):
        group = by_side[side]
        streak = int(direction_loss_streak.get(side, 0))
        if streak >= CLUSTER_LOSS_TRIGGER and len(group) >= CLUSTER_MIN_SAME_SIDE:
            keep_n = min(CLUSTER_KEEP_AFTER_TRIGGER, len(group))
            kept = group[:keep_n]
            out.extend(kept)
            blocked.extend(group[keep_n:])
            diagnostics["cluster_control_events"] += 1
            diagnostics[f"cluster_control_blocked_{side.lower()}"] += len(group) - keep_n
            for c in group[keep_n:]:
                diagnostics["cluster_control_block_log"].append({
                    "Timestamp": ts,
                    "Symbol": c.get("symbol"),
                    "Side": side,
                    "LossEventStreak": streak,
                    "SameSideCandidates": len(group),
                    "Action": "BLOCK_CORRELATED_CLUSTER",
                })
        else:
            out.extend(group)

    # Preserve V74 candidate ordering as much as possible.
    rank = {id(c): i for i, c in enumerate(candidates)}
    out.sort(key=lambda c: rank[id(c)])
    return out


# ============================================================
# MAIN — TWO-SIDED V74 + LOSS-REGIME SHIELD TEST
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 76)
    print("HUNTER-V74 — TWO-SIDED CORRELATED LOSS CLUSTER CONTROL TEST")
    print("=" * 76)
    print("BASELINE = EXACT V74 LONG + SHORT CORE")
    print("TEST     = EXACT V74 CORE + CORRELATED LOSS CLUSTER CONTROL")
    print("The control only limits NEW same-direction clusters after repeated losses.")
    print("It keeps the best-ranked V74 candidate and leaves exits/positions untouched.")
    print("No signal / ranking / ATR-SL / trailing / timeout / sizing parameter is changed.")
    print("=" * 76)

    baseline_trades, baseline_equity, baseline_diag = run_backtest(
        processed_data,
        use_lsp=False,
        use_lsp3=False,
        use_shock_shield=False,
        long_only=False,
        use_loss_regime_shield=False,
        use_cluster_control=False,
    )

    shield_trades, shield_equity, shield_diag = run_backtest(
        processed_data,
        use_lsp=False,
        use_lsp3=False,
        use_shock_shield=False,
        long_only=False,
        use_loss_regime_shield=False,
        use_cluster_control=True,
    )

    baseline_summary = report(
        "V74 BASELINE — TWO SIDED",
        baseline_trades,
        baseline_equity,
        baseline_diag,
    )

    shield_summary = report(
        "V74 + CORRELATED CLUSTER CONTROL — TWO SIDED",
        shield_trades,
        shield_equity,
        shield_diag,
    )

    print("\n" + "=" * 76)
    print("DIRECT COMPARISON")
    print("=" * 76)
    print(f"{'Metric':28s} {'V74':>14s} {'SHIELD':>14s} {'Delta':>14s}")
    print("-" * 74)
    if baseline_summary and shield_summary:
        comparison_rows = [
            ("Trades", baseline_summary["trades"], shield_summary["trades"]),
            ("Wins", baseline_summary["wins"], shield_summary["wins"]),
            ("Losses", baseline_summary["losses"], shield_summary["losses"]),
            ("Win Rate %", baseline_summary["wr"], shield_summary["wr"]),
            ("Net PnL $", baseline_summary["pnl"], shield_summary["pnl"]),
            ("Net R", baseline_summary["net_r"], shield_summary["net_r"]),
            ("Max DD $", baseline_summary["max_dd"], shield_summary["max_dd"]),
            ("Max DD %", baseline_summary["max_dd_pct"], shield_summary["max_dd_pct"]),
            ("Max Loss Streak", baseline_summary["max_loss_streak"], shield_summary["max_loss_streak"]),
        ]
        for metric, b, v in comparison_rows:
            d = v - b
            if metric in ("Net PnL $", "Max DD $"):
                print(f"{metric:28s} ${b:13,.2f} ${v:13,.2f} ${d:13,.2f}")
            elif metric in ("Win Rate %", "Max DD %"):
                print(f"{metric:28s} {b:13.2f} {v:13.2f} {d:13.2f}")
            else:
                print(f"{metric:28s} {b:14.2f} {v:14.2f} {d:14.2f}")

    print("\nCORRELATED CLUSTER CONTROL DIAGNOSTICS")
    print(f"Blocked LONG       : {shield_diag.get('cluster_control_blocked_long', 0)}")
    print(f"Blocked SHORT      : {shield_diag.get('cluster_control_blocked_short', 0)}")
    print(f"Blocked TOTAL      : {shield_diag.get('cluster_control_blocked_long', 0) + shield_diag.get('cluster_control_blocked_short', 0)}")
    print(f"Control events     : {shield_diag.get('cluster_control_events', 0)}")

    # Explicit per-symbol audit for the TEST result.
    print("\nSHIELD PER-SYMBOL DETAIL")
    print("Symbol    | Trades | Wins | Losses | Win Rate | PnL")
    print("-" * 64)
    for symbol in SYMBOLS:
        sdf = shield_trades[shield_trades["Symbol"] == symbol]
        if sdf.empty:
            print(f"{symbol:8s} |      0 |    0 |      0 |    0.00% | $       0.00")
            continue
        st = len(sdf)
        sw = int((sdf["Outcome"] == "WIN").sum())
        sl = st - sw
        swr = sw / st * 100.0
        sp = float(sdf["Dollar_PnL"].sum())
        print(f"{symbol:8s} | {st:6d} | {sw:4d} | {sl:6d} | {swr:8.2f}% | ${sp:12,.2f}")

    # Forensics on both runs: this makes it obvious whether the shield
    # actually changes the consecutive-loss structure instead of merely
    # changing the final PnL.
    print("\n" + "=" * 76)
    print("BASELINE LOSS-STREAK FORENSICS")
    print("=" * 76)
    enhanced_loss_streak_forensics(baseline_trades)

    print("\n" + "=" * 76)
    print("SHIELD LOSS-STREAK FORENSICS")
    print("=" * 76)
    enhanced_loss_streak_forensics(shield_trades)

    # Save both trade/equity sets plus the shield block log.
    save_csvs(
        baseline_trades,
        baseline_equity,
        shield_trades,
        shield_equity,
    )
    if shield_diag.get("cluster_control_block_log"):
        pd.DataFrame(shield_diag["cluster_control_block_log"]).to_csv(
            os.path.join(OUTPUT_DIR, "correlated_cluster_control_blocks.csv"),
            index=False,
        )

    print("\n" + "=" * 76)
    print("TWO-SIDED CORRELATED LOSS CLUSTER CONTROL TEST COMPLETE")
    print("=" * 76)
    print("Use the DIRECT COMPARISON + both forensics sections to judge whether")
    print("the shield reduces consecutive losses without materially changing")
    print("the original V74 trade count / win rate / PnL profile.")

