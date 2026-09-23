from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import os


# ============================================================
# 1. CONFIG & RISK ENGINE
# ============================================================

@dataclass
class CLEConfig:
    risk_reward: float = 2.0
    trade_margin: float = 100.0
    leverage: float = 50.0
    fee_rate: float = 0.0007
    min_confluence_score: int = 5  # آستانه امتیاز منعطف (از ۱۰ امتیاز)
    max_consecutive_losses: int = 5


class CLERiskManager:
    def __init__(self, initial_equity: float, config: CLEConfig = None):
        self.cfg = config or CLEConfig()
        self.equity = initial_equity
        self.initial_equity = initial_equity
        self.open_trade = None
        self.consecutive_losses = 0
        self.trading_paused = False
        self.pause_timer = 0
        self.trade_history = []

    def can_open_new_trade(self) -> bool:
        if self.trading_paused or self.open_trade is not None or self.equity < self.cfg.trade_margin:
            return False
        return True

    def position_size(self, entry_price: float, stop_price: float) -> float:
        notional_value = self.cfg.trade_margin * self.cfg.leverage
        return notional_value / entry_price

    def register_result(self, pnl_amount: float, is_win: bool):
        notional = self.cfg.trade_margin * self.cfg.leverage
        fee_cost = notional * self.cfg.fee_rate * 2.0  # کارمزد رفت و برگشت
        net_pnl = pnl_amount - fee_cost
        
        self.equity += net_pnl
        if self.equity < 0:
            self.equity = 0.0

        self.trade_history.append({"win": is_win, "net_pnl": net_pnl})

        if not is_win:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.cfg.max_consecutive_losses:
                self.trading_paused = True
                self.pause_timer = 10
        else:
            self.consecutive_losses = 0

    def tick_pause(self):
        if self.trading_paused:
            self.pause_timer -= 1
            if self.pause_timer <= 0:
                self.trading_paused = False
                self.consecutive_losses = 0

    def get_stats(self) -> Dict[str, Any]:
        total = len(self.trade_history)
        if total == 0:
            return {"trades": 0, "win_rate": 0.0, "equity": self.equity}
        wins = sum(1 for t in self.trade_history if t["win"])
        return {
            "trades": total,
            "win_rate": round((wins / total) * 100, 2),
            "equity": round(self.equity, 2)
        }


# ============================================================
# 2. CAUSAL MARKET STRUCTURE & REGIME DETECTOR (بدون Lookahead)
# ============================================================

class MarketContextAnalyzer:
    @staticmethod
    def get_market_regime(df_4h: pd.DataFrame, current_idx: int) -> str:
        if current_idx < 50:
            return "Neutral"
        
        subset = df_4h.iloc[:current_idx + 1]
        close = subset["close"].iloc[-1]
        ema_50 = subset["close"].ewm(span=50, adjust=False).mean().iloc[-1]
        
        recent_highs = subset["high"].rolling(10).max().iloc[-1]
        recent_lows = subset["low"].rolling(10).min().iloc[-1]
        
        if close > ema_50 and close >= recent_highs * 0.99:
            return "Bull"
        elif close < ema_50 and close <= recent_lows * 1.01:
            return "Bear"
        
        return "Neutral"


# ============================================================
# 3. CLE-1 SIGNAL GENERATOR
# ============================================================

class CLEStrategyCore:
    def __init__(self, risk_manager: CLERiskManager, config: CLEConfig = None):
        self.rm = risk_manager
        self.cfg = config or CLEConfig()

    def evaluate_signal(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, idx_15m: int) -> Optional[Dict[str, Any]]:
        if idx_15m < 30:
            return None

        subset = df_15m.iloc[:idx_15m + 1]
        current_candle = subset.iloc[-1]
        
        # تطبیق ایندکس 15 دقیقه‌ای با تایم‌فریم 4 ساعته (هر 16 کندل 15م = یک کندل 4ساعت)
        idx_4h = max(0, idx_15m // 16)
        regime = MarketContextAnalyzer.get_market_regime(df_4h, idx_4h)
        if regime == "Neutral":
            return None

        score = 0
        direction = "long" if regime == "Bull" else "short"

        # 1. رژیم بازار -> 3 امتیاز
        score += 3

        # 2. تشخیص نقدینگی و Sweep -> 3 امتیاز
        lookback_liq = 15
        recent_high = subset["high"].iloc[-lookback_liq:-1].max()
        recent_low = subset["low"].iloc[-lookback_liq:-1].min()

        if direction == "long" and current_candle["low"] < recent_low and current_candle["close"] > recent_low:
            score += 3
        elif direction == "short" and current_candle["high"] > recent_high and current_candle["close"] < recent_high:
            score += 3

        # 3. شتاب و Displacement -> 2 امتیاز
        high_low = subset["high"] - subset["low"]
        atr = high_low.rolling(14).mean().iloc[-1]
        candle_range = current_candle["high"] - current_candle["low"]

        if np.isfinite(atr) and atr > 0 and candle_range > (1.1 * atr):
            score += 2

        # 4. حجم LBank -> 2 امتیاز
        if "volume" in subset.columns:
            vol_mean = subset["volume"].rolling(14).mean().iloc[-1]
            if np.isfinite(vol_mean) and current_candle["volume"] > (1.1 * vol_mean):
                score += 2

        if score < self.cfg.min_confluence_score:
            return None

        entry_price = current_candle["close"]
        if not np.isfinite(atr) or atr <= 0:
            atr = candle_range if candle_range > 0 else 1.0

        if direction == "long":
            stop_price = recent_low - (atr * 0.5)
            risk_dist = entry_price - stop_price
            if risk_dist <= 0:
                stop_price = entry_price - atr
                risk_dist = atr
            target_price = entry_price + (risk_dist * self.cfg.risk_reward)
        else:
            stop_price = recent_high + (atr * 0.5)
            risk_dist = stop_price - entry_price
            if risk_dist <= 0:
                stop_price = entry_price + atr
                risk_dist = atr
            target_price = entry_price - (risk_dist * self.cfg.risk_reward)

        size = self.rm.position_size(entry_price, stop_price)

        return {
            "direction": direction,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "size": size,
            "score": score
        }


# ============================================================
# 4. BACKTEST RUNNER (مختص دیتای واقعی LBank)
# ============================================================

def run_cle_backtest(df_15m: pd.DataFrame, df_4h: pd.DataFrame, initial_equity: float = 1000.0):
    cfg = CLEConfig()
    rm = CLERiskManager(initial_equity, cfg)
    strategy = CLEStrategyCore(rm, cfg)

    open_trade = None

    for i in range(30, len(df_15m)):
        candle = df_15m.iloc[i]
        
        if open_trade is not None:
            hit_stop = False
            hit_target = False

            if open_trade["direction"] == "long":
                hit_stop = candle["low"] <= open_trade["stop_price"]
                hit_target = candle["high"] >= open_trade["target_price"]
            else:
                hit_stop = candle["high"] >= open_trade["stop_price"]
                hit_target = candle["low"] <= open_trade["target_price"]

            if hit_stop or hit_target:
                is_win = hit_target
                if is_win:
                    pnl = abs(open_trade["target_price"] - open_trade["entry_price"]) * open_trade["size"]
                else:
                    pnl = -abs(open_trade["entry_price"] - open_trade["stop_price"]) * open_trade["size"]
                
                rm.register_result(pnl, is_win)
                open_trade = None

        elif rm.can_open_new_trade():
            signal = strategy.evaluate_signal(df_15m, df_4h, i)
            if signal is not None:
                open_trade = signal

        rm.tick_pause()

    stats = rm.get_stats()
    print("=" * 60)
    print("== CLE-1 LBANK REAL DATA BACKTEST RESULTS ==")
    print("=" * 60)
    print(f"تعداد معاملات کل:           {stats['trades']}")
    print(f"وین‌ریت (Win Rate):          {stats['win_rate']}%")
    print(f"سرمایه‌ی نهایی (USD):        ${stats['equity']}")
    print("=" * 60)
    return stats


if __name__ == "__main__":
    # مسیر فایل‌های واقعی LBank خودت را اینجا وارد کن (مثلاً فایل‌های CSV یا Parquet موجود در پروژه)
    data_path_15m = "lbank_data_15m.csv"  # نام فایل دیتای ۱۵ دقیقه‌ای ال‌بنک
    data_path_4h = "lbank_data_4h.csv"    # نام فایل دیتای ۴ ساعته ال‌بنک

    if os.path.exists(data_path_15m) and os.path.exists(data_path_4h):
        df_15 = pd.read_csv(data_path_15m)
        df_4 = pd.read_csv(data_path_4h)
        
        # اطمینان از نام ستون‌ها (استاندارد ال‌بنک)
        for df in [df_15, df_4]:
            df.columns = [c.lower() for c in df.columns]

        run_cle_backtest(df_15, df_4)
    else:
        print(f"⚠️ فایل‌های دیتا در مسیر پیدا نشدند. لطفاً مسیر فایل‌های LBank را اصلاح کن.")
