from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np


# ============================================================
# 1. CONFIG & RISK ENGINE
# ============================================================

@dataclass
class CLEConfig:
    risk_reward: float = 2.0
    trade_margin: float = 100.0
    leverage: float = 50.0
    fee_rate: float = 0.0007
    min_confluence_score: int = 7  # امتیاز حداقلی برای ورود (قابل تنظیم با بهینه‌سازی)
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
        # حجم بر اساس مارجین ۱۰۰ و اهرم ۵۰
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
        """تشخیص رژیم بازار در تایم‌فریم 4 ساعته به‌صورت کاملاً علّی (Causal)"""
        if current_idx < 50:
            return "Neutral"
        
        subset = df_4h.iloc[:current_idx + 1]
        close = subset["close"].iloc[-1]
        
        # محاسبه EMA 50
        ema_50 = subset["close"].ewm(span=50, adjust=False).mean().iloc[-1]
        
        # تشخیص ساختار ساده (مقایسه سقف و کف‌های اخیر بدون نگاه به آینده)
        recent_highs = subset["high"].rolling(10).max().iloc[-1]
        recent_lows = subset["low"].rolling(10).min().iloc[-1]
        
        if close > ema_50 and close >= recent_highs * 0.99:
            return "Bull"
        elif close < ema_50 and close <= recent_lows * 1.01:
            return "Bear"
        
        return "Neutral"


# ============================================================
# 3. CLE-1 SIGNAL GENERATOR (سیستم امتیازدهی Confluence Score)
# ============================================================

@dataclass
class TradeOrder:
    direction: str
    entry_price: float
    stop_price: float
    target_price: float
    size: float
    score: int


class CLEStrategyCore:
    def __init__(self, risk_manager: CLERiskManager, config: CLEConfig = None):
        self.rm = risk_manager
        self.cfg = config or CLEConfig()

    def evaluate_signal(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, idx_15m: int) -> Optional[Dict[str, Any]]:
        if idx_15m < 30:
            return None

        subset = df_15m.iloc[:idx_15m + 1]
        current_candle = subset.iloc[-1]
        prev_candle = subset.iloc[-2]
        
        # معادل‌سازی زمان 4 ساعته برای رژیم
        regime = MarketContextAnalyzer.get_market_regime(df_4h, max(0, idx_15m // 16)) # فرض تقریب نسبت تایم‌فریم‌ها
        if regime == "Neutral":
            return None

        score = 0
        direction = "long" if regime == "Bull" else "short"

        # 1. رژیم بازار (+2 امتیاز)
        score += 2

        # 2. تشخیص نقدینگی و Sweep (بررسی نفوذ به کف یا سقف قبلی و بازگشت)
        lookback_liq = 15
        recent_high = subset["high"].iloc[-lookback_liq:-1].max()
        recent_low = subset["low"].iloc[-lookback_liq:-1].min()

        liquidity_swept = False
        if direction == "long" and current_candle["low"] < recent_low and current_candle["close"] > recent_low:
            liquidity_swept = True
            score += 2
        elif direction == "short" and current_candle["high"] > recent_high and current_candle["close"] < recent_high:
            liquidity_swept = True
            score += 2

        # 3. شتاب و Displacement (قدرت کندل فعلی نسبت به ATR)
        high_low = subset["high"] - subset["low"]
        atr = high_low.rolling(14).mean().iloc[-1]
        candle_range = current_candle["high"] - current_candle["low"]

        is_displacement = candle_range > (1.2 * atr)
        if is_displacement:
            score += 2

        # 4. حجم یا پروکسی حجم (+1 امتیاز)
        if "volume" in subset.columns:
            vol_mean = subset["volume"].rolling(14).mean().iloc[-1]
            if current_candle["volume"] > (1.2 * vol_mean):
                score += 1

        # بررسی حد نصاب امتیاز Confluence Score
        if score < self.cfg.min_confluence_score:
            return None

        # تعیین حد ضرر و حد سود با رعایت دقیق ریوارد ۱:۲
        entry_price = current_candle["close"]
        if direction == "long":
            stop_price = recent_low - (atr * 0.5)  # پشت نقدینگی + بافر ATR
            risk_dist = entry_price - stop_price
            if risk_dist <= 0: return None
            target_price = entry_price + (risk_dist * self.cfg.risk_reward)
        else:
            stop_price = recent_high + (atr * 0.5)
            risk_dist = stop_price - entry_price
            if risk_dist <= 0: return None
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
# 4. BACKTEST RUNNER (اجرای ارزیابی ساختاری)
# ============================================================

def run_cle_backtest(df_15m: pd.DataFrame, df_4h: pd.DataFrame, initial_equity: float = 1000.0):
    cfg = CLEConfig()
    rm = CLERiskManager(initial_equity, cfg)
    strategy = CLEStrategyCore(rm, cfg)

    open_trade = None

    for i in range(30, len(df_15m)):
        candle = df_15m.iloc[i]
        
        # مدیریت پوزیشن باز
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
                pnl = (open_trade["target_price"] - open_trade["entry_price"]) * open_trade["size"] if is_win else \
                      (open_trade["stop_price"] - open_trade["entry_price"]) * open_trade["size"]
                pnl = abs(pnl) if is_win else -abs(pnl)
                
                rm.register_result(pnl, is_win)
                open_trade = None

        # بررسی ورود جدید
        elif rm.can_open_new_trade():
            signal = strategy.evaluate_signal(df_15m, df_4h, i)
            if signal is not None:
                open_trade = signal

        rm.tick_pause()

    stats = rm.get_stats()
    print("=" * 60)
    print("== CLE-1 BACKTEST RESULTS (CAUSAL ENGINE) ==")
    print("=" * 60)
    print(f"تعداد معاملات کل:           {stats['trades']}")
    print(f"وین‌ریت (Win Rate):          {stats['win_rate']}%")
    print(f"سرمایه‌ی نهایی (USD):        ${stats['equity']}")
    print("=" * 60)
    return stats


if __name__ == "__main__":
    # تست اولیه با دیتای مصنوعی استاندارد
    np.random.seed(42)
    n = 2000
    price = 100 + np.cumsum(np.random.randn(n) * 0.25)
    
    df_15 = pd.DataFrame({
        "open": price,
        "high": price + np.random.rand(n) * 0.5,
        "low": price - np.random.rand(n) * 0.5,
        "close": price + np.random.randn(n) * 0.1,
        "volume": np.random.randint(500, 2000, n)
    })
    
    # شبیه‌سازی 4 ساعته با ریتم کندتر
    df_4 = df_15.iloc[::16].copy()

    run_cle_backtest(df_15, df_4)
