if __name__ == "__main__":
    print("=" * 72)
    print("HUNTER-V74 — OPTIMIZED STREAK MITIGATION & 34K PNL PRESERVATION")
    print("=" * 72)

    # 1. حالت پایه (Baseline) برای سنجش سود ۳۴ هزار دلاری و استریک ۸تایی
    base, base_eq, _ = run_backtest(
        processed_data,
        use_lsp=False, use_lsp3=False, gate_mode=None,
        use_firewall=False, use_single_loss_crowd=False
    )
    
    # 2. حالت فایروال مشروط
    fw223, _, fd223 = run_backtest(
        processed_data,
        use_lsp=False, use_lsp3=False, gate_mode=None,
        use_firewall=True, use_single_loss_crowd=False
    )
    
    # 3. حالت فایروال + کنترل تجمع بهینه برای شکستن استریک بدون افت سود
    fw4, _, fd4 = run_backtest(
        processed_data,
        use_lsp=False, use_lsp3=False, gate_mode=None,
        use_firewall=True, use_single_loss_crowd=True
    )

    def analyze_results(df):
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

    res_base = analyze_results(base)
    res_fw223 = analyze_results(fw223)
    res_fw4 = analyze_results(fw4)

    print(f"\n[Baseline] Trades: {res_base[0]} | WinRate: {res_base[1]:.2f}% | Net PnL: ${res_base[2]:,.2f} | Max Loss Streak: {res_base[3]}")
    print(f"[Firewall] Trades: {res_fw223[0]} | WinRate: {res_fw223[1]:.2f}% | Net PnL: ${res_fw223[2]:,.2f} | Max Loss Streak: {res_fw223[3]}")
    print(f"[FW + Crowd] Trades: {res_fw4[0]} | WinRate: {res_fw4[1]:.2f}% | Net PnL: ${res_fw4[2]:,.2f} | Max Loss Streak: {res_fw4[3]}")
    
    print("\nتحلیل نهایی:")
    print("هدف این است که گزینه‌ای انتخاب شود که Net PnL آن نزدیک به حد نصاب ۳۴ هزار دلار بماند و Max Loss Streak از ۸ کمتر شود.")
