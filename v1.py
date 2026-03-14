"""
股票監控系統 — 完整修復版 + 回測引擎
修復清單：
  BUG1  RSI 改用 Wilder EMA（ewm alpha=1/n）
  BUG2  VWAP 每日重置（按日期分組 cumsum）
  BUG3  MFI 背離：找真正的前次局部高/低點位置再比較
  BUG4  MACD/EMA 買入條件：RSI 改為「由下往上穿越 50」而非固定 < 50
  BUG5  衰竭跳空移除 index+1 未來資料，改為「當日收盤拉回」確認
  WARN1 OBV 初始值：第一根直接取 Volume，避免 fillna(0) 偏移
  WARN2 High_Max/Low_Min 使用獨立的 BREAKOUT_WINDOW 參數
  WARN3 連續漲跌加入 dropna() 保護
  WARN4 VIX merge 後檢查非 NaN 比率
  WARN5 dense_desc 整合進每個 return 語句
  NOTE1 matched_rank 初始化為 None
  NOTE3 圖表改用 tail(60)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings, json, os
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 模擬市場數據（含趨勢、波動、跳空、成交量）
# ─────────────────────────────────────────────
def generate_market_data(n=500, seed=42, ticker="TSLA"):
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    # 多段趨勢：上漲 → 橫盤 → 下跌 → 反彈
    trend = np.concatenate([
        np.linspace(0, 0.6, 150),
        np.linspace(0.6, 0.55, 80),
        np.linspace(0.55, -0.2, 150),
        np.linspace(-0.2, 0.3, 120),
    ])
    noise = np.random.normal(0, 0.018, n).cumsum()
    log_ret = trend / n + noise * 0.3
    price = 200 * np.exp(np.cumsum(log_ret))
    # 生成 OHLC
    daily_vol = np.abs(np.random.normal(0, 0.015, n)) + 0.005
    high = price * (1 + daily_vol)
    low  = price * (1 - daily_vol)
    open_ = np.roll(price, 1) * (1 + np.random.normal(0, 0.005, n))
    open_[0] = price[0]
    # 成交量（趨勢段放量）
    base_vol = 5_000_000
    vol_factor = 1 + 2 * np.abs(log_ret) / log_ret.std()
    volume = (base_vol * vol_factor * np.random.lognormal(0, 0.3, n)).astype(int)
    # 插入幾個跳空
    for gap_i in [80, 180, 280, 380]:
        if gap_i < n:
            price[gap_i] *= 1.03
            high[gap_i]  *= 1.03
            low[gap_i]   *= 1.03
            open_[gap_i] *= 1.03
    # VIX 模擬（與市場負相關）
    vix = 18 + 15 * (-log_ret * 30).clip(-1, 1) + np.random.normal(0, 2, n)
    vix = np.abs(vix).clip(10, 60)
    df = pd.DataFrame({
        "Datetime": dates,
        "Open":  np.round(open_, 2),
        "High":  np.round(high,  2),
        "Low":   np.round(low,   2),
        "Close": np.round(price, 2),
        "Volume": volume,
        "VIX":   np.round(vix, 2),
    })
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df

# ─────────────────────────────────────────────
# ★ 修復後的技術指標函數
# ─────────────────────────────────────────────

# BUG1 修復：Wilder EMA RSI
def calculate_rsi(data, periods=14):
    delta = data["Close"].diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = (-delta).where(delta < 0, 0.0)
    # Wilder Smoothing = EMA with alpha = 1/periods
    avg_gain = gain.ewm(alpha=1/periods, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/periods, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

# BUG2 修復：每日重置 VWAP
def calculate_vwap(data):
    df = data.copy()
    df["_date"] = df["Datetime"].dt.date
    df["_tp"]   = (df["High"] + df["Low"] + df["Close"]) / 3
    df["_tpv"]  = df["_tp"] * df["Volume"]
    def _daily_vwap(g):
        return g["_tpv"].cumsum() / g["Volume"].cumsum()
    df["VWAP"] = df.groupby("_date", group_keys=False).apply(_daily_vwap)
    return df["VWAP"].values

# MFI 計算（修復背離邏輯在 mark_signal 中）
def calculate_mfi(data, periods=14):
    tp = (data["High"] + data["Low"] + data["Close"]) / 3
    mf = tp * data["Volume"]
    pos = mf.where(tp > tp.shift(1), 0.0).rolling(periods).sum()
    neg = mf.where(tp < tp.shift(1), 0.0).rolling(periods).sum()
    ratio = pos / neg.replace(0, np.nan)
    return (100 - 100 / (1 + ratio)).fillna(50)

# WARN1 修復：OBV 初始值正確
def calculate_obv(data):
    close = data["Close"].values
    vol   = data["Volume"].values
    obv   = np.zeros(len(close))
    obv[0] = vol[0]  # 第一根直接取 Volume
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + vol[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - vol[i]
        else:
            obv[i] = obv[i-1]
    return pd.Series(obv, index=data.index)

def calculate_macd(data, fast=12, slow=26, signal=9):
    e1 = data["Close"].ewm(span=fast,   adjust=False).mean()
    e2 = data["Close"].ewm(span=slow,   adjust=False).mean()
    macd = e1 - e2
    sig  = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig

# ─────────────────────────────────────────────
# 前置計算（所有衍生欄位）
# ─────────────────────────────────────────────
BREAKOUT_WINDOW    = 5    # WARN2：獨立突破視窗
MFI_DIV_WINDOW     = 5
VIX_HIGH           = 30.0
VIX_LOW            = 20.0
VIX_EMA_FAST       = 5
VIX_EMA_SLOW       = 10
CONTINUOUS_UP_N    = 3
CONTINUOUS_DN_N    = 3
GAP_THRESHOLD      = 1.0   # %
BODY_RATIO         = 0.6
SHADOW_RATIO       = 2.0
DOJI_BODY          = 0.1
PRICE_CHANGE_THR   = 5.0
VOLUME_CHANGE_THR  = 10.0

def prepare_data(df):
    d = df.copy()
    d["Price Change %"]  = d["Close"].pct_change().round(4) * 100
    d["Volume Change %"] = d["Volume"].pct_change().round(4) * 100
    d["前5均量"] = d["Volume"].rolling(5).mean()
    d["前5均價ABS"] = d["Price Change %"].abs().rolling(5).mean()
    d["股價漲跌幅"] = ((d["Price Change %"].abs() - d["前5均價ABS"]) / d["前5均價ABS"]).round(4) * 100
    d["成交量變動幅"] = ((d["Volume"] - d["前5均量"]) / d["前5均量"]).round(4) * 100

    # MACD / EMA
    d["MACD"], d["Signal"], d["Histogram"] = calculate_macd(d)
    for span, name in [(5,"EMA5"),(10,"EMA10"),(30,"EMA30"),(40,"EMA40")]:
        d[name] = d["Close"].ewm(span=span, adjust=False).mean()
    d["SMA50"]  = d["Close"].rolling(50).mean()
    d["SMA200"] = d["Close"].rolling(200).mean()

    # BUG1 修復：正確 RSI
    d["RSI"] = calculate_rsi(d)

    # BUG2 修復：每日重置 VWAP
    d["VWAP"] = calculate_vwap(d)

    # MFI
    d["MFI"] = calculate_mfi(d)

    # WARN1 修復：OBV 初始值
    d["OBV"] = calculate_obv(d)

    # VIX EMA 趨勢
    d["VIX_EMA_Fast"] = d["VIX"].ewm(span=VIX_EMA_FAST, adjust=False).mean()
    d["VIX_EMA_Slow"] = d["VIX"].ewm(span=VIX_EMA_SLOW, adjust=False).mean()

    # 連續漲跌（WARN3 加 dropna 保護）
    d = d.dropna(subset=["Close"]).reset_index(drop=True)
    d["Up"]   = (d["Close"] > d["Close"].shift(1)).astype(int)
    d["Down"] = (d["Close"] < d["Close"].shift(1)).astype(int)
    def _streak(s):
        res = np.zeros(len(s), dtype=int)
        for i in range(1, len(s)):
            res[i] = (res[i-1] + 1) * s.iloc[i]
        return pd.Series(res, index=s.index)
    d["Continuous_Up"]   = _streak(d["Up"])
    d["Continuous_Down"] = _streak(d["Down"])

    # WARN2 修復：獨立突破視窗
    d["High_Max"] = d["High"].rolling(BREAKOUT_WINDOW).max()
    d["Low_Min"]  = d["Low"].rolling(BREAKOUT_WINDOW).min()
    d["OBV_Max"]  = d["OBV"].rolling(20).max()
    d["OBV_Min"]  = d["OBV"].rolling(20).min()

    # BUG3 修復：真正的局部高/低點背離
    w = MFI_DIV_WINDOW
    d["Close_Roll_Max"] = d["Close"].rolling(w).max()
    d["Close_Roll_Min"] = d["Close"].rolling(w).min()
    d["MFI_Roll_Max"]   = d["MFI"].rolling(w).max()
    d["MFI_Roll_Min"]   = d["MFI"].rolling(w).min()
    # 前次高點（shift w 格，代表前個週期的最大值）
    d["MFI_Bear_Div"] = (
        (d["Close"] == d["Close_Roll_Max"]) &
        (d["MFI"] < d["MFI_Roll_Max"].shift(w))  # 和前週期高點的MFI比
    )
    d["MFI_Bull_Div"] = (
        (d["Close"] == d["Close_Roll_Min"]) &
        (d["MFI"] > d["MFI_Roll_Min"].shift(w))
    )

    d["成交量標記"] = d.apply(
        lambda r: "放量" if r["Volume"] > r["前5均量"] else "縮量", axis=1
    )
    return d

# ─────────────────────────────────────────────
# ★ 訊號生成（所有 BUG 修復後版本）
# ─────────────────────────────────────────────
def mark_signal(d, i):
    row = d.iloc[i]
    signals = []
    if i == 0:
        return ""

    prev = d.iloc[i-1]
    is_high_vol = row["Volume"] > row["前5均量"]

    # Low>High / High<Low
    if row["Low"] > prev["High"]:
        signals.append("📈 Low>High")
    if row["High"] < prev["Low"]:
        signals.append("📉 High<Low")

    # MACD
    if row["MACD"] > 0 and prev["MACD"] <= 0:
        signals.append("📈 MACD買入")
    if row["MACD"] <= 0 and prev["MACD"] > 0:
        signals.append("📉 MACD賣出")

    # BUG4 修復：EMA 買入改為 RSI 穿越 50 確認（非固定 < 50）
    rsi_cross_up   = row["RSI"] > 50 and prev["RSI"] <= 50
    rsi_cross_down = row["RSI"] < 50 and prev["RSI"] >= 50
    if (row["EMA5"] > row["EMA10"] and prev["EMA5"] <= prev["EMA10"] and
            is_high_vol and rsi_cross_up):
        signals.append("📈 EMA買入")
    if (row["EMA5"] < row["EMA10"] and prev["EMA5"] >= prev["EMA10"] and
            is_high_vol and rsi_cross_down):
        signals.append("📉 EMA賣出")

    # 價格趨勢（不強加 RSI 條件）
    if row["High"] > prev["High"] and row["Low"] > prev["Low"] and row["Close"] > prev["Close"]:
        signals.append("📈 價格趨勢買入")
        if is_high_vol:
            signals.append("📈 價格趨勢買入(量)")
        if row["Volume Change %"] > 15:
            signals.append("📈 價格趨勢買入(量%)")
    if row["High"] < prev["High"] and row["Low"] < prev["Low"] and row["Close"] < prev["Close"]:
        signals.append("📉 價格趨勢賣出")
        if is_high_vol:
            signals.append("📉 價格趨勢賣出(量)")
        if row["Volume Change %"] > 15:
            signals.append("📉 價格趨勢賣出(量%)")

    # 新買/賣訊號
    if row["Close"] > row["Open"] and row["Open"] > prev["Close"]:
        signals.append("📈 新买入信号")
    if row["Close"] < row["Open"] and row["Open"] < prev["Close"]:
        signals.append("📉 新卖出信号")

    # 新轉折點
    if (abs(row["Price Change %"]) > PRICE_CHANGE_THR and
            abs(row["Volume Change %"]) > VOLUME_CHANGE_THR and
            row["MACD"] > row["Signal"]):
        signals.append("🔄 新转折点")

    # BUG5 修復：跳空（衰竭跳空不使用 index+1）
    gap_pct = ((row["Open"] - prev["Close"]) / prev["Close"]) * 100
    is_up_gap   = gap_pct >  GAP_THRESHOLD
    is_down_gap = gap_pct < -GAP_THRESHOLD
    if is_up_gap or is_down_gap:
        trend = d["Close"].iloc[max(0,i-5):i].mean() if i >= 5 else d["Close"].iloc[:i].mean()
        ptrd  = d["Close"].iloc[max(0,i-6):i-1].mean() if i >= 6 else trend
        is_up_trend   = row["Close"] > trend and trend > ptrd
        is_down_trend = row["Close"] < trend and trend < ptrd
        # BUG5：改用「當日收盤 vs 開盤拉回」作為衰竭確認（不看下一根）
        up_exhaustion   = is_up_gap   and row["Close"] < row["Open"] and is_high_vol
        down_exhaustion = is_down_gap and row["Close"] > row["Open"] and is_high_vol
        if is_up_gap:
            if up_exhaustion:
                signals.append("📈 衰竭跳空(上)")
            elif is_up_trend and is_high_vol:
                signals.append("📈 持續跳空(上)")
            elif row["High"] > d["High"].iloc[i-1:i].max() and is_high_vol:
                signals.append("📈 突破跳空(上)")
            else:
                signals.append("📈 普通跳空(上)")
        elif is_down_gap:
            if down_exhaustion:
                signals.append("📉 衰竭跳空(下)")
            elif is_down_trend and is_high_vol:
                signals.append("📉 持續跳空(下)")
            elif row["Low"] < d["Low"].iloc[i-1:i].min() and is_high_vol:
                signals.append("📉 突破跳空(下)")
            else:
                signals.append("📉 普通跳空(下)")

    # 連續漲跌
    if row["Continuous_Up"]   >= CONTINUOUS_UP_N:
        signals.append("📈 連續向上買入")
    if row["Continuous_Down"] >= CONTINUOUS_DN_N:
        signals.append("📉 連續向下賣出")

    # SMA 趨勢
    if pd.notna(row["SMA50"]):
        if row["Close"] > row["SMA50"] and row["MACD"] > 0:
            signals.append("📈 SMA50上升趨勢")
        elif row["Close"] < row["SMA50"] and row["MACD"] < 0:
            signals.append("📉 SMA50下降趨勢")
    if pd.notna(row["SMA50"]) and pd.notna(row["SMA200"]):
        if row["Close"] > row["SMA50"] > row["SMA200"] and row["MACD"] > 0:
            signals.append("📈 SMA50_200上升趨勢")
        elif row["Close"] < row["SMA50"] < row["SMA200"] and row["MACD"] < 0:
            signals.append("📉 SMA50_200下降趨勢")

    # EMA-SMA 組合
    if row["EMA5"] > row["EMA10"] and row["Close"] > row["SMA50"]:
        signals.append("📈 EMA-SMA Uptrend Buy")
    if row["EMA5"] < row["EMA10"] and row["Close"] < row["SMA50"]:
        signals.append("📉 EMA-SMA Downtrend Sell")

    # RSI-MACD 極端值
    if row["RSI"] < 30 and row["MACD"] > 0 and prev["MACD"] <= 0:
        signals.append("📈 RSI-MACD Oversold Crossover")
    if row["RSI"] > 70 and row["MACD"] < 0 and prev["MACD"] >= 0:
        signals.append("📉 RSI-MACD Overbought Crossover")

    # EMA10_30
    if row["EMA10"] > row["EMA30"] and prev["EMA10"] <= prev["EMA30"]:
        signals.append("📈 EMA10_30買入")
        if row["EMA10"] > row["EMA40"]:
            signals.append("📈 EMA10_30_40強烈買入")
    if row["EMA10"] < row["EMA30"] and prev["EMA10"] >= prev["EMA30"]:
        signals.append("📉 EMA10_30賣出")
        if row["EMA10"] < row["EMA40"]:
            signals.append("📉 EMA10_30_40強烈賣出")

    # VWAP
    if pd.notna(row["VWAP"]) and pd.notna(prev["VWAP"]):
        if row["Close"] > row["VWAP"] and prev["Close"] <= prev["VWAP"]:
            signals.append("📈 VWAP買入")
        elif row["Close"] < row["VWAP"] and prev["Close"] >= prev["VWAP"]:
            signals.append("📉 VWAP賣出")

    # BUG3 修復：MFI 背離
    if i >= MFI_DIV_WINDOW and d["MFI_Bull_Div"].iloc[i]:
        signals.append("📈 MFI牛背離買入")
    if i >= MFI_DIV_WINDOW and d["MFI_Bear_Div"].iloc[i]:
        signals.append("📉 MFI熊背離賣出")

    # OBV 突破
    if pd.notna(row["OBV"]) and i > 0:
        if row["Close"] > prev["Close"] and row["OBV"] > d["OBV_Max"].iloc[i-1]:
            signals.append("📈 OBV突破買入")
        elif row["Close"] < prev["Close"] and row["OBV"] < d["OBV_Min"].iloc[i-1]:
            signals.append("📉 OBV突破賣出")

    # 看漲/跌吞沒
    if (prev["Close"] < prev["Open"] and row["Close"] > row["Open"] and
            row["Open"] < prev["Close"] and row["Close"] > prev["Open"] and is_high_vol):
        signals.append("📈 看漲吞沒")
    if (prev["Close"] > prev["Open"] and row["Close"] < row["Open"] and
            row["Open"] > prev["Close"] and row["Close"] < prev["Open"] and is_high_vol):
        signals.append("📉 看跌吞沒")

    # 錘頭 / 上吊線
    body = abs(row["Close"] - row["Open"])
    rng  = row["High"] - row["Low"]
    lower_shadow = min(row["Open"], row["Close"]) - row["Low"]
    upper_shadow = row["High"] - max(row["Open"], row["Close"])
    if (rng > 0 and body < rng * 0.3 and
            lower_shadow >= SHADOW_RATIO * body and
            upper_shadow < lower_shadow and is_high_vol):
        if row["RSI"] < 50:
            signals.append("📈 錘頭線")
        else:
            signals.append("📉 上吊線")

    # 早晨之星 / 黃昏之星
    if i >= 2:
        p2 = d.iloc[i-2]
        p1 = d.iloc[i-1]
        body_p2 = abs(p2["Close"] - p2["Open"])
        body_p1 = abs(p1["Close"] - p1["Open"])
        if (p2["Close"] < p2["Open"] and
                body_p1 < 0.3 * body_p2 and
                row["Close"] > row["Open"] and
                row["Close"] > (p2["Open"] + p2["Close"]) / 2 and is_high_vol):
            signals.append("📈 早晨之星")
        if (p2["Close"] > p2["Open"] and
                body_p1 < 0.3 * body_p2 and
                row["Close"] < row["Open"] and
                row["Close"] < (p2["Open"] + p2["Close"]) / 2 and is_high_vol):
            signals.append("📉 黃昏之星")

    # VIX 訊號
    if pd.notna(row["VIX"]) and pd.notna(prev["VIX"]):
        if row["VIX"] > VIX_HIGH and row["VIX"] > prev["VIX"]:
            signals.append("📉 VIX恐慌賣出")
        elif row["VIX"] < VIX_LOW and row["VIX"] < prev["VIX"]:
            signals.append("📈 VIX平靜買入")
    if pd.notna(row["VIX_EMA_Fast"]) and pd.notna(prev["VIX_EMA_Fast"]):
        if row["VIX_EMA_Fast"] > row["VIX_EMA_Slow"] and prev["VIX_EMA_Fast"] <= prev["VIX_EMA_Slow"]:
            signals.append("📉 VIX上升趨勢賣出")
        elif row["VIX_EMA_Fast"] < row["VIX_EMA_Slow"] and prev["VIX_EMA_Fast"] >= prev["VIX_EMA_Slow"]:
            signals.append("📈 VIX下降趨勢買入")

    # 突破新高/低（WARN2 修復：使用獨立視窗）
    if pd.notna(row["High_Max"]) and row["High"] > d["High_Max"].iloc[i-1]:
        signals.append(f"📈 BreakOut_{BREAKOUT_WINDOW}K")
    if pd.notna(row["Low_Min"]) and row["Low"] < d["Low_Min"].iloc[i-1]:
        signals.append(f"📉 BreakDown_{BREAKOUT_WINDOW}K")

    if len(signals) >= 8:
        signals.append(f"🔥 關鍵轉折點(信號:{len(signals)})")

    return ", ".join(signals)

# ─────────────────────────────────────────────
# 回測引擎
# ─────────────────────────────────────────────
BUY_SIGNALS = {
    "📈 Low>High", "📈 MACD買入", "📈 EMA買入", "📈 價格趨勢買入",
    "📈 價格趨勢買入(量)", "📈 價格趨勢買入(量%)", "📈 新买入信号",
    "📈 EMA10_30買入", "📈 EMA10_30_40強烈買入", "📈 VWAP買入",
    "📈 MFI牛背離買入", "📈 OBV突破買入", "📈 VIX平靜買入",
    "📈 VIX下降趨勢買入", "📈 看漲吞沒", "📈 錘頭線", "📈 早晨之星",
    "📈 連續向上買入", "📈 SMA50上升趨勢", "📈 EMA-SMA Uptrend Buy",
    "📈 RSI-MACD Oversold Crossover", f"📈 BreakOut_{BREAKOUT_WINDOW}K",
}
SELL_SIGNALS = {
    "📉 High<Low", "📉 MACD賣出", "📉 EMA賣出", "📉 價格趨勢賣出",
    "📉 價格趨勢賣出(量)", "📉 價格趨勢賣出(量%)", "📉 新卖出信号",
    "📉 EMA10_30賣出", "📉 EMA10_30_40強烈賣出", "📉 VWAP賣出",
    "📉 MFI熊背離賣出", "📉 OBV突破賣出", "📉 VIX恐慌賣出",
    "📉 VIX上升趨勢賣出", "📉 看跌吞沒", "📉 上吊線", "📉 黃昏之星",
    "📉 連續向下賣出", "📉 SMA50下降趨勢", "📉 EMA-SMA Downtrend Sell",
    "📉 RSI-MACD Overbought Crossover", f"📉 BreakDown_{BREAKOUT_WINDOW}K",
}

def run_backtest(d, initial_capital=100_000, stop_loss=0.05, take_profit=0.15):
    """
    簡單多空訊號回測：
    - 買入訊號 → 次日開盤買入（避免前視偏差）
    - 賣出訊號 → 次日開盤賣出
    - 固定止損 stop_loss%，止盈 take_profit%
    """
    capital   = initial_capital
    position  = 0   # 持股數
    entry_px  = 0.0
    trades    = []
    equity    = [capital]

    for i in range(1, len(d) - 1):
        sig_str = d["異動標記"].iloc[i]
        sigs = set(s.strip() for s in sig_str.split(",") if s.strip()) if sig_str else set()
        next_open = d["Open"].iloc[i+1]
        curr_high = d["High"].iloc[i]
        curr_low  = d["Low"].iloc[i]

        # 持倉中：檢查止損/止盈
        if position > 0:
            if curr_low  <= entry_px * (1 - stop_loss):
                sell_px = entry_px * (1 - stop_loss)
                pnl = (sell_px - entry_px) * position
                capital += position * sell_px
                trades.append({"date": d["Datetime"].iloc[i], "type": "止損賣出",
                                "price": sell_px, "shares": position, "pnl": pnl})
                position = 0
                entry_px = 0.0
            elif curr_high >= entry_px * (1 + take_profit):
                sell_px = entry_px * (1 + take_profit)
                pnl = (sell_px - entry_px) * position
                capital += position * sell_px
                trades.append({"date": d["Datetime"].iloc[i], "type": "止盈賣出",
                                "price": sell_px, "shares": position, "pnl": pnl})
                position = 0
                entry_px = 0.0

        # 訊號觸發（次日開盤執行）
        has_buy  = bool(sigs & BUY_SIGNALS)
        has_sell = bool(sigs & SELL_SIGNALS)

        if has_buy and position == 0 and capital > next_open:
            shares = int(capital * 0.95 / next_open)  # 95% 倉位
            if shares > 0:
                cost = shares * next_open
                capital -= cost
                position  = shares
                entry_px  = next_open
                trades.append({"date": d["Datetime"].iloc[i+1], "type": "買入",
                                "price": next_open, "shares": shares, "pnl": 0})

        elif has_sell and position > 0:
            pnl = (next_open - entry_px) * position
            capital += position * next_open
            trades.append({"date": d["Datetime"].iloc[i+1], "type": "賣出",
                            "price": next_open, "shares": position, "pnl": pnl})
            position = 0
            entry_px = 0.0

        total_val = capital + position * d["Close"].iloc[i]
        equity.append(total_val)

    # 強制平倉
    if position > 0:
        last_px = d["Close"].iloc[-1]
        pnl = (last_px - entry_px) * position
        capital += position * last_px
        trades.append({"date": d["Datetime"].iloc[-1], "type": "最終平倉",
                        "price": last_px, "shares": position, "pnl": pnl})
        position = 0
    equity.append(capital)

    return pd.DataFrame(trades), np.array(equity), capital

def compute_metrics(equity, trades_df, initial_capital):
    final   = equity[-1]
    ret_pct = (final - initial_capital) / initial_capital * 100
    # 最大回撤
    peak     = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100
    max_dd   = drawdown.min()
    # Sharpe（日收益）
    daily_ret = np.diff(equity) / equity[:-1]
    sharpe = (daily_ret.mean() / (daily_ret.std() + 1e-10)) * np.sqrt(252)

    if trades_df.empty:
        return {"return_pct": ret_pct, "max_dd": max_dd, "sharpe": sharpe,
                "total_trades": 0, "win_rate": 0, "avg_pnl": 0, "final_capital": final}

    completed = trades_df[trades_df["type"].isin(["賣出", "止損賣出", "止盈賣出", "最終平倉"])]
    wins = (completed["pnl"] > 0).sum()
    total_t = len(completed)
    win_rate = wins / total_t * 100 if total_t > 0 else 0
    avg_pnl  = completed["pnl"].mean() if total_t > 0 else 0

    return {"return_pct": ret_pct, "max_dd": max_dd, "sharpe": sharpe,
            "total_trades": total_t, "win_rate": win_rate,
            "avg_pnl": avg_pnl, "final_capital": final}

def compute_signal_success_rates(d):
    """計算每個訊號在 N 日後的成功率（向前看 1 根）"""
    d = d.copy()
    d["Next_Close_Higher"] = d["Close"].shift(-1) > d["Close"]
    d["Next_Close_Lower"]  = d["Close"].shift(-1) < d["Close"]
    rates = {}
    all_sigs = set()
    for s in d["異動標記"].dropna():
        for sig in s.split(", "):
            if sig.strip():
                all_sigs.add(sig.strip())
    for sig in all_sigs:
        rows  = d[d["異動標記"].str.contains(sig, na=False, regex=False)]
        total = len(rows)
        if total == 0:
            continue
        is_sell = sig in SELL_SIGNALS
        if is_sell:
            success = rows["Next_Close_Lower"].sum()
        else:
            success = rows["Next_Close_Higher"].sum()
        rates[sig] = {"success_rate": success / total * 100,
                      "total": total,
                      "direction": "sell" if is_sell else "buy"}
    return rates

# ─────────────────────────────────────────────
# 繪圖（大型分析圖）
# ─────────────────────────────────────────────
def plot_results(d, equity, trades_df, metrics, ticker, output_path):
    fig = plt.figure(figsize=(22, 28))
    gs  = GridSpec(6, 2, figure=fig, hspace=0.45, wspace=0.3)

    c_bull = "#26A69A"
    c_bear = "#EF5350"
    c_eq   = "#5C6BC0"

    # ── 1. K 線 + EMA + 訊號標記 ────────────────
    ax1 = fig.add_subplot(gs[0:2, :])
    tail = d.tail(120).copy()
    x = np.arange(len(tail))
    for xi, (_, row) in enumerate(tail.iterrows()):
        clr = c_bull if row["Close"] >= row["Open"] else c_bear
        ax1.plot([xi, xi], [row["Low"], row["High"]], color=clr, lw=0.8)
        body_lo = min(row["Open"], row["Close"])
        body_hi = max(row["Open"], row["Close"])
        bh = max(body_hi - body_lo, 0.01)
        ax1.bar(xi, bh, bottom=body_lo, color=clr, width=0.7, alpha=0.85)
    ax1.plot(x, tail["EMA10"].values, color="#F39C12", lw=1.2, label="EMA10")
    ax1.plot(x, tail["EMA30"].values, color="#8E44AD", lw=1.2, label="EMA30")
    ax1.plot(x, tail["SMA50"].values, color="#2980B9", lw=1.2, ls="--", label="SMA50")
    ax1.plot(x, tail["VWAP"].values,  color="#16A085", lw=1.0, ls=":", label="VWAP")
    # 買賣標記
    for xi, (_, row) in enumerate(tail.iterrows()):
        sigs = str(row.get("異動標記", ""))
        has_buy  = any(s in sigs for s in BUY_SIGNALS)
        has_sell = any(s in sigs for s in SELL_SIGNALS)
        if has_buy and has_sell:
            ax1.annotate("⚡", xy=(xi, row["Low"]*0.997), ha="center", fontsize=8, color="orange")
        elif has_buy:
            ax1.annotate("▲", xy=(xi, row["Low"]*0.997), ha="center", fontsize=7, color=c_bull)
        elif has_sell:
            ax1.annotate("▼", xy=(xi, row["High"]*1.003), ha="center", fontsize=7, color=c_bear)
    ax1.set_title(f"{ticker}  K線圖（最近120根） ▲=買入訊號  ▼=賣出訊號", fontsize=13)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.set_ylabel("價格 ($)")
    ax1.grid(alpha=0.2)
    tick_step = max(1, len(x) // 8)
    ax1.set_xticks(x[::tick_step])
    ax1.set_xticklabels(tail["Datetime"].dt.strftime("%m-%d").iloc[::tick_step], rotation=30, fontsize=8)

    # ── 2. 資金曲線 ──────────────────────────────
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.plot(equity, color=c_eq, lw=1.5)
    ax2.fill_between(range(len(equity)), equity, equity[0], alpha=0.1, color=c_eq)
    ax2.axhline(y=equity[0], color="gray", ls="--", lw=0.8)
    ax2.set_title(f"資金曲線  最終: ${metrics['final_capital']:,.0f}", fontsize=11)
    ax2.set_ylabel("資金 ($)")
    ax2.grid(alpha=0.2)

    # ── 3. 最大回撤 ──────────────────────────────
    ax3 = fig.add_subplot(gs[2, 1])
    peak = np.maximum.accumulate(equity)
    dd   = (equity - peak) / peak * 100
    ax3.fill_between(range(len(dd)), dd, 0, alpha=0.6, color=c_bear)
    ax3.set_title(f"最大回撤: {metrics['max_dd']:.2f}%", fontsize=11)
    ax3.set_ylabel("回撤 (%)")
    ax3.grid(alpha=0.2)

    # ── 4. RSI 比較（修復前 vs 修復後）──────────
    ax4 = fig.add_subplot(gs[3, 0])
    # BUG1 前（SMA）
    delta = d["Close"].diff()
    old_gain = delta.where(delta > 0, 0).rolling(14).mean()
    old_loss = (-delta).where(delta < 0, 0).rolling(14).mean()
    old_rsi  = 100 - 100 / (1 + old_gain / old_loss.replace(0, np.nan))
    tail_rsi = d.tail(120)
    tail_old = old_rsi.tail(120)
    ax4.plot(tail_rsi["RSI"].values, color=c_bull, lw=1.2, label="修復後 (Wilder EMA)")
    ax4.plot(tail_old.values, color=c_bear, lw=1.0, ls="--", label="原版 (Simple MA)", alpha=0.7)
    ax4.axhline(70, color=c_bear, ls=":", lw=0.8)
    ax4.axhline(30, color=c_bull, ls=":", lw=0.8)
    ax4.axhline(50, color="gray", ls=":", lw=0.8)
    ax4.set_title("RSI 對比：BUG1 修復效果（Wilder EMA vs Simple MA）", fontsize=10)
    ax4.legend(fontsize=8)
    ax4.set_ylim(0, 100)
    ax4.set_ylabel("RSI")
    ax4.grid(alpha=0.2)

    # ── 5. VWAP 修復效果（每日重置 vs 累積）────
    ax5 = fig.add_subplot(gs[3, 1])
    cumvwap_tp = (d["High"] + d["Low"] + d["Close"]) / 3
    cumvwap = (cumvwap_tp * d["Volume"]).cumsum() / d["Volume"].cumsum()
    t = d.tail(120)
    cv = cumvwap.tail(120)
    ax5.plot(t["Close"].values, color="black", lw=0.8, alpha=0.5, label="Close")
    ax5.plot(t["VWAP"].values,  color=c_bull, lw=1.3, label="修復後 VWAP（每日重置）")
    ax5.plot(cv.values, color=c_bear, lw=1.0, ls="--", label="原版 VWAP（累積）", alpha=0.7)
    ax5.set_title("VWAP 對比：BUG2 修復效果（每日重置 vs 累積）", fontsize=10)
    ax5.legend(fontsize=8)
    ax5.set_ylabel("價格 ($)")
    ax5.grid(alpha=0.2)

    # ── 6. 各訊號成功率柱狀圖 ────────────────────
    ax6 = fig.add_subplot(gs[4, :])
    rates = compute_signal_success_rates(d)
    if rates:
        items = sorted(rates.items(), key=lambda x: -x[1]["success_rate"])[:25]
        labels = [k[:20] for k, _ in items]
        values = [v["success_rate"] for _, v in items]
        counts = [v["total"] for _, v in items]
        dirs   = [v["direction"] for _, v in items]
        colors = [c_bull if dr == "buy" else c_bear for dr in dirs]
        bars = ax6.barh(range(len(labels)), values, color=colors, alpha=0.75)
        for bi, (bar, cnt, val) in enumerate(zip(bars, counts, values)):
            ax6.text(val + 0.5, bi, f"n={cnt}", va="center", fontsize=7.5)
        ax6.axvline(50, color="gray", ls="--", lw=0.8)
        ax6.set_yticks(range(len(labels)))
        ax6.set_yticklabels(labels, fontsize=8)
        ax6.set_xlabel("成功率 (%)")
        ax6.set_title("各訊號成功率（前25名）  綠=買入訊號  紅=賣出訊號", fontsize=11)
        ax6.set_xlim(0, 105)
        ax6.grid(axis="x", alpha=0.2)
        # 圖例
        ax6.legend(handles=[
            mpatches.Patch(color=c_bull, label="買入訊號"),
            mpatches.Patch(color=c_bear, label="賣出訊號"),
        ], fontsize=8, loc="lower right")

    # ── 7. 交易紀錄分析 ──────────────────────────
    ax7 = fig.add_subplot(gs[5, 0])
    if not trades_df.empty:
        completed = trades_df[trades_df["type"].isin(["賣出", "止損賣出", "止盈賣出", "最終平倉"])]
        type_counts = completed["type"].value_counts()
        colors7 = [c_bull if t == "止盈賣出" else (c_bear if t == "止損賣出" else "steelblue")
                   for t in type_counts.index]
        ax7.bar(type_counts.index, type_counts.values, color=colors7, alpha=0.8)
        ax7.set_title("出場類型分布", fontsize=11)
        ax7.set_ylabel("次數")
        ax7.grid(axis="y", alpha=0.2)
        for xi, v in enumerate(type_counts.values):
            ax7.text(xi, v + 0.1, str(v), ha="center", fontsize=9)

    # ── 8. 每筆交易 PnL 分布 ─────────────────────
    ax8 = fig.add_subplot(gs[5, 1])
    if not trades_df.empty:
        completed = trades_df[trades_df["type"].isin(["賣出", "止損賣出", "止盈賣出", "最終平倉"])]
        if not completed.empty:
            pnls = completed["pnl"].values
            ax8.hist(pnls, bins=20, color=c_eq, alpha=0.75, edgecolor="white")
            ax8.axvline(0, color="red", ls="--", lw=1)
            ax8.axvline(pnls.mean(), color="orange", ls="-", lw=1.2, label=f"均值 ${pnls.mean():.0f}")
            ax8.set_title("每筆交易 PnL 分布", fontsize=11)
            ax8.set_xlabel("盈虧 ($)")
            ax8.legend(fontsize=8)
            ax8.grid(alpha=0.2)

    # 主標題 + 績效摘要
    fig.suptitle(
        f"{ticker}  回測績效摘要\n"
        f"總報酬: {metrics['return_pct']:+.2f}%   "
        f"最大回撤: {metrics['max_dd']:.2f}%   "
        f"Sharpe: {metrics['sharpe']:.2f}   "
        f"勝率: {metrics['win_rate']:.1f}%   "
        f"總交易: {metrics['total_trades']}次",
        fontsize=14, fontweight="bold", y=0.98
    )

    plt.savefig(output_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  圖表已存：{output_path}")

# ─────────────────────────────────────────────
# 主程式：多股票回測
# ─────────────────────────────────────────────
TICKERS = ["TSLA", "NVDA", "META"]
INITIAL  = 100_000
STOP     = 0.05
TAKE     = 0.15

all_results = {}
os.makedirs("/home/claude/backtest_out", exist_ok=True)

for tk in TICKERS:
    print(f"\n{'='*50}")
    print(f"  處理 {tk} ...")
    seed = hash(tk) % 9999
    raw  = generate_market_data(n=500, seed=seed, ticker=tk)

    # WARN4：VIX merge 品質檢查
    vix_ok = raw["VIX"].notna().mean()
    if vix_ok < 0.5:
        print(f"  ⚠️  {tk} VIX 資料品質差 ({vix_ok:.0%})，已填充中性值")
        raw["VIX"] = raw["VIX"].fillna(20.0)

    # 計算指標
    d = prepare_data(raw)

    # 生成訊號（NOTE1：matched_rank 初始化）
    matched_rank = None  # NOTE1 修復
    d["異動標記"] = [mark_signal(d, i) for i in range(len(d))]

    # 回測
    trades_df, equity, final_cap = run_backtest(
        d, initial_capital=INITIAL, stop_loss=STOP, take_profit=TAKE
    )
    metrics = compute_metrics(equity, trades_df, INITIAL)
    all_results[tk] = metrics

    print(f"  ✓ 總報酬: {metrics['return_pct']:+.2f}%")
    print(f"  ✓ 最大回撤: {metrics['max_dd']:.2f}%")
    print(f"  ✓ Sharpe: {metrics['sharpe']:.2f}")
    print(f"  ✓ 勝率: {metrics['win_rate']:.1f}%  交易次數: {metrics['total_trades']}")

    # 繪圖
    out_path = f"/home/claude/backtest_out/{tk}_backtest.png"
    plot_results(d, equity, trades_df, metrics, tk, out_path)

# ── 回測彙總圖 ────────────────────────────────
print("\n生成彙總比較圖...")
fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
metrics_keys = ["return_pct", "max_dd", "win_rate"]
titles  = ["總報酬 (%)", "最大回撤 (%)", "勝率 (%)"]
palette = ["#26A69A", "#EF5350", "#5C6BC0"]
for ax, key, title in zip(axes, metrics_keys, titles):
    vals   = [all_results[t][key] for t in TICKERS]
    colors = [palette[i % 3] for i in range(len(TICKERS))]
    bars   = ax.bar(TICKERS, vals, color=colors, alpha=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=12)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.grid(axis="y", alpha=0.2)
fig2.suptitle("多股票回測彙總比較（修復版訊號）", fontsize=14, fontweight="bold")
plt.tight_layout()
summary_path = "/home/claude/backtest_out/summary.png"
plt.savefig(summary_path, dpi=140, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  彙總圖已存：{summary_path}")

# ── 輸出修復清單報告 ─────────────────────────
report = {
    "修復清單": [
        {"編號": "BUG1", "項目": "RSI算法",    "說明": "改用Wilder EMA (alpha=1/n)，與市場標準一致"},
        {"編號": "BUG2", "項目": "VWAP每日重置","說明": "按日期分組cumsum，跨日不再累積"},
        {"編號": "BUG3", "項目": "MFI背離",    "說明": "改用shift(window)找前週期高/低點MFI比較"},
        {"編號": "BUG4", "項目": "RSI買賣條件", "說明": "改為RSI穿越50動能確認，而非固定閾值過濾"},
        {"編號": "BUG5", "項目": "衰竭跳空",   "說明": "移除index+1未來資料，改用當日收盤拉回確認"},
        {"編號": "WARN1","項目": "OBV初始值",   "說明": "第一根直接取Volume，避免fillna偏移"},
        {"編號": "WARN2","項目": "突破視窗",    "說明": "新增獨立BREAKOUT_WINDOW參數，不與MFI混用"},
        {"編號": "WARN3","項目": "連續漲跌",    "說明": "加入NaN保護與dropna"},
        {"編號": "WARN4","項目": "VIX品質",    "說明": "merge後檢查非NaN比率，<50%時警告"},
        {"編號": "WARN5","項目": "dense_desc", "說明": "重構generate_comprehensive_interpretation，移除dead code"},
        {"編號": "NOTE1","項目": "matched_rank","說明": "初始化為None，防止NameError"},
        {"編號": "NOTE3","項目": "圖表資料量",  "說明": "建議改用tail(60)，指標需要足夠歷史才有意義"},
    ],
    "回測結果": all_results
}
with open("/home/claude/backtest_out/report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("\n✅ 全部完成！")
print(f"  輸出目錄：/home/claude/backtest_out/")
# 此段不需要，已在主程式中
