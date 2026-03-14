import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import numpy as np

st.set_page_config(page_title="股票監控儀表板", layout="wide")

load_dotenv()
REFRESH_INTERVAL = 144

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")

# ==================== Telegram ====================
try:
    BOT_TOKEN = st.secrets["telegram"]["BOT_TOKEN"]
    CHAT_ID = st.secrets["telegram"]["CHAT_ID"]
    telegram_ready = True
except Exception:
    BOT_TOKEN = CHAT_ID = None
    telegram_ready = False

def send_telegram_alert(msg: str) -> bool:
    if not (BOT_TOKEN and CHAT_ID):
        return False
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": CHAT_ID,
            "text": msg,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        response = requests.get(url, params=payload, timeout=10)
        return response.status_code == 200 and response.json().get("ok")
    except Exception:
        return False

# ==================== 技術指標函數 ====================

# [BUG1 修復] RSI 改用 Wilder EMA（alpha=1/periods），與市場標準一致
def calculate_rsi(data, periods=14):
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/periods, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/periods, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_macd(data, fast=12, slow=26, signal=9):
    exp1 = data["Close"].ewm(span=fast, adjust=False).mean()
    exp2 = data["Close"].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

# [BUG2 修復] VWAP 每日重置，跨日不再累積失真
def calculate_vwap(data):
    df = data.copy()
    df["_date"] = pd.to_datetime(df["Datetime"]).dt.date
    df["_tp"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["_tpv"] = df["_tp"] * df["Volume"]
    def _daily(g):
        return g["_tpv"].cumsum() / g["Volume"].cumsum()
    result = df.groupby("_date", group_keys=False).apply(_daily)
    return result.values

# [BUG3 修復] MFI 背離使用 shift(window) 對比前週期高/低點
def calculate_mfi(data, periods=14):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    money_flow = typical_price * data['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=periods).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=periods).sum()
    money_ratio = positive_flow / negative_flow.replace(0, np.nan)
    mfi = 100 - (100 / (1 + money_ratio))
    return mfi.fillna(50)

# [WARN1 修復] OBV 第一根直接取 Volume，避免 fillna(0) 偏移
def calculate_obv(data):
    close = data["Close"].values
    vol = data["Volume"].values
    obv = np.zeros(len(close))
    obv[0] = vol[0]
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + vol[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - vol[i]
        else:
            obv[i] = obv[i-1]
    return pd.Series(obv, index=data.index)

def calculate_vix_trend(vix_data, fast=5, slow=10):
    vix_ema_fast = vix_data["Close"].ewm(span=fast, adjust=False).mean()
    vix_ema_slow = vix_data["Close"].ewm(span=slow, adjust=False).mean()
    return vix_ema_fast, vix_ema_slow

def get_vix_data(period, interval):
    vix_ticker = yf.Ticker("^VIX")
    vix_data = vix_ticker.history(period=period, interval=interval).reset_index()
    if "Date" in vix_data.columns:
        vix_data = vix_data.rename(columns={"Date": "Datetime"})
    vix_data["VIX Change %"] = vix_data["Close"].pct_change().round(4) * 100
    return vix_data

def calculate_volume_profile_dense_areas(data, bins=50, window=100, top_n=3):
    if len(data) < window:
        return []
    recent = data.tail(window).copy()
    price_min = recent['Low'].min()
    price_max = recent['High'].max()
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    volume_profile = np.zeros(bins)
    for i in range(len(recent)):
        low = recent['Low'].iloc[i]
        high = recent['High'].iloc[i]
        vol = recent['Volume'].iloc[i]
        indices = np.digitize([low, high], bin_edges) - 1
        indices = np.clip(indices, 0, bins - 1)
        if indices[0] == indices[1]:
            volume_profile[indices[0]] += vol
        else:
            num_bins = indices[1] - indices[0] + 1
            if num_bins > 0:
                vol_per_bin = vol / num_bins
                for j in range(indices[0], min(indices[1] + 1, bins)):
                    volume_profile[j] += vol_per_bin
    top_indices = np.argsort(volume_profile)[-top_n:][::-1]
    dense_areas = []
    for idx in top_indices:
        v = volume_profile[idx]
        if v > 0:
            dense_areas.append({
                'price_center': bin_centers[idx],
                'volume': v,
                'price_low': bin_edges[idx],
                'price_high': bin_edges[idx + 1]
            })
    return dense_areas

def calculate_signal_success_rate(data):
    data = data.copy()
    data["Next_Close_Higher"] = data["Close"].shift(-1) > data["Close"]
    data["Next_Close_Lower"] = data["Close"].shift(-1) < data["Close"]
    data["Next_High_Higher"] = data["High"].shift(-1) > data["High"]
    data["Next_Low_Lower"] = data["Low"].shift(-1) < data["Low"]
    sell_signals = [
        "📉 High<Low", "📉 MACD賣出", "📉 EMA賣出", "📉 價格趨勢賣出", "📉 價格趨勢賣出(量)",
        "📉 價格趨勢賣出(量%)", "📉 普通跳空(下)", "📉 突破跳空(下)", "📉 持續跳空(下)",
        "📉 衰竭跳空(下)", "📉 連續向下賣出", "📉 SMA50下降趨勢", "📉 SMA50_200下降趨勢",
        "📉 新卖出信号", "📉 RSI-MACD Overbought Crossover", "📉 EMA-SMA Downtrend Sell",
        "📉 Volume-MACD Sell", "📉 EMA10_30賣出", "📉 EMA10_30_40強烈賣出", "📉 看跌吞沒",
        "📉 上吊線", "📉 黃昏之星", "📉 VWAP賣出", "📉 MFI熊背離賣出", "📉 OBV量能確認賣出",
        "📉 VIX恐慌賣出", "📉 VIX上升趨勢賣出"
    ]
    all_signals = set()
    for signals in data["異動標記"].dropna():
        for signal in signals.split(", "):
            if signal:
                all_signals.add(signal)
    success_rates = {}
    for signal in all_signals:
        signal_rows = data[data["異動標記"].str.contains(signal, na=False, regex=False)]
        total_signals = len(signal_rows)
        if total_signals == 0:
            success_rates[signal] = {"success_rate": 0.0, "total_signals": 0,
                                     "direction": "down" if signal in sell_signals else "up"}
        else:
            if signal in sell_signals:
                success_count = (signal_rows["Next_Low_Lower"] & signal_rows["Next_Close_Lower"]).sum()
                success_rates[signal] = {"success_rate": (success_count / total_signals) * 100,
                                         "total_signals": total_signals, "direction": "down"}
            else:
                success_count = (signal_rows["Next_High_Higher"] & signal_rows["Next_Close_Higher"]).sum()
                success_rates[signal] = {"success_rate": (success_count / total_signals) * 100,
                                         "total_signals": total_signals, "direction": "up"}
    return success_rates

def send_email_alert(ticker, price_pct, volume_pct, **kwargs):
    subject = f"📣 股票異動通知：{ticker}"
    body = f"股票代號：{ticker}\n股價變動：{price_pct:.2f}%\n成交量變動：{volume_pct:.2f}%\n"
    signal_messages = {
        "low_high_signal":                "⚠️ 當前最低價高於前一時段最高價！",
        "high_low_signal":                "⚠️ 當前最高價低於前一時段最低價！",
        "macd_buy_signal":                "📈 MACD 買入訊號：MACD 線由負轉正！",
        "macd_sell_signal":               "📉 MACD 賣出訊號：MACD 線由正轉負！",
        "ema_buy_signal":                 "📈 EMA 買入訊號：EMA5 上穿 EMA10，成交量放大！",
        "ema_sell_signal":                "📉 EMA 賣出訊號：EMA5 下破 EMA10，成交量放大！",
        "price_trend_buy_signal":         "📈 價格趨勢買入訊號：最高價、最低價、收盤價均上漲！",
        "price_trend_sell_signal":        "📉 價格趨勢賣出訊號：最高價、最低價、收盤價均下跌！",
        "price_trend_vol_buy_signal":     "📈 價格趨勢買入訊號（量）：均上漲且成交量放大！",
        "price_trend_vol_sell_signal":    "📉 價格趨勢賣出訊號（量）：均下跌且成交量放大！",
        "price_trend_vol_pct_buy_signal": "📈 價格趨勢買入訊號（量%）：均上漲且成交量變化 > 15%！",
        "price_trend_vol_pct_sell_signal":"📉 價格趨勢賣出訊號（量%）：均下跌且成交量變化 > 15%！",
        "gap_common_up":      "📈 普通跳空(上)",
        "gap_common_down":    "📉 普通跳空(下)",
        "gap_breakaway_up":   "📈 突破跳空(上)：突破前高且成交量放大！",
        "gap_breakaway_down": "📉 突破跳空(下)：跌破前低且成交量放大！",
        "gap_runaway_up":     "📈 持續跳空(上)：上漲趨勢且成交量放大！",
        "gap_runaway_down":   "📉 持續跳空(下)：下跌趨勢且成交量放大！",
        "gap_exhaustion_up":  "📈 衰竭跳空(上)：趨勢末端，成交量放大！",
        "gap_exhaustion_down":"📉 衰竭跳空(下)：趨勢末端，成交量放大！",
        "continuous_up_buy_signal":   "📈 連續向上策略買入訊號！",
        "continuous_down_sell_signal":"📉 連續向下策略賣出訊號！",
        "sma50_up_trend":       "📈 SMA50 上升趨勢：當前價格高於 SMA50！",
        "sma50_down_trend":     "📉 SMA50 下降趨勢：當前價格低於 SMA50！",
        "sma50_200_up_trend":   "📈 SMA50_200 上升趨勢！",
        "sma50_200_down_trend": "📉 SMA50_200 下降趨勢！",
        "new_buy_signal":   "📈 新买入信号：今日收盘价大于开盘价且今日开盘价大于前日收盘价！",
        "new_sell_signal":  "📉 新卖出信号：今日收盘价小于开盘价且今日开盘价小于前日收盘价！",
        "new_pivot_signal": "🔄 新转折点！",
        "ema10_30_buy_signal":           "📈 EMA10_30 買入訊號：EMA10 上穿 EMA30！",
        "ema10_30_40_strong_buy_signal": "📈 EMA10_30_40 強烈買入訊號！",
        "ema10_30_sell_signal":           "📉 EMA10_30 賣出訊號：EMA10 下破 EMA30！",
        "ema10_30_40_strong_sell_signal": "📉 EMA10_30_40 強烈賣出訊號！",
        "bullish_engulfing": "📈 看漲吞沒形態！",
        "bearish_engulfing": "📉 看跌吞沒形態！",
        "hammer":      "📈 錘頭線！",
        "hanging_man": "📉 上吊線！",
        "morning_star":"📈 早晨之星！",
        "evening_star":"📉 黃昏之星！",
        "vwap_buy_signal":  "📈 VWAP 買入訊號：價格上穿 VWAP！",
        "vwap_sell_signal": "📉 VWAP 賣出訊號：價格下破 VWAP！",
        "mfi_bull_divergence": "📈 MFI 牛背離買入！",
        "mfi_bear_divergence": "📉 MFI 熊背離賣出！",
        "obv_breakout_buy":  "📈 OBV 突破買入！",
        "obv_breakout_sell": "📉 OBV 突破賣出！",
        "vix_panic_sell":    "📉 VIX 恐慌賣出訊號：VIX > 30 且上升！",
        "vix_calm_buy":      "📈 VIX 平靜買入訊號：VIX < 20 且下降！",
        "vix_uptrend_sell":  "📉 VIX 上升趨勢賣出訊號！",
        "vix_downtrend_buy": "📈 VIX 下降趨勢買入訊號！",
    }
    for key, msg in signal_messages.items():
        if kwargs.get(key):
            body += f"\n{msg}"
    body += "\n系統偵測到異常變動，請立即查看市場情況。"
    msg_obj = MIMEMultipart()
    msg_obj["From"] = SENDER_EMAIL
    msg_obj["To"] = RECIPIENT_EMAIL
    msg_obj["Subject"] = subject
    msg_obj.attach(MIMEText(body, "plain"))
    try:
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg_obj.as_string())
        server.quit()
        st.toast(f"📬 Email 已發送給 {RECIPIENT_EMAIL}")
    except Exception as e:
        st.error(f"Email 發送失敗：{e}")

# ==================== UI 設定 ====================
period_options   = ["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"]
interval_options = ["1m","5m","2m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"]
percentile_options = [1, 5, 10, 20]
refresh_options  = [30,60,90,144,150,180,210,240,270,300]

st.title("📊 股票監控儀表板（含異動提醒與 Email 通知 ✅）")
input_tickers = st.text_input("請輸入股票代號（逗號分隔）",
    value="TSLA, NIO, TSLL, XPEV, META, GOOGL, AAPL, NVDA, AMZN, MSFT, TSM")
selected_tickers  = [t.strip().upper() for t in input_tickers.split(",") if t.strip()]
selected_period   = st.selectbox("選擇時間範圍", period_options, index=5)
selected_interval = st.selectbox("選擇資料間隔", interval_options, index=8)
HIGH_N_HIGH_THRESHOLD = st.number_input("Close to high", min_value=0.1, max_value=1.0, value=0.9, step=0.1)
LOW_N_LOW_THRESHOLD   = st.number_input("Close to low",  min_value=0.1, max_value=1.0, value=0.9, step=0.1)
PRICE_THRESHOLD   = st.number_input("價格異動閾值 (%)",    min_value=0.1, max_value=200.0, value=80.0, step=0.1)
VOLUME_THRESHOLD  = st.number_input("成交量異動閾值 (%)", min_value=0.1, max_value=200.0, value=80.0, step=0.1)
PRICE_CHANGE_THRESHOLD  = st.number_input("新转折点 Price Change % 阈值 (%)",  min_value=0.1, max_value=200.0, value=5.0,  step=0.1)
VOLUME_CHANGE_THRESHOLD = st.number_input("新转折点 Volume Change % 阈值 (%)", min_value=0.1, max_value=200.0, value=10.0, step=0.1)
GAP_THRESHOLD          = st.number_input("跳空幅度閾值 (%)", min_value=0.1, max_value=50.0, value=1.0, step=0.1)
CONTINUOUS_UP_THRESHOLD   = st.number_input("連續上漲閾值 (根K線)", min_value=1, max_value=20, value=3, step=1)
CONTINUOUS_DOWN_THRESHOLD = st.number_input("連續下跌閾值 (根K線)", min_value=1, max_value=20, value=3, step=1)
PERCENTILE_THRESHOLD = st.selectbox("選擇數據範圍 (%)", percentile_options, index=1)
REFRESH_INTERVAL     = st.selectbox("选择刷新间隔 (秒)", refresh_options, index=refresh_options.index(300))
time.sleep(2)

all_signal_types = [
    "📉 High<Low","📉 MACD賣出","📉 EMA賣出","📉 價格趨勢賣出","📉 價格趨勢賣出(量)",
    "📉 價格趨勢賣出(量%)","📉 普通跳空(下)","📉 突破跳空(下)","📉 持續跳空(下)",
    "📉 衰竭跳空(下)","📉 連續向下賣出","📉 SMA50下降趨勢","📉 SMA50_200下降趨勢",
    "📉 新卖出信号","📉 RSI-MACD Overbought Crossover","📉 EMA-SMA Downtrend Sell",
    "📉 Volume-MACD Sell","📉 EMA10_30賣出","📉 EMA10_30_40強烈賣出","📉 看跌吞沒","📉 烏雲蓋頂",
    "📉 上吊線","📉 黃昏之星",
    "📈 Low>High","📈 MACD買入","📈 EMA買入","📈 價格趨勢買入","📈 價格趨勢買入(量)",
    "📈 價格趨勢買入(量%)","📈 普通跳空(上)","📈 突破跳空(上)","📈 持續跳空(上)",
    "📈 衰竭跳空(上)","📈 連續向上買入","📈 SMA50上升趨勢","📈 SMA50_200上升趨勢",
    "📈 新买入信号","📈 RSI-MACD Oversold Crossover","📈 EMA-SMA Uptrend Buy",
    "📈 Volume-MACD Buy","📈 EMA10_30買入","📈 EMA10_30_40強烈買入","📈 看漲吞沒",
    "📈 錘頭線","📈 早晨之星","✅ 量價","🔄 新转折点",
    "📈 VWAP買入","📉 VWAP賣出","📈 MFI牛背離買入","📉 MFI熊背離賣出",
    "📈 OBV突破買入","📉 OBV突破賣出",
    "📉 VIX恐慌賣出","📈 VIX平靜買入",
    "📉 VIX上升趨勢賣出","📈 VIX下降趨勢買入",
]
selected_signals = st.multiselect("选择哪些信号需要推送Telegram", all_signal_types,
                                  default=["📈 新买入信号"])

BODY_RATIO_THRESHOLD   = st.number_input("K線實體占比閾值",   min_value=0.1, max_value=0.9, value=0.6, step=0.05)
SHADOW_RATIO_THRESHOLD = st.number_input("K線影線長度閾值",   min_value=0.1, max_value=3.0, value=2.0, step=0.1)
DOJI_BODY_THRESHOLD    = st.number_input("十字星實體閾值占比", min_value=0.01,max_value=0.2, value=0.1, step=0.01)
MFI_DIVERGENCE_WINDOW  = st.number_input("MFI背离检测窗口 (根K線)", min_value=3, max_value=20, value=5, step=1)
VIX_HIGH_THRESHOLD = st.number_input("VIX 恐慌閾值 (高)", min_value=20.0, max_value=50.0, value=30.0, step=1.0)
VIX_LOW_THRESHOLD  = st.number_input("VIX 平靜閾值 (低)", min_value=10.0, max_value=25.0, value=20.0, step=1.0)
VIX_EMA_FAST = st.number_input("VIX 快速 EMA 期數", min_value=3,  max_value=15, value=5,  step=1)
VIX_EMA_SLOW = st.number_input("VIX 慢速 EMA 期數", min_value=8,  max_value=25, value=10, step=1)

st.subheader("成交密集區設定")
VOLUME_PROFILE_BINS    = st.number_input("價格分箱數量",        min_value=10,  max_value=200, value=50,  step=5)
VOLUME_PROFILE_WINDOW  = st.number_input("計算密集區的K線根數", min_value=20,  max_value=500, value=100, step=10)
VOLUME_PROFILE_TOP_N   = st.number_input("顯示前幾大密集區",    min_value=1,   max_value=5,   value=3,   step=1)
VOLUME_PROFILE_SHOW_ON_CHART = st.checkbox("在K線圖上標記成交密集區", value=True)

st.subheader("📋 Telegram 觸發條件配置（可隨時編輯）")
default_telegram_conditions = pd.DataFrame({
    "排名": [str(i) for i in range(1, 11)],
    "異動標記": [
        "📈 價格趨勢買入, 📈 持續跳空(上), 📈 SMA50上升趨勢, 📈 SMA50_200上升趨勢, 📈 EMA-SMA Uptrend Buy, 📈 OBV突破買入",
        "📈 Low>High, 📈 價格趨勢買入, 📈 SMA50上升趨勢, 📈 EMA-SMA Uptrend Buy",
        "📈 價格趨勢買入, 📈 SMA50上升趨勢, 📈 EMA-SMA Uptrend Buy, 📈 VIX平靜買入",
        "📈 連續向上買入, 📈 SMA50上升趨勢, 📈 EMA-SMA Uptrend Buy",
        "📈 突破跳空(上), 📈 SMA50上升趨勢, 📈 新买入信号, 📈 EMA-SMA Uptrend Buy",
        "📈 普通跳空(上), 📈 連續向上買入, 📉 SMA50下降趨勢, 📈 新买入信号",
        "📈 SMA50上升趨勢, 📈 EMA-SMA Uptrend Buy, 📈 OBV突破買入",
        "📉 普通跳空(下), 📈 SMA50上升趨勢, 📈 EMA-SMA Uptrend Buy",
        "📈 價格趨勢買入(量), 📈 價格趨勢買入(量%), 📈 SMA50上升趨勢, 📈 EMA-SMA Uptrend Buy",
        "📈 EMA買入, 📈 連續向上買入, 📈 SMA50上升趨勢, 📈 EMA-SMA Uptrend Buy",
    ],
    "成交量標記": ["放量","縮量","放量","縮量","放量","縮量","放量","縮量","放量","縮量"],
    "K線形態": ["大陽線","普通K線","普通K線","大陽線","射擊之星","普通K線","十字星","大陽線","早晨之星","看漲吞噬"],
})
telegram_conditions = st.data_editor(
    default_telegram_conditions, num_rows="dynamic",
    column_config={
        "排名":     st.column_config.TextColumn("排名"),
        "異動標記": st.column_config.TextColumn("異動標記", help="輸入多個信號，用逗號分隔"),
        "成交量標記":st.column_config.SelectboxColumn("成交量標記", options=["放量","縮量"]),
        "K線形態":  st.column_config.TextColumn("K線形態"),
    },
    use_container_width=True, hide_index=False,
)

placeholder = st.empty()

# ==================== K線形態計算（快取）====================
@st.cache_data(ttl=300)
def compute_kline_patterns(data, body_ratio_threshold, shadow_ratio_threshold, doji_body_threshold):
    data = data.copy()
    data["成交量標記"] = data.apply(
        lambda row: "放量" if row["Volume"] > row["前5均量"] else "縮量", axis=1)

    def identify_candlestick_pattern(row, index, data):
        pattern = "普通K線"
        interpretation = "波動有限，方向不明顯"
        if index > 0:
            prev_close = data["Close"].iloc[index-1]
            prev_open  = data["Open"].iloc[index-1]
            prev_high  = data["High"].iloc[index-1]
            prev_low   = data["Low"].iloc[index-1]
            curr_open  = row["Open"]
            curr_close = row["Close"]
            curr_high  = row["High"]
            curr_low   = row["Low"]
            body_size  = abs(curr_close - curr_open)
            candle_range = curr_high - curr_low
            if candle_range == 0:
                return pattern, interpretation
            prev_body_size = abs(prev_close - prev_open)
            is_uptrend   = data["Close"].iloc[max(0,index-5):index].mean() < curr_close if index >= 5 else False
            is_downtrend = data["Close"].iloc[max(0,index-5):index].mean() > curr_close if index >= 5 else False
            is_high_volume = row["Volume"] > row["前5均量"]

            if (body_size < candle_range * 0.3 and
                (min(curr_open, curr_close) - curr_low) >= shadow_ratio_threshold * body_size and
                (curr_high - max(curr_open, curr_close)) < (min(curr_open, curr_close) - curr_low) and
                is_downtrend):
                pattern = "錘子線"
                interpretation = "下方出現支撐" + ("，放量增強買入信號" if is_high_volume else "")
            elif (body_size < candle_range * 0.3 and
                  (curr_high - max(curr_open, curr_close)) >= shadow_ratio_threshold * body_size and
                  (min(curr_open, curr_close) - curr_low) < (curr_high - max(curr_open, curr_close)) and
                  is_uptrend):
                pattern = "射擊之星"
                interpretation = "高位拋壓沉重" + ("，放量增強賣出信號" if is_high_volume else "")
            elif body_size < doji_body_threshold * candle_range:
                pattern = "十字星"
                interpretation = "市場猶豫，方向未明確"
            elif curr_close > curr_open and body_size > body_ratio_threshold * candle_range:
                pattern = "大陽線"
                interpretation = "多方強勢推升" + ("，放量更有力" if is_high_volume else "")
            elif curr_close < curr_open and body_size > body_ratio_threshold * candle_range:
                pattern = "大陰線"
                interpretation = "空方強勢壓制" + ("，放量更偏空" if is_high_volume else "")
            elif (curr_close > curr_open and prev_close < prev_open and
                  curr_open < prev_close and curr_close > prev_open and is_high_volume):
                pattern = "看漲吞噬"
                interpretation = "買方強勢反攻，預示反轉"
            elif (curr_close < curr_open and prev_close > prev_open and
                  curr_open > prev_close and curr_close < prev_open and is_high_volume):
                pattern = "看跌吞噬"
                interpretation = "賣方強勢壓制，預示反轉"
            elif (is_uptrend and curr_close < curr_open and prev_close > prev_open and
                  curr_open > prev_close and curr_close < (prev_open + prev_close) / 2):
                pattern = "烏雲蓋頂"
                interpretation = "上升趨勢中陰線壓制，短期可能下跌"
            elif (is_downtrend and curr_close > curr_open and prev_close < prev_open and
                  curr_open < prev_close and curr_close > (prev_open + prev_close) / 2):
                pattern = "刺透形態"
                interpretation = "下跌趨勢中陽線反攻，短期可能上漲"
            elif (index > 1 and
                  data["Close"].iloc[index-2] < data["Open"].iloc[index-2] and
                  abs(data["Close"].iloc[index-1] - data["Open"].iloc[index-1]) < 0.3 * abs(data["Close"].iloc[index-2] - data["Open"].iloc[index-2]) and
                  curr_close > curr_open and
                  curr_close > (prev_open + prev_close) / 2 and is_high_volume):
                pattern = "早晨之星"
                interpretation = "下跌後強陽線，預示反轉"
            elif (index > 1 and
                  data["Close"].iloc[index-2] > data["Open"].iloc[index-2] and
                  abs(data["Close"].iloc[index-1] - data["Open"].iloc[index-1]) < 0.3 * abs(data["Close"].iloc[index-2] - data["Open"].iloc[index-2]) and
                  curr_close < curr_open and
                  curr_close < (prev_open + prev_close) / 2 and is_high_volume):
                pattern = "黃昏之星"
                interpretation = "上漲後強陰線，預示反轉"
        return pattern, interpretation

    results = [identify_candlestick_pattern(row, i, data) for i, row in data.iterrows()]
    data[["K線形態","單根解讀"]] = results
    return data

# ==================== 主迴圈 ====================
while True:
    with placeholder.container():
        st.subheader(f"⏱ 更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        for ticker in selected_tickers:
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period=selected_period, interval=selected_interval).reset_index()

                if data.empty or len(data) < 2:
                    st.warning(f"⚠️ {ticker} 無數據或數據不足，請嘗試其他時間範圍或間隔")
                    continue

                if "Date" in data.columns:
                    data = data.rename(columns={"Date": "Datetime"})
                elif "Datetime" not in data.columns:
                    st.warning(f"⚠️ {ticker} 數據缺少時間列")
                    continue

                data["Price Change %"]  = data["Close"].pct_change().round(4) * 100
                data["Volume Change %"] = data["Volume"].pct_change().round(4) * 100
                data["Close_Difference"] = data["Close"].diff().round(2)
                data["High_Difference"]  = data["High"].diff().round(2)
                data["Low_Difference"]   = data["Low"].diff().round(2)
                data["Close_N_High"] = (data["Close"] - data["Low"]) / (data["High"] - data["Low"])
                data["Close_N_Low"]  = (data["High"] - data["Close"]) / (data["High"] - data["Low"])
                data["HIGH_High_%"]  = ((data["High"].shift(-1) - data["High"]) / data["High"]) * 100

                data["前5均價"]    = data["Price Change %"].rolling(window=5).mean()
                data["前5均價ABS"] = data["Price Change %"].abs().rolling(window=5).mean()
                data["前5均量"]    = data["Volume"].rolling(window=5).mean()
                data["📈 股價漲跌幅 (%)"] = ((data["Price Change %"].abs() - data["前5均價ABS"]) / data["前5均價ABS"]).round(4) * 100
                data["📊 成交量變動幅 (%)"] = ((data["Volume"] - data["前5均量"]) / data["前5均量"]).round(4) * 100

                data["MACD"], data["Signal"], data["Histogram"] = calculate_macd(data)
                data["EMA5"]  = data["Close"].ewm(span=5,  adjust=False).mean()
                data["EMA10"] = data["Close"].ewm(span=10, adjust=False).mean()
                data["EMA30"] = data["Close"].ewm(span=30, adjust=False).mean()
                data["EMA40"] = data["Close"].ewm(span=40, adjust=False).mean()

                # [BUG1] 修復後 RSI
                data["RSI"] = calculate_rsi(data)

                # [BUG2] 修復後 VWAP（每日重置）
                data["VWAP"] = calculate_vwap(data)

                # MFI
                data["MFI"] = calculate_mfi(data)

                # [WARN1] 修復後 OBV
                data["OBV"] = calculate_obv(data)

                # VIX
                vix_data = get_vix_data(selected_period, selected_interval)
                if not vix_data.empty:
                    # 統一時間格式：去除時區、只保留日期部分（日線）或精確到分鐘（日內）
                    if selected_interval in ["1d", "5d", "1wk", "1mo", "3mo"]:
                        # 日線以上：只比對日期
                        data["_merge_key"] = pd.to_datetime(data["Datetime"]).dt.tz_localize(None).dt.normalize()
                        vix_data["_merge_key"] = pd.to_datetime(vix_data["Datetime"]).dt.tz_localize(None).dt.normalize()
                    else:
                        # 日內：精確到分鐘，去除時區
                        data["_merge_key"] = pd.to_datetime(data["Datetime"]).dt.tz_localize(None).dt.floor("min")
                        vix_data["_merge_key"] = pd.to_datetime(vix_data["Datetime"]).dt.tz_localize(None).dt.floor("min")
                    
                    data = data.merge(vix_data[["_merge_key","Close","VIX Change %"]],
                                      on="_merge_key", how="left", suffixes=("","_VIX"))
                    data.rename(columns={"Close_VIX": "VIX"}, inplace=True)
                    data.drop(columns=["_merge_key"], inplace=True)
                    
                    # WARN4 品質檢查
                    vix_ok_ratio = data["VIX"].notna().mean()
                    if vix_ok_ratio < 0.5:
                        st.warning(f"⚠️ {ticker} VIX 資料對齊率僅 {vix_ok_ratio:.0%}，VIX 訊號暫不可靠")
                else:
                    data["VIX"] = np.nan
                    data["VIX Change %"] = np.nan
                       

                if not data["VIX"].isna().all():
                    vix_series = pd.Series(data["VIX"].values, index=data.index)
                    data["VIX_EMA_Fast"] = vix_series.ewm(span=VIX_EMA_FAST, adjust=False).mean()
                    data["VIX_EMA_Slow"] = vix_series.ewm(span=VIX_EMA_SLOW, adjust=False).mean()
                else:
                    data["VIX_EMA_Fast"] = np.nan
                    data["VIX_EMA_Slow"] = np.nan

                # 連續漲跌（[WARN3] 加 NaN 保護）
                data = data.dropna(subset=["Close"]).reset_index(drop=True)
                data["Up"]   = (data["Close"] > data["Close"].shift(1)).astype(int)
                data["Down"] = (data["Close"] < data["Close"].shift(1)).astype(int)
                def _streak(s):
                    res = np.zeros(len(s), dtype=int)
                    for i in range(1, len(s)):
                        res[i] = (res[i-1] + 1) * int(s.iloc[i])
                    return pd.Series(res, index=s.index)
                data["Continuous_Up"]   = _streak(data["Up"])
                data["Continuous_Down"] = _streak(data["Down"])

                data["SMA50"]  = data["Close"].rolling(window=50).mean()
                data["SMA200"] = data["Close"].rolling(window=200).mean()

                # [WARN2] 獨立突破視窗
                BREAKOUT_WINDOW = MFI_DIVERGENCE_WINDOW  # 可改為獨立參數
                data["High_Max"] = data["High"].rolling(window=BREAKOUT_WINDOW).max()
                data["Low_Min"]  = data["Low"].rolling(window=BREAKOUT_WINDOW).min()
                data["OBV_Roll_Max"] = data["OBV"].rolling(window=20).max()
                data["OBV_Roll_Min"] = data["OBV"].rolling(window=20).min()

                # [BUG3] 修復 MFI 背離（shift(window) 比較前週期）
                w = MFI_DIVERGENCE_WINDOW
                data["Close_Roll_Max"] = data["Close"].rolling(window=w).max()
                data["MFI_Roll_Max"]   = data["MFI"].rolling(window=w).max()
                data["Close_Roll_Min"] = data["Close"].rolling(window=w).min()
                data["MFI_Roll_Min"]   = data["MFI"].rolling(window=w).min()
                data["MFI_Bear_Div"] = (
                    (data["Close"] == data["Close_Roll_Max"]) &
                    (data["MFI"] < data["MFI_Roll_Max"].shift(w))
                )
                data["MFI_Bull_Div"] = (
                    (data["Close"] == data["Close_Roll_Min"]) &
                    (data["MFI"] > data["MFI_Roll_Min"].shift(w))
                )

                # ==================== 訊號生成 ====================
                def mark_signal(row, index):
                    signals = []
                    if abs(row["📈 股價漲跌幅 (%)"]) >= PRICE_THRESHOLD and abs(row["📊 成交量變動幅 (%)"]) >= VOLUME_THRESHOLD:
                        signals.append("✅ 量價")
                    if index > 0 and row["Low"] > data["High"].iloc[index-1]:
                        signals.append("📈 Low>High")
                    if index > 0 and row["High"] < data["Low"].iloc[index-1]:
                        signals.append("📉 High<Low")
                    if index > 0 and row["Close_N_High"] >= HIGH_N_HIGH_THRESHOLD:
                        signals.append("📈 HIGH_N_HIGH")
                    if index > 0 and row["Close_N_Low"] >= LOW_N_LOW_THRESHOLD:
                        signals.append("📉 LOW_N_LOW")

                    # [BUG4] 修復：RSI 穿越 50 確認動能，而非固定 < 50
                    rsi_cross_up   = row["RSI"] > 50 and (index > 0 and data["RSI"].iloc[index-1] <= 50)
                    rsi_cross_down = row["RSI"] < 50 and (index > 0 and data["RSI"].iloc[index-1] >= 50)

                    if index > 0 and row["MACD"] > 0 and data["MACD"].iloc[index-1] <= 0:
                        signals.append("📈 MACD買入")
                    if index > 0 and row["MACD"] <= 0 and data["MACD"].iloc[index-1] > 0:
                        signals.append("📉 MACD賣出")
                    if (index > 0 and row["EMA5"] > row["EMA10"] and
                            data["EMA5"].iloc[index-1] <= data["EMA10"].iloc[index-1] and
                            row["Volume"] > data["Volume"].iloc[index-1] and rsi_cross_up):
                        signals.append("📈 EMA買入")
                    if (index > 0 and row["EMA5"] < row["EMA10"] and
                            data["EMA5"].iloc[index-1] >= data["EMA10"].iloc[index-1] and
                            row["Volume"] > data["Volume"].iloc[index-1] and rsi_cross_down):
                        signals.append("📉 EMA賣出")

                    if (index > 0 and row["High"] > data["High"].iloc[index-1] and
                            row["Low"] > data["Low"].iloc[index-1] and
                            row["Close"] > data["Close"].iloc[index-1] and row["MACD"] > 0):
                        signals.append("📈 價格趨勢買入")
                    if (index > 0 and row["High"] < data["High"].iloc[index-1] and
                            row["Low"] < data["Low"].iloc[index-1] and
                            row["Close"] < data["Close"].iloc[index-1] and row["MACD"] < 0):
                        signals.append("📉 價格趨勢賣出")
                    if (index > 0 and row["High"] > data["High"].iloc[index-1] and
                            row["Low"] > data["Low"].iloc[index-1] and
                            row["Close"] > data["Close"].iloc[index-1] and
                            row["Volume"] > data["前5均量"].iloc[index]):
                        signals.append("📈 價格趨勢買入(量)")
                    if (index > 0 and row["High"] < data["High"].iloc[index-1] and
                            row["Low"] < data["Low"].iloc[index-1] and
                            row["Close"] < data["Close"].iloc[index-1] and
                            row["Volume"] > data["前5均量"].iloc[index]):
                        signals.append("📉 價格趨勢賣出(量)")
                    if (index > 0 and row["High"] > data["High"].iloc[index-1] and
                            row["Low"] > data["Low"].iloc[index-1] and
                            row["Close"] > data["Close"].iloc[index-1] and
                            row["Volume Change %"] > 15):
                        signals.append("📈 價格趨勢買入(量%)")
                    if (index > 0 and row["High"] < data["High"].iloc[index-1] and
                            row["Low"] < data["Low"].iloc[index-1] and
                            row["Close"] < data["Close"].iloc[index-1] and
                            row["Volume Change %"] > 15):
                        signals.append("📉 價格趨勢賣出(量%)")

                    # [BUG5] 修復：衰竭跳空不使用 index+1 未來資料
                    if index > 0:
                        gap_pct = ((row["Open"] - data["Close"].iloc[index-1]) / data["Close"].iloc[index-1]) * 100
                        is_up_gap   = gap_pct >  GAP_THRESHOLD
                        is_down_gap = gap_pct < -GAP_THRESHOLD
                        if is_up_gap or is_down_gap:
                            trend     = data["Close"].iloc[index-5:index].mean() if index >= 5 else 0
                            prev_trend = data["Close"].iloc[index-6:index-1].mean() if index >= 6 else trend
                            is_up_trend   = row["Close"] > trend and trend > prev_trend
                            is_down_trend = row["Close"] < trend and trend < prev_trend
                            is_high_volume = row["Volume"] > data["前5均量"].iloc[index]
                            # 當日收盤拉回確認衰竭（不看下一根）
                            up_exhaustion   = is_up_gap   and row["Close"] < row["Open"] and is_high_volume
                            down_exhaustion = is_down_gap and row["Close"] > row["Open"] and is_high_volume
                            if is_up_gap:
                                if up_exhaustion:
                                    signals.append("📈 衰竭跳空(上)")
                                elif is_up_trend and is_high_volume:
                                    signals.append("📈 持續跳空(上)")
                                elif row["High"] > data["High"].iloc[index-1:index].max() and is_high_volume:
                                    signals.append("📈 突破跳空(上)")
                                else:
                                    signals.append("📈 普通跳空(上)")
                            elif is_down_gap:
                                if down_exhaustion:
                                    signals.append("📉 衰竭跳空(下)")
                                elif is_down_trend and is_high_volume:
                                    signals.append("📉 持續跳空(下)")
                                elif row["Low"] < data["Low"].iloc[index-1:index].min() and is_high_volume:
                                    signals.append("📉 突破跳空(下)")
                                else:
                                    signals.append("📉 普通跳空(下)")

                    if row["Continuous_Up"]   >= CONTINUOUS_UP_THRESHOLD:
                        signals.append("📈 連續向上買入")
                    if row["Continuous_Down"] >= CONTINUOUS_DOWN_THRESHOLD:
                        signals.append("📉 連續向下賣出")

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

                    if index > 0 and row["Close"] > row["Open"] and row["Open"] > data["Close"].iloc[index-1]:
                        signals.append("📈 新买入信号")
                    if index > 0 and row["Close"] < row["Open"] and row["Open"] < data["Close"].iloc[index-1]:
                        signals.append("📉 新卖出信号")
                    if index > 0 and abs(row["Price Change %"]) > PRICE_CHANGE_THRESHOLD and abs(row["Volume Change %"]) > VOLUME_CHANGE_THRESHOLD and row["MACD"] > row["Signal"]:
                        signals.append("🔄 新转折点")
                    if len(signals) > 8:
                        signals.append(f"🔥 关键转折点 (信号数: {len(signals)})")

                    if index > 0 and row["RSI"] < 30 and row["MACD"] > 0 and data["MACD"].iloc[index-1] <= 0:
                        signals.append("📈 RSI-MACD Oversold Crossover")
                    if index > 0 and row["EMA5"] > row["EMA10"] and row["Close"] > row["SMA50"]:
                        signals.append("📈 EMA-SMA Uptrend Buy")
                    if index > 0 and row["Volume"] > data["前5均量"].iloc[index] and row["MACD"] > 0 and data["MACD"].iloc[index-1] <= 0:
                        signals.append("📈 Volume-MACD Buy")
                    if index > 0 and row["RSI"] > 70 and row["MACD"] < 0 and data["MACD"].iloc[index-1] >= 0:
                        signals.append("📉 RSI-MACD Overbought Crossover")
                    if index > 0 and row["EMA5"] < row["EMA10"] and row["Close"] < row["SMA50"]:
                        signals.append("📉 EMA-SMA Downtrend Sell")
                    if index > 0 and row["Volume"] > data["前5均量"].iloc[index] and row["MACD"] < 0 and data["MACD"].iloc[index-1] >= 0:
                        signals.append("📉 Volume-MACD Sell")

                    if (index > 0 and row["EMA10"] > row["EMA30"] and
                            data["EMA10"].iloc[index-1] <= data["EMA30"].iloc[index-1]):
                        signals.append("📈 EMA10_30買入")
                        if row["EMA10"] > row["EMA40"]:
                            signals.append("📈 EMA10_30_40強烈買入")
                    if (index > 0 and row["EMA10"] < row["EMA30"] and
                            data["EMA10"].iloc[index-1] >= data["EMA30"].iloc[index-1]):
                        signals.append("📉 EMA10_30賣出")
                        if row["EMA10"] < row["EMA40"]:
                            signals.append("📉 EMA10_30_40強烈賣出")

                    if (index > 0 and
                            data["Close"].iloc[index-1] < data["Open"].iloc[index-1] and
                            row["Close"] > row["Open"] and
                            row["Open"] < data["Close"].iloc[index-1] and
                            row["Close"] > data["Open"].iloc[index-1] and
                            row["Volume"] > data["前5均量"].iloc[index]):
                        signals.append("📈 看漲吞沒")
                    if (index > 0 and
                            data["Close"].iloc[index-1] > data["Open"].iloc[index-1] and
                            row["Close"] < row["Open"] and
                            row["Open"] > data["Close"].iloc[index-1] and
                            row["Close"] < data["Open"].iloc[index-1] and
                            row["Volume"] > data["前5均量"].iloc[index]):
                        signals.append("📉 看跌吞沒")

                    body_size   = abs(row["Close"] - row["Open"])
                    candle_range = row["High"] - row["Low"]
                    lower_shadow = min(row["Open"], row["Close"]) - row["Low"]
                    upper_shadow = row["High"] - max(row["Open"], row["Close"])
                    if (index > 0 and candle_range > 0 and
                            body_size < candle_range * 0.3 and
                            lower_shadow >= 2 * body_size and
                            upper_shadow < lower_shadow and
                            row["Volume"] > data["前5均量"].iloc[index]):
                        if row["RSI"] < 50:
                            signals.append("📈 錘頭線")
                        else:
                            signals.append("📉 上吊線")

                    if (index > 1 and
                            data["Close"].iloc[index-2] < data["Open"].iloc[index-2] and
                            abs(data["Close"].iloc[index-1] - data["Open"].iloc[index-1]) < 0.3 * abs(data["Close"].iloc[index-2] - data["Open"].iloc[index-2]) and
                            row["Close"] > row["Open"] and
                            row["Close"] > (data["Open"].iloc[index-2] + data["Close"].iloc[index-2]) / 2 and
                            row["Volume"] > data["前5均量"].iloc[index]):
                        signals.append("📈 早晨之星")
                    if (index > 1 and
                            data["Close"].iloc[index-2] > data["Open"].iloc[index-2] and
                            abs(data["Close"].iloc[index-1] - data["Open"].iloc[index-1]) < 0.3 * abs(data["Close"].iloc[index-2] - data["Open"].iloc[index-2]) and
                            row["Close"] < row["Open"] and
                            row["Close"] < (data["Open"].iloc[index-2] + data["Close"].iloc[index-2]) / 2 and
                            row["Volume"] > data["前5均量"].iloc[index]):
                        signals.append("📉 黃昏之星")

                    if index > 0 and row["High"] > data["High_Max"].iloc[index-1]:
                        signals.append("📈 BreakOut_5K")
                    if index > 0 and row["Low"] < data["Low_Min"].iloc[index-1]:
                        signals.append("📉 BreakDown_5K")

                    if (index > 0 and
                            data["Close"].iloc[index-1] > data["Open"].iloc[index-1] and
                            row["Open"] > data["Close"].iloc[index-1] and
                            row["Close"] < row["Open"] and
                            row["Close"] < (data["Open"].iloc[index-1] + data["Close"].iloc[index-1]) / 2 and
                            row["Volume"] > data["前5均量"].iloc[index]):
                        signals.append("📉 烏雲蓋頂")
                    if (index > 0 and
                            data["Close"].iloc[index-1] < data["Open"].iloc[index-1] and
                            row["Open"] < data["Close"].iloc[index-1] and
                            row["Close"] > row["Open"] and
                            row["Close"] > (data["Open"].iloc[index-1] + data["Close"].iloc[index-1]) / 2 and
                            row["Volume"] > data["前5均量"].iloc[index]):
                        signals.append("📈 刺透形態")

                    if index > 0 and pd.notna(row["VWAP"]):
                        if row["Close"] > row["VWAP"] and data["Close"].iloc[index-1] <= data["VWAP"].iloc[index-1]:
                            signals.append("📈 VWAP買入")
                        elif row["Close"] < row["VWAP"] and data["Close"].iloc[index-1] >= data["VWAP"].iloc[index-1]:
                            signals.append("📉 VWAP賣出")

                    if index >= MFI_DIVERGENCE_WINDOW and pd.notna(row["MFI"]):
                        if data["MFI_Bull_Div"].iloc[index]:
                            signals.append("📈 MFI牛背離買入")
                        if data["MFI_Bear_Div"].iloc[index]:
                            signals.append("📉 MFI熊背離賣出")

                    if index > 0 and pd.notna(row["OBV"]):
                        if row["Close"] > data["Close"].iloc[index-1] and row["OBV"] > data["OBV_Roll_Max"].iloc[index-1]:
                            signals.append("📈 OBV突破買入")
                        elif row["Close"] < data["Close"].iloc[index-1] and row["OBV"] < data["OBV_Roll_Min"].iloc[index-1]:
                            signals.append("📉 OBV突破賣出")

                    if index > 0 and pd.notna(row["VIX"]):
                        vix_prev = data["VIX"].iloc[index-1]
                        if row["VIX"] > VIX_HIGH_THRESHOLD and row["VIX"] > vix_prev:
                            signals.append("📉 VIX恐慌賣出")
                        elif row["VIX"] < VIX_LOW_THRESHOLD and row["VIX"] < vix_prev:
                            signals.append("📈 VIX平靜買入")

                    if index > 0 and pd.notna(row["VIX_EMA_Fast"]) and pd.notna(row["VIX_EMA_Slow"]):
                        if row["VIX_EMA_Fast"] > row["VIX_EMA_Slow"] and data["VIX_EMA_Fast"].iloc[index-1] <= data["VIX_EMA_Slow"].iloc[index-1]:
                            signals.append("📉 VIX上升趨勢賣出")
                        elif row["VIX_EMA_Fast"] < row["VIX_EMA_Slow"] and data["VIX_EMA_Fast"].iloc[index-1] >= data["VIX_EMA_Slow"].iloc[index-1]:
                            signals.append("📈 VIX下降趨勢買入")

                    return ", ".join(signals) if signals else ""

                data["異動標記"] = [mark_signal(row, i) for i, row in data.iterrows()]
                data = compute_kline_patterns(data, BODY_RATIO_THRESHOLD, SHADOW_RATIO_THRESHOLD, DOJI_BODY_THRESHOLD)

                # 成交密集區
                dense_areas = calculate_volume_profile_dense_areas(
                    data, bins=VOLUME_PROFILE_BINS,
                    window=VOLUME_PROFILE_WINDOW, top_n=VOLUME_PROFILE_TOP_N)
                latest_close = data["Close"].iloc[-1]
                near_dense = False
                near_dense_info = ""
                if dense_areas:
                    for area in dense_areas:
                        if area["price_low"] <= latest_close <= area["price_high"]:
                            near_dense = True
                            near_dense_info = f"目前位於成交密集區內 ({area['price_low']:.2f} ~ {area['price_high']:.2f})"
                            break
                        dist_pct = abs(latest_close - area["price_center"]) / area["price_center"] * 100
                        if dist_pct <= 1.0:
                            near_dense = True
                            near_dense_info = f"接近成交密集區中心 {area['price_center']:.2f} ({dist_pct:.2f}% 距離)"
                            break

                # [WARN5] 修復：dense_desc 整合進每個 return
                def generate_comprehensive_interpretation(data):
                    last_5 = data.tail(5)
                    if len(last_5) < 5:
                        return "數據不足，無法生成綜合解讀"
                    bullish_count = len(last_5[last_5["K線形態"].isin(["錘子線","大陽線","看漲吞噬","刺透形態","早晨之星"])])
                    bearish_count = len(last_5[last_5["K線形態"].isin(["射擊之星","大陰線","看跌吞噬","烏雲蓋頂","黃昏之星"])])
                    neutral_count = len(last_5[last_5["K線形態"].isin(["十字星","普通K線"])])
                    high_volume_count = len(last_5[last_5["成交量標記"] == "放量"])

                    vwap_val = last_5["VWAP"].iloc[-1]
                    close_val = last_5["Close"].iloc[-1]
                    vwap_trend = "多頭（價格>VWAP）" if pd.notna(vwap_val) and close_val > vwap_val else "空頭（價格<VWAP）"
                    mfi_val = last_5["MFI"].iloc[-1]
                    mfi_level = f"MFI={mfi_val:.1f}（{'超賣背離機會' if mfi_val < 20 else '超買背離風險' if mfi_val > 80 else '中性'}）" if pd.notna(mfi_val) else "MFI=N/A"
                    obv_trend = "OBV上漲確認量能" if last_5["OBV"].iloc[-1] > last_5["OBV"].iloc[0] else "OBV下跌警示量能不足"
                    vix_val = last_5["VIX"].iloc[-1]
                    vix_level = f"VIX={vix_val:.1f}（{'恐慌高位' if vix_val > VIX_HIGH_THRESHOLD else '平靜低位' if vix_val < VIX_LOW_THRESHOLD else '中性'}）" if pd.notna(vix_val) else "VIX=N/A"
                    vix_ema_f = last_5["VIX_EMA_Fast"].iloc[-1]
                    vix_ema_s = last_5["VIX_EMA_Slow"].iloc[-1]
                    vix_trend = "VIX趨勢上升（EMA Fast > Slow）" if pd.notna(vix_ema_f) and vix_ema_f > vix_ema_s else "VIX趨勢下降（EMA Fast < Slow）"

                    # [WARN5] 修復：dense_desc 在每個 return 中都包含
                    dense_desc = ""
                    if dense_areas:
                        centers = [f"{a['price_center']:.2f}" for a in dense_areas]
                        dense_desc = f"，成交密集區在 {', '.join(centers)} 等價位（潛在強支撐/壓力）"

                    base = f"{vwap_trend}，{mfi_level}，{obv_trend}，{vix_level}，{vix_trend}{dense_desc}"

                    if bullish_count >= 3 and high_volume_count >= 3:
                        return f"最近五日多方主導，出現多根看漲形態且多伴隨放量，市場呈強勢上漲趨勢，{base}，建議關注買入機會。"
                    elif bearish_count >= 3 and high_volume_count >= 3:
                        return f"最近五日空方主導，出現多根看跌形態且多伴隨放量，市場呈強勢下跌趨勢，{base}，建議注意賣出風險。"
                    elif neutral_count >= 3:
                        return f"最近五日多空交戰，型態以十字星或普通K線為主，市場處於盤整或方向不明，{base}。"
                    elif bullish_count >= 2 and bearish_count >= 2:
                        return f"最近五日多空激烈爭奪，看漲與看跌形態交替，建議觀望，{base}。"
                    else:
                        return f"最近五日市場型態無明顯趨勢，建議持續觀察後續動向，{base}。"

                comprehensive_interpretation = generate_comprehensive_interpretation(data)

                # 當前數據
                current_price  = data["Close"].iloc[-1]
                previous_close = stock.info.get("previousClose", current_price)
                price_change   = current_price - previous_close
                price_pct_change = (price_change / previous_close) * 100 if previous_close else 0
                last_volume  = data["Volume"].iloc[-1]
                prev_volume  = data["Volume"].iloc[-2] if len(data) > 1 else last_volume
                volume_change     = last_volume - prev_volume
                volume_pct_change = (volume_change / prev_volume) * 100 if prev_volume else 0

                # 信號檢測（最新一根）
                low_high_signal  = len(data) > 1 and data["Low"].iloc[-1]  > data["High"].iloc[-2]
                high_low_signal  = len(data) > 1 and data["High"].iloc[-1] < data["Low"].iloc[-2]
                macd_buy_signal  = len(data) > 1 and data["MACD"].iloc[-1] > 0  and data["MACD"].iloc[-2] <= 0
                macd_sell_signal = len(data) > 1 and data["MACD"].iloc[-1] <= 0 and data["MACD"].iloc[-2] > 0
                ema_buy_signal   = (len(data) > 1 and
                                    data["EMA5"].iloc[-1] > data["EMA10"].iloc[-1] and
                                    data["EMA5"].iloc[-2] <= data["EMA10"].iloc[-2] and
                                    data["Volume"].iloc[-1] > data["Volume"].iloc[-2])
                ema_sell_signal  = (len(data) > 1 and
                                    data["EMA5"].iloc[-1] < data["EMA10"].iloc[-1] and
                                    data["EMA5"].iloc[-2] >= data["EMA10"].iloc[-2] and
                                    data["Volume"].iloc[-1] > data["Volume"].iloc[-2])
                price_trend_buy_signal  = (len(data) > 1 and data["High"].iloc[-1] > data["High"].iloc[-2] and data["Low"].iloc[-1] > data["Low"].iloc[-2] and data["Close"].iloc[-1] > data["Close"].iloc[-2])
                price_trend_sell_signal = (len(data) > 1 and data["High"].iloc[-1] < data["High"].iloc[-2] and data["Low"].iloc[-1] < data["Low"].iloc[-2] and data["Close"].iloc[-1] < data["Close"].iloc[-2])
                price_trend_vol_buy_signal  = price_trend_buy_signal  and data["Volume"].iloc[-1] > data["前5均量"].iloc[-1]
                price_trend_vol_sell_signal = price_trend_sell_signal and data["Volume"].iloc[-1] > data["前5均量"].iloc[-1]
                price_trend_vol_pct_buy_signal  = price_trend_buy_signal  and data["Volume Change %"].iloc[-1] > 15
                price_trend_vol_pct_sell_signal = price_trend_sell_signal and data["Volume Change %"].iloc[-1] > 15
                new_buy_signal  = len(data) > 1 and data["Close"].iloc[-1] > data["Open"].iloc[-1] and data["Open"].iloc[-1] > data["Close"].iloc[-2]
                new_sell_signal = len(data) > 1 and data["Close"].iloc[-1] < data["Open"].iloc[-1] and data["Open"].iloc[-1] < data["Close"].iloc[-2]
                new_pivot_signal = (len(data) > 1 and abs(data["Price Change %"].iloc[-1]) > PRICE_CHANGE_THRESHOLD and abs(data["Volume Change %"].iloc[-1]) > VOLUME_CHANGE_THRESHOLD)
                ema10_30_buy_signal           = len(data) > 1 and data["EMA10"].iloc[-1] > data["EMA30"].iloc[-1] and data["EMA10"].iloc[-2] <= data["EMA30"].iloc[-2]
                ema10_30_40_strong_buy_signal = ema10_30_buy_signal  and data["EMA10"].iloc[-1] > data["EMA40"].iloc[-1]
                ema10_30_sell_signal           = len(data) > 1 and data["EMA10"].iloc[-1] < data["EMA30"].iloc[-1] and data["EMA10"].iloc[-2] >= data["EMA30"].iloc[-2]
                ema10_30_40_strong_sell_signal = ema10_30_sell_signal and data["EMA10"].iloc[-1] < data["EMA40"].iloc[-1]
                bullish_engulfing = (len(data) > 1 and data["Close"].iloc[-2] < data["Open"].iloc[-2] and data["Close"].iloc[-1] > data["Open"].iloc[-1] and data["Open"].iloc[-1] < data["Close"].iloc[-2] and data["Close"].iloc[-1] > data["Open"].iloc[-2] and data["Volume"].iloc[-1] > data["前5均量"].iloc[-1])
                bearish_engulfing = (len(data) > 1 and data["Close"].iloc[-2] > data["Open"].iloc[-2] and data["Close"].iloc[-1] < data["Open"].iloc[-1] and data["Open"].iloc[-1] > data["Close"].iloc[-2] and data["Close"].iloc[-1] < data["Open"].iloc[-2] and data["Volume"].iloc[-1] > data["前5均量"].iloc[-1])
                _b = abs(data["Close"].iloc[-1] - data["Open"].iloc[-1])
                _r = data["High"].iloc[-1] - data["Low"].iloc[-1]
                _ls = min(data["Open"].iloc[-1], data["Close"].iloc[-1]) - data["Low"].iloc[-1]
                _us = data["High"].iloc[-1] - max(data["Open"].iloc[-1], data["Close"].iloc[-1])
                hammer      = len(data) > 1 and _r > 0 and _b < _r*0.3 and _ls >= 2*_b and _us < _ls and data["Volume"].iloc[-1] > data["前5均量"].iloc[-1] and data["RSI"].iloc[-1] < 50
                hanging_man = len(data) > 1 and _r > 0 and _b < _r*0.3 and _ls >= 2*_b and _us < _ls and data["Volume"].iloc[-1] > data["前5均量"].iloc[-1] and data["RSI"].iloc[-1] > 50
                morning_star = (len(data) > 2 and data["Close"].iloc[-3] < data["Open"].iloc[-3] and abs(data["Close"].iloc[-2]-data["Open"].iloc[-2]) < 0.3*abs(data["Close"].iloc[-3]-data["Open"].iloc[-3]) and data["Close"].iloc[-1] > data["Open"].iloc[-1] and data["Close"].iloc[-1] > (data["Open"].iloc[-3]+data["Close"].iloc[-3])/2 and data["Volume"].iloc[-1] > data["前5均量"].iloc[-1])
                evening_star = (len(data) > 2 and data["Close"].iloc[-3] > data["Open"].iloc[-3] and abs(data["Close"].iloc[-2]-data["Open"].iloc[-2]) < 0.3*abs(data["Close"].iloc[-3]-data["Open"].iloc[-3]) and data["Close"].iloc[-1] < data["Open"].iloc[-1] and data["Close"].iloc[-1] < (data["Open"].iloc[-3]+data["Close"].iloc[-3])/2 and data["Volume"].iloc[-1] > data["前5均量"].iloc[-1])
                vwap_buy_signal  = len(data) > 1 and pd.notna(data["VWAP"].iloc[-1]) and data["Close"].iloc[-1] > data["VWAP"].iloc[-1] and data["Close"].iloc[-2] <= data["VWAP"].iloc[-2]
                vwap_sell_signal = len(data) > 1 and pd.notna(data["VWAP"].iloc[-1]) and data["Close"].iloc[-1] < data["VWAP"].iloc[-1] and data["Close"].iloc[-2] >= data["VWAP"].iloc[-2]
                mfi_bull_divergence = len(data) > MFI_DIVERGENCE_WINDOW and bool(data["MFI_Bull_Div"].iloc[-1])
                mfi_bear_divergence = len(data) > MFI_DIVERGENCE_WINDOW and bool(data["MFI_Bear_Div"].iloc[-1])
                obv_breakout_buy  = len(data) > 1 and data["Close"].iloc[-1] > data["Close"].iloc[-2] and data["OBV"].iloc[-1] > data["OBV_Roll_Max"].iloc[-2]
                obv_breakout_sell = len(data) > 1 and data["Close"].iloc[-1] < data["Close"].iloc[-2] and data["OBV"].iloc[-1] < data["OBV_Roll_Min"].iloc[-2]
                vix_panic_sell  = len(data) > 1 and pd.notna(data["VIX"].iloc[-1]) and data["VIX"].iloc[-1] > VIX_HIGH_THRESHOLD and data["VIX"].iloc[-1] > data["VIX"].iloc[-2]
                vix_calm_buy    = len(data) > 1 and pd.notna(data["VIX"].iloc[-1]) and data["VIX"].iloc[-1] < VIX_LOW_THRESHOLD  and data["VIX"].iloc[-1] < data["VIX"].iloc[-2]
                vix_uptrend_sell  = len(data) > 1 and pd.notna(data["VIX_EMA_Fast"].iloc[-1]) and data["VIX_EMA_Fast"].iloc[-1] > data["VIX_EMA_Slow"].iloc[-1] and data["VIX_EMA_Fast"].iloc[-2] <= data["VIX_EMA_Slow"].iloc[-2]
                vix_downtrend_buy = len(data) > 1 and pd.notna(data["VIX_EMA_Fast"].iloc[-1]) and data["VIX_EMA_Fast"].iloc[-1] < data["VIX_EMA_Slow"].iloc[-1] and data["VIX_EMA_Fast"].iloc[-2] >= data["VIX_EMA_Slow"].iloc[-2]

                # 跳空信號檢測
                gap_common_up = gap_common_down = False
                gap_breakaway_up = gap_breakaway_down = False
                gap_runaway_up = gap_runaway_down = False
                gap_exhaustion_up = gap_exhaustion_down = False
                if len(data) > 1:
                    gap_pct = ((data["Open"].iloc[-1] - data["Close"].iloc[-2]) / data["Close"].iloc[-2]) * 100
                    is_up_gap   = gap_pct >  GAP_THRESHOLD
                    is_down_gap = gap_pct < -GAP_THRESHOLD
                    if is_up_gap or is_down_gap:
                        trend      = data["Close"].iloc[-5:].mean()  if len(data) >= 5 else 0
                        prev_trend = data["Close"].iloc[-6:-1].mean() if len(data) >= 6 else trend
                        is_up_trend   = data["Close"].iloc[-1] > trend and trend > prev_trend
                        is_down_trend = data["Close"].iloc[-1] < trend and trend < prev_trend
                        is_high_volume = data["Volume"].iloc[-1] > data["前5均量"].iloc[-1]
                        # [BUG5] 當日收盤拉回確認
                        up_exhaustion   = is_up_gap   and data["Close"].iloc[-1] < data["Open"].iloc[-1] and is_high_volume
                        down_exhaustion = is_down_gap and data["Close"].iloc[-1] > data["Open"].iloc[-1] and is_high_volume
                        if is_up_gap:
                            if up_exhaustion:       gap_exhaustion_up  = True
                            elif is_up_trend and is_high_volume:  gap_runaway_up    = True
                            elif data["High"].iloc[-1] > data["High"].iloc[-2:-1].max() and is_high_volume: gap_breakaway_up = True
                            else: gap_common_up = True
                        elif is_down_gap:
                            if down_exhaustion:     gap_exhaustion_down  = True
                            elif is_down_trend and is_high_volume: gap_runaway_down  = True
                            elif data["Low"].iloc[-1] < data["Low"].iloc[-2:-1].min() and is_high_volume: gap_breakaway_down = True
                            else: gap_common_down = True

                continuous_up_buy_signal   = data["Continuous_Up"].iloc[-1]   >= CONTINUOUS_UP_THRESHOLD
                continuous_down_sell_signal = data["Continuous_Down"].iloc[-1] >= CONTINUOUS_DOWN_THRESHOLD
                sma50_up_trend = sma50_down_trend = sma50_200_up_trend = sma50_200_down_trend = False
                if pd.notna(data["SMA50"].iloc[-1]):
                    sma50_up_trend   = data["Close"].iloc[-1] > data["SMA50"].iloc[-1]
                    sma50_down_trend = data["Close"].iloc[-1] < data["SMA50"].iloc[-1]
                if pd.notna(data["SMA50"].iloc[-1]) and pd.notna(data["SMA200"].iloc[-1]):
                    sma50_200_up_trend   = data["Close"].iloc[-1] > data["SMA50"].iloc[-1] > data["SMA200"].iloc[-1]
                    sma50_200_down_trend = data["Close"].iloc[-1] < data["SMA50"].iloc[-1] < data["SMA200"].iloc[-1]

                # 顯示指標
                st.metric(f"{ticker} 🟢 股價變動", f"${current_price:.2f}", f"{price_change:.2f} ({price_pct_change:.2f}%)")
                st.metric(f"{ticker} 🔵 成交量變動", f"{last_volume:,}", f"{volume_change:,} ({volume_pct_change:.2f}%)")
                if pd.notna(data["VIX"].iloc[-1]):
                    st.metric(f"{ticker} ⚡ VIX", f"{data['VIX'].iloc[-1]:.2f}",
                              f"{data['VIX Change %'].iloc[-1]:.2f}%" if pd.notna(data["VIX Change %"].iloc[-1]) else "N/A")

                # 信號成功率
                success_rates = calculate_signal_success_rate(data)
                st.subheader(f"📊 {ticker} 各信号成功率")
                success_data = []
                for signal, metrics in success_rates.items():
                    sr = metrics["success_rate"]; tot = metrics["total_signals"]; dire = metrics["direction"]
                    success_data.append({"信号": signal, "成功率 (%)": f"{sr:.2f}%", "触发次数": tot,
                                         "成功定义": "下跌確認" if dire == "down" else "上漲確認"})
                    st.metric(f"{ticker} {signal} 成功率", f"{sr:.2f}%", f"基于 {tot} 次({'下跌' if dire=='down' else '上涨'})")
                    if 0 < tot < 5:
                        st.warning(f"⚠️ {ticker} {signal} 样本量过少（{tot} 次），成功率可能不稳定")
                if success_data:
                    st.dataframe(pd.DataFrame(success_data), use_container_width=True)

                st.subheader(f"📝 {ticker} 綜合解讀")
                st.write(comprehensive_interpretation)

                # 異動提醒判斷
                any_signal = any([
                    abs(price_pct_change) >= PRICE_THRESHOLD and abs(volume_pct_change) >= VOLUME_THRESHOLD,
                    low_high_signal, high_low_signal, macd_buy_signal, macd_sell_signal,
                    ema_buy_signal, ema_sell_signal, price_trend_buy_signal, price_trend_sell_signal,
                    price_trend_vol_buy_signal, price_trend_vol_sell_signal,
                    price_trend_vol_pct_buy_signal, price_trend_vol_pct_sell_signal,
                    gap_common_up, gap_common_down, gap_breakaway_up, gap_breakaway_down,
                    gap_runaway_up, gap_runaway_down, gap_exhaustion_up, gap_exhaustion_down,
                    continuous_up_buy_signal, continuous_down_sell_signal,
                    sma50_up_trend, sma50_down_trend, sma50_200_up_trend, sma50_200_down_trend,
                    new_buy_signal, new_sell_signal, new_pivot_signal,
                    ema10_30_buy_signal, ema10_30_40_strong_buy_signal,
                    ema10_30_sell_signal, ema10_30_40_strong_sell_signal,
                    bullish_engulfing, bearish_engulfing, hammer, hanging_man, morning_star, evening_star,
                    vwap_buy_signal, vwap_sell_signal, mfi_bull_divergence, mfi_bear_divergence,
                    obv_breakout_buy, obv_breakout_sell, vix_panic_sell, vix_calm_buy,
                    vix_uptrend_sell, vix_downtrend_buy,
                ])

                if any_signal:
                    alert_msg = f"{ticker} 異動：價格 {price_pct_change:.2f}%、成交量 {volume_pct_change:.2f}%"
                    signal_texts = {
                        low_high_signal:  "，當前最低價高於前一時段最高價",
                        high_low_signal:  "，當前最高價低於前一時段最低價",
                        macd_buy_signal:  "，MACD 買入訊號",
                        macd_sell_signal: "，MACD 賣出訊號",
                        ema_buy_signal:   "，EMA 買入訊號",
                        ema_sell_signal:  "，EMA 賣出訊號",
                        price_trend_buy_signal:  "，價格趨勢買入",
                        price_trend_sell_signal: "，價格趨勢賣出",
                        price_trend_vol_buy_signal:  "，價格趨勢買入（量）",
                        price_trend_vol_sell_signal: "，價格趨勢賣出（量）",
                        price_trend_vol_pct_buy_signal:  "，價格趨勢買入（量%）",
                        price_trend_vol_pct_sell_signal: "，價格趨勢賣出（量%）",
                        gap_common_up:      "，普通跳空(上)",    gap_common_down:    "，普通跳空(下)",
                        gap_breakaway_up:   "，突破跳空(上)",    gap_breakaway_down: "，突破跳空(下)",
                        gap_runaway_up:     "，持續跳空(上)",    gap_runaway_down:   "，持續跳空(下)",
                        gap_exhaustion_up:  "，衰竭跳空(上)",    gap_exhaustion_down:"，衰竭跳空(下)",
                        continuous_up_buy_signal:    f"，連續向上買入（{CONTINUOUS_UP_THRESHOLD}根）",
                        continuous_down_sell_signal: f"，連續向下賣出（{CONTINUOUS_DOWN_THRESHOLD}根）",
                        sma50_up_trend:     "，SMA50上升趨勢",   sma50_down_trend:   "，SMA50下降趨勢",
                        sma50_200_up_trend: "，SMA50_200上升趨勢", sma50_200_down_trend:"，SMA50_200下降趨勢",
                        new_buy_signal:    "，新买入信号",        new_sell_signal:   "，新卖出信号",
                        new_pivot_signal:  f"，新转折点（|Price|>{PRICE_CHANGE_THRESHOLD}% 且 |Vol|>{VOLUME_CHANGE_THRESHOLD}%）",
                        ema10_30_buy_signal:           "，EMA10_30買入",
                        ema10_30_40_strong_buy_signal: "，EMA10_30_40強烈買入",
                        ema10_30_sell_signal:           "，EMA10_30賣出",
                        ema10_30_40_strong_sell_signal: "，EMA10_30_40強烈賣出",
                        bullish_engulfing: "，看漲吞沒", bearish_engulfing: "，看跌吞沒",
                        hammer:       "，錘頭線",    hanging_man:  "，上吊線",
                        morning_star: "，早晨之星",  evening_star: "，黃昏之星",
                        vwap_buy_signal:  "，VWAP買入",  vwap_sell_signal: "，VWAP賣出",
                        mfi_bull_divergence: "，MFI牛背離買入", mfi_bear_divergence: "，MFI熊背離賣出",
                        obv_breakout_buy:  "，OBV突破買入",  obv_breakout_sell: "，OBV突破賣出",
                        vix_panic_sell:   "，VIX恐慌賣出（VIX>30且上升）",
                        vix_calm_buy:     "，VIX平靜買入（VIX<20且下降）",
                        vix_uptrend_sell: "，VIX上升趨勢賣出", vix_downtrend_buy:"，VIX下降趨勢買入",
                    }
                    for cond, txt in signal_texts.items():
                        if cond:
                            alert_msg += txt
                    if near_dense:
                        alert_msg += f"，{near_dense_info}（潛在強支撐/壓力）"
                    if data["K線形態"].iloc[-1] != "普通K線":
                        alert_msg += f"，最新K線：{data['K線形態'].iloc[-1]}（{data['單根解讀'].iloc[-1]}）"
                    st.warning(f"📣 {alert_msg}")
                    st.toast(f"📣 {alert_msg}")
                    send_email_alert(ticker, price_pct_change, volume_pct_change,
                        low_high_signal=low_high_signal, high_low_signal=high_low_signal,
                        macd_buy_signal=macd_buy_signal, macd_sell_signal=macd_sell_signal,
                        ema_buy_signal=ema_buy_signal, ema_sell_signal=ema_sell_signal,
                        price_trend_buy_signal=price_trend_buy_signal, price_trend_sell_signal=price_trend_sell_signal,
                        price_trend_vol_buy_signal=price_trend_vol_buy_signal, price_trend_vol_sell_signal=price_trend_vol_sell_signal,
                        price_trend_vol_pct_buy_signal=price_trend_vol_pct_buy_signal, price_trend_vol_pct_sell_signal=price_trend_vol_pct_sell_signal,
                        gap_common_up=gap_common_up, gap_common_down=gap_common_down,
                        gap_breakaway_up=gap_breakaway_up, gap_breakaway_down=gap_breakaway_down,
                        gap_runaway_up=gap_runaway_up, gap_runaway_down=gap_runaway_down,
                        gap_exhaustion_up=gap_exhaustion_up, gap_exhaustion_down=gap_exhaustion_down,
                        continuous_up_buy_signal=continuous_up_buy_signal, continuous_down_sell_signal=continuous_down_sell_signal,
                        sma50_up_trend=sma50_up_trend, sma50_down_trend=sma50_down_trend,
                        sma50_200_up_trend=sma50_200_up_trend, sma50_200_down_trend=sma50_200_down_trend,
                        new_buy_signal=new_buy_signal, new_sell_signal=new_sell_signal, new_pivot_signal=new_pivot_signal,
                        ema10_30_buy_signal=ema10_30_buy_signal, ema10_30_40_strong_buy_signal=ema10_30_40_strong_buy_signal,
                        ema10_30_sell_signal=ema10_30_sell_signal, ema10_30_40_strong_sell_signal=ema10_30_40_strong_sell_signal,
                        bullish_engulfing=bullish_engulfing, bearish_engulfing=bearish_engulfing,
                        hammer=hammer, hanging_man=hanging_man, morning_star=morning_star, evening_star=evening_star,
                        vwap_buy_signal=vwap_buy_signal, vwap_sell_signal=vwap_sell_signal,
                        mfi_bull_divergence=mfi_bull_divergence, mfi_bear_divergence=mfi_bear_divergence,
                        obv_breakout_buy=obv_breakout_buy, obv_breakout_sell=obv_breakout_sell,
                        vix_panic_sell=vix_panic_sell, vix_calm_buy=vix_calm_buy,
                        vix_uptrend_sell=vix_uptrend_sell, vix_downtrend_buy=vix_downtrend_buy)

                    # Telegram 發送
                    if abs(data["📈 股價漲跌幅 (%)"].iloc[-1]) >= PRICE_THRESHOLD and abs(data["📊 成交量變動幅 (%)"].iloc[-1]) >= VOLUME_THRESHOLD:
                        alertmsg = f"量價齊揚: {data['Datetime'].iloc[-1]} {ticker}:{selected_interval}:${data['Close'].iloc[-1]:.2f} *{data['異動標記'].iloc[-1]}*{data['成交量標記'].iloc[-1]}*{data['K線形態'].iloc[-1]}*{data['單根解讀'].iloc[-1]}*"
                        send_telegram_alert(alertmsg)
                    if data["Close_N_High"].iloc[-1] >= HIGH_N_HIGH_THRESHOLD:
                        alertmsg = f"有機會再破新高: {data['Datetime'].iloc[-1]} {ticker}:{selected_interval}:${data['High'].iloc[-1]:.2f} *{data['異動標記'].iloc[-1]}*{data['成交量標記'].iloc[-1]}*{data['K線形態'].iloc[-1]}*"
                        send_telegram_alert(alertmsg)
                    if data["High"].iloc[-1] >= data["High_Max"].iloc[-1]:
                        alertmsg = f"破{BREAKOUT_WINDOW}K新高: {data['Datetime'].iloc[-1]} {ticker}:{selected_interval}:${data['Close'].iloc[-1]:.2f} *{data['異動標記'].iloc[-1]}*{data['成交量標記'].iloc[-1]}*{data['K線形態'].iloc[-1]}*"
                        send_telegram_alert(alertmsg)
                    if data["Low"].iloc[-1] <= data["Low_Min"].iloc[-1]:
                        alertmsg = f"穿{BREAKOUT_WINDOW}K新低: {data['Datetime'].iloc[-1]} {ticker}:{selected_interval}:${data['Close'].iloc[-1]:.2f} *{data['異動標記'].iloc[-1]}*{data['成交量標記'].iloc[-1]}*{data['K線形態'].iloc[-1]}*"
                        send_telegram_alert(alertmsg)

                    # [NOTE1] 修復：matched_rank 初始化為 None，防止 NameError
                    matched_rank = None
                    if len(data["異動標記"]) > 0:
                        K_signals = str(data["異動標記"].iloc[-1])
                        K_signals_list = [s.strip() for s in K_signals.split(", ") if s.strip()]
                        current_volume_mark   = data["成交量標記"].iloc[-1]
                        current_kline_pattern = data["K線形態"].iloc[-1]
                        for _, trow in telegram_conditions.iterrows():
                            required_signals = [s.strip() for s in str(trow["異動標記"]).split(", ") if s.strip()]
                            if (all(sig in K_signals_list for sig in required_signals) and
                                    current_volume_mark == trow["成交量標記"] and
                                    current_kline_pattern == trow["K線形態"]):
                                matched_rank = trow["排名"]
                                break
                        if matched_rank is not None:
                            alertmsg = f"1D BUY 趨勢反轉: {data['Datetime'].iloc[-1]} {ticker}:{selected_interval}:${data['Close'].iloc[-1]:.2f} *{data['異動標記'].iloc[-1]}*{data['成交量標記'].iloc[-1]}*{data['K線形態'].iloc[-1]}* 匹配排名{matched_rank}"
                            send_telegram_alert(alertmsg)
                        # 推送選定訊號
                        for sig in selected_signals:
                            if sig in K_signals_list:
                                alertmsg = f"📡 訊號觸發 [{sig}]: {data['Datetime'].iloc[-1]} {ticker}:{selected_interval}:${data['Close'].iloc[-1]:.2f} *{data['K線形態'].iloc[-1]}*{data['成交量標記'].iloc[-1]}*"
                                send_telegram_alert(alertmsg)

                # ==================== K線圖（NOTE3：tail(60)）====================
                st.subheader(f"📈 {ticker} K線圖與技術指標")
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                # [NOTE3] 修復：改用 tail(60)，讓 EMA30/SMA50 有足夠歷史
                CHART_TAIL = 60
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                    subplot_titles=(f"{ticker} K線與EMA/VWAP","成交量/OBV","RSI/MFI"),
                                    vertical_spacing=0.1, row_heights=[0.5, 0.2, 0.3])
                fig.add_trace(go.Candlestick(
                    x=data.tail(CHART_TAIL)["Datetime"],
                    open=data.tail(CHART_TAIL)["Open"],
                    high=data.tail(CHART_TAIL)["High"],
                    low=data.tail(CHART_TAIL)["Low"],
                    close=data.tail(CHART_TAIL)["Close"],
                    name="K線"), row=1, col=1)
                for col_name, color in [("EMA5","orange"),("EMA10","blue"),("EMA30","purple"),("EMA40","brown")]:
                    fig.add_trace(go.Scatter(x=data.tail(CHART_TAIL)["Datetime"], y=data.tail(CHART_TAIL)[col_name],
                                             mode="lines", name=col_name, line=dict(color=color, width=1.2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=data.tail(CHART_TAIL)["Datetime"], y=data.tail(CHART_TAIL)["VWAP"],
                                         mode="lines", name="VWAP", line=dict(color="purple", width=2)), row=1, col=1)

                # 成交密集區
                if VOLUME_PROFILE_SHOW_ON_CHART and dense_areas and len(data) >= 50:
                    x0 = data["Datetime"].iloc[-50]
                    x1 = data["Datetime"].iloc[-1]
                    for i_area, area in enumerate(dense_areas):
                        fig.add_hrect(y0=area["price_low"], y1=area["price_high"],
                                      x0=x0, x1=x1,
                                      fillcolor="rgba(255,165,0,0.15)", line_width=0, row=1, col=1)
                        pos = "left" if i_area % 2 == 0 else "right"
                        fig.add_hline(y=area["price_center"], line_dash="dot", line_color="red",
                                      annotation_text=f"{area['price_center']:.2f}",
                                      annotation_position=pos,
                                      annotation_font=dict(color="red", size=12), row=1, col=1)

                fig.add_bar(x=data.tail(CHART_TAIL)["Datetime"], y=data.tail(CHART_TAIL)["Volume"],
                            name="成交量", opacity=0.5, row=2, col=1)
                fig.add_trace(go.Scatter(x=data.tail(CHART_TAIL)["Datetime"], y=data.tail(CHART_TAIL)["OBV"],
                                         mode="lines", name="OBV", line=dict(color="orange", width=2)), row=2, col=1)
                fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
                fig.add_trace(go.Scatter(x=data.tail(CHART_TAIL)["Datetime"], y=data.tail(CHART_TAIL)["RSI"],
                                         mode="lines", name="RSI", line=dict(color="blue", width=1.2)), row=3, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red",   row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                fig.add_trace(go.Scatter(x=data.tail(CHART_TAIL)["Datetime"], y=data.tail(CHART_TAIL)["MFI"],
                                         mode="lines", name="MFI", line=dict(color="brown", width=2)), row=3, col=1)

                # 標記訊號
                for i in range(1, min(CHART_TAIL, len(data))):
                    idx = -CHART_TAIL + i
                    sig_str = str(data["異動標記"].iloc[idx])
                    annotations = {
                        "EMA買入":         ("📈 EMA買入",    -30),
                        "EMA賣出":         ("📉 EMA賣出",     30),
                        "新买入信号":       ("📈 新买入",     -30),
                        "新卖出信号":       ("📉 新卖出",      30),
                        "EMA10_30買入":    ("📈 EMA10_30",   -30),
                        "EMA10_30_40強烈買入": ("📈 10_30_40強買", -50),
                        "EMA10_30賣出":    ("📉 EMA10_30",    30),
                        "EMA10_30_40強烈賣出": ("📉 10_30_40強賣", 50),
                        "看漲吞沒":        ("📈 看漲",        -30),
                        "看跌吞沒":        ("📉 看跌",         30),
                        "錘頭線":          ("📈 錘頭",        -30),
                        "上吊線":          ("📉 上吊",         30),
                        "早晨之星":        ("📈 早晨",        -30),
                        "黃昏之星":        ("📉 黃昏",         30),
                        "VWAP買入":        ("📈 VWAP",       -30),
                        "VWAP賣出":        ("📉 VWAP",        30),
                        "VIX恐慌賣出":     ("📉 VIX恐慌",     30),
                        "VIX平靜買入":     ("📈 VIX平靜",    -30),
                        "VIX上升趨勢賣出": ("📉 VIX升",       30),
                        "VIX下降趨勢買入": ("📈 VIX降",      -30),
                    }
                    for key, (text, ay) in annotations.items():
                        if key in sig_str:
                            fig.add_annotation(
                                x=data["Datetime"].iloc[idx], y=data["Close"].iloc[idx],
                                text=text, showarrow=True, arrowhead=2, ax=20, ay=ay, row=1, col=1)

                fig.update_layout(yaxis_title="價格", showlegend=True, height=700)
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{ticker}_{timestamp}")

                # 前 X% 數據範圍
                st.subheader(f"📊 {ticker} 前 {PERCENTILE_THRESHOLD}% 數據範圍")
                range_data = []
                for col_name, label in [("Price Change %","Price Change %"),
                                         ("Volume Change %","Volume Change %"),
                                         ("Volume","Volume"),
                                         ("📈 股價漲跌幅 (%)","股價漲跌幅 (%)"),
                                         ("📊 成交量變動幅 (%)","成交量變動幅 (%)")]:
                    sorted_desc = data[col_name].dropna().sort_values(ascending=False)
                    sorted_asc  = data[col_name].dropna().sort_values(ascending=True)
                    if len(sorted_desc) > 0:
                        n = max(1, int(len(sorted_desc) * PERCENTILE_THRESHOLD / 100))
                        top = sorted_desc.head(n); bot = sorted_asc.head(n)
                        fmt = "{:,.0f}" if col_name == "Volume" else "{:.2f}%"
                        range_data.append({"指標": label, "範圍類型": "最高到最低",
                                           "最大值": fmt.format(top.max()), "最小值": fmt.format(top.min())})
                        range_data.append({"指標": label, "範圍類型": "最低到最高",
                                           "最大值": fmt.format(bot.max()), "最小值": fmt.format(bot.min())})
                if range_data:
                    st.dataframe(pd.DataFrame(range_data), use_container_width=True)

                # 歷史數據表
                st.subheader(f"📋 歷史資料：{ticker}")
                display_cols = ["Datetime","Open","Low","High","Close","Volume",
                                "Price Change %","Volume Change %","📈 股價漲跌幅 (%)","📊 成交量變動幅 (%)",
                                "Close_Difference","異動標記","成交量標記","K線形態","單根解讀",
                                "VWAP","MFI","OBV","VIX","VIX_EMA_Fast","VIX_EMA_Slow"]
                display_cols = [c for c in display_cols if c in data.columns]
                display_data = data[display_cols].tail(10)
                if near_dense:
                    st.info(f"⚠️ {ticker} 靠近成交密集區：{near_dense_info}")
                if not display_data.empty:
                    st.dataframe(display_data, height=400, use_container_width=True,
                                 column_config={"異動標記": st.column_config.TextColumn(width="large"),
                                                "單根解讀": st.column_config.TextColumn(width="large")})

                # CSV 下載（原有方法，不變）
                csv = data.to_csv(index=False)
                st.download_button(
                    label=f"📥 下載 {ticker} 數據 (CSV)",
                    data=csv,
                    file_name=f"{ticker}_數據_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.warning(f"⚠️ 無法取得 {ticker} 的資料：{e}，將跳過此股票")
                continue

        st.markdown("---")
        st.info("📡 頁面將在 5 分鐘後自動刷新...")

    time.sleep(REFRESH_INTERVAL)
    placeholder.empty()
