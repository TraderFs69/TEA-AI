import pandas as pd
import numpy as np
import requests
import datetime
import os
import streamlit as st
import streamlit.components.v1 as components
from polygon import RESTClient
from openai import OpenAI
import json

# ==============================
# CONFIG
# ==============================
API_KEY = os.environ.get("POLYGON_API_KEY")
WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

client = RESTClient(API_KEY)
gpt = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

st.set_page_config(layout="wide")
st.title("🟫 TEA — HEDGE FUND DASHBOARD")

# ==============================
# DATA
# ==============================
def get_data(ticker):
    try:
        today = datetime.date.today()
        past = today - datetime.timedelta(days=180)

        aggs = client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=past,
            to=today,
            limit=500
        )

        if not aggs:
            return None

        df = pd.DataFrame([{
            "time": pd.to_datetime(a.timestamp, unit="ms"),
            "open": a.open,
            "high": a.high,
            "low": a.low,
            "close": a.close,
            "volume": a.volume
        } for a in aggs])

        df.set_index("time", inplace=True)
        return df

    except:
        return None


# ==============================
# INDICATORS (FIXED)
# ==============================
def compute_indicators(df):

    df["EMA9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["close"].ewm(span=200, adjust=False).mean()

    # RSI WILDER
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # RVOL
    df["Volume_MA"] = df["volume"].rolling(20).mean()
    df["RVOL"] = df["volume"] / df["Volume_MA"]

    return df


# ==============================
# CHART (LIGHTWEIGHT)
# ==============================
def plot_candles(df):
    data = [
        {
            "time": int(i.timestamp()),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
        for i, row in df.iterrows()
    ]

    html = f"""
    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
    <div id="chart"></div>
    <script>
    const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
        width: 800,
        height: 400
    }});
    const series = chart.addCandlestickSeries();
    series.setData({json.dumps(data)});
    </script>
    """
    components.html(html, height=420)


# ==============================
# STRUCTURE
# ==============================
def compute_rr(df):
    price = df["close"].iloc[-1]
    support = df["low"].rolling(20).min().iloc[-1]
    resistance = df["high"].rolling(20).max().iloc[-1]

    risk = price - support
    reward = resistance - price

    return reward / risk if risk > 0 else 0


# ==============================
# GPT
# ==============================
def generate_gpt_analysis(row):
    if gpt is None:
        return "GPT non configuré"

    try:
        response = gpt.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Analyse ce stock en 3 lignes: {row['ticker']}, score {row['ai_score']}, prob {round(row['prob']*100)}%"
            }],
            temperature=0.4
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)


# ==============================
# SECTORS
# ==============================
sector_etfs = {
    "Tech": "XLK",
    "Energy": "XLE",
    "Finance": "XLF",
    "Health": "XLV",
    "Consumer": "XLY",
    "Industrial": "XLI"
}

def analyze_sectors():
    results = []

    for name, ticker in sector_etfs.items():
        df = get_data(ticker)
        if df is None:
            continue

        df = compute_indicators(df)
        latest = df.iloc[-1]

        score = 0
        if latest["EMA9"] > latest["EMA20"]:
            score += 1
        if latest["MACD"] > latest["MACD_signal"]:
            score += 1
        if latest["RSI"] > 55:
            score += 1

        results.append({"sector": name, "ticker": ticker, "score": score})

    return pd.DataFrame(results).sort_values(by="score", ascending=False)


# ==============================
# DISCORD FIX
# ==============================
def send_discord(message):
    if not WEBHOOK_URL:
        return

    parts = [message[i:i+1900] for i in range(0, len(message), 1900)]
    for p in parts:
        requests.post(WEBHOOK_URL, json={"content": p})


# ==============================
# MAIN
# ==============================
tickers = ["AAPL","MSFT","NVDA","TSLA","META","AMD","AMZN"]

results = []

for t in tickers:
    df = get_data(t)
    if df is None:
        continue

    df = compute_indicators(df)
    rr = compute_rr(df)
    latest = df.iloc[-1]

    prob = 0.55 + (0.05 if latest["MACD"] > latest["MACD_signal"] else 0)

    results.append({
        "ticker": t,
        "ai_score": round(prob * 10, 1),
        "prob": prob,
        "rr": round(rr, 2)
    })

df = pd.DataFrame(results)

if not df.empty:

    top5 = df.sort_values(by="ai_score", ascending=False).head(5)

    st.subheader("Top 5")
    st.dataframe(top5)

    # SECTORS
    st.subheader("Secteurs")
    sector_df = analyze_sectors()
    st.dataframe(sector_df)

    # CHARTS
    st.subheader("Charts")
    for _, row in top5.iterrows():
        st.write(row["ticker"])
        df_chart = get_data(row["ticker"])
        plot_candles(df_chart.tail(100))

    # REPORT
    text = "TEA REPORT\n\n"

    for _, row in top5.iterrows():
        text += f"{row['ticker']} Score {row['ai_score']}\n"
        text += generate_gpt_analysis(row) + "\n\n"

    st.code(text)

    send_discord(text)
    st.subheader("Rapport")
    st.code(text)

    send_discord(text)
