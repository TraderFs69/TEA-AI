import pandas as pd
import requests
import datetime
import os
import streamlit as st
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor
from polygon import RESTClient
from openai import OpenAI

# ==============================
# CONFIG
# ==============================
API_KEY = os.environ.get("POLYGON_API_KEY")
WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

client = RESTClient(API_KEY)
gpt = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

st.set_page_config(layout="wide")
st.title("🟫 TEA — LEVEL 4 DASHBOARD")

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
            limit=200
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
# INDICATORS
# ==============================
def compute_indicators(df):

    df["EMA20"] = df["close"].ewm(span=20).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/14).mean()
    avg_loss = loss.ewm(alpha=1/14).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Volume_MA"] = df["volume"].rolling(20).mean()
    df["RVOL"] = df["volume"] / df["Volume_MA"]

    return df


# ==============================
# SCAN FUNCTION
# ==============================
def analyze_ticker(ticker):
    df = get_data(ticker)
    if df is None:
        return None

    df = compute_indicators(df)
    latest = df.iloc[-1]

    score = 0

    # Trend
    if latest["close"] > latest["EMA20"]:
        score += 1

    # Momentum
    if 55 < latest["RSI"] < 70:
        score += 1

    # Volume
    if latest["RVOL"] > 1.3:
        score += 1

    # Breakout
    high20 = df["high"].rolling(20).max().iloc[-2]
    if latest["close"] > high20:
        score += 2

    if score < 3:
        return None

    return {
        "ticker": ticker,
        "score": score,
        "rsi": round(latest["RSI"], 1),
        "rvol": round(latest["RVOL"], 2)
    }


# ==============================
# FAST SCAN
# ==============================
def scan_market(tickers):

    results = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        data = executor.map(analyze_ticker, tickers)

    for r in data:
        if r:
            results.append(r)

    df = pd.DataFrame(results)

    if df.empty:
        return df

    return df.sort_values(by="score", ascending=False).head(5)


# ==============================
# PROBABILITY
# ==============================
def compute_probability(score):
    if score >= 5:
        return 0.65
    elif score == 4:
        return 0.60
    else:
        return 0.57


# ==============================
# MACRO GPT
# ==============================
def get_macro():
    if gpt is None:
        return "GPT non configuré"

    try:
        res = gpt.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": "Analyse le marché en 3 lignes (style hedge fund)."
            }],
            temperature=0.4
        )
        return res.choices[0].message.content.strip()
    except:
        return "Macro indisponible"


# ==============================
# SECTORS
# ==============================
sector_etfs = ["XLK","XLE","XLF"]

def analyze_sectors():
    rows = []

    for etf in sector_etfs:
        df = get_data(etf)
        if df is None:
            continue

        df = compute_indicators(df)
        latest = df.iloc[-1]

        score = 1 if latest["close"] > latest["EMA20"] else 0

        rows.append({"ETF": etf, "score": score})

    return pd.DataFrame(rows)


# ==============================
# CHART
# ==============================
def plot_chart(df, ticker):

    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"]
    )])

    fig.update_layout(
        title=ticker,
        template="plotly_dark",
        height=400,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)


# ==============================
# DISCORD
# ==============================
def send_discord(msg):
    if not WEBHOOK_URL:
        return

    parts = [msg[i:i+1900] for i in range(0, len(msg), 1900)]

    for p in parts:
        requests.post(WEBHOOK_URL, json={"content": p})


# ==============================
# MAIN
# ==============================
tickers = ["AAPL","MSFT","NVDA","TSLA","META","AMD","AMZN","GOOGL","NFLX"]

# Macro
st.subheader("🌍 Macro")
macro = get_macro()
st.write(macro)

# Sectors
st.subheader("🏭 Secteurs")
sector_df = analyze_sectors()
st.dataframe(sector_df)

# Scan
st.subheader("🎯 Top 5")
top5 = scan_market(tickers)
st.dataframe(top5)

# Charts
st.subheader("📊 Charts")

if not top5.empty:
    for _, row in top5.iterrows():
        df_chart = get_data(row["ticker"])
        if df_chart is not None:
            plot_chart(df_chart.tail(100), row["ticker"])

# Report
report = "🟫 TEA LEVEL 4\n\n"
report += "Macro:\n" + macro + "\n\n"

if not top5.empty:
    for _, row in top5.iterrows():
        prob = compute_probability(row["score"])
        report += f"{row['ticker']} — Score {row['score']} | Prob {int(prob*100)}%\n"

send_discord(report)
