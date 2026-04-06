import pandas as pd
import requests
import datetime
import os
import streamlit as st
import plotly.graph_objects as go
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
st.title("🟫 TEA — LEVEL 5 (STABLE)")

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

    except Exception as e:
        return None


# ==============================
# INDICATORS (FIX)
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

    return df.dropna()  # 🔥 CRUCIAL


# ==============================
# SCAN (FIX)
# ==============================
def analyze_ticker(ticker):

    df = get_data(ticker)
    if df is None:
        return None

    df = compute_indicators(df)

    if len(df) < 50:
        return None

    latest = df.iloc[-1]

    score = 0

    if latest["close"] > latest["EMA20"]:
        score += 1

    if 55 < latest["RSI"] < 70:
        score += 1

    if latest["RVOL"] > 1.2:
        score += 1

    # 🔥 BREAKOUT FIX
    high20 = df["high"].rolling(20).max().iloc[-1]
    if latest["close"] >= high20:
        score += 2

    if score < 2:
        return None

    return {
        "ticker": ticker,
        "score": score,
        "rsi": round(latest["RSI"], 1),
        "rvol": round(latest["RVOL"], 2)
    }


# ==============================
# SCAN MARKET (SAFE)
# ==============================
def scan_market(tickers):

    results = []

    for t in tickers:
        r = analyze_ticker(t)
        if r:
            results.append(r)

    if len(results) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    return df.sort_values(by="score", ascending=False).head(5)


# ==============================
# PROBABILITY
# ==============================
def compute_probability(score):
    if score >= 4:
        return 0.65
    elif score == 3:
        return 0.60
    else:
        return 0.55


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
                "content": "Analyse le marché en 3 lignes."
            }],
            temperature=0.4
        )
        return res.choices[0].message.content.strip()
    except:
        return "Macro indisponible"


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

# MACRO
st.subheader("🌍 Macro")
macro = get_macro()
st.write(macro)

# SCAN
st.subheader("🎯 Top 5")

top5 = scan_market(tickers)

if top5.empty:
    st.warning("⚠️ Aucun stock trouvé — conditions marché faibles")
else:
    st.dataframe(top5)

    # CHARTS
    st.subheader("📊 Charts")

    for _, row in top5.iterrows():
        df_chart = get_data(row["ticker"])
        if df_chart is not None:
            plot_chart(df_chart.tail(100), row["ticker"])

    # REPORT
    report = "🟫 TEA LEVEL 5\n\n"
    report += "Macro:\n" + macro + "\n\n"

    for _, row in top5.iterrows():
        prob = compute_probability(row["score"])
        report += f"{row['ticker']} — Score {row['score']} | Prob {int(prob*100)}%\n"

    send_discord(report)
