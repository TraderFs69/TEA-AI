import pandas as pd
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
st.title("🟫 TEA — REBUILD PRO DASHBOARD")

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
# INDICATORS (PRO)
# ==============================
def compute_indicators(df):

    df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()

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
# SPY + MACRO
# ==============================
def get_spy():
    df = get_data("SPY")
    df = compute_indicators(df)
    latest = df.iloc[-1]

    return {
        "price": round(latest["close"], 2),
        "rsi": round(latest["RSI"], 1),
        "trend": latest["close"] > latest["EMA20"]
    }


def compute_breadth(tickers):
    count = 0
    total = 0

    for t in tickers[:50]:
        df = get_data(t)
        if df is None:
            continue

        df = compute_indicators(df)
        latest = df.iloc[-1]

        total += 1
        if latest["close"] > latest["EMA20"]:
            count += 1

    return round((count / total) * 100, 1) if total else 0


def generate_macro(spy, breadth):
    if gpt is None:
        return "GPT non configuré"

    prompt = f"""
Analyse le marché en 3 lignes.

SPY RSI: {spy['rsi']}
Breadth: {breadth}%
"""

    try:
        res = gpt.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return res.choices[0].message.content.strip()
    except:
        return "Macro indisponible"


# ==============================
# SECTORS
# ==============================
sector_etfs = {
    "Tech": "XLK",
    "Energy": "XLE",
    "Finance": "XLF"
}

def analyze_sectors():
    rows = []

    for name, ticker in sector_etfs.items():
        df = get_data(ticker)
        if df is None:
            continue

        df = compute_indicators(df)
        latest = df.iloc[-1]

        score = 0
        if latest["close"] > latest["EMA20"]:
            score += 1
        if latest["RSI"] > 55:
            score += 1

        rows.append({
            "sector": name,
            "ticker": ticker,
            "score": score
        })

    return pd.DataFrame(rows).sort_values(by="score", ascending=False)


# ==============================
# SCANNER TOP 5
# ==============================
def scan_market(tickers):

    results = []

    for t in tickers:
        df = get_data(t)
        if df is None:
            continue

        df = compute_indicators(df)
        latest = df.iloc[-1]

        score = 0

        if latest["close"] > latest["EMA20"]:
            score += 1
        if latest["RSI"] > 55:
            score += 1
        if latest["RVOL"] > 1.2:
            score += 1

        if score >= 2:
            results.append({
                "ticker": t,
                "score": score,
                "rsi": latest["RSI"]
            })

    df = pd.DataFrame(results)

    if df.empty:
        return df

    return df.sort_values(by="score", ascending=False).head(5)


# ==============================
# CHART
# ==============================
def plot_candles(df):
    data = [{
        "time": int(ts.timestamp()),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
    } for ts, row in df.iterrows()]

    html = f"""
    <div id="chart" style="width:100%; height:400px;"></div>
    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
    <script>
    const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
        width: document.getElementById('chart').clientWidth,
        height: 400
    }});
    const series = chart.addCandlestickSeries();
    series.setData({json.dumps(data)});
    </script>
    """
    components.html(html, height=420)


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
tickers = ["AAPL","MSFT","NVDA","TSLA","META","AMD","AMZN"]

# Macro
spy = get_spy()
breadth = compute_breadth(tickers)
macro = generate_macro(spy, breadth)

st.subheader("🌍 Macro")
st.write(macro)

st.write(f"SPY: {spy['price']} | RSI: {spy['rsi']} | Breadth: {breadth}%")

# Secteurs
st.subheader("🏭 Secteurs")
sector_df = analyze_sectors()
st.dataframe(sector_df)

# Scanner
st.subheader("🎯 Top 5")
top5 = scan_market(tickers)
st.dataframe(top5)

# Charts
st.subheader("📊 Charts")
for _, row in top5.iterrows():
    st.write(row["ticker"])
    df_chart = get_data(row["ticker"])
    if df_chart is not None:
        plot_candles(df_chart.tail(100))

# Discord report
report = f"TEA REPORT\n\nMacro:\n{macro}\n\nTop Picks:\n{top5}"

send_discord(report)
