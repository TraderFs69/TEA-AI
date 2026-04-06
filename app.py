import pandas as pd
import requests
import datetime
import os
import streamlit as st
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor
from polygon import RESTClient
from openai import OpenAI
import tempfile

# ==============================
# CONFIG
# ==============================
API_KEY = os.environ.get("POLYGON_API_KEY")
WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

client = RESTClient(API_KEY)
gpt = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

st.set_page_config(layout="wide")
st.title("🟫 TEA — FINAL INSTITUTIONAL SYSTEM")

# ==============================
# LOAD SP500
# ==============================
@st.cache_data
def load_sp500():
    url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
    df = pd.read_csv(url)
    return df["Symbol"].tolist()

tickers = load_sp500()

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

    return df.dropna()

# ==============================
# SPY
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

# ==============================
# BREADTH
# ==============================
def compute_breadth():

    count = 0
    total = 0

    sample = tickers[:200]

    for t in sample:
        df = get_data(t)
        if df is None:
            continue

        df = compute_indicators(df)
        latest = df.iloc[-1]

        total += 1
        if latest["close"] > latest["EMA20"]:
            count += 1

    return round((count / total) * 100, 1) if total else 0

# ==============================
# MARKET SCORE
# ==============================
def compute_market_score(spy, breadth):

    score = 0

    if spy["trend"]:
        score += 4

    if spy["rsi"] > 55:
        score += 3

    if breadth > 60:
        score += 3

    return score

# ==============================
# SECTORS
# ==============================
sector_map = {
    "Tech": "XLK",
    "Energy": "XLE",
    "Finance": "XLF",
    "Healthcare": "XLV",
    "Consumer": "XLY",
    "Industrial": "XLI"
}

def analyze_sectors():

    rows = []

    for name, ticker in sector_map.items():
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
        if latest["RVOL"] > 1.2:
            score += 1

        rows.append({
            "sector": name,
            "etf": ticker,
            "score": score
        })

    return pd.DataFrame(rows).sort_values(by="score", ascending=False)

# ==============================
# SCAN
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

    high20 = df["high"].rolling(20).max().iloc[-1]
    if latest["close"] >= high20:
        score += 2

    if score < 3:
        return None

    entry = latest["close"]
    stop = df["low"].rolling(10).min().iloc[-1]
    target = entry + (entry - stop) * 2

    return {
        "ticker": ticker,
        "score": score,
        "rsi": round(latest["RSI"],1),
        "rvol": round(latest["RVOL"],2),
        "entry": round(entry,2),
        "stop": round(stop,2),
        "target": round(target,2)
    }

def scan_market():

    results = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        data = executor.map(analyze_ticker, tickers)

    for r in data:
        if r:
            results.append(r)

    if len(results) == 0:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values(by="score", ascending=False).head(5)

# ==============================
# GPT ANALYSES
# ==============================
def generate_macro_sector(spy, breadth, sectors):

    if gpt is None:
        return "Analyse indisponible"

    top_sec = ", ".join(sectors.head(3)["sector"])

    prompt = f"""
Analyse le marché comme un analyste Goldman Sachs.

SPY RSI: {spy['rsi']}
Breadth: {breadth}%
Secteurs dominants: {top_sec}

4 lignes max.
"""

    res = gpt.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.5
    )

    return res.choices[0].message.content.strip()


def generate_stock_analysis(row):

    if gpt is None:
        return "Analyse indisponible"

    prompt = f"""
Analyse ce stock comme un analyste senior Goldman Sachs.

Ticker: {row['ticker']}
RSI: {row['rsi']}
RVOL: {row['rvol']}

3 lignes max.
"""

    res = gpt.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.5
    )

    return res.choices[0].message.content.strip()

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

    fig.update_layout(template="plotly_dark", height=400)

    st.plotly_chart(fig, use_container_width=True)

# ==============================
# DISCORD
# ==============================
def send_discord(report, top5):

    if not WEBHOOK_URL:
        return

    requests.post(WEBHOOK_URL, json={"content": report})

# ==============================
# MAIN
# ==============================
spy = get_spy()
breadth = compute_breadth()
market_score = compute_market_score(spy, breadth)

sector_df = analyze_sectors()
top5 = scan_market()

macro = generate_macro_sector(spy, breadth, sector_df)

# UI
st.subheader("📊 Market Internals")
st.write(f"RSI: {spy['rsi']} | Breadth: {breadth}% | Score: {market_score}/10")

st.subheader("🌍 Macro & Secteurs")
st.write(macro)

st.subheader("🏭 Secteurs")
st.dataframe(sector_df)

st.subheader("🎯 Top 5")
st.dataframe(top5)

st.subheader("📊 Charts")
for _, row in top5.iterrows():
    df_chart = get_data(row["ticker"])
    if df_chart is not None:
        plot_chart(df_chart.tail(100), row["ticker"])

st.subheader("🧠 Analyse institutionnelle")
for _, row in top5.iterrows():
    st.write(f"### {row['ticker']}")
    st.write(generate_stock_analysis(row))

# REPORT
report = f"""
🟫 TEA REPORT

Market Score: {market_score}/10
Breadth: {breadth}%

Macro:
{macro}
"""

send_discord(report, top5)
