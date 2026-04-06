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
st.title("🟫 TEA — FINAL SYSTEM")

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
# ANALYZE STOCK
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
        "rsi": round(latest["RSI"], 1),
        "rvol": round(latest["RVOL"], 2),
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target": round(target, 2)
    }


# ==============================
# SCAN MARKET (FAST)
# ==============================
def scan_market():

    results = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        data = executor.map(analyze_ticker, tickers)

    for r in data:
        if r:
            results.append(r)

    if len(results) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    return df.sort_values(by="score", ascending=False).head(5)


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
                "content": "Analyse le marché global en 3 lignes style hedge fund."
            }],
            temperature=0.4
        )
        return res.choices[0].message.content.strip()
    except:
        return "Macro indisponible"


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
            "score": score,
            "rsi": round(latest["RSI"], 1)
        })

    df = pd.DataFrame(rows)

    return df.sort_values(by="score", ascending=False)


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
# SAVE IMAGE FOR DISCORD
# ==============================
def save_chart_image(df, ticker):

    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"]
    )])

    fig.update_layout(template="plotly_dark", title=ticker)

    file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.write_image(file.name)

    return file.name


# ==============================
# DISCORD WITH IMAGES
# ==============================
def send_discord(report, top5):

    if not WEBHOOK_URL:
        return

    requests.post(WEBHOOK_URL, json={"content": report})

    for _, row in top5.iterrows():

        df_chart = get_data(row["ticker"])
        if df_chart is None:
            continue

        img = save_chart_image(df_chart.tail(100), row["ticker"])

        with open(img, "rb") as f:
            requests.post(WEBHOOK_URL, files={"file": f})


# ==============================
# MAIN
# ==============================
st.subheader("🌍 Macro")
macro = get_macro()
st.write(macro)

st.subheader("🏭 Secteurs")
sector_df = analyze_sectors()
st.dataframe(sector_df)

st.subheader("🎯 Top 5 S&P500")
top5 = scan_market()

if top5.empty:
    st.warning("⚠️ Aucun setup aujourd’hui")
else:
    st.dataframe(top5)

    st.subheader("📊 Charts")
    for _, row in top5.iterrows():
        df_chart = get_data(row["ticker"])
        if df_chart is not None:
            plot_chart(df_chart.tail(100), row["ticker"])

    report = "🟫 TEA FINAL REPORT\n\n"
    report += "🌍 Macro:\n" + macro + "\n\n"

    report += "🏭 Secteurs dominants:\n"
    for _, row in sector_df.head(3).iterrows():
        report += f"{row['sector']} ({row['etf']}) Score {row['score']}\n"

    report += "\n🎯 Top Picks:\n"

    for _, row in top5.iterrows():
        report += f"""{row['ticker']}
Score: {row['score']}
Entry: {row['entry']}
Stop: {row['stop']}
Target: {row['target']}

"""

    send_discord(report, top5)
