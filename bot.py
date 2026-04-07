def run():

    print("🚀 BOT RUNNING")

import pandas as pd
import requests
import datetime
import os
import streamlit as st
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
st.title("🟫 TEA — DESK MODE")

# ==============================
# LOAD SP500
# ==============================
@st.cache_data
def load_sp500():
    url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
    return pd.read_csv(url)["Symbol"].tolist()

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
# INDICATORS (INTERNAL ONLY)
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
# MARKET
# ==============================
def get_spy():
    df = compute_indicators(get_data("SPY"))
    latest = df.iloc[-1]
    return {
        "rsi": round(latest["RSI"], 1),
        "trend": latest["close"] > latest["EMA20"]
    }

def compute_breadth():
    count, total = 0, 0
    for t in tickers[:150]:
        df = get_data(t)
        if df is None:
            continue
        df = compute_indicators(df)
        latest = df.iloc[-1]
        total += 1
        if latest["close"] > latest["EMA20"]:
            count += 1
    return round((count / total) * 100, 1) if total else 0

def compute_market_score(spy, breadth):
    score = 0
    if spy["trend"]: score += 4
    if spy["rsi"] > 55: score += 3
    if breadth > 60: score += 3
    return score

# ==============================
# SECTORS
# ==============================
sector_map = {
    "Tech": "XLK",
    "Energy": "XLE",
    "Finance": "XLF",
    "Healthcare": "XLV"
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
        if latest["close"] > latest["EMA20"]: score += 1
        if latest["RSI"] > 55: score += 1
        if latest["RVOL"] > 1.2: score += 1
        rows.append({"sector": name, "etf": ticker, "score": score})
    return pd.DataFrame(rows).sort_values(by="score", ascending=False)

# ==============================
# SCAN
# ==============================
def analyze_ticker(t):
    df = get_data(t)
    if df is None:
        return None

    df = compute_indicators(df)
    if len(df) < 50:
        return None

    latest = df.iloc[-1]
    score = 0

    if latest["close"] > latest["EMA20"]: score += 1
    if 55 < latest["RSI"] < 70: score += 1
    if latest["RVOL"] > 1.2: score += 1

    high20 = df["high"].rolling(20).max().iloc[-1]
    if latest["close"] >= high20:
        score += 2

    if score < 3:
        return None

    return {
        "ticker": t,
        "score": score,
        "entry": round(latest["close"],2),
        "stop": round(df["low"].rolling(10).min().iloc[-1],2),
        "target": round(latest["close"]*1.1,2)
    }

def scan_market():
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        data = executor.map(analyze_ticker, tickers)
    for r in data:
        if r:
            results.append(r)
    return pd.DataFrame(results).sort_values(by="score", ascending=False).head(5) if results else pd.DataFrame()

# ==============================
# GPT — DESK MODE MINIMAL
# ==============================
def generate_stock_analysis(row):

    if gpt is None:
        return "Analyse indisponible"

    prompt = f"""
Tu es un analyste hedge fund.

Donne une thèse en 2 phrases max.

- Pas d’indicateurs
- Pas de jargon technique
- Focus : pourquoi maintenant + scénario

Ticker: {row['ticker']}
"""

    try:
        res = gpt.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.6
        )
        return res.choices[0].message.content.strip()
    except:
        return "Analyse indisponible"

def generate_market_analysis(spy, breadth, score):

    if gpt is None:
        return "Analyse indisponible"

    prompt = f"""
Analyse marché en 3 lignes max.

RSI: {spy['rsi']}
Breadth: {breadth}
Score: {score}/10

Style hedge fund, concis.
"""

    return gpt.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    ).choices[0].message.content.strip()

# ==============================
# DISCORD
# ==============================
def send_discord(report):

    if not WEBHOOK_URL:
        return

    chunks = [report[i:i+1800] for i in range(0, len(report), 1800)]

    for chunk in chunks:
        requests.post(WEBHOOK_URL, json={"content": chunk})

# ==============================
# MAIN
# ==============================
spy = get_spy()
breadth = compute_breadth()
market_score = compute_market_score(spy, breadth)

sector_df = analyze_sectors()
top5 = scan_market()

market_analysis = generate_market_analysis(spy, breadth, market_score)

# UI
st.subheader("Market")
st.write(market_analysis)

st.subheader("Top Picks")
st.dataframe(top5)

# REPORT
report = f"""
🟫 TEA DESK

Score: {market_score}/10
Breadth: {breadth}%

{market_analysis}

"""

if top5.empty:
    report += "No setup today\n"
else:
    for _, row in top5.iterrows():
        analysis = generate_stock_analysis(row)

        report += f"""{row['ticker']}
Entry: {row['entry']} | Stop: {row['stop']}

{analysis}

-----
"""

send_discord(report)

if __name__ == "__main__":
    run()
