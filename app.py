import pandas as pd
import numpy as np
import requests
import datetime
import os
import streamlit as st
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

TICKERS_URL = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"

st.set_page_config(layout="wide")
st.title("🟫 TEA — HEDGE FUND DASHBOARD")

# ==============================
# DEBUG
# ==============================
st.write("Polygon:", "OK" if API_KEY else "❌")
st.write("OpenAI:", "OK" if OPENAI_KEY else "❌")
st.write("Discord:", "OK" if WEBHOOK_URL else "❌")

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
            "high": a.high,
            "low": a.low,
            "close": a.close,
            "volume": a.volume
        } for a in aggs])

        return df
    except:
        return None


# ==============================
# INDICATORS
# ==============================
def compute_indicators(df):
    df["EMA9"] = df["close"].ewm(span=9).mean()
    df["EMA20"] = df["close"].ewm(span=20).mean()
    df["EMA50"] = df["close"].ewm(span=50).mean()
    df["EMA200"] = df["close"].ewm(span=200).mean()

    df["RSI"] = 100 - (100 / (1 + df["close"].pct_change().rolling(14).mean()))

    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()

    df["Volume_MA"] = df["volume"].rolling(20).mean()

    return df


# ==============================
# STRUCTURE
# ==============================
def compute_rr(df):
    price = df["close"].iloc[-1]
    support = df["low"].rolling(20).min().iloc[-1]
    resistance = df["high"].rolling(20).max().iloc[-1]

    risk = price - support
    reward = resistance - price

    if risk <= 0:
        return 0

    return reward / risk


def compute_distance(df):
    resistance = df["high"].rolling(20).max().iloc[-1]
    price = df["close"].iloc[-1]
    return (resistance - price) / price


# ==============================
# PROBABILITY
# ==============================
def compute_probability(row, rr):
    prob = 0.55

    if row["MACD"] > row["MACD_signal"]:
        prob += 0.04

    if row["EMA9"] > row["EMA20"]:
        prob += 0.03

    if row["RSI"] > 60 or row["RSI"] < 35:
        prob += 0.02

    if row["volume"] > row["Volume_MA"]:
        prob += 0.03

    if rr > 1.5:
        prob += 0.03

    return prob


# ==============================
# GPT STOCK
# ==============================
def generate_gpt_analysis(row):
    if gpt is None:
        return "⚠️ GPT non configuré"

    try:
        response = gpt.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tu es un trader professionnel."},
                {"role": "user", "content": f"""
Analyse ce stock en 3 lignes max.

Ticker: {row['ticker']}
Score: {row['ai_score']}
Probabilité: {round(row['prob']*100)}%
RR: {row['rr']}
"""}
            ],
            temperature=0.4
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ GPT ERROR: {str(e)}"


# ==============================
# GPT MACRO
# ==============================
def generate_macro(spy, breadth):
    if gpt is None:
        return "Macro indisponible"

    try:
        response = gpt.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""
Analyse le marché global en 3 lignes.

SPY RSI: {spy['rsi']}
Breadth: {breadth}%
"""
            }],
            temperature=0.4
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Macro error: {e}"


# ==============================
# TEA OVERLAY
# ==============================
def generate_tea_overlay(row):
    return f"""Structure {'forte' if row['rr'] > 1.5 else 'correcte'}.
Momentum {'aligné' if row['prob'] > 0.60 else 'à confirmer'}.
Approche: {'continuation' if row['edge'] > 5 else 'prudente'}."""


# ==============================
# SPY
# ==============================
def get_spy_data():
    df = get_data("SPY")
    df = compute_indicators(df)
    latest = df.iloc[-1]

    return {
        "price": round(latest["close"], 2),
        "rsi": round(latest["RSI"], 1),
        "trend": latest["EMA9"] > latest["EMA20"]
    }


# ==============================
# BREADTH
# ==============================
def compute_breadth(tickers):
    count = 0
    total = 0

    for t in tickers[:100]:
        df = get_data(t)
        if df is None or len(df) < 50:
            continue

        df = compute_indicators(df)
        latest = df.iloc[-1]

        total += 1
        if latest["close"] > latest["EMA50"]:
            count += 1

    return round((count / total) * 100, 1) if total > 0 else 0


# ==============================
# DISCORD SAFE
# ==============================
def send_discord(message):
    if not WEBHOOK_URL:
        return

    parts = [message[i:i+1900] for i in range(0, len(message), 1900)]

    for part in parts:
        r = requests.post(WEBHOOK_URL, json={"content": part})
        print("Discord status:", r.status_code)


# ==============================
# ANALYSE
# ==============================
def analyze_ticker(ticker):
    df = get_data(ticker)

    if df is None or len(df) < 50:
        return None

    df = compute_indicators(df)

    rr = compute_rr(df)
    distance = compute_distance(df)

    if rr < 1.0:
        return None

    if distance > 0.15:
        return None

    latest = df.iloc[-1]
    prob = compute_probability(latest, rr)

    return {
        "ticker": ticker,
        "ai_score": round(prob * 10, 1),
        "prob": prob,
        "edge": round((prob - 0.55) * 100, 2),
        "rr": round(rr, 2)
    }


# ==============================
# MAIN
# ==============================
tickers = pd.read_csv(TICKERS_URL)["Symbol"].tolist()

results = []
for t in tickers[:200]:
    r = analyze_ticker(t)
    if r:
        results.append(r)

df = pd.DataFrame(results)

if df.empty:
    st.error("❌ Aucun stock trouvé")
else:
    top5 = df.sort_values(by="ai_score", ascending=False).head(5)

    spy = get_spy_data()
    breadth = compute_breadth(tickers)

    macro = generate_macro(spy, breadth)

    text = "🟫 TEA — HEDGE FUND REPORT\n\n"

    text += "🌍 Macro\n" + macro + "\n\n"
    text += f"📊 SPY: {spy['price']} | RSI {spy['rsi']}\n"
    text += f"🔥 Breadth: {breadth}%\n\n"
    text += "----------------------\n\n"

    for _, row in top5.iterrows():
        text += f"{row['ticker']} — Score {row['ai_score']}\n"
        text += f"Probabilité: {round(row['prob']*100)}%\n"
        text += f"RR: {row['rr']}\n\n"

        text += generate_gpt_analysis(row) + "\n\n"
        text += "🧠 Lecture TEA\n" + generate_tea_overlay(row) + "\n\n"

        text += "----------------------\n\n"

    st.subheader("Top 5")
    st.dataframe(top5)

    st.subheader("Rapport")
    st.code(text)

    send_discord(text)
