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

# 🔥 GPT CLIENT (sécurisé)
gpt = None
if OPENAI_KEY:
    try:
        gpt = OpenAI(api_key=OPENAI_KEY)
    except Exception as e:
        st.error(f"❌ GPT init error: {e}")

TICKERS_URL = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"

st.set_page_config(layout="wide")
st.title("🟫 TEA — HEDGE FUND DASHBOARD")

# ==============================
# DEBUG CLÉS
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

    except Exception as e:
        return None


# ==============================
# INDICATEURS
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
# PROBABILITÉ
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
# GPT ANALYSE
# ==============================
def generate_gpt_analysis(row):
    if gpt is None:
        return "⚠️ GPT non configuré"

    try:
        response = gpt.chat.completions.create(
            model="gpt-4o-mini",  # 🔥 stable
            messages=[
                {"role": "system", "content": "Tu es un trader professionnel."},
                {"role": "user", "content": f"""
Analyse ce stock en 3 lignes max.

Ticker : {row['ticker']}
Score : {row['ai_score']}
Probabilité : {round(row['prob']*100)}%
RR : {row['rr']}

Structure, momentum, plan implicite.
Style clair, professionnel.
"""}
            ],
            temperature=0.4
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ GPT ERROR: {str(e)}"


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
# LOAD TICKERS
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

    st.subheader("Top 5")
    st.dataframe(top5)

    # ==============================
    # TEXTE
    # ==============================
    text = "🟫 TEA — REPORT\n\n"

    for _, row in top5.iterrows():
        text += f"{row['ticker']} — Score {row['ai_score']}\n"
        text += f"Probabilité : {round(row['prob']*100)}%\n"
        text += f"RR : {row['rr']}\n\n"

        text += generate_gpt_analysis(row)
        text += "\n\n---\n\n"

    st.subheader("Rapport")
    st.code(text)

    # ==============================
    # DISCORD
    # ==============================
    if WEBHOOK_URL and not top5.empty:
        requests.post(WEBHOOK_URL, json={"content": text})
