import pandas as pd
import numpy as np
import requests
import datetime
from polygon import RESTClient

# ==============================
# CONFIG
# ==============================
API_KEY = "TA_CLE_POLYGON"
WEBHOOK_URL = "TON_WEBHOOK_DISCORD"

client = RESTClient(API_KEY)

TICKERS_URL = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"


# ==============================
# LOAD TICKERS
# ==============================
tickers = pd.read_csv(TICKERS_URL)["Symbol"].tolist()


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
# STRUCTURE TRADING
# ==============================
def compute_rr(df):
    price = df["close"].iloc[-1]
    support = df["low"].rolling(20).min().iloc[-1]
    resistance = df["high"].rolling(20).max().iloc[-1]

    risk = price - support
    reward = resistance - price

    if risk == 0:
        return 0

    return reward / risk


def compute_distance_resistance(df):
    resistance = df["high"].rolling(20).max().iloc[-1]
    price = df["close"].iloc[-1]
    return (resistance - price) / price


def compute_atr(df):
    df["tr"] = df["high"] - df["low"]
    return df["tr"].rolling(14).mean().iloc[-1]


# ==============================
# SCORE TEA
# ==============================
def compute_score(row):
    score = 0

    if row["EMA9"] > row["EMA20"] > row["EMA50"]:
        score += 4
    elif row["EMA9"] > row["EMA20"]:
        score += 2

    if row["MACD"] > row["MACD_signal"]:
        score += 2

    if row["RSI"] > 55:
        score += 1
    if row["RSI"] < 35:
        score += 1

    if row["close"] > row["EMA200"]:
        score += 2

    if row["volume"] > row["Volume_MA"]:
        score += 1

    return round(score, 2)


# ==============================
# PROBABILITÉ (AI STYLE)
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

    return round(prob, 4)


def compute_ai_score(prob):
    return round(prob * 10, 1)


def compute_edge(prob):
    return round((prob - 0.55) * 100, 2)


# ==============================
# ANALYSE
# ==============================
def analyze_ticker(ticker):
    df = get_data(ticker)
    if df is None or len(df) < 50:
        return None

    df = compute_indicators(df)

    rr = compute_rr(df)
    distance = compute_distance_resistance(df)

    # FILTRE QUALITÉ
    if rr < 1.3:
        return None
    if distance > 0.08:
        return None

    latest = df.iloc[-1]

    prob = compute_probability(latest, rr)

    return {
        "ticker": ticker,
        "score": compute_score(latest),
        "prob": prob,
        "ai_score": compute_ai_score(prob),
        "edge": compute_edge(prob),
        "rr": round(rr, 2),
        "rsi": latest["RSI"],
        "momentum": latest["MACD"] > latest["MACD_signal"]
    }


# ==============================
# SCAN
# ==============================
results = []

for t in tickers:
    data = analyze_ticker(t)
    if data:
        results.append(data)

df = pd.DataFrame(results)

df = df[df["ai_score"] >= 7]

top5 = df.sort_values(by="ai_score", ascending=False).head(5)


# ==============================
# TEXTE FINAL TEA
# ==============================
def generate_text(df):
    text = "🟫 TEA — RECAP SWING (AI)\n\n"

    for _, row in df.iterrows():
        text += f"{row['ticker']} — Score {row['ai_score']}\n"
        text += f"Probabilité : {round(row['prob']*100)}%\n"
        text += f"Edge : +{row['edge']}%\n"
        text += f"RR : {row['rr']}\n"

        if row["momentum"]:
            text += "Setup : Momentum\n"
        elif row["rsi"] < 40:
            text += "Setup : Reversal\n"
        else:
            text += "Setup : Continuation\n"

        text += "\n"

    text += "🚨 5 opportunités retenues sur +500 analysées\n"
    text += "💡 Moins de trades = plus d’argent."

    return text


msg = generate_text(top5)


# ==============================
# DISCORD
# ==============================
requests.post(WEBHOOK_URL, json={"content": msg})

print("✅ TEA AI SYSTEM DONE")
