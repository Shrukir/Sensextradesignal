# 📓 Sensex Enriched Gamma + Macro Strategy Real-Time Trade Signal (Single Trade Mode)

# 📌 SECTION 1: Install Required Packages
# Removed for script context (handled in requirements.txt)

# 📌 SECTION 2: Import Libraries
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands
from datetime import datetime, timedelta
import requests
import random
import time
from scipy.stats import norm

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# 📌 SECTION 2B: Telegram Alert Function
TELEGRAM_TOKEN = "7364483334:AAEMJeGxlUkFspVTw4oRYTymK_BwM6pS3no"
TELEGRAM_CHAT_ID = "<1006078163>"

def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        r = requests.post(url, json=payload)
        if r.status_code != 200:
            print("❌ Telegram error:", r.text)
    except Exception as e:
        print("❌ Telegram send exception:", e)

# 📌 SECTION 3: Load Historical Data

def load_data(ticker, period="5y", interval="1d"):
    print(f"📱 Fetching data for {ticker} @ {datetime.now()}")
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    time.sleep(1)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    return data

sensex_df = load_data("^BSESN")
us_df = load_data("^GSPC")
crude_df = load_data("BZ=F")
vix_df = load_data("^VIX")
dxy_df = load_data("DX-Y.NYB")
bond_df = load_data("^TNX")
copper_df = load_data("HG=F")
shanghai_df = load_data("000001.SS")
btc_df = load_data("BTC-USD")

# 📌 SECTION 4: Technical + Macro Feature Engineering

# ... (unchanged)

# 📌 SECTION 6: Single High-Confidence Trade Suggestion

latest = features[selected_columns].iloc[-1:]
prob = ensemble.predict_proba(latest)[0][1]
label = ensemble.predict(latest)[0]
latest_spot = sensex_df["Close"].iloc[-1]
base_strike = round(latest_spot / 100) * 100
expiry = (pd.to_datetime(df.index[-1]) + pd.DateOffset(days=(7 - pd.to_datetime(df.index[-1]).weekday()))).strftime("%d %b")

option_type = "CE" if label == 1 else "PE"
confidence = "High" if prob > 0.8 else ("Moderate" if prob > 0.6 else "Low")

if prob > 0.6:
    msg = (
        f"\n📈 *Live Trade Signal*\n"
        f"→ *{expiry} {int(base_strike)} {option_type}*\n"
        f"→ Entry: *NOW* | Exit: *15:15 or SL*\n"
        f"→ Confidence: *{confidence}*\n"
        f"→ Est. Success: *{round(prob*100, 2)}%*"
    )
    print(msg)
    send_telegram_message(msg)
else:
    msg = (
        f"⚠️ *No Trade Suggested* — Confidence too Low\n"
        f"→ Predicted Probability: *{round(prob*100,2)}%*\n"
        f"→ Waiting for better signal."
    )
    print(msg)
    send_telegram_message(msg)
