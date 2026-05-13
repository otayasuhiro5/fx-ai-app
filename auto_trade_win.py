import json
import time
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
import os
import requests

SYMBOL = "USDJPY=X"
MODEL_FILE = "/root/fx-ai-app/fx_model.pkl"
SCALER_FILE = "/root/fx-ai-app/fx_scaler.pkl"
SIGNAL_FILE = "/root/fx-ai-app/signal.json"

FEATURES = [
    "RSI", "RSI_prev", "RSI_diff", "STOCH_k", "STOCH_d",
    "MACD", "MACD_signal", "MACD_hist",
    "BB_upper", "BB_lower", "BB_width", "BB_pct",
    "EMA20", "EMA50", "EMA_diff", "EMA_trend",
    "ATR", "ATR_pct", "momentum", "momentum_10",
    "volatility", "volatility_20",
    "return_1", "return_5", "return_10", "return_20",
    "high_low", "close_open", "upper_shadow", "lower_shadow",
    "volume_ratio"
]

def build_features(df):
    df = df.copy()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    df["RSI_prev"] = df["RSI"].shift(1)
    df["RSI_diff"] = df["RSI"] - df["RSI_prev"]
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(df["Close"], window=20)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_mid"] = bb.bollinger_mavg()
    df["BB_lower"] = bb.bollinger_lband()
    df["BB_width"] = df["BB_upper"] - df["BB_lower"]
    df["BB_pct"] = bb.bollinger_pband()
    df["EMA20"] = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
    df["EMA50"] = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()
    df["EMA200"] = ta.trend.EMAIndicator(df["Close"], window=200).ema_indicator()
    df["EMA_diff"] = df["EMA20"] - df["EMA50"]
    df["EMA_trend"] = df["EMA50"] - df["EMA200"]
    df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
    df["ATR_pct"] = df["ATR"] / df["Close"]
    stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"])
    df["STOCH_k"] = stoch.stoch()
    df["STOCH_d"] = stoch.stoch_signal()
    df["momentum"] = df["Close"] - df["Close"].shift(5)
    df["momentum_10"] = df["Close"] - df["Close"].shift(10)
    df["volatility"] = df["Close"].rolling(10).std()
    df["volatility_20"] = df["Close"].rolling(20).std()
    df["return_1"] = df["Close"].pct_change(1)
    df["return_5"] = df["Close"].pct_change(5)
    df["return_10"] = df["Close"].pct_change(10)
    df["return_20"] = df["Close"].pct_change(20)
    df["high_low"] = df["High"] - df["Low"]
    df["close_open"] = df["Close"] - df["Open"]
    df["upper_shadow"] = df["High"] - df[["Close","Open"]].max(axis=1)
    df["lower_shadow"] = df[["Close","Open"]].min(axis=1) - df["Low"]
    df["volume_ma"] = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / (df["volume_ma"] + 1e-10)
    return df

def get_signal():
    ticker = yf.Ticker(SYMBOL)
    raw = ticker.history(period="5d", interval="5m")
    if raw.empty:
        return None
    df = build_features(raw)
    df = df.dropna()
    base = round(float(df["Close"].iloc[-1]), 4)
    rsi = round(float(df["RSI"].iloc[-1]), 1)
    macd_val = round(float(df["MACD"].iloc[-1]), 4)
    macd_sig = round(float(df["MACD_signal"].iloc[-1]), 4)
    atr = round(float(df["ATR"].iloc[-1]), 4)
    ema20 = round(float(df["EMA20"].iloc[-1]), 4)
    ema50 = round(float(df["EMA50"].iloc[-1]), 4)
    bb_upper = float(df["BB_upper"].iloc[-1])
    bb_lower = float(df["BB_lower"].iloc[-1])
    stoch_k = round(float(df["STOCH_k"].iloc[-1]), 1)

    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        last = scaler.transform(df[FEATURES].iloc[-1].values.reshape(1, -1))
        ai_pred = model.predict(last)[0]
        ai_proba = model.predict_proba(last)[0]
        ai_direction = "UP" if ai_pred == 1 else "DOWN"
        ai_confidence = round(max(ai_proba) * 100, 1)
    else:
        ai_direction = "UNKNOWN"
        ai_confidence = 0.0

    score = 0
    if rsi < 30: score += 2
    elif rsi > 70: score -= 2
    if macd_val > macd_sig: score += 1
    else: score -= 1
    if ema20 > ema50: score += 1
    else: score -= 1
    if base < bb_lower: score += 1
    elif base > bb_upper: score -= 1
    if stoch_k < 20: score += 1
    elif stoch_k > 80: score -= 1
    if ai_direction == "UP": score += 3
    elif ai_direction == "DOWN": score -= 3

    if score >= 2:
        sign = "BUY"
    elif score <= -2:
        sign = "SELL"
    else:
        sign = "WAIT"

    confidence = min(95, 45 + abs(score) * 5 + int(ai_confidence * 0.3))

    spread = 0.016
    if sign == "BUY":
        entry = round(base + spread, 4)
        tp = round(entry + atr * 2, 4)
        sl = round(entry - atr * 1.5, 4)
    elif sign == "SELL":
        entry = round(base - spread, 4)
        tp = round(entry - atr * 2, 4)
        sl = round(entry + atr * 1.5, 4)
    else:
        entry = base
        tp = 0
        sl = 0

    return {
        "signal": sign,
        "atr": atr,
        "confidence": confidence,
        "price": base,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "time": datetime.now().strftime("%H:%M:%S")
    }

print("AI自動シグナル生成開始")
while True:
    try:
        data = get_signal()
        if data:
            with open(SIGNAL_FILE, 'w') as f:
                json.dump(data, f)
            print(datetime.now().strftime("%H:%M:%S"), "シグナル:", data["signal"], "信頼度:", data["confidence"], "%")
    except Exception as e:
        print("エラー:", e)
    time.sleep(300)
