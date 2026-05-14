import MetaTrader5 as mt5
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
import time
import json

SYMBOL = "USDJPY"
RISK_PERCENT = 0.001
MAGIC = 12345
MODEL_FILE = "C:/fx_model_v2.pkl"
SCALER_FILE = "C:/fx_scaler_v2.pkl"
LEARN_FILE = "C:/fx_learn_data.json"
SPREAD = 0.026

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

def train_model(df, extra_data=None):
    df = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna(subset=FEATURES + ["target"])
    if extra_data and len(extra_data) > 0:
        extra_df = pd.DataFrame(extra_data)
        df = pd.concat([df, extra_df], ignore_index=True)
    if len(df) < 50:
        return None, None, 0.0
    X = df[FEATURES].values
    y = df["target"].values
    split = int(len(X) * 0.8)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss")
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    ensemble = VotingClassifier(estimators=[("rf", rf), ("xgb", xgb), ("gb", gb)], voting="soft", weights=[1,2,1])
    ensemble.fit(X_scaled[:split], y[:split])
    from sklearn.metrics import accuracy_score
    acc = round(accuracy_score(y[split:], ensemble.predict(X_scaled[split:])) * 100, 1) if split < len(X) else 0.0
    joblib.dump(ensemble, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(datetime.now().strftime("%H:%M:%S"), "モデル更新 精度:", acc, "%")
    return ensemble, scaler, acc

def load_learn_data():
    try:
        with open(LEARN_FILE) as f:
            return json.load(f)
    except:
        return []

def save_learn_data(data):
    with open(LEARN_FILE, 'w') as f:
        json.dump(data[-500:], f)

def get_lot():
    account = mt5.account_info()
    if account is None:
        return 0.01
    balance = account.balance
    lot = round(balance * RISK_PERCENT, 2)
    return max(0.01, min(10.0, lot))

def close_all_orders():
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions:
        for pos in positions:
            if pos.type == mt5.ORDER_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(SYMBOL).bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(SYMBOL).ask
            mt5.order_send({
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": SYMBOL,
                "volume": pos.volume,
                "type": order_type,
                "position": pos.ticket,
                "price": price,
                "magic": MAGIC,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            })
            print(datetime.now().strftime("%H:%M:%S"), "クローズ完了")

def place_order(signal, tp, sl):
    lot = get_lot()
    if signal == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(SYMBOL).ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(SYMBOL).bid
    result = mt5.order_send({
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": MAGIC,
        "comment": "FX AI v2",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    })
    print(datetime.now().strftime("%H:%M:%S"), "注文:", signal, "ロット:", lot, "結果:", result.retcode)
    return result

print("FX AI 自動売買 v2 開始（継続学習あり）")
print("通貨ペア:", SYMBOL)
print("リスク:", RISK_PERCENT * 100, "%")

mt5.initialize()
last_signal = None
last_position_price = None
learn_data = load_learn_data()
retrain_counter = 0

ticker = yf.Ticker("USDJPY=X")
raw = ticker.history(period="5d", interval="5m")
df_base = build_features(raw)
df_base = df_base.dropna()
model, scaler, acc = train_model(df_base, learn_data)
print("初回学習完了 精度:", acc, "%")

while True:
    try:
        ticker = yf.Ticker("USDJPY=X")
        raw = ticker.history(period="5d", interval="5m")
        if raw.empty:
            time.sleep(60)
            continue

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

        if model and scaler:
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

        if sign == "BUY":
            entry = round(base + SPREAD, 4)
            tp = round(entry + atr * 2, 4)
            sl = round(entry - atr * 1.5, 4)
        elif sign == "SELL":
            entry = round(base - SPREAD, 4)
            tp = round(entry - atr * 2, 4)
            sl = round(entry + atr * 1.5, 4)
        else:
            tp = sl = 0

        print(datetime.now().strftime("%H:%M:%S"), "シグナル:", sign, "信頼度:", confidence, "% 価格:", base)

        if last_signal in ["BUY", "SELL"] and last_position_price:
            current_price = base
            if last_signal == "BUY":
                result = 1 if current_price > last_position_price else 0
            else:
                result = 1 if current_price < last_position_price else 0
            row = df[FEATURES].iloc[-1].to_dict()
            row["target"] = result
            learn_data.append(row)
            save_learn_data(learn_data)
            retrain_counter += 1
            if retrain_counter >= 10:
                model, scaler, acc = train_model(df, learn_data)
                retrain_counter = 0

        positions = mt5.positions_get(symbol=SYMBOL)
        has_position = positions is not None and len(positions) > 0

        if sign in ["BUY", "SELL"] and not has_position:
            print("新しいサイン！注文実行中...")
            if tp > 0 and sl > 0:
                place_order(sign, tp, sl)
                last_signal = sign
                last_position_price = base
        elif has_position:
            print("ポジション保有中 TP/SL待機...")

    except Exception as e:
        print(datetime.now().strftime("%H:%M:%S"), "エラー:", e)
    time.sleep(300)
