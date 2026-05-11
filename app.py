import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import requests
import joblib
import os

st.set_page_config(page_title="FX AI サイン", page_icon="📈")
st.title("FX AI サインアプリ")

if "history" not in st.session_state:
    st.session_state.history = []

MODEL_FILE = "fx_model.pkl"
SCALER_FILE = "fx_scaler.pkl"

def get_news_sentiment(pair):
    try:
        keywords = {"USD/JPY": "dollar yen", "EUR/USD": "euro dollar", "GBP/JPY": "pound yen", "EUR/JPY": "euro yen"}
        kw = keywords.get(pair, "forex")
        url = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=" + kw + "&region=US&lang=en-US"
        r = requests.get(url, timeout=5)
        text = r.text.lower()
        pos = text.count("rise") + text.count("gain") + text.count("bull") + text.count("up") + text.count("strong")
        neg = text.count("fall") + text.count("drop") + text.count("bear") + text.count("down") + text.count("weak")
        total = pos + neg
        if total == 0:
            return 0.0, "中立"
        score = (pos - neg) / total
        if score > 0.1:
            label = "強気"
        elif score < -0.1:
            label = "弱気"
        else:
            label = "中立"
        return round(score, 2), label
    except:
        return 0.0, "取得不可"

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

def train_and_save(df):
    df = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna(subset=FEATURES + ["target"])
    if len(df) < 80:
        return None, None, 0.0
    X = df[FEATURES].values
    y = df["target"].values
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss")
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    ensemble = VotingClassifier(estimators=[("rf", rf), ("xgb", xgb), ("gb", gb)], voting="soft", weights=[1, 2, 1])
    ensemble.fit(X_train_s, y_train)
    acc = round(accuracy_score(y_test, ensemble.predict(X_test_s)) * 100, 1)
    joblib.dump(ensemble, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    return ensemble, scaler, acc

def load_or_train(df):
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            scaler = joblib.load(SCALER_FILE)
            df2 = df.copy()
            df2["target"] = (df2["Close"].shift(-1) > df2["Close"]).astype(int)
            df2 = df2.dropna(subset=FEATURES + ["target"])
            if len(df2) < 10:
                return train_and_save(df)
            X = scaler.transform(df2[FEATURES].values)
            y = df2["target"].values
            acc = round(accuracy_score(y, model.predict(X)) * 100, 1)
            if acc < 52:
                st.warning("精度低下のため再学習中... (" + str(acc) + "%)")
                return train_and_save(df)
            return model, scaler, acc
        except:
            return train_and_save(df)
    return train_and_save(df)

def run_backtest(df, model, scaler):
    df = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna(subset=FEATURES + ["target"])
    if len(df) < 60:
        return None
    X = scaler.transform(df[FEATURES].values)
    preds = model.predict(X)
    probas = model.predict_proba(X)
    results = []
    capital = 10000.0
    wins = 0
    losses = 0
    for i in range(len(preds) - 1):
        proba = max(probas[i])
        if proba < 0.58:
            continue
        pred = preds[i]
        actual = df["target"].iloc[i]
        price = df["Close"].iloc[i]
        atr = df["ATR"].iloc[i]
        won = (pred == 1 and actual == 1) or (pred == 0 and actual == 0)
        pnl = atr * 2 if won else -atr * 1.5
        if won:
            wins += 1
        else:
            losses += 1
        capital += pnl * 100
        results.append({
            "時刻": df.index[i],
            "価格": round(price, 4),
            "予測": "BUY" if pred == 1 else "SELL",
            "結果": "勝" if won else "負",
            "損益(pips)": round(pnl, 4),
            "累計資金": round(capital, 2)
        })
    if not results:
        return None
    rdf = pd.DataFrame(results)
    total = wins + losses
    win_rate = round(wins / total * 100, 1) if total > 0 else 0
    return rdf, win_rate, wins, losses, round(capital - 10000, 2)

pair = st.selectbox("通貨ペア", ["USD/JPY", "EUR/USD", "GBP/JPY", "EUR/JPY"])
timeframe = st.selectbox("時間足", ["5分", "15分", "1時間", "4時間", "日足"])

symbols = {"USD/JPY": "USDJPY=X", "EUR/USD": "EURUSD=X", "GBP/JPY": "GBPJPY=X", "EUR/JPY": "EURJPY=X"}
periods = {"5分": ("5d", "5m"), "15分": ("5d", "15m"), "1時間": ("1mo", "1h"), "4時間": ("3mo", "1h"), "日足": ("1y", "1d")}


# フォワードテスト機能
if "forward_tests" not in st.session_state:
    st.session_state.forward_tests = []

def check_forward_results(current_price, pair):
    updated = []
    for test in st.session_state.forward_tests:
        if test["結果"] != "待機中":
            updated.append(test)
            continue
        if test["通貨ペア"] != pair:
            updated.append(test)
            continue
        entry = test["エントリー価格"]
        tp = test["TP"]
        sl = test["SL"]
        sign = test["サイン"]
        if sign == "BUY":
            if current_price >= tp:
                test["結果"] = "勝"
                test["損益pips"] = round(tp - entry, 4)
            elif current_price <= sl:
                test["結果"] = "負"
                test["損益pips"] = round(sl - entry, 4)
        elif sign == "SELL":
            if current_price <= tp:
                test["結果"] = "勝"
                test["損益pips"] = round(entry - tp, 4)
            elif current_price >= sl:
                test["結果"] = "負"
                test["損益pips"] = round(entry - sl, 4)
        updated.append(test)
    st.session_state.forward_tests = updated

tabs = st.tabs(["サイン", "バックテスト", "フォワードテスト"])

with tabs[0]:
    with st.spinner("価格取得・AI学習中..."):
        try:
            symbol = symbols[pair]
            period, interval = periods[timeframe]
            ticker = yf.Ticker(symbol)
            raw = ticker.history(period=period, interval=interval)
            if raw.empty:
                st.error("価格データが取得できませんでした")
                st.stop()
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
            bb_pct = round(float(df["BB_pct"].iloc[-1]), 2)
            news_score, news_label = get_news_sentiment(pair)
            model, scaler, acc = load_or_train(df)
            if model:
                last = scaler.transform(df[FEATURES].iloc[-1].values.reshape(1, -1))
                ai_pred = model.predict(last)[0]
                ai_proba = model.predict_proba(last)[0]
                ai_direction = "UP" if ai_pred == 1 else "DOWN"
                ai_confidence = round(max(ai_proba) * 100, 1)
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
                if news_score > 0.1: score += 1
                elif news_score < -0.1: score -= 1
                if score >= 2:
                    sign = "BUY"
                elif score <= -2:
                    sign = "SELL"
                else:
                    sign = "WAIT"
                confidence = min(95, 45 + abs(score) * 5 + int(ai_confidence * 0.3))
            else:
                ai_direction = "UNKNOWN"
                ai_confidence = 0.0
                sign = "WAIT"
                confidence = 50

            st.metric(pair + " 現在価格", base)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RSI", rsi)
            col2.metric("STOCH", stoch_k)
            col3.metric("BB%", bb_pct)
            col4.metric("AI精度", str(acc) + "%")
            col5, col6 = st.columns(2)
            col5.metric("EMA20", ema20)
            col6.metric("ATR", atr)
            st.divider()
            col7, col8 = st.columns(2)
            arrow = "上昇予測" if ai_direction == "UP" else "下降予測"
            col7.info("AI: " + arrow + " " + str(ai_confidence) + "%")
            col8.info("ニュース: " + news_label)
            if sign == "BUY":
                st.success("BUY  信頼度: " + str(confidence) + "%")
                tp = round(base + atr * 2, 4)
                sl = round(base - atr * 1.5, 4)
                col1, col2 = st.columns(2)
                col1.metric("TP利確", tp)
                col2.metric("SL損切", sl)
            elif sign == "SELL":
                st.error("SELL  信頼度: " + str(confidence) + "%")
                tp = round(base - atr * 2, 4)
                sl = round(base + atr * 1.5, 4)
                col1, col2 = st.columns(2)
                col1.metric("TP利確", tp)
                col2.metric("SL損切", sl)
            else:
                st.warning("WAIT  信頼度: " + str(confidence) + "%")
            st.caption("最終更新: " + datetime.now().strftime("%H:%M:%S"))
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], increasing_line_color="green", decreasing_line_color="red", name="ローソク足"))
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], line=dict(color="blue", width=1, dash="dash"), name="BB上限"))
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], line=dict(color="blue", width=1, dash="dash"), name="BB下限"))
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], line=dict(color="orange", width=1), name="EMA20"))
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], line=dict(color="purple", width=1), name="EMA50"))
            fig.update_layout(title=pair + " " + timeframe + "チャート", height=400, xaxis_rangeslider_visible=False, paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="green"), name="RSI"))
            fig2.add_hline(y=70, line_dash="dash", line_color="red")
            fig2.add_hline(y=30, line_dash="dash", line_color="blue")
            fig2.update_layout(title="RSI", height=200, paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True)
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df.index, y=df["MACD"], line=dict(color="blue"), name="MACD"))
            fig3.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], line=dict(color="red"), name="シグナル"))
            fig3.update_layout(title="MACD", height=200, paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.error("エラー: " + str(e))
            st.stop()

    if st.button("手動でサインを再確認", use_container_width=True):
        if sign == "BUY":
            st.success("BUY  信頼度: " + str(confidence) + "%")
        elif sign == "SELL":
            st.error("SELL  信頼度: " + str(confidence) + "%")
        else:
            st.warning("WAIT  信頼度: " + str(confidence) + "%")

    st.divider()
    st.subheader("シグナル履歴")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.info("まだ履歴がありません")
    st.caption("このアプリは投資アドバイスではありません")

with tabs[1]:
    st.subheader("バックテスト")
    if st.button("バックテスト実行", use_container_width=True):
        with st.spinner("バックテスト実行中..."):
            try:
                symbol = symbols[pair]
                period, interval = periods[timeframe]
                ticker = yf.Ticker(symbol)
                raw = ticker.history(period=period, interval=interval)
                df = build_features(raw)
                df = df.dropna()
                model, scaler, acc = load_or_train(df)
                result = run_backtest(df, model, scaler)
                if result is None:
                    st.warning("データが少なすぎます。時間足を長くしてください。")
                else:
                    rdf, win_rate, wins, losses, total_pnl = result
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("勝率", str(win_rate) + "%")
                    col2.metric("勝ち", str(wins) + "回")
                    col3.metric("負け", str(losses) + "回")
                    col4.metric("総損益", str(total_pnl) + "円")
                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(x=rdf["時刻"], y=rdf["累計資金"], line=dict(color="green" if total_pnl >= 0 else "red", width=2), name="累計資金"))
                    fig_bt.update_layout(title="資金推移", height=300, paper_bgcolor="white", plot_bgcolor="white")
                    st.plotly_chart(fig_bt, use_container_width=True)
                    st.dataframe(rdf, use_container_width=True)
            except Exception as e:
                st.error("エラー: " + str(e))
    if st.button("モデルを強制再学習", use_container_width=True):
        if os.path.exists(MODEL_FILE):
            os.remove(MODEL_FILE)
        if os.path.exists(SCALER_FILE):
            os.remove(SCALER_FILE)
        st.success("次回更新時に再学習します")
    st.caption("このアプリは投資アドバイスではありません")

with tabs[2]:
    st.subheader("フォワードテスト")

    if "base" in dir() or True:
        try:
            check_forward_results(base, pair)
        except:
            pass

    if st.button("現在のサインをフォワードテストに登録", use_container_width=True):
        try:
            if sign != "WAIT":
                if sign == "BUY":
                    tp_f = round(base + atr * 2, 4)
                    sl_f = round(base - atr * 1.5, 4)
                else:
                    tp_f = round(base - atr * 2, 4)
                    sl_f = round(base + atr * 1.5, 4)
                st.session_state.forward_tests.insert(0, {
                    "登録時刻": datetime.now().strftime("%H:%M:%S"),
                    "通貨ペア": pair,
                    "サイン": sign,
                    "エントリー価格": base,
                    "TP": tp_f,
                    "SL": sl_f,
                    "信頼度": str(confidence) + "%",
                    "結果": "待機中",
                    "損益pips": "-"
                })
                st.success(sign + " を登録しました！価格が TP/SL に達したら自動判定されます")
            else:
                st.warning("WAITサインは登録できません")
        except:
            st.error("まずサインタブで価格を取得してください")

    st.divider()

    if st.session_state.forward_tests:
        ft_df = pd.DataFrame(st.session_state.forward_tests)
        wins = len(ft_df[ft_df["結果"] == "勝"])
        losses = len(ft_df[ft_df["結果"] == "負"])
        waiting = len(ft_df[ft_df["結果"] == "待機中"])
        total = wins + losses

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("勝ち", str(wins) + "回")
        col2.metric("負け", str(losses) + "回")
        col3.metric("待機中", str(waiting) + "回")
        col4.metric("勝率", str(round(wins/total*100, 1)) + "%" if total > 0 else "-")

        if total > 0:
            completed = ft_df[ft_df["結果"] != "待機中"].copy()
            if len(completed) > 0:
                completed["損益pips"] = pd.to_numeric(completed["損益pips"], errors="coerce")
                fig_fw = go.Figure()
                fig_fw.add_trace(go.Bar(
                    x=list(range(len(completed))),
                    y=completed["損益pips"],
                    marker_color=["green" if v > 0 else "red" for v in completed["損益pips"]],
                    name="損益"
                ))
                fig_fw.update_layout(title="フォワードテスト損益", height=250, paper_bgcolor="white", plot_bgcolor="white")
                st.plotly_chart(fig_fw, use_container_width=True)

        st.dataframe(ft_df, use_container_width=True)

        if st.button("フォワードテスト履歴をクリア", use_container_width=True):
            st.session_state.forward_tests = []
            st.success("クリアしました")
    else:
        st.info("まだ登録がありません。サインタブでサインを確認後、登録ボタンを押してください。")

    st.caption("このアプリは投資アドバイスではありません")
