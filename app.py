import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from ta import add_all_ta_features
from lightgbm import LGBMClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ST_PAIR = "USDJPY=X"
SPREAD = 0.016 

st.set_page_config(page_title="AI Trading SOTA Pro-Eye", layout="wide")

@st.cache_data(ttl=60)
def get_data():
    df = yf.download(ST_PAIR, period="25d", interval="5m")
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    for col in df.columns:
        df[col] = np.array(df[col]).flatten()
    df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    df['hour'] = df.index.hour
    df['change_30m'] = df['Close'].pct_change(6)
    return df.dropna()

def train_ai(df):
    train_size = int(len(df) * 0.8)
    train_data = df.iloc[:train_size]
    target_move = SPREAD * 2.5
    future_diff = np.roll(train_data['Close'], -5) - train_data['Close']
    target = np.where(future_diff > target_move, 1, np.where(future_diff < -target_move, 2, 0))
    features = ['momentum_rsi', 'trend_macd', 'volatility_atr', 'change_30m', 'hour']
    model = LGBMClassifier(n_estimators=100, learning_rate=0.05, verbose=-1, random_state=42)
    model.fit(np.array(train_data[features]), target)
    return model, features

st.title("🦊 SOTA Pro-Eye")

try:
    data = get_data()
    model, features = train_ai(data)
    
    # 全データに対して予測
    X_all = np.array(data[features])
    probs = model.predict_proba(X_all)
    data['pred'] = np.argmax(probs, axis=1)
    data['prob'] = np.max(probs, axis=1)
    
    # 1. 設定（サイドバーの代わりに上部に配置）
    threshold = st.slider("AI信頼度のしきい値", 0.50, 0.95, 0.75)
    
    tab1, tab2 = st.tabs(["🔥 ライブ解析", "📊 戦略バックテスト"])

    with tab1:
        latest = data.iloc[-1]
        c1, c2 = st.columns(2)
        if latest['prob'] > threshold and latest['pred'] != 0:
            if latest['pred'] == 1: c1.success(f"🚀 BUY ({latest['prob']*100:.1f}%)")
            else: c1.error(f"📉 SELL ({latest['prob']*100:.1f}%)")
        else:
            c1.info(f"💤 WAIT ({latest['prob']*100:.1f}%)")
        c2.metric("現在価格", f"{latest['Close']:.3f}")

        # チャート表示
        plot_data = data.tail(100)
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Candlestick(x=plot_data.index, open=plot_data['Open'], high=plot_data['High'], low=plot_data['Low'], close=plot_data['Close'], name="価格"))
        
        # サインの描画
        buys = plot_data[(plot_data['pred'] == 1) & (plot_data['prob'] > threshold)]
        sells = plot_data[(plot_data['pred'] == 2) & (plot_data['prob'] > threshold)]
        fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.9998, mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'), name='BUY'))
        fig.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.0002, mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name='SELL'))
        
        fig.update_layout(height=450, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("直近データの検証結果")
        test_data = data.iloc[int(len(data)*0.8):].copy()
        
        # 簡易損益計算
        diff = np.roll(test_data['Close'], -5) - test_data['Close']
        pnl = []
        for i in range(len(test_data)-5):
            if test_data['prob'].iloc[i] > threshold:
                if test_data['pred'].iloc[i] == 1: pnl.append(diff[i] - SPREAD)
                elif test_data['pred'].iloc[i] == 2: pnl.append(-diff[i] - SPREAD)
        
        if pnl:
            pnl_ser = pd.Series(pnl)
            st.line_chart(pnl_ser.cumsum())
            m1, m2, m3 = st.columns(3)
            m1.metric("合計利益", f"{pnl_ser.sum()*100:.1f} pips")
            m2.metric("勝率", f"{(pnl_ser > 0).mean()*100:.1f}%")
            m3.metric("トレード数", f"{len(pnl)}回")
        else:
            st.warning("このしきい値では取引がありませんでした。")

except Exception as e:
    st.error(f"エラー: {e}")

if st.button("更新", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

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
