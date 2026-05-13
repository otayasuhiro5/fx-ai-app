import MetaTrader5 as mt5
import requests
import json
import time
from datetime import datetime

mt5.initialize()
SYMBOL = "USDJPY"
MAGIC = 12345
RISK_PERCENT = 0.001

def get_lot():
    account = mt5.account_info()
    balance = account.balance
    lot = round(balance * RISK_PERCENT, 2)
    return max(0.01, min(10.0, lot))

def get_signal():
    try:
        r = requests.get("http://133.117.72.33:5000/signal", timeout=10)
        return r.json()
    except:
        return None

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
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": SYMBOL,
                "volume": pos.volume,
                "type": order_type,
                "position": pos.ticket,
                "price": price,
                "magic": MAGIC,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            mt5.order_send(request)
            print(datetime.now(), "ポジションクローズ")

def place_order(signal, tp, sl):
    lot = get_lot()
    if signal == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(SYMBOL).ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(SYMBOL).bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": MAGIC,
        "comment": "FX AI Auto",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    print(datetime.now(), "注文:", signal, "ロット:", lot, "結果:", result.retcode)
    return result

print("FX AI 自動売買開始")
print("通貨ペア:", SYMBOL)
print("リスク:", RISK_PERCENT * 100, "%")
last_signal = None

while True:
    try:
        data = get_signal()
        if data:
            signal = data.get("signal", "WAIT")
            tp = data.get("tp", 0)
            sl = data.get("sl", 0)
            confidence = data.get("confidence", 0)
            print(datetime.now().strftime("%H:%M:%S"), "シグナル:", signal, "信頼度:", confidence, "%")

            if signal in ["BUY", "SELL"] and signal != last_signal:
                print("新しいサイン検出！注文実行中...")
                close_all_orders()
                time.sleep(1)
                if tp > 0 and sl > 0:
                    place_order(signal, tp, sl)
                    last_signal = signal
            elif signal == "WAIT" and last_signal in ["BUY", "SELL"]:
                print("WAITサイン - クローズ")
                close_all_orders()
                last_signal = "WAIT"
    except Exception as e:
        print(datetime.now(), "エラー:", e)
    time.sleep(300)
