import MetaTrader5 as mt5
import requests
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

def get_ai_signal():
    try:
        r = requests.get("http://133.117.72.33/signal", timeout=10)
        return r.json()
    except:
        return None

def close_all_orders():
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions:
        for pos in positions:
            order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(SYMBOL).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(SYMBOL).ask
            mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": pos.volume, "type": order_type, "position": pos.ticket, "price": price, "magic": MAGIC, "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC})

def place_order(signal, tp, sl):
    lot = get_lot()
    order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
    price = mt5.symbol_info_tick(SYMBOL).ask if signal == "BUY" else mt5.symbol_info_tick(SYMBOL).bid
    mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": lot, "type": order_type, "price": price, "sl": sl, "tp": tp, "magic": MAGIC, "comment": "FX AI Auto", "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC})

print("FX AI 自動売買開始")
last_signal = None
while True:
    try:
        signal_data = get_ai_signal()
        if signal_data and signal_data.get("signal") in ["BUY", "SELL"]:
            signal = signal_data["signal"]
            if signal != last_signal:
                close_all_orders()
                time.sleep(1)
                tick = mt5.symbol_info_tick(SYMBOL)
                price = tick.ask if signal == "BUY" else tick.bid
                atr = signal_data.get("atr", 0.05)
                tp = round(price + atr * 2, 3) if signal == "BUY" else round(price - atr * 2, 3)
                sl = round(price - atr * 1.5, 3) if signal == "BUY" else round(price + atr * 1.5, 3)
                place_order(signal, tp, sl)
                last_signal = signal
        elif signal_data and signal_data.get("signal") == "WAIT" and last_signal in ["BUY", "SELL"]:
            close_all_orders()
            last_signal = "WAIT"
    except Exception as e:
        print(datetime.now(), "エラー:", e)
    time.sleep(300)
