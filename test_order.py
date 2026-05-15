import MetaTrader5 as mt5
mt5.initialize()
mt5.symbol_select('USDJPYm', True)
tick = mt5.symbol_info_tick('USDJPYm')
print("現在価格:", tick.ask)
r = mt5.order_send({
    'action': mt5.TRADE_ACTION_DEAL,
    'symbol': 'USDJPYm',
    'volume': 0.01,
    'type': mt5.ORDER_TYPE_BUY,
    'price': tick.ask,
    'magic': 12345,
    'type_time': mt5.ORDER_TIME_GTC,
    'type_filling': mt5.ORDER_FILLING_IOC
})
print("結果:", r.retcode)
print("詳細:", r.comment)
