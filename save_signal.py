import json
import os

def save_signal(sign, atr, confidence, base):
    data = {
        "signal": sign,
        "atr": atr,
        "confidence": confidence,
        "price": base
    }
    with open('/root/fx-ai-app/signal.json', 'w') as f:
        json.dump(data, f)
