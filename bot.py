import websocket
import json
import ssl
import time

URL = "wss://fstream.binance.com/ws/ethusdt@kline_4h"

print("Testing Binance Futures WebSocket...")
print(URL)

try:
    ws = websocket.create_connection(
        URL,
        timeout=15,
        sslopt={"cert_reqs": ssl.CERT_REQUIRED}
    )

    print("CONNECTED ✅")

    for i in range(5):
        message = ws.recv()
        data = json.loads(message)

        print(
            "MESSAGE:",
            data
        )

    ws.close()

    print("WebSocket test completed ✅")

except Exception as e:

    print(
        "WEBSOCKET ERROR:",
        repr(e)
    )
