# https://github.com/polygon-io/client-python
from polygon import WebSocketClient
from polygon.websocket.models import WebSocketMessage
from typing import List
from vectorbtpro import Portfolio, Ranges
from vectorbtpro._typing import SeriesFrame, MaybeTuple

from configs.configSecrets import ConfigSecrets

ws = WebSocketClient(
    api_key=ConfigSecrets.POLYGON_API_KEY,
    feed='socket.polygon.io',
    market='crypto',
    subscriptions=["XA.X:BTC-USD"] # xa is minute aggregate, xt each trade, xq quote etc
)


def handle_msg(msg: List[WebSocketMessage]):
    for m in msg:
        print(m)

ws.run(handle_msg=handle_msg)


#longs: MaybeTuple[SeriesFrame] = resultLongEntries.vbt.signals.clean(resultShortEntries)
#shorts: MaybeTuple[SeriesFrame] = resultShortEntries.vbt.signals.clean(resultLongEntries)
#cleanLongEntries: SeriesFrame = longs[0]
#cleanLongExits: SeriesFrame = longs[1]
#cleanShortEntries: SeriesFrame = shorts[0]
#cleanShortExits: SeriesFrame = shorts[1]
#%%
