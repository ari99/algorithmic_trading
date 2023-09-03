
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.trading import Order
from .predictor import Prediction
from alpaca.data import CryptoSnapshotRequest, Snapshot
from alpaca.trading.enums import PositionSide
from .provider import Provider


class Trader:

    def __init__(self, provider: Provider):
        self.provider = provider
        self.historyClient = CryptoHistoricalDataClient()

    def doTrade(self, prediction: Prediction):

        currentPosition = self.provider.getOpenPosition(prediction.providerOpenPositionSymbol)
        #unrealizedPnlPercent: str = currentPosition.unrealized_plpc
        #exchange: AssetExchange = currentPosition.exchange
        #positionSymbol: str = currentPosition.symbol
        #quantity: str = currentPosition.qty

        side: PositionSide = currentPosition.side

        if side == PositionSide.LONG and prediction.shouldShort:
            self.flipLongToShort(prediction.providerSymbol)
        elif side == PositionSide.SHORT and prediction.shouldLong:
            self.flipShortToLong(prediction.providerSymbol)
        elif prediction.shouldLiquidate:
            self.provider.liquidate(prediction.providerSymbol)


    def flipLongToShort(self, symbol):
        self.provider.cancel(symbol)
        liquidateOrder: Order = self.provider.liquidate(symbol)
        self.provider.waitForFill(str(liquidateOrder.id))
        snapshot: Snapshot = self.historyClient.get_crypto_snapshot(
            CryptoSnapshotRequest(symbol_or_symbols=symbol))  # "BTC/USD"
        symbol_price = snapshot[symbol].minute_bar.close
        shortOrder: Order = self.provider.shortOrder(symbol, 10000)
        self.provider.waitForFill(str(shortOrder.id))
        stopOrder: Order = self.provider.shortStopOrder(symbol, symbol_price)

    def flipShortToLong(self, symbol):
        self.provider.cancel(symbol)
        liquidateOrder: Order = self.provider.liquidate(symbol)
        self.provider.waitForFill(str(liquidateOrder.id))
        snapshot: Snapshot = self.historyClient.get_crypto_snapshot(
            CryptoSnapshotRequest(symbol_or_symbols=symbol))  # "BTC/USD"
        symbol_price = snapshot[symbol].minute_bar.close
        longOrder: Order = self.provider.longOrder(symbol, 10000)
        self.provider.waitForFill(str(longOrder.id))
        stopOrder: Order = self.provider.longStopOrder(symbol, symbol_price)




