from abc import ABC
from typing import List
from configs.configSecrets import ConfigSecrets
from .provider import Provider
import time
from alpaca.data import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data import CryptoLatestQuoteRequest
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
from alpaca.trading import GetAssetsRequest, AssetClass, Position, TradeAccount, Order
from alpaca.trading.client import TradingClient
from alpaca.trading.client import TradingClient
from .predictor import Prediction
from alpaca.data import CryptoSnapshotRequest, Snapshot
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest, CancelOrderResponse, \
    GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, OrderType, AssetExchange, PositionSide, \
    QueryOrderStatus
from alpaca.trading import StopLimitOrderRequest


API_KEY = ConfigSecrets.ALPACA_API_KEY #os.environ['API_KEY']
API_SECRET = ConfigSecrets.ALPACA_API_SECRET #os.environ['API_SECRET']
BASE_URL = "https://paper-api.alpaca.markets"


class AlpacaProvider(Provider):

    def __init__(self):
        self.trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
        self.account: TradeAccount = self.trading_client.get_account()

    def getOpenOrders(self, symbol: str) -> List[Order]:
        ordersFilter: GetOrdersRequest = GetOrdersRequest()
        ordersFilter.status = QueryOrderStatus.OPEN
        ordersFilter.symbols = [symbol]
        orders: List[Order] = self.trading_client.get_orders(ordersFilter)
        return orders

    def getOpenPosition(self, symbol: str) -> Position:
        currentPosition: Position = self.trading_client.get_open_position(
            symbol)
        return currentPosition

    def liquidate(self, symbol: str) -> Order:
        order: Order = self.trading_client.close_position(symbol)
        return order

    def liquidateAll(self) -> None:
        self.trading_client.close_all_positions(cancel_orders=True)

    def cancel(self, symbol: str):
        orders: List[Order] = self.getOpenOrders(symbol)
        for order in orders:
            self.cancel(order.symbol)

    def cancelAll(self) -> None:
        print(" Cancelling all")
        cancelStatuses: List[CancelOrderResponse] = self.trading_client.cancel_orders()
        for status in cancelStatuses:
            print(" cancelling order id " + str(status.id) + " http response " + str(status.status))
            #order: Order = status.body
            #print(" cancelling order symbol " + order.symbol + " side " + order.side + " qty " + order.qty)
        return None

    def waitForFillOrder(self, orderId: str) -> bool:
        while True:
            time.sleep(10)

            filled = self.trading_client.get_order_by_id(orderId)
            if filled.filled_qty is not None:
                return True

    def shortOrder(self, symbol: str, notional: int) -> Order:
        # symbol='BTC/USD'
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            #qty=0.023,
            notional=notional,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            #order_class=OrderClass.,
            type=OrderType.MARKET,
            #stop_loss=stopLossRequest,
            #take_profit=takeProfitRequest,
            #client_order_id="first_algo"
        )

        order: Order = self.trading_client.submit_order(
            order_data=market_order_data
        )

        return order

    def longOrder(self, symbol: str, notional: int) -> Order:
        # symbol='BTC/USD'
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            #qty=0.023,
            notional=notional,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            #order_class=OrderClass.,
            type=OrderType.MARKET,
            #stop_loss=stopLossRequest,
            #take_profit=takeProfitRequest,
            #client_order_id="first_algo"
        )

        order: Order = self.trading_client.submit_order(
            order_data=market_order_data
        )

        return order

    def longStopOrder(self, symbol: str, positionPrice: float) -> Order:
        # "BTCUSD"
        qty = self.trading_client.get_open_position(symbol).qty
        stopOrderRequest: StopLimitOrderRequest = StopLimitOrderRequest(symbol=symbol, qty=qty,
                                                                        side=OrderSide.SELL,
                                                                        type=OrderType.STOP,
                                                                        time_in_force=TimeInForce.GTC,
                                                                        stop_price = positionPrice * .97,
                                                                        limit_price = positionPrice * 1.05)
        stopOrder = self.trading_client.submit_order(
            order_data=stopOrderRequest
        )

        return stopOrder

    def shortStopOrder(self, symbol: str, positionPrice: float) -> Order:
        # "BTCUSD"
        qty = self.trading_client.get_open_position(symbol).qty
        stopOrderRequest: StopLimitOrderRequest = StopLimitOrderRequest(symbol=symbol, qty=qty,
                                                                        side=OrderSide.SELL,
                                                                        type=OrderType.STOP,
                                                                        time_in_force=TimeInForce.GTC,
                                                                        stop_price = positionPrice * 1.03,
                                                                        limit_price = positionPrice * .95)
        stopOrder = self.trading_client.submit_order(
                order_data=stopOrderRequest
        )

        return stopOrder



