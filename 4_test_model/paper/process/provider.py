from abc import ABC, abstractmethod
from alpaca.trading import GetAssetsRequest, AssetClass, Position, Order


class Provider(ABC):

    @abstractmethod
    def getOpenPosition(self, symbol: str) -> Position:
        pass

    @abstractmethod
    def getOpenOrders(self, symbol: str) -> Position:
        pass

    @abstractmethod
    def liquidate(self, symbol: str) -> Order:
        pass

    @abstractmethod
    def liquidateAll(self) -> None:
        pass

    @abstractmethod
    def cancel(self, symbol: str) -> None:
        pass

    @abstractmethod
    def cancelAll(self) -> None:
        pass

    @abstractmethod
    def waitForFill(self, symbol: str) -> Position:
        pass

    @abstractmethod
    def shortOrder(self, symbol: str, notional: int) -> Order:
        pass

    @abstractmethod
    def longOrder(self, symbol: str, notional: int) -> Order:
        pass

    @abstractmethod
    def longStopOrder(self, symbol: str, positionPrice: float) -> Order:
        pass

    @abstractmethod
    def shortStopOrder(self, symbol: str, positionPrice: float) -> Order:
        pass