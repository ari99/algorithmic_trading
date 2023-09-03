
from vectorbtpro import Portfolio

class PortfolioWrapper:
    def __init__(self, close, longEntries, longExits, shortEntries, shortExits, portfolio: Portfolio):
        self.close = close
        self.longEntries = longEntries
        self.longExits = longExits
        self.shortEntries = shortEntries
        self.shortExits = shortExits
        self.portfolio = portfolio

