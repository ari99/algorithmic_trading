import vectorbtpro as vbt
import pandas as pd
from pandas import DataFrame, Series
from vectorbtpro import Portfolio

from .cleanEntriesExits import CleanEntriesExits
from .portfolioWrapper import PortfolioWrapper

idx = pd.IndexSlice


class PortfolioMaker:

    # def __init__(self, allData: DataFrame):

    # self.pf10: Portfolio = self.makePortfolio(allData, 10)
    # self.pf20: Portfolio = self.makePortfolio(allData, 20)
    # self.pf50: Portfolio = self.makePortfolio(allData, 50)
    # self.pf100: Portfolio = self.makePortfolio(allData, 100)
    # pf100.stats()

    # https://vectorbt.pro/pvt_d904e513/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals
    def makePortfolio(self, allData: DataFrame, period: int) -> PortfolioWrapper:
        cleanedEntriesExits = CleanEntriesExits(allData, period)

        pf = self.doPortfolio(allData.get('Close'),
                              cleanedEntriesExits.cleanLongEntries,
                              cleanedEntriesExits.cleanLongExits,
                              cleanedEntriesExits.cleanShortEntries,
                              cleanedEntriesExits.cleanShortExits)

        return PortfolioWrapper(allData.get('Close'),
                                cleanedEntriesExits.cleanLongEntries,
                                cleanedEntriesExits.cleanLongExits,
                                cleanedEntriesExits.cleanShortEntries,
                                cleanedEntriesExits.cleanShortExits,
                                pf
                                )

    def doPortfolio(self, close: Series, longEntries: Series,
                    longExits: Series, shortEntries: Series, shortExits: Series):

        pf: Portfolio = vbt.Portfolio.from_signals(
            close=close,
            short_entries=shortEntries,
            short_exits=shortExits,
            entries=longEntries,
            exits=longExits,
            size=10000,
            size_type='value',
            init_cash='auto',
            tp_stop=0.3,
            sl_stop=0.03,
            # https://vectorbt.pro/pvt_d904e513/documentation/portfolio/from-signals/#stop-orders

        )

        return pf
