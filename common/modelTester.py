from typing import List

import warnings
from pandas import DataFrame, Series

warnings.filterwarnings('ignore')

from .portfolioMaker import PortfolioMaker
import vectorbtpro as vbt
from common.labelMaker import LabelMaker
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .portfolioWrapper import PortfolioWrapper

class ModelTester:
    def __init__(self):
        self.portfolioMaker = PortfolioMaker()

    def createPortfolioData(self, close: Series, comparisonDf, period, longMin, shortMin) -> DataFrame:
        allData: DataFrame = DataFrame()
        allData[LabelMaker.makeLongEntryLabel(period)] = comparisonDf['predictedLongEntry_'+str(longMin)].astype(bool)
        allData[LabelMaker.makeShortEntryLabel(period)] = comparisonDf['predictedShortEntry_'+str(shortMin)].astype(bool)
        allData.index = comparisonDf.index
        allData['Close'] = close
        return allData

    def createPortfolio(self, close: Series, comparisonDf,
                        period: int, longMin, shortMin) -> PortfolioWrapper:
        allData: DataFrame = self.createPortfolioData(close, comparisonDf, period, longMin, shortMin)
        pf: PortfolioWrapper = self.portfolioMaker.makePortfolio(allData, period)
        return pf

    def createPortfolioComparisonDf(self, close: Series, comparisonDf, period: int) -> DataFrame:
        d = []
        for longMin in [x / 100.0 for x in range(0, 100, 5)]:
            for shortMin in [x / 100.0 for x in range(0, 100, 5)]:
                pf = self.createPortfolio(close, comparisonDf, period, longMin, shortMin).portfolio
                stats=pf.stats()
                d.append({
                    'LongMin': longMin,
                    'ShortMin': shortMin,
                    'Trades': stats['Total Trades'],
                    'TotalReturn': stats['Total Return [%]'],
                    'MaxDrawdown': stats['Max Drawdown [%]'],
                    'MaxDrawdownDuration': stats['Max Drawdown Duration'],
                    'WinRate': stats['Win Rate [%]']
                })
        return pd.DataFrame(d)

    def createMinScoreHistogram(self, portfolioStats: DataFrame, values: str) -> None:
        #https://blog.quantinsti.com/creating-heatmap-using-python-seaborn/

        result: DataFrame = portfolioStats.pivot(index='LongMin', columns='ShortMin', values=values)
        fig, ax = plt.subplots(figsize=(13,7))
        title = values

        # Set the font size and the distance of the title from the plot
        plt.title(title,fontsize=18)
        ttl = ax.title
        ttl.set_position([0.5,1.05])
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.axis('off')
        sns.heatmap(result, fmt="", cmap='RdYlGn', linewidths=0.30, ax=ax)
        plt.show()

    def randomPortfolioProbability(self, close):
        pf = vbt.PF.from_random_signals(
            close,
            prob=.007,
            run_kwargs=dict(hide_params=True),
            tp_stop=0.2,
            sl_stop=0.05,
            direction="both",# "both" "longonly"
            size=10000,
            size_type='value',
            init_cash='auto'
        )
        #pf.trade_history
        #pf.stats()
        return pf

    def createRandomPortfolios(self, close: Series) -> List:
        d = []
        for iteration in range(2000):
            pf = self.randomPortfolioProbability(close)
            stats = pf.stats()
            d.append({
                'Trades': stats['Total Trades'],
                'TotalReturn': stats['Total Return [%]'],
                'MaxDrawdown': stats['Max Drawdown [%]'],
                'MaxDrawdownDuration': stats['Max Drawdown Duration'],
                'WinRate': stats['Win Rate [%]']
            })
        randomPortfolios = pd.DataFrame(d)
        randomPortfolios = randomPortfolios.reset_index()
        #randomPortfolios["TotalReturn"] = randomPortfolios["TotalReturn"] * 100
        #randomPortfolios.plot.scatter(x="index", y="TotalReturn")
        #randomPortfolios.plot.hist(bins=100, column=["TotalReturn"])#, "WinRate", "MaxDrawdown"])
        return randomPortfolios

