from common.portfolioMaker import PortfolioMaker
from pandas import DataFrame
from common.labelMaker import LabelMaker
import vectorbtpro as vbt
from vectorbtpro.data.custom import LocalDataT

from common.portfolioWrapper import PortfolioWrapper


class StatsMaker():

    def __init__(self, read10Df: DataFrame, read20Df: DataFrame,
                 read50Df: DataFrame, read100Df: DataFrame,):
        self.read10Df = read10Df
        self.read20Df = read20Df
        self.read50Df = read50Df
        self.read100Df = read100Df

    def __createAllData(self) -> DataFrame:
        allData = DataFrame()
        allData[LabelMaker.makeLongEntryLabel(10)] = self.read10Df['predictedLongEntry'].astype(bool)
        allData[LabelMaker.makeShortEntryLabel(10)] = self.read10Df['predictedShortEntry'].astype(bool)
        allData[LabelMaker.makeLongEntryLabel(20)] = self.read20Df['predictedLongEntry'].astype(bool)
        allData[LabelMaker.makeShortEntryLabel(20)] = self.read20Df['predictedShortEntry'].astype(bool)
        allData[LabelMaker.makeLongEntryLabel(50)] = self.read50Df['predictedLongEntry'].astype(bool)
        allData[LabelMaker.makeShortEntryLabel(50)] = self.read50Df['predictedShortEntry'].astype(bool)
        allData[LabelMaker.makeLongEntryLabel(100)] = self.read100Df['predictedLongEntry'].astype(bool)
        allData[LabelMaker.makeShortEntryLabel(100)] = self.read100Df['predictedShortEntry'].astype(bool)
        allData.index = self.read10Df.index
        barData: LocalDataT = vbt.HDFData.fetch('../my_data_2017-2023.h5')
        symbolData = barData.data['X:BTCUSD']
        allData['Close'] = symbolData['Close']

        return allData

    def __create10Data(self) -> DataFrame:
        allData = DataFrame()
        allData[LabelMaker.makeLongEntryLabel(10)] = self.read10Df['predictedLongEntry'].astype(bool)
        allData[LabelMaker.makeShortEntryLabel(10)] = self.read10Df['predictedShortEntry'].astype(bool)
        allData.index = self.read10Df.index
        barData: LocalDataT = vbt.HDFData.fetch('../my_data_2017-2023.h5')
        symbolData = barData.data['X:BTCUSD']
        allData['Close'] = symbolData['Close']

        return allData

    def make10Stats(self) -> PortfolioWrapper:
        # allData = self.__createAllData()
        allData = self.__create10Data()

        portfolioHolder: PortfolioMaker = PortfolioMaker()
        wrap: PortfolioWrapper = portfolioHolder.makePortfolio(allData, 10)
        return wrap

