
from common.modelTester import ModelTester
import vectorbtpro as vbt
from vectorbtpro.data.custom import LocalDataT
from configs.config import Config
import numpy
from pandas import DataFrame
from configs.comparisonConfig import ComparisonConfig
from .comparisonCreator import ComparisonCreator


modelTester = ModelTester()

def createHistogram(comparison, barDataPath: str, ticker: str, testIndexStart) -> None:
    barData: LocalDataT = vbt.HDFData.fetch(barDataPath)
    symbolData = barData.data[ticker] # Config.tickerTrain]
    if testIndexStart is not None:
        testData = symbolData.loc[Config.testIndexStart:]
    else:
        testData = symbolData
    testClose = testData['Close']
    portfolioStats = modelTester.createPortfolioComparisonDf(testClose, comparison, 10)
    modelTester.createMinScoreHistogram(portfolioStats, 'TotalReturn')




def createComparison(dataConfig: tuple[numpy.ndarray, DataFrame, DataFrame],
                     comparisonConfig: ComparisonConfig , longEntrySt: str, shortEntryStr: str):
    (testLaggedReturns, testFeatures2, testTargets)=dataConfig
    conf = comparisonConfig
    conf.createComparisonPath()
    comparisonCreator = ComparisonCreator(longEntrySt, shortEntryStr,
                                          testLaggedReturns, testFeatures2, testTargets)
    comparison = comparisonCreator.fetchOrCreateSave(conf.comparisonPath, conf.comparisonKey)
    return comparison