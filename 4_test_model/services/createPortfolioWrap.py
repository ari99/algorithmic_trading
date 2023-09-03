from vectorbtpro import Data, symbol_dict
from vectorbtpro.data.custom import LocalDataT

from common.portfolioWrapper import PortfolioWrapper
from configs.comparisonConfig import ComparisonConfig

import vectorbtpro as vbt
from common.modelTester import ModelTester


def fetchComparison(confDir: str, period, symbol: str):
    conf = ComparisonConfig(confDir, str(period), symbol) # Note ticker was added here
    # so that will not look for any __ files without a symbol anymore. Ticker was always a param
    comparisonHDF: LocalDataT = vbt.HDFData.fetch(conf.relativeComparisonPath)
    comparison = comparisonHDF.data[conf.comparisonKey]
    return comparison


def createPortfolioWrap( confDir: str, barDataPath: str, symbol, period, minLong, minShort) -> PortfolioWrapper:
    comparison = fetchComparison(confDir, period, symbol)
    modelTester = ModelTester()
    barData: LocalDataT = vbt.HDFData.fetch(barDataPath)
    symbolData = barData.data[symbol]
    pfWrap: PortfolioWrapper = modelTester.createPortfolio(symbolData.Close, comparison, period, minLong, minShort)
    return pfWrap