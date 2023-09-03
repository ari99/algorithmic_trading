from numpy import ndarray
from vectorbtpro.data.base import DataT

from configs.config import Config
from common.polygonDownloader import downloadMonthHourly
from common.featuresTargetsUnshiftedMaker import FeaturesTargetsUnshiftedMaker
from pandas import DataFrame
from common.featurePreparer import FeaturePreparer
from common.hourlyReturns import HourlyReturns
from common.predict import Predict
from common.modelTester import ModelTester


class Prediction:
    def __init__(self):
        self.polygonSymbol: str = ""
        self.providerSymbol: str = ""
        # to check the snapshot and use market order use "BTC/USD"
        # to check the currently held position use "BTCUSD"
        self.providerOpenPositionSymbol: str = ""
        self.shouldLong = False
        self.shouldLiquidate = False
        self.shouldShort = False


class Predictor:
    def makeShiftedFeaturesTargets(self, latest)-> DataFrame:
        featuresTargetsMaker: FeaturesTargetsUnshiftedMaker = FeaturesTargetsUnshiftedMaker()
        featuresTargetsUnshifted: DataFrame = featuresTargetsMaker.loadData(latest, 'X:BTCUSD')
        featureMaker = FeaturePreparer()
        featureDataShifted = featureMaker.makeFeatureDf(featuresTargetsUnshifted)
        return featureDataShifted

    def createComparisonPredictions(self, featureDataShifted: DataFrame,
                                    longTarget: str,
                                    shortTarget: str) -> DataFrame:
        closeSeries = featureDataShifted.Close

        returnMaker = HourlyReturns(closeSeries)
        returns = returnMaker.createHourlyReturns()
        sequence = list(range(1, Config.hourlyReturnsWindowSize+1))
        npReturns: ndarray = returns.loc[:, sequence].values.reshape(-1, Config.hourlyReturnsWindowSize , 1)
        testFeatures: DataFrame = featureDataShifted.loc[:, Config.featuresColumnStart:Config.featuresColumnEnd]
        testTargets: DataFrame = featureDataShifted.loc[:, Config.targetsColumnStart:Config.targetsColumnEnd]
        testFeatures = testFeatures[testFeatures.index.isin(returns.index)]
        testTargets = testTargets[testTargets.index.isin(returns.index)]
        predictor = Predict(longTarget, shortTarget,
                            npReturns, testFeatures, testTargets)
        predictor.runPredicts()
        comparison = predictor.createComparisonDf()
        return comparison


    # latest= downloadMonthHourly(['X:BTCUSD'])
    # symbol= 'X:BTCUSD'
    # longTarget = 'longEntry50'
    # shortTarget = 'shortEntry50'
    # targetPeriod = 50
    def makePrediction(self, latest: DataT, symbol: str,
                       longTarget: str, shortTarget: str,
                       targetPeriod: int ) -> Prediction:
        #latest= downloadMonthHourly(['X:BTCUSD'])
        featureDataShifted: DataFrame = self.makeShiftedFeaturesTargets(latest)
        comparison = self.createComparisonPredictions(featureDataShifted, longTarget, shortTarget)
        closes = latest.data[symbol].Close
        modelTester = ModelTester()
        portfolioStats = modelTester.createPortfolioComparisonDf(closes, comparison, targetPeriod)
        best=portfolioStats.loc[portfolioStats['TotalReturn'].idxmax()]
        recentHour = comparison.iloc[-1:, :]
        #df2=df.loc[df['Fee'].idxmax()]
        longMin=best['LongMin']
        shortMin=best['ShortMin']
        prediction = Prediction()
        if recentHour.iloc[0]['predictedLongEntry'] > longMin:
            prediction.shouldLong = True

        if recentHour.iloc[0]['predictedShortEntry'] > shortMin:
            prediction.shouldShort = True

        return prediction