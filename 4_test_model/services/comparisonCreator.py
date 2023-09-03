from numpy import ndarray
from pandas import DataFrame
from vectorbtpro import Data
from common.checkpointHandler import CheckpointHandler
from common.predict import Predict


class ComparisonCreator(CheckpointHandler):

    def __init__(self, longTargetKey: str, shortTargetKey: str, testLaggedReturns: ndarray,
                 testFeatures: DataFrame, testTargets: DataFrame):
        self.longTargetKey: str = longTargetKey
        self.shortTargetKey: str = shortTargetKey
        self.testLaggedReturns: ndarray = testLaggedReturns
        self.testFeatures: DataFrame = testFeatures
        self.testTargets: DataFrame = testTargets

    def createData(self) -> Data:
        data = self.createComparison()
        return data

    def createComparison(self):
        predictor = Predict(self.longTargetKey, self.shortTargetKey,
                            self.testLaggedReturns, self.testFeatures, self.testTargets)
        predictor.runPredicts()
        comparison = predictor.createComparisonDf()
        return comparison

