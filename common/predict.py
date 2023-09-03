from numpy import ndarray
from pandas import DataFrame
from .modelLoader import ModelLoader
from .predictComparer import PredictComparer


class Predict:

    def __init__(self, longTargetKey: str, shortTargetKey: str,
                 testLaggedReturns: ndarray, testFeatures: DataFrame, allTestTargets: DataFrame):
        self.allTestTargets = allTestTargets
        self.longTargetKey = longTargetKey
        self.shortTargetKey = shortTargetKey
        self.X_test = [
                testLaggedReturns,
                testFeatures,
            ]
        self.y_test_longs = self.allTestTargets[longTargetKey]
        self.y_test_shorts = self.allTestTargets[shortTargetKey]
        self.longRnn, self.longHistory = ModelLoader.loadModel(longTargetKey)
        self.shortRnn, self.shortHistory = ModelLoader.loadModel(shortTargetKey)
        self.predictComparer = PredictComparer()
        self.test_predict_long = None
        self.test_predict_short = None

    def runPredicts(self):
        self.test_predict_long = self.longRnn.predict(self.X_test).squeeze()
        self.test_predict_short = self.shortRnn.predict(self.X_test).squeeze()

    def createComparisonDf(self) -> DataFrame:
        comparisonDf: DataFrame = self.predictComparer.createAllComparisonDf(
                                    self.test_predict_long, self.test_predict_short,
                                    self.y_test_longs, self.y_test_shorts)
        return comparisonDf




