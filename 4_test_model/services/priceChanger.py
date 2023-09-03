from pandas import DataFrame

from common.predict import Predict

class PriceChanger:
    def convertLaggedReturnsToDataFrameOfDetrendedData(self, predictor: Predict) -> DataFrame:
        testLaggedReturns = predictor.X_test[0]
        columns = range(1, 121, 1)
        testLaggedReturnsDf: DataFrame = DataFrame(testLaggedReturns.tolist(), columns=columns)
        for column in columns:
            testLaggedReturnsDf[column] = testLaggedReturnsDf[column].map(lambda x: x[0]).diff()

        return testLaggedReturnsDf
