
from numpy import ndarray
from pandas import DataFrame

from configs.allFeaturesTargetsConfig import AllConfig
from .mergedShiftedFeaturesAndReturns import Merged
from common.checkpointHandler import CheckpointHandler
from configs.config import Config


class TestLaggedReturns(CheckpointHandler):
    def __init__(self, allConfig: AllConfig):
        self.allConfig = allConfig

    def createData(self) -> ndarray:
        data = self.createTestLaggedReturns()
        return data

    def createTestLaggedReturns(self) -> ndarray:
        merged: DataFrame = Merged.createMerged(self.allConfig)
        testData: DataFrame = merged.loc[Config.testIndexStart:]
        sequence = list(range(1, Config.hourlyReturnsWindowSize+1))
        testLaggedReturns: ndarray = testData.loc[:, sequence].values.reshape(-1, Config.hourlyReturnsWindowSize, 1)
        return testLaggedReturns
