from pandas import DataFrame
from vectorbtpro import Data
from configs.allFeaturesTargetsConfig import AllConfig
from .mergedShiftedFeaturesAndReturns import Merged
from common.checkpointHandler import CheckpointHandler
from configs.config import Config


class TestTargets(CheckpointHandler):

    def __init__(self, allConfig: AllConfig):
        self.allConfig = allConfig

    def createData(self) -> Data:
        data = self.createTestTargets()
        return data

    def createTestTargets(self):
        merged: DataFrame = Merged.createMerged(self.allConfig)
        testData: DataFrame = merged.loc[Config.testIndexStart:]
        testTargets = testData.loc[:, Config.targetsColumnStart:Config.targetsColumnEnd]
        testTargets['fwd_returns_label'] = testData['fwd_returns_label']
        return testTargets
