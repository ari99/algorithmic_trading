from pandas import DataFrame
from vectorbtpro import Data
from configs.allFeaturesTargetsConfig import AllConfig
from .mergedShiftedFeaturesAndReturns import Merged
from common.checkpointHandler import CheckpointHandler
from configs.config import Config


class TestFeatures(CheckpointHandler):

    def __init__(self, allConfig: AllConfig):
        self.allConfig = allConfig

    def createData(self) -> Data:
        data = self.createTestFeatures()
        return data

    def createTestFeatures(self):
        merged: DataFrame = Merged.createMerged(self.allConfig)
        testData: DataFrame = merged.loc[Config.testIndexStart:]
        testFeatures = testData.loc[:, Config.featuresColumnStart:Config.featuresColumnEnd]
        return testFeatures


'''
 TODO count how many features you are generating
 in prepare_data. figure out how to import merged there and compy and paste the above code.
 trying to solve this error which was talking about the features count:
ValueError: Input 1 of layer "model" is incompatible with the layer: expected shape=(None, 178), found shape=(None, 124)
'''
