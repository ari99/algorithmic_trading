from pandas import DataFrame
from vectorbtpro import Data
from configs.allFeaturesTargetsConfig import AllConfig
from .mergedShiftedFeaturesAndReturns import Merged
from common.checkpointHandler import CheckpointHandler
from configs.config import Config


class TrainFeatures(CheckpointHandler):

    def __init__(self, allConfig: AllConfig):
        self.allConfig = allConfig

    def createData(self) -> Data:
        data = self.createTrainFeatures()
        return data

    def createTrainFeatures(self):
        merged: DataFrame = Merged.createMerged(self.allConfig)
        trainData: DataFrame = merged.loc[:Config.trainIndexEnd]
        trainFeatures = trainData.loc[:, Config.featuresColumnStart:Config.featuresColumnEnd]
        return trainFeatures
