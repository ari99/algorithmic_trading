from pandas import DataFrame
from vectorbtpro import Data
from configs.allFeaturesTargetsConfig import AllConfig
from .mergedShiftedFeaturesAndReturns import Merged
from common.checkpointHandler import CheckpointHandler
from configs.config import Config


class TrainTargets(CheckpointHandler):

    def __init__(self, allConfig: AllConfig):
        self.allConfig = allConfig

    def createData(self) -> Data:
        data = self.createTrainTargets()
        return data

    def createTrainTargets(self):
        merged: DataFrame = Merged.createMerged(self.allConfig)
        trainData: DataFrame = merged.loc[:Config.trainIndexEnd]
        trainTargets = trainData.loc[:, Config.targetsColumnStart:Config.targetsColumnEnd]
        trainTargets['fwd_returns_label'] = trainData['fwd_returns_label']
        return trainTargets
