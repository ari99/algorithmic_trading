
from numpy import ndarray
from pandas import DataFrame

from configs.allFeaturesTargetsConfig import AllConfig
from .mergedShiftedFeaturesAndReturns import Merged
from common.checkpointHandler import CheckpointHandler
from configs.config import Config


class TrainLaggedReturns(CheckpointHandler):

    def __init__(self, allConfig: AllConfig):
        self.allConfig = allConfig

    def createData(self) -> ndarray:
        data = self.createTrainLaggedReturns()
        return data

    def createTrainLaggedReturns(self):
        merged: DataFrame = Merged.createMerged(self.allConfig)
        trainData: DataFrame = merged.loc[:Config.trainIndexEnd]
        sequence = list(range(1, Config.hourlyReturnsWindowSize+1))
        trainLaggedReturns = trainData.loc[:, sequence].values.reshape(-1, Config.hourlyReturnsWindowSize, 1)
        return trainLaggedReturns
