
from numpy import ndarray
from pandas import DataFrame
from common.checkpointHandler import CheckpointHandler
from configs.config import Config


class AllLaggedReturns(CheckpointHandler):

    def __init__(self, merged: DataFrame):
        self.merged = merged

    def createData(self) -> ndarray:
        data: ndarray = self.createAllLaggedReturns()
        return data

    def createAllLaggedReturns(self) -> ndarray:
        sequence = list(range(1, Config.hourlyReturnsWindowSize+1))
        allLaggedReturns = self.merged.loc[:, sequence].values.reshape(-1, Config.hourlyReturnsWindowSize, 1)
        return allLaggedReturns
