from pandas import DataFrame
from vectorbtpro import Data
from common.checkpointHandler import CheckpointHandler
from configs.config import Config


class AllTargets(CheckpointHandler):

    def __init__(self, merged: DataFrame):
        self.merged = merged

    def createData(self) -> Data:
        data = self.createAllTargets()
        return data

    def createAllTargets(self) -> Data:
        testTargets: Data = self.merged.loc[:, Config.targetsColumnStart:Config.targetsColumnEnd]
        testTargets['fwd_returns_label'] = self.merged['fwd_returns_label']
        return testTargets
