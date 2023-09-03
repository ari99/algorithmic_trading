from pandas import DataFrame
from vectorbtpro import Data

from common.checkpointHandler import CheckpointHandler
from configs.config import Config


class AllFeatures(CheckpointHandler):

    def __init__(self, merged: DataFrame):
        self.merged: DataFrame = merged

    def createData(self) -> Data:
        data: Data = self.createAllFeatures()
        return data

    def createAllFeatures(self):
        allFeatures: Data = self.merged.loc[:, Config.featuresColumnStart:Config.featuresColumnEnd]
        return allFeatures
