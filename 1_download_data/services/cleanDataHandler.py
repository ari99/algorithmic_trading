
from vectorbtpro import Data
from vectorbtpro.data.custom import LocalDataT

from common.checkpointHandler import CheckpointHandler
from configs.config import Config
import vectorbtpro as vbt


class CleanDataHandler(CheckpointHandler):

    @classmethod
    def createCleanedData(cls) -> Data:
        barData: LocalDataT = vbt.HDFData.fetch(Config.relativeDownloadDataPath)
        newBarData = barData.copy()
        for sym in barData.data:
            # drop rows with nan
            # 2022-04-26 00:00:00+00:00 to 2022-04-26 13:00:00+00:00 are nan for ETH from polygon
            newBarData.data[sym] = barData.data[sym].dropna()

        print("returning from function")
        print(newBarData.data['X:ETHUSD'].isna().sum().sum())
        return newBarData

    def createData(self) -> Data:
        return CleanDataHandler.createCleanedData()

