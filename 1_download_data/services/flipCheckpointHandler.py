from vectorbtpro import Data
from .flipBarData import FlipBarData
from common.checkpointHandler import CheckpointHandler
from .cleanDataHandler import CleanDataHandler


class FlipHandler(CheckpointHandler):

    def createData(self) -> Data:
        barData: Data = CleanDataHandler.createCleanedData()
        newBarData: Data = barData.copy()
        flipper: FlipBarData = FlipBarData()
        for sym in barData.data:
            print(sym)
            newBarData.data[sym] = flipper.flipBarData(barData.data[sym])

        return newBarData
