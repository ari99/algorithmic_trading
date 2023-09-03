from typing import List
from vectorbtpro import Data
from vectorbtpro.data.base import DataT
from common.checkpointHandler import CheckpointHandler
from configs.config import Config
import vectorbtpro as vbt


# https://vectorbt.pro/pvt_d904e513/tutorials/stop-signals/#__codelineno-59-1
class DownloadDataHandler(CheckpointHandler):
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        if symbols is None:
            self.symbols = Config.tickersDownload

    def createData(self) -> Data:
        vbt.PolygonData.set_custom_settings(
            client_config=dict(
                api_key=Config.polygonApiKey
            )
        )
        data: DataT = vbt.PolygonData.fetch(
            self.symbols,
            start=Config.allDataStart,
            end=Config.allDataEnd,
            timeframe=Config.allDataTimeFrame
        )
        return data
