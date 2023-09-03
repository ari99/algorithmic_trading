from typing import List, Union, Iterable

from numpy import ndarray
from pandas import DataFrame
from vectorbtpro import Data
from vectorbtpro.data.custom import FileDataT

import vectorbtpro as vbt
import numpy as np
import os
from abc import ABC, abstractmethod


class CheckpointHandler(ABC):

    def fetchOrCreateSave(self, dataPath: str, key=None):
        print(" in fetchOrCreateSaveData")

        if os.path.exists(dataPath):
            print(dataPath + " file already exists")
            data = self.fetchData(dataPath)
            if key is not None:
                data: Data = data.data[key]
        else:
            print(" creating data at " + dataPath)

            data: Data = self.createData()
            if key is not None:
                data.to_hdf(dataPath, key)
            else:
                data.to_hdf(dataPath)
        # TODO convert Data to DataFrame , seems to work as it is now anyway
        return data

    def fetchOrCreateSaveNumpy(self, dataPath: str) -> Union[ndarray, Iterable, int, float, tuple, dict]:
        print(" in fetchOrCreateSaveNumpy")

        if os.path.exists(dataPath):
            print(" download data file already exists")
            data: ndarray = np.load(dataPath)
        else:
            print(" creating data")
            data: ndarray = self.createData()
            np.save(dataPath, data)
        return data

    def fetchData(self, dataPath: str) -> FileDataT:
        fetched: FileDataT = vbt.HDFData.fetch(dataPath)
        return fetched


    @abstractmethod
    def createData(self) -> Union[ndarray, Iterable, int, float, tuple, dict, Data]:
        ...


    # https://vectorbt.pro/pvt_d904e513/tutorials/stop-signals/#__codelineno-59-1

