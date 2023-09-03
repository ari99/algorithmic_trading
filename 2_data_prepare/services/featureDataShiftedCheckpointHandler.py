
from vectorbtpro import Data

from common.checkpointHandler import CheckpointHandler
from common.featurePreparer import FeaturePreparer


# https://vectorbt.pro/pvt_d904e513/tutorials/stop-signals/#__codelineno-59-1
class FeatureDataShiftedHandler(CheckpointHandler):
    def __init__(self, allData):
        self.allData = allData

    def createData(self) -> Data:
        featureMaker = FeaturePreparer()
        featureDataShifted = featureMaker.makeFeatureDf(self.allData)
        return featureDataShifted
