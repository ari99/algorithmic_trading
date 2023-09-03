from pandas import DataFrame
from vectorbtpro import HDFData

from .featureDataShiftedCheckpointHandler import FeatureDataShiftedHandler
from common.featuresTargetsUnshiftedMaker import FeaturesTargetsUnshiftedMaker
from configs.featuresShiftedConfig import FeaturesShiftedConfig


def createFeatureDataShifted(symbol: str, config: FeaturesShiftedConfig,
                             barData: HDFData) -> DataFrame:
    featuresTargetsMaker: FeaturesTargetsUnshiftedMaker = FeaturesTargetsUnshiftedMaker()
    # this creates the features and targets from talib
    featuresTargetsUnshifted: DataFrame = featuresTargetsMaker.loadData(barData, symbol)
    checkpointHandler: FeatureDataShiftedHandler = FeatureDataShiftedHandler(featuresTargetsUnshifted)
    featureDataSecondShifted: DataFrame = checkpointHandler.fetchOrCreateSave(
        config.featuresShiftedPath,
        config.featuresShiftedKey)
    return featureDataSecondShifted

