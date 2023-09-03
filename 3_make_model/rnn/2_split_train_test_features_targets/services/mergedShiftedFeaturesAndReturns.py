import vectorbtpro as vbt
import pandas as pd
from pandas import DataFrame
from vectorbtpro import Data

from configs.allFeaturesTargetsConfig import AllConfig


class Merged:

    @classmethod
    def createMerged(cls, allConfig: AllConfig) -> DataFrame:
        featureDataPath: str = allConfig.featuresConfig.relativeFeaturesShiftedPath
        hourlyReturnsPath: str = allConfig.hourlyConfig.relativeHourlyReturnsPath
        fetched: Data = vbt.HDFData.fetch(featureDataPath)
        featureDataShifted: DataFrame = fetched.data[allConfig.featuresConfig.featuresShiftedKey]

        fetched: Data = vbt.HDFData.fetch(hourlyReturnsPath)
        returnsHourly: DataFrame = fetched.data[allConfig.hourlyConfig.hourlyReturnsKey]

        merged: DataFrame = pd.merge(
            featureDataShifted, returnsHourly, how="left", left_index=True, right_index=True
        )

        merged = merged.dropna(axis=0)
        merged = merged.drop('fwd_returns', axis=1)

        return merged

