import numpy
import numpy as np
import vectorbtpro as vbt
from pandas import DataFrame

from configs.allFeaturesTargetsConfig import AllConfig
from configs.config import Config


class TestDataConfig:

    @classmethod
    def btcSplitConfig(cls) -> tuple[numpy.ndarray, DataFrame, DataFrame]:
        testLaggedReturns: numpy.ndarray = np.load(Config.relativeTestLaggedReturnsPath)
        testFeatures: DataFrame = vbt.HDFData.fetch(Config.relativeTestFeaturesPath).data['test_features']
        testTargets: DataFrame = vbt.HDFData.fetch(Config.relativeTestTargetsPath).data['test_targets']
        return testLaggedReturns, testFeatures, testTargets

    @classmethod
    def fromAllConfig(cls, allConfig: AllConfig) -> tuple[numpy.ndarray, DataFrame, DataFrame]:
        testLaggedReturns = np.load(allConfig.relativeAllLaggedReturnsPath)
        testFeatures = vbt.HDFData.fetch(allConfig.relativeAllFeaturesPath).data[allConfig.allFeaturesKey]
        testTargets = vbt.HDFData.fetch(allConfig.relativeAllTargetsPath).data[allConfig.allTargetsKey]
        return testLaggedReturns, testFeatures, testTargets

    @classmethod
    def flippedConfig(cls) -> tuple[numpy.ndarray, DataFrame, DataFrame]:
        allConfig = AllConfig(Config.flippedDir)
        return cls.fromAllConfig(allConfig)

    @classmethod
    def ethConfig(cls) -> tuple[numpy.ndarray, DataFrame, DataFrame]:
        allConfig = AllConfig(Config.secondDir)
        return cls.fromAllConfig(allConfig)

    @classmethod
    def btcConfig(cls) -> tuple[numpy.ndarray, DataFrame, DataFrame]:
        allConfig = AllConfig(Config.originalDir)
        return cls.fromAllConfig(allConfig)

    @classmethod
    def symbolConfig(cls, symbol: str) -> tuple[numpy.ndarray, DataFrame, DataFrame]:
        allConfig = AllConfig(Config.otherDir, symbol)
        return cls.fromAllConfig(allConfig)
