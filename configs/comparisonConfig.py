from configs import config
from configs.configSecrets import ConfigSecrets


class ComparisonConfig:

    def __init__(self, dirStr: str, targetsPeriod: str, symbol: str = ""):
        self.dirStr = dirStr
        self.targetsPeriod = targetsPeriod
        # self.featuresConfig = FeaturesShiftedConfig(dirStr)
        # self.hourlyConfig = HourlyReturnsConfig(dirStr)

        checkpointsStr = 'checkpoints/comparison/' + dirStr
        ###
        self.comparisonPath: str = (checkpointsStr
                                    + "/4_1_comparison_" + symbol + "_period_" + targetsPeriod + "_2017-2023.h5")

        self.relativeComparisonPath: str = (ConfigSecrets.PROJECT_DIR
                                            + "4_test_model/"
                                            + self.comparisonPath)

        self.comparisonKey: str = 'comparison'

    #######################
    # combined data instead
    def createComparisonPath(self) -> str:
        return config.makeDirectoryPath(['checkpoints', 'comparison', self.dirStr])
