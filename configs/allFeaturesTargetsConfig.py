from configs import config
from configs.configSecrets import ConfigSecrets
from configs.featuresShiftedConfig import FeaturesShiftedConfig
from configs.hourlyReturnsConfig import HourlyReturnsConfig


class AllConfig:
    # original: str = "original_all"
    # flipped: str = "flipped_all"
    # second: str = "second_all"

    def __init__(self, dirStr: str, symbol: str = ""):
        self.dirStr = dirStr
        self.featuresConfig = FeaturesShiftedConfig(dirStr, symbol)
        self.hourlyConfig = HourlyReturnsConfig(dirStr, symbol)

        checkpointsStr = 'checkpoints/' + dirStr
        ###
        self.allFeaturesPath: str = (checkpointsStr
                                     + "/3_2_all_features_"+symbol+"_2017-2023.h5")
        self.relativeAllFeaturesPath: str = (ConfigSecrets.PROJECT_DIR
                                             + "3_make_model/rnn/2_split_train_test_features_targets/"
                                             + self.allFeaturesPath)
        self.allFeaturesKey: str = 'all_features'
        ###
        self.allTargetsPath: str = (checkpointsStr
                                    + "/3_2_all_targets_"+symbol+"_2017-2023.h5")
        self.relativeAllTargetsPath: str = (ConfigSecrets.PROJECT_DIR
                                            + "3_make_model/rnn/2_split_train_test_features_targets/"
                                            + self.allTargetsPath)
        self.allTargetsKey: str = 'all_targets'
        ###
        self.allLaggedReturnsPath: str = (checkpointsStr
                                          + "/3_2_all_lagged_returns_"+symbol+"_2017-2023.npy")
        self.relativeAllLaggedReturnsPath: str = (ConfigSecrets.PROJECT_DIR
                                                  + "3_make_model/rnn/2_split_train_test_features_targets/"
                                                  + self.allLaggedReturnsPath)

    #######################
    # combined data instead
    def createAllPath(self) -> str:
        return config.makeDirectoryPath(['checkpoints', self.dirStr])
