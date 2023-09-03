from configs import config
from configs.configSecrets import ConfigSecrets


##########
# 2 - shift data
class FeaturesShiftedConfig:
    # original: str = "original_shifted_features"
    # flipped: str = "flipped_shifted_features"
    # second: str = "second_shifted_features"

    def __init__(self, dirStr: str, symbolStr: str = ""):
        self.dirStr = dirStr
        self.featuresShiftedKey = 'featuresShifted'

        checkpointsStr = 'checkpoints/' + dirStr

        self.featuresShiftedPath: str = (checkpointsStr + "/2_feature_data_shifted_" + symbolStr + "_2017-2023.h5")
        self.relativeFeaturesShiftedPath: str = (ConfigSecrets.PROJECT_DIR +
                                                 "2_data_prepare/" + self.featuresShiftedPath)

    #######################
    # combined data instead
    def createFeaturesShiftedPath(self) -> str:
        return config.makeDirectoryPath(['checkpoints', self.dirStr])
