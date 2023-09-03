from configs import config
from configs.configSecrets import ConfigSecrets


##########
# 2 - shift data
class HourlyReturnsConfig:
    def __init__(self, dirStr: str, symbol: str = ""):
        self.dirStr = dirStr
        self.hourlyReturnsKey = 'returns_hourly'

        checkpointsStr = 'checkpoints/'+dirStr

        self.hourlyReturnsPath: str = (checkpointsStr + "/3_1_returns_hourly_"+symbol+"_2017-2023.h5")
        self.relativeHourlyReturnsPath: str = (ConfigSecrets.PROJECT_DIR +
                                               "3_make_model/rnn/1_prepare_returns/"+self.hourlyReturnsPath)

    # combined data instead
    def createHourlyReturnsPath(self) -> str:
        return config.makeDirectoryPath(['checkpoints', self.dirStr])

