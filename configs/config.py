from typing import List
from pathlib import Path

from configs.configSecrets import ConfigSecrets


def makeDirectoryPath(path: List[str]) -> str:
    # 'results', 'lstm_embeddings'  , * unpacks https://note.nkmk.me/en/python-argument-expand/
    results_path = Path(*path)
    if not results_path.exists():
        results_path.mkdir(parents=True)
    return results_path.as_posix()


# naming conventions
# downloadData this is the data initially downloaded from polygon
class Config:
    originalDir = "original"
    secondDir = "second"
    flippedDir = "flipped"
    otherDir = "other"
    # 1 - download data
    tickersDownload: List[str] = ["X:BTCUSD"]
    # tickersDownload: List[str] = ["X:BTCUSD", "X:ETHUSD", "X:XMRUSD","X:DOGEUSD","AMZN","BAC","SPY", "SH"]

    @classmethod
    def createDownloadDataPath(cls) -> str:
        return makeDirectoryPath(['checkpoints', 'downloaded_data'])

    downloadDataPath: str = ('checkpoints/downloaded_data' + "/1_downloaded_data_2017-2023.h5")

    @classmethod
    def symbolDownloadDataPath(cls, symbol: str) -> str:
        return ('checkpoints/downloaded_data' + "/1_downloaded_" + symbol + "_2017-2023.h5")

    relativeDownloadDataPath: str = ConfigSecrets.PROJECT_DIR \
                                    + "1_download_data/checkpoints/downloaded_data/1_downloaded_data_2017-2023.h5"

    cleanedDataPath: str = ('checkpoints/downloaded_data' + "/2_cleaned_data_2017-2023.h5")
    relativeCleanedDataPath: str = ConfigSecrets.PROJECT_DIR \
                                   + "1_download_data/checkpoints/downloaded_data/2_cleaned_data_2017-2023.h5"

    flippedDataPath: str = ('checkpoints/downloaded_data' + "/2_flipped_data_2017-2023.h5")
    relativeFlippedDataPath: str = ConfigSecrets.PROJECT_DIR \
                                   + "1_download_data/checkpoints/downloaded_data/2_flipped_data_2017-2023.h5"

    @classmethod
    def relativeSymbolDownloadDataPath(cls, symbol: str) -> str:
        return (ConfigSecrets.PROJECT_DIR +
                '1_download_data/checkpoints/downloaded_data' + "/1_downloaded_" + symbol + "_2017-2023.h5")

    ##########
    # 2 - shift data
    # in configs / featuresShiftedConfig.py
    ####
    # 3 -1 - hourly returns
    # in hourlyReturnsConfig

    ####
    ####

    # skipping flipped and second for now
    # 3 - 1 - daily returns
    @classmethod
    def createDailyReturnsPath(cls) -> str:
        return makeDirectoryPath(['checkpoints', 'returns_daily_data'])

    dailyReturnsPath: str = ('checkpoints/returns_daily_data'
                             + "/3_1_returns_daily_data_2017-2023.h5")
    relativeDailyReturnsPath: str = "../3_make_model/rnn/1_prepare_returns/" + dailyReturnsPath
    dailyReturnsKey: str = 'returns_daily'

    # hourlyData.to_hdf('./my_returns_hourly_data_2017-2023.h5', 'returns_hourly')

    ####
    # 3 - 2 - split train test features targets
    @classmethod
    def createTrainFeaturesPath(cls) -> str:
        return makeDirectoryPath(['checkpoints', 'train_features'])

    trainFeaturesPath: str = ('checkpoints/train_features'
                              + "/3_2_train_features_2017-2023.h5")
    relativeTrainFeaturesPath: str = "../3_make_model/rnn/2_split_train_test_features_targets/" + trainFeaturesPath
    trainFeaturesKey: str = 'train_features'

    ##
    @classmethod
    def createTrainTargetsPath(cls) -> str:
        return makeDirectoryPath(['checkpoints', 'train_targets'])

    trainTargetsPath: str = ('checkpoints/train_targets'
                             + "/3_2_train_targets_2017-2023.h5")
    relativeTrainTargetsPath: str = "../3_make_model/rnn/2_split_train_test_features_targets/" + trainTargetsPath
    trainTargetsKey: str = 'train_targets'

    ####
    @classmethod
    def createTestFeaturesPath(cls) -> str:
        return makeDirectoryPath(['checkpoints', 'test_features'])

    testFeaturesPath: str = ('checkpoints/test_features'
                             + "/3_2_test_features_2017-2023.h5")
    relativeTestFeaturesPath: str = (ConfigSecrets.PROJECT_DIR +
                                     "3_make_model/rnn/2_split_train_test_features_targets/" + testFeaturesPath)

    testFeaturesKey: str = 'test_features'

    ##
    @classmethod
    def createTestTargetsPath(cls) -> str:
        return makeDirectoryPath(['checkpoints', 'test_targets'])

    testTargetsPath: str = ('checkpoints/test_targets'
                            + "/3_2_test_targets_2017-2023.h5")
    relativeTestTargetsPath: str = "../3_make_model/rnn/2_split_train_test_features_targets/" + testTargetsPath
    testTargetsKey: str = 'test_targets'

    ####
    @classmethod
    def createTrainLaggedReturnsPath(cls) -> str:
        makeDirectoryPath(['checkpoints', 'train_lagged_returns'])

    trainLaggedReturnsPath: str = ('checkpoints/train_lagged_returns'
                                   + "/3_2_train_lagged_returns_2017-2023.npy")

    relativeTrainLaggedReturnsPath: str = (ConfigSecrets.PROJECT_DIR +
                                           "3_make_model/rnn/2_split_train_test_features_targets/" +
                                           trainLaggedReturnsPath)

    ####
    @classmethod
    def createTestLaggedReturnsPath(cls) -> str:
        makeDirectoryPath(['checkpoints', 'test_lagged_returns'])

    testLaggedReturnsPath: str = ('checkpoints/test_lagged_returns'
                                  + "/3_2_test_lagged_returns_2017-2023.npy")
    relativeTestLaggedReturnsPath: str = ("../3_make_model/rnn/2_split_train_test_features_targets/"
                                          + testLaggedReturnsPath)
    #################### end 3-2

    trainIndexEnd = '2021-06-01'
    featuresColumnStart = 'CurrentOpen'
    featuresColumnEnd = 'outWQA1'
    targetsColumnStart = 'longEntry10'
    targetsColumnEnd = 'shortEntry100'
    testIndexStart = '2021-06-02'
    tickerTrain: str = "X:BTCUSD"
    secondTicker: str = "X:ETHUSD"

    # allDataStart: str = '2022-06-01 UTC'
    # allDataEnd: str = '2022-06-30 UTC'
    allDataStart: str = '2017-01-01 UTC'
    allDataEnd: str = '2023-01-01 UTC'
    allDataTimeFrame: str = '1h'
    hourlyReturnsWindowSize = 120
    dailyReturnsWindowSize = 52

    @classmethod
    def modelPath(cls, tag: str) -> str:
        modelsPath: str = (ConfigSecrets.PROJECT_DIR
                           + "4_test_model/models" + "/3_3_model_" + tag)
        return modelsPath

    @classmethod
    def modelCheckpointPath(cls, tag: str) -> str:
        modelsPath: str = (makeDirectoryPath(['checkpoints', 'modelCheckpoints'])
                           + "/3_3_model_" + tag + ".h5")
        return modelsPath
