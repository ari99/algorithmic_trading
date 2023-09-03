from typing import List
from pathlib import Path


def makeDirectoryPath(path: List[str]) -> str:
    # 'results', 'lstm_embeddings'  , * unpacks https://note.nkmk.me/en/python-argument-expand/
    results_path = Path(*path)
    if not results_path.exists():
        results_path.mkdir(parents=True)
    return results_path.as_posix()


class Config:
    # 1 - download data
    tickersDownload: List[str] = ["X:BTCUSD", "X:ETHUSD"]
    @classmethod
    def createDownloadDataPath(cls):
        makeDirectoryPath(['checkpoints', 'downloaded_data'])
    downloadDataPath: str = ('checkpoints/downloaded_data' + "/1_downloaded_data_2017-2023.h5")
    # this is the path from other steps
    relativeDownloadDataPath: str = "../1_download_data/"+downloadDataPath
    ##########
    # 2 - shift data
    @classmethod
    def createFeatureDataShiftedPath(cls):
        makeDirectoryPath(['checkpoints', 'feature_data_shifted'])
    featureDataShiftedPath: str = ('checkpoints/feature_data_shifted'
                                    + "/2_feature_data_shifted_2017-2023.h5")
    featureDataShiftedKey = 'featureDataShifted'
    relativeFeatureDataShiftedPath: str = "../2_data_prepare/"+featureDataShiftedPath
    ####
    # 3 -1 - hourly returns
    @classmethod
    def createHourlyReturnsPath(cls):
        makeDirectoryPath(['checkpoints', 'returns_hourly_data'])
    hourlyReturnsPath: str = ('checkpoints/returns_hourly_data'
                                   + "/3_1_returns_hourly_data_2017-2023.h5")
    relativeHourlyReturnsPath: str = "../3_make_model/rnn/1_prepare_returns/"+hourlyReturnsPath
    hourlyReturnsKey = 'returns_hourly'

    #hourlyData.to_hdf('./my_returns_hourly_data_2017-2023.h5', 'returns_hourly')
    ####
    ####
    # 3 - 1 - daily returns
    @classmethod
    def createDailyReturnsPath(cls):
        makeDirectoryPath(['checkpoints', 'returns_daily_data'])
    dailyReturnsPath: str = ('checkpoints/returns_daily_data'
                              + "/3_1_returns_daily_data_2017-2023.h5")
    relativeDailyReturnsPath: str = "../3_make_model/rnn/1_prepare_returns/"+dailyReturnsPath
    dailyReturnsKey: str = 'returns_daily'
    #hourlyData.to_hdf('./my_returns_hourly_data_2017-2023.h5', 'returns_hourly')
    ####
    # 3 - 2 - split train test features targets
    @classmethod
    def createTrainFeaturesPath(cls):
        makeDirectoryPath(['checkpoints', 'train_features'])
    trainFeaturesPath: str = ('checkpoints/train_features'
                                + "/3_2_train_features_2017-2023.h5")
    relativeTrainFeaturesPath: str = "../3_make_model/rnn/2_split_train_test_features_targets/"+trainFeaturesPath
    trainFeaturesKey: str = 'train_features'
    ##
    @classmethod
    def createTrainTargetsPath(cls):
        makeDirectoryPath(['checkpoints', 'train_targets'])
    trainTargetsPath: str = ('checkpoints/train_targets'
                              + "/3_2_train_targets_2017-2023.h5")
    relativeTrainTargetsPath: str = "../3_make_model/rnn/2_split_train_test_features_targets/"+trainTargetsPath
    trainTargetsKey: str = 'train_targets'
    ####
    @classmethod
    def createTestFeaturesPath(cls):
        makeDirectoryPath(['checkpoints', 'test_features'])
    testFeaturesPath: str = ('checkpoints/test_features'
                              + "/3_2_test_features_2017-2023.h5")
    relativeTestFeaturesPath: str = "../3_make_model/rnn/2_split_train_test_features_targets/"+testFeaturesPath
    testFeaturesKey: str = 'test_features'
    ##
    @classmethod
    def createTestTargetsPath(cls):
        makeDirectoryPath(['checkpoints', 'test_targets'])
    testTargetsPath: str = ('checkpoints/test_targets'
                             + "/3_2_test_targets_2017-2023.h5")
    relativeTestTargetsPath: str = "../3_make_model/rnn/2_split_train_test_features_targets/"+testTargetsPath
    testTargetsKey: str = 'test_targets'
    ####
    @classmethod
    def createTrainLaggedReturnsPath(cls):
        makeDirectoryPath(['checkpoints', 'train_lagged_returns'])
    trainLaggedReturnsPath: str = ('checkpoints/train_lagged_returns'
                            + "/3_2_train_lagged_returns_2017-2023.npy")
    relativeTrainLaggedReturnsPath: str = ("../3_make_model/rnn/2_split_train_test_features_targets/"
                                           + trainLaggedReturnsPath)
    ####
    @classmethod
    def createTestLaggedReturnsPath(cls):
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
    allDataStart: str = '2022-06-01 UTC'
    allDataEnd: str = '2022-06-30 UTC'
    #allDataStart: str = '2017-01-01 UTC'
    #@    allDataEnd: str = '2023-01-01 UTC'
    allDataTimeFrame: str = '1h'
    hourlyReturnsWindowSize = 120
    dailyReturnsWindowSize = 52

    @classmethod
    def modelPath(cls, tag: str):
        modelsPath: str = (makeDirectoryPath(['models'])
                                      + "/3_3_model_"+tag)
        return modelsPath

    @classmethod
    def modelCheckpointPath(cls, tag: str):
        modelsPath: str = (makeDirectoryPath(['checkpoints', 'modelCheckpoints'])
                           + "/3_3_model_"+tag+".h5")
        return modelsPath


