import warnings
from typing import Tuple

from keras import Model
from numpy import ndarray
from pandas import DataFrame

from config import Config
import pickle
warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd

from .modelMaker import ModelMaker
from keras.callbacks import History
from tensorflow import keras


class CreateModel:

    def __init__(self):
        self.trainLaggedReturns = np.load("checkpoints/3_2_train_lagged_returns_2017-2023.npy")
        self.trainFeatures = pd.read_hdf("checkpoints/3_2_train_features_2017-2023.h5")
        self.allTrainTargets = pd.read_hdf("checkpoints/3_2_train_targets_2017-2023.h5")
        self.testLaggedReturns = np.load("checkpoints/3_2_test_lagged_returns_2017-2023.npy")
        self.testFeatures = pd.read_hdf("checkpoints/3_2_test_features_2017-2023.h5")
        self.allTestTargets = pd.read_hdf("checkpoints/3_2_test_targets_2017-2023.h5")

    def __createBias(self, trainTargets, key: str) -> ndarray:
        neg, pos = np.bincount(trainTargets[key])
        total = neg + pos
        print(key + ' Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
            total, pos, 100 * pos / total))
        return np.log([pos/neg])

    def __createClassWeights(self, trainTargets, key: str) -> dict[int, float]:
        neg, pos = np.bincount(trainTargets[key])
        total = neg + pos
        # Scaling by total/2 helps keep the loss to a similar magnitude.
        # The sum of the weights of all examples stays the same.
        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)

        class_weight: dict[int, float] = {0: weight_for_0, 1: weight_for_1}

        return class_weight


    def __createRnn(self, targetKey: str) -> Tuple[Model, History]:
        trainTargets = self.allTrainTargets[[targetKey]]
        testTargets = self.allTestTargets[[targetKey]]

        X_train = [
            # removes the month columns, label, ticker columns
            self.trainLaggedReturns,
            self.trainFeatures
        ]
        y_train = trainTargets
        #[x.shape for x in X_train], y_train_longs.shape, y_train_shorts.shape

        X_test = [
            self.testLaggedReturns,
            self.testFeatures,
        ]
        y_test = testTargets

        #[x.shape for x in X_test], y_test_longs.shape, y_test_shorts.shape


        checkpointPath = Config.modelCheckpointPath(targetKey)

        modelMaker = ModelMaker(checkpointPath)
        initial_bias = self.__createBias(trainTargets, targetKey)
        weights = self.__createClassWeights(trainTargets, targetKey)

        print(targetKey + ' Weight for class 0: {:.2f}'.format(weights[0]))
        print(targetKey + ' Weight for class 1: {:.2f}'.format(weights[1]))

        rnn: Model = modelMaker.createModel(len(self.trainFeatures.columns),
                                             len(trainTargets.columns), output_bias=initial_bias)
        earlyStopping, checkpointer = modelMaker.createCallbacks()
        # see rnn_try_2 for tensorboard callback usage
        history: History = rnn.fit(X_train,
                                           y_train,
                                           epochs=200,
                                           batch_size=2200, # with a large batch size on 10, it hardly gives any trues
                                           validation_data=(X_test, y_test),
                                           callbacks=[checkpointer] ,#earlyStopping, checkpointer],
                                           verbose=1,
                                           class_weight=weights)
        return rnn, history



    def createModel(self, targetKey: str) -> Tuple[Model, History]:
        #  https://www.tensorflow.org/api_docs/python/tf/keras/saving/save_model
        # https://www.tensorflow.org/guide/keras/save_and_serialize
        modelPath = Config.modelPath(targetKey)
        print("model path is " + modelPath)
        # Calling `save('my_model')` creates a SavedModel folder `my_model`.
        if os.path.exists(modelPath):
            print(" Using Existing model")
            rnn = keras.models.load_model(modelPath)
            with open(modelPath+'/trainHistoryDict', "rb") as file_pi:
                history = pickle.load(file_pi)
        else:
            rnn, history = self.__createRnn(targetKey)
            keras.models.save_model(rnn, modelPath)
            with open(modelPath+'/trainHistoryDict', 'wb') as file_pi:
                pickle.dump(history.history,  file_pi)
           # rnn.save(modelPath)

        return rnn, history

