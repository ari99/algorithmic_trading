import warnings
from configs.config import Config
warnings.filterwarnings('ignore')
import os
import vectorbtpro as vbt
import pickle
import numpy as np
from .modelMaker import ModelMaker
from keras.callbacks import History
from tensorflow import keras


class CreateModel:

    def __init__(self):
        self.trainLaggedReturns = np.load("../../" + Config.relativeTrainLaggedReturnsPath)
        self.trainFeatures = vbt.HDFData.fetch("../../" + Config.relativeTrainFeaturesPath).data['train_features']
        self.allTrainTargets = vbt.HDFData.fetch("../../" + Config.relativeTrainTargetsPath).data['train_targets']
        self.testLaggedReturns = np.load("../../" + Config.relativeTestLaggedReturnsPath)
        self.testFeatures = vbt.HDFData.fetch("../../" + Config.relativeTestFeaturesPath).data['test_features']
        self.allTestTargets = vbt.HDFData.fetch("../../" + Config.relativeTestTargetsPath).data['test_targets']

    def __createBias(self, trainTargets, key: str):
        neg, pos = np.bincount(trainTargets[key])
        total = neg + pos
        print(key + ' Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
            total, pos, 100 * pos / total))
        return np.log([pos/neg])

    def __createClassWeights(self, trainTargets, key: str):
        neg, pos = np.bincount(trainTargets[key])
        total = neg + pos
        # Scaling by total/2 helps keep the loss to a similar magnitude.
        # The sum of the weights of all examples stays the same.
        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)

        class_weight = {0: weight_for_0, 1: weight_for_1}

        return class_weight

    def __createRnn(self, targetKey: str):

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

        rnn = modelMaker.createModel(len(self.trainFeatures.columns),
                                             len(trainTargets.columns), output_bias=initial_bias)
        earlyStopping, checkpointer = modelMaker.createCallbacks()
        # see rnn_try_2 for tensorboard callback usage
        history: History = rnn.fit(X_train,
                                           y_train,
                                           epochs=1,
                                           batch_size=2000, # with a large batch size on 10, it hardly gives any trues
                                           validation_data=(X_test, y_test),
                                           callbacks=[earlyStopping, checkpointer],
                                           verbose=1,
                                           class_weight=weights)
        return rnn, history



    def createModel(self, targetKey: str):
        #  https://www.tensorflow.org/api_docs/python/tf/keras/saving/save_model
        # https://www.tensorflow.org/guide/keras/save_and_serialize
        modelPath = Config.modelPath(targetKey)
        print("model path is "+ modelPath)
        # Calling `save('my_model')` creates a SavedModel folder `my_model`.
        if os.path.exists(modelPath):
            print(" Using existing model")
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

