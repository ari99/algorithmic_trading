from typing import Tuple

from keras.callbacks import ModelCheckpoint, EarlyStopping
# import warnings
# warnings.filterwarnings('ignore')
from keras.models import Model
from keras.layers import Dense, LSTM, Input, concatenate, BatchNormalization, Dropout
import tensorflow as tf
from tensorflow import keras
# for local computer use:
# from keras.optimizers import SGD
# for sagemaker:
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop


class ModelMaker():

    def __init__(self, lstmPath: str):
        self.n_features = 1
        self.window_size = 120
        self.lstmPath = lstmPath

    def createCallbacks(self):
        checkpointer = ModelCheckpoint(filepath=self.lstmPath,
                                       verbose=1,
                                       monitor='auc',
                                       mode='max',
                                       save_best_only=True)
        early_stopping = EarlyStopping(monitor='auc',
                                       patience=500,
                                       restore_best_weights=True,
                                       mode='max')

        return early_stopping, checkpointer

    def __createInput(self, trainFeaturesColumns: int) -> Tuple[Input, Input]:
        returnsInput = Input(shape=(self.window_size, self.n_features), name='Returns')
        featuresInput = Input(shape=(trainFeaturesColumns,), name='Features')
        return returnsInput, featuresInput

    def __createLSTMLayers(self, returnsInput: Input):
        lstm1_units = 200
        lstm2_units = 100
        lstm1 = LSTM(units=lstm1_units,
                     input_shape=(self.window_size,
                                  self.n_features),
                     name='LSTM1',
                     dropout=.1,
                     return_sequences=True)(returnsInput)

        lstm_model = LSTM(units=lstm2_units,
                          dropout=.1,
                          name='LSTM2')(lstm1)

        return lstm_model

    def createModel(self, trainFeaturesColumns: int, trainTargetsColumns: int, output_bias=None) -> Model:
        # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#calculate_class_weights

        returnsInput, featuresInput = self.__createInput(trainFeaturesColumns)
        lstm_model = self.__createLSTMLayers(returnsInput)
        merged = concatenate([lstm_model,
                              featuresInput], name='Merged')

        bn = BatchNormalization()(merged)
        # hidden_dense = Dense(10, name='FC1')(bn)
        hidden_dense = Dense(300, name='FC1')(bn)
        drop1 = Dropout(0.1)(hidden_dense)
        dense1 = Dense(100, name='FC2')(drop1)
        drop2 = Dropout(0.1)(dense1)
        dense2 = Dense(50, name='FC3')(drop2)

        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)

        output: Dense = Dense(trainTargetsColumns, name='Output',
                              activation='sigmoid', bias_initializer=output_bias)(dense2)

        # softmax doesnt seem to work, the AUC stays at .5 and accuracy is around .1
        # output = Dense(len(trainTargets.columns), name='Output', activation='softmax')(hidden_dense)

        rnnModel: Model = Model(inputs=[returnsInput, featuresInput], outputs=output)
        optimizer: RMSprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0001)
        # optimizer = SGD(lr=0.01)
        print(" Compiling Model")

        METRICS = [
            keras.metrics.TruePositives(name='true-positives'),
            keras.metrics.FalsePositives(name='false-positives'),
            keras.metrics.TrueNegatives(name='true-negatives'),
            keras.metrics.FalseNegatives(name='false-negatives'),
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='precision-recall-curve-auc', curve='PR'),  # precision-recall curve
        ]

        rnnModel.compile(  # loss=BinaryCrossentropy(from_logits=False), #'binary_crossentropy',
            # loss='categorical_crossentropy',
            # loss='sparse_categorical_crossentropy',
            loss='binary_crossentropy',
            # optimizer='adam',
            optimizer=optimizer,
            metrics=METRICS)

        return rnnModel
