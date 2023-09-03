
import os
from tensorflow import keras
from configs.config import Config
import pickle

class ModelLoader:

    @classmethod
    def loadModel(self, targetKey: str):
        #  https://www.tensorflow.org/api_docs/python/tf/keras/saving/save_model
        #  https://www.tensorflow.org/guide/keras/save_and_serialize
        modelPath = Config.modelPath(targetKey)
        print("model path is "+ modelPath)
        if os.path.exists(modelPath):
            print(" Using Existing model")
            rnn = keras.models.load_model(modelPath)
            with open(modelPath+'/trainHistoryDict', "rb") as file_pi:
                history = pickle.load(file_pi)
            return rnn, history
        else:
            print(" No Existing model")
            return None, None

