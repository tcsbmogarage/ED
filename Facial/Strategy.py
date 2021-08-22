import os
import time
import tflearn
import tensorflow as tf
from sklearn.model_selection import train_test_split

from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.estimator import regression

from Facial.Utils import Utils
class Strategy:

    __ = None
    _name = None
    _model_name = None
    _model = None
    _category_count = 0

    def __init__(self, name, categoryCount, config) -> None:

        self._name = name
        self._category_count = categoryCount
        self._model_name = None
        self._model = None
        self.__ = config

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, modelName):
        self._model_name = modelName
    
    def __buildCNNNetwork(self):

        #Input Layer
        convnet = input_data(name="input", shape=[None, self.__.getint(self._name, "IMAGE_SIZE"), self.__.getint(self._name, "IMAGE_SIZE"), self.__.getint(self._name, "CHANNEL")])

        if self._name == "Conv_10":

            #Enabling Filters
            convnet = conv_2d(convnet, 32, 5, activation = self.__.get(self._name, "ACTIVATION"))
            convnet = max_pool_2d(convnet, 5)

            convnet = conv_2d(convnet, 64, 5, activation = self.__.get(self._name, "ACTIVATION"))
            convnet = max_pool_2d(convnet, 5)

            convnet = conv_2d(convnet, 128, 5, activation = self.__.get(self._name, "ACTIVATION"))
            convnet = max_pool_2d(convnet, 5)

            convnet = conv_2d(convnet, 64, 5, activation = self.__.get(self._name, "ACTIVATION"))
            convnet = max_pool_2d(convnet, 5)

            convnet = conv_2d(convnet, 32, 5, activation = self.__.get(self._name, "ACTIVATION"))
            convnet = max_pool_2d(convnet, 5)

            convnet = fully_connected(convnet, 1024, activation = self.__.get(self._name, "ACTIVATION"))
            convnet = dropout(convnet, self.__.getfloat(self._name, "DROP_OUT_VALUE"))

        #Output Layer
        convnet = fully_connected(convnet, self._category_count, activation = "softmax")
        convnet = regression(convnet, optimizer = self.__.get(self._name, "OPTIMIZER"), learning_rate = self.__.getfloat(self._name, "LR"), loss = self.__.get(self._name, "LOSS"), name = "targets")

        return convnet

    def buildCNN(self):

        convnet = self.__buildCNNNetwork()

        self._model_name = "{0}-{1}_{2}-{3}_{4}-{5}-{6}.model".format(
                            self.__.get("Models", "MODEL_NAME"),
                            self.__.get(self._name, "OPTIMIZER"),
                            self.__.getfloat(self._name, "LR"),
                            self.__.get(self._name, "ACTIVATION"),
                            self.__.getfloat(self._name, "DROP_OUT_VALUE"),
                            self.__.getint(self._name, "EPOCH"),
                            int(time.time())
                )

        log_path = os.path.join(self.__.get("TensorBoard", "LOG_DIR"), self._model_name)

        #Create log dir
        Utils.makePath(log_path, mode=0o777)

        model_path = os.path.join(self.__.get("Models", "MODELS_DIR"), self._model_name)
        #Create model dir
        Utils.makePath(model_path, mode=0o777)

        best_cp_path = os.path.join(self.__.get("Models", "BEST_CHECKPOINT_PATH"), self._model_name, "best")

        #Create best checkpoint dir
        Utils.makePath(best_cp_path, mode=0o777)

        self._model = tflearn.DNN(convnet, 
            best_checkpoint_path = best_cp_path + '/', 
            best_val_accuracy = self.__.getfloat("Models", "BEST_VAL_ACCURACY"),
            tensorboard_dir = log_path, 
            tensorboard_verbose = self.__.getint("TensorBoard", "VERBOSE"))

        return self._model


    def fit(self, train_x, train_y, test_x, test_y):

        self._model.fit(
            train_x,
            train_y, 
            validation_set = (test_x, test_y),
            n_epoch = self.__.getint(self._name, "EPOCH"),
            snapshot_step = self.__.getint(self._name, "SNAPSHOT_STEP"),
            show_metric = True,
            run_id = self._model_name,
            snapshot_epoch = True
        )

    def save(self):

        model_full_path = os.path.join(self.__.get("Models", "MODELS_DIR"), self._model_name, self._model_name)
        self._model.save(model_full_path)