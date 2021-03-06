import os
import time
import tflearn
import tensorflow as tf
from sklearn.model_selection import train_test_split

from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.estimator import regression
from tflearn.optimizers import Momentum, Adam, SGD


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

    def __pickOptimizer(self, network, optimizer):

        if optimizer == 'Momentum':

            optimizer = Momentum(learning_rate = self.__.getfloat(self._name, "LR"), 
                    momentum = self.__.getfloat(self._name, "OPTIMIZER_PARAM"), 
                    lr_decay = self.__.getfloat(self._name, "LEARNING_RATE_DECAY"), 
                    decay_step = self.__.getfloat(self._name, "DECAY_STEP"))
        elif optimizer == 'Adam':

            optimizer = Adam(learning_rate = self.__.getfloat(self._name, "LR"), 
                    beta1 = self.__.getfloat(self._name, "OPTIMIZER_PARAM"), 
                    beta2 = self.__.getfloat(self._name, "LEARNING_RATE_DECAY"))
        elif optimizer == 'SGD':

            optimizer = SGD(learning_rate = self.__.getfloat(self._name, "LR"))
        else:
            print( "Unknown optimizer: {}".format(optimizer))
        
        reg = regression(network, optimizer = optimizer, loss = self.__.get(self._name, "LOSS"), learning_rate = self.__.getfloat(self._name, "LR"), name = "targets")

        return reg

    def __buildCNNNetwork(self):

        #Input Layer
        convnet = input_data(name="input", shape=[None, self.__.getint(self._name, "IMAGE_SIZE"), self.__.getint(self._name, "IMAGE_SIZE"), self.__.getint(self._name, "CHANNEL")])

        if self._name == "Conv_10":

            #Enabling Filters
            convnet = conv_2d(convnet, 32, 3, strides=2, activation = self.__.get(self._name, "ACTIVATION"))
            convnet = max_pool_2d(convnet, 2)

            convnet = conv_2d(convnet, 64, 3, strides=2, activation = self.__.get(self._name, "ACTIVATION"))
            convnet = max_pool_2d(convnet, 2)

            convnet = conv_2d(convnet, 128, 3, strides=1, activation = self.__.get(self._name, "ACTIVATION"))
            convnet = max_pool_2d(convnet, 2)

            convnet = conv_2d(convnet, 64, 3, strides=2, activation = self.__.get(self._name, "ACTIVATION"))
            convnet = max_pool_2d(convnet, 2)

            convnet = conv_2d(convnet, 32, 3, strides=2, activation = self.__.get(self._name, "ACTIVATION"))
            convnet = max_pool_2d(convnet, 2)

            convnet = fully_connected(convnet, 1024, activation = self.__.get(self._name, "ACTIVATION"))


        elif self._name == "Conv_6_64":

            #Enabling Filters
            convnet = conv_2d(convnet, 64, 3, strides=1, activation = self.__.get(self._name, "ACTIVATION"))
            convnet = max_pool_2d(convnet, 3, strides=2)

            convnet = conv_2d(convnet, 128, 3, strides=1, activation = self.__.get(self._name, "ACTIVATION"))
            convnet = max_pool_2d(convnet, 3, strides=2)

            convnet = conv_2d(convnet, 256, 3, strides=1, activation = self.__.get(self._name, "ACTIVATION"))
            convnet = max_pool_2d(convnet, 3, strides=2)

            # convnet = conv_2d(convnet, 64, 3, strides=2, activation = self.__.get(self._name, "ACTIVATION"))
            # convnet = max_pool_2d(convnet, 2)

            # convnet = conv_2d(convnet, 32, 3, strides=2, activation = self.__.get(self._name, "ACTIVATION"))
            # convnet = max_pool_2d(convnet, 2)

            convnet = fully_connected(convnet, 4096, activation = self.__.get(self._name, "ACTIVATION"))
            convnet = fully_connected(convnet, 1024, activation = self.__.get(self._name, "ACTIVATION"))
        
        else:
                raise Exception("Invalid Strategy Name: " + self._name)

        #if dropout value greater than 0.0
        if self.__.getfloat(self._name, "DROP_OUT_VALUE") > 0.0:
            convnet = dropout(convnet, self.__.getfloat(self._name, "DROP_OUT_VALUE"))

        #Output Layer
        convnet = fully_connected(convnet, self._category_count, activation = "softmax")
        convnet = self.__pickOptimizer(convnet, self.__.get(self._name, "OPTIMIZER"))

        return convnet

    def buildCNN(self):

        convnet = self.__buildCNNNetwork()

        self._model_name = "{0}-{1}_{2}_{3}_{4}-{5}-{6}-{7}-{8}.model".format(
                            self.__.get("Models", "MODEL_NAME"),
                            self.__.get(self._name, "OPTIMIZER"),
                            self.__.getfloat(self._name, "LR"),
                            self.__.getfloat(self._name, "OPTIMIZER_PARAM"),
                            self.__.getfloat(self._name, "LEARNING_RATE_DECAY"),
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
            snapshot_epoch = True,
        )

    def save(self):

        model_full_path = os.path.join(self.__.get("Models", "MODELS_DIR"), self._model_name, self._model_name)
        self._model.save(model_full_path)