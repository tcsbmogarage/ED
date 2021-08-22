import os
import time
import argparse
from configparser import ConfigParser, ExtendedInterpolation
import pandas as pd
import numpy as np
import tensorflow as tf

#Local Packages
from Facial.Utils import Utils
from Facial.Data import Data
from Facial.Strategy import Strategy

#Initialize Argument Parser
parser = argparse.ArgumentParser()

#Adding Optional parameters
parser.add_argument("-g", "--gpu", type=float, default=0.0, dest="gpu_mem_size", help="gpu memory usage")
args = parser.parse_args()

#Initialize Config Parser
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read("./Config/Rule.ini")

#Emotions
EMOTIONS = dict(config["Emotions"])

#Report GPU Status
if args.gpu_mem_size > 0.0:

    from numba import jit, cuda
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    Utils.reportGPU(tf)

#Initialize Data Util
data = Data(EMOTIONS.values(), config)

#Load Training data
training_files = data.getImagesListWithLimit(config.get("Data", "TRAIN_DIR"), config.getint("Data", "MAX_TRAIN_SAMPLES"))
print("Training Set: ")
Utils.printPandasGroupBy(training_files, [ "Path", "Emotion" ], "Emotion")
train_data = data.preProcess(training_files)
print("Total Training Files: {0}".format(len(train_data)))

#Load Validation data
testing_files = data.getImagesListWithLimit(config.get("Data", "TEST_DIR"), config.getint("Data", "MAX_TEST_SAMPLES"))
print("Testing Set: ")
Utils.printPandasGroupBy(testing_files, [ "Path", "Emotion" ], "Emotion")
test_data = data.preProcess(testing_files)
print("Total Validation Files: {0}".format(len(test_data)))

#Initiate CNN Strategy
ed_strategy_name = config.get("Rules", "EDStrategyName")
strategy = Strategy(ed_strategy_name, len(EMOTIONS.values()), config)
network = strategy.buildCNN()

#Model Path
model_path = os.path.join(config.get("Models", "MODELS_DIR"), strategy.model_name)

#Write randomly picked files list
if config.get("Default", "ENV").lower() == "prod":

    #Save files
    pd.DataFrame(training_files).to_csv(model_path + "/t.csv")
    pd.DataFrame(testing_files).to_csv(model_path + "/v.csv")


#Split train and test data
train = train_data
test = test_data

image_size = config.getint(ed_strategy_name, "IMAGE_SIZE")
channel = config.getint(ed_strategy_name, "CHANNEL")
#Split for training and testing
train_x = np.array([index[0] for index in train]).reshape(-1, image_size, image_size, channel)

train_y = np.array([index[1] for index in train])

test_x = np.array([index[0] for index in test]).reshape(-1, image_size, image_size, channel)

test_y = np.array([index[1] for index in test])

print("TrainX: {}".format(train_x.shape))
print("TrainY: {}".format(train_y.shape))
print("TestX: {}".format(test_x.shape))
print("TestY: {}".format(test_y.shape))

#Fit the splitted data
start = time.time()
strategy.fit(train_x, train_y, test_x, test_y)
end = time.time()
print("Completed time: {0}".format(end - start))

#Save the model
strategy.save()

