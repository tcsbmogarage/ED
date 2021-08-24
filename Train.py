import os
import time
import argparse
from configparser import ConfigParser, ExtendedInterpolation
from cv2 import normalize
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
parser.add_argument("-r", "--rules", type=str, default="./Config/Rules.ini", dest="rules_ini", help="Config file path")

args = parser.parse_args()

#Initialize Config Parser
config = ConfigParser(interpolation=ExtendedInterpolation())
if os.path.exists(args.rules_ini):
    print("Used Rule file: {0}".format(args.rules_ini))
    config.read(args.rules_ini)
else:
    raise Exception("Couldn't find the given config file: {0}".format(args.rules_ini))

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

#Create Infref path if not exists
Utils.makePath(config.get("Data", "IN_REF_PATH"), mode=0o755)

#Load Training data
t_dump_file = os.path.join(config.get("Data", "IN_REF_PATH"), config.get("Data", "TRAIN_DUMP_FILE")) + ".npy"

#Load files from second time
if os.path.exists(t_dump_file):
    print("Loaded Training File: {0}".format(t_dump_file))
    training_files = np.load(t_dump_file)
else:
    training_files = data.getImagesListWithLimit(config.get("Data", "TRAIN_DIR"), config.getint("Data", "MAX_TRAIN_SAMPLES"))
    np.save(t_dump_file, training_files)
    print("Saved Training File: {0}".format(t_dump_file))


print("Training Set: ")
Utils.printPandasGroupBy(training_files, [ "Path", "Emotion" ], "Emotion")
train_data = data.preProcess(training_files, normalize=True)
print("Total Training Files: {0}".format(len(train_data)))

#Load Validation data
v_dump_file = os.path.join(config.get("Data", "IN_REF_PATH"), config.get("Data", "TEST_DUMP_FILE")) + ".npy"

#Load files from second time
if os.path.exists(v_dump_file):
    print("Loaded Testing File: {0}".format(v_dump_file))
    testing_files = np.load(v_dump_file)
else:
    testing_files = data.getImagesListWithLimit(config.get("Data", "TEST_DIR"), config.getint("Data", "MAX_TEST_SAMPLES"))
    np.save(v_dump_file, testing_files)
    print("Saved Testing File: {0}".format(v_dump_file))

print("Testing Set: ")
Utils.printPandasGroupBy(testing_files, [ "Path", "Emotion" ], "Emotion")
test_data = data.preProcess(testing_files, normalize=True)
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

