#!/usr/bin/env python
# coding: utf-8

# # Usecase: AI driven Emotion Detection: HNAPS Classification

# In[1]:


import os
import re
import html
import base64
import numpy as np
import collections
import time
import pandas as pd
from random import shuffle
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from tqdm import tqdm
from numba import jit, cuda

import tensorflow as tf
from mtcnn.mtcnn import MTCNN
from IPython.core.display import HTML
from moviepy.editor import *
from prettytable import PrettyTable


# ### Test huge dataset

# In[2]:


ROOT_DIR = "/home/azureuser/"
#ROOT_DIR = "/Users/garage/Cohort/"
USECASE_NAME = "TF_Usecase"
USECASE_DIR = ROOT_DIR + USECASE_NAME + "/"
DATA_DIR = ROOT_DIR + "Data/"
TRAIN_DIR = DATA_DIR + "fer2013-skimmed/train/"
TEST_DIR = DATA_DIR + "fer2013-skimmed/test/"
IMG_FORMAT = ".jpg"
CHANNEL = 1


MODELS_DIR = ROOT_DIR + "TrainedModels/" + USECASE_NAME + "/"
TRAIN_DUMP_FILE = "Train-fer2013.np"
TEST_DUMP_FILE = "Test-fer2013.np"
IMAGE_SIZE = 48
EMOTIONS = {0: "neutral", 1: "happy", 2: "angry", 3: "perplex", 4: "sad"}
DROP_OUT_VALUE = 0.5
ACTIVATION = "relu"
OPTIMIZER = "SGD"
LR = 1e-3
EPOCH = 500
MODEL_NAME = "EmotionDetection-{}_{}-{}_{}-{}-{}.model".format(OPTIMIZER, LR, '10-conv-HNAPS', DROP_OUT_VALUE, EPOCH, int(time.time()))

if not os.path.exists(MODELS_DIR + MODEL_NAME):
    os.mkdir(MODELS_DIR + MODEL_NAME)

GPU_MEM_LIMIT = 0.5
BEST_VAL_ACCURACY = 0.5
BEST_CHECKPOINT_PATH = os.path.join(MODELS_DIR, MODEL_NAME, "best")

if not os.path.exists(BEST_CHECKPOINT_PATH):
    os.mkdir(BEST_CHECKPOINT_PATH)

#GPU or CPU Mode
if tf.test.gpu_device_name():
    print('GPU: {}'.format(tf.test.gpu_device_name()))
else:
    print("No GPU found")

# ### Test min dataset

# In[3]:


# ROOT_DIR = "/Users/garage/Cohort/"
# UTIL_DIR = ROOT_DIR + "Utils/"
# USECASE_DIR = ROOT_DIR + "Usecase/"
# TRAIN_DIR = UTIL_DIR + "vision/fer2013-min/train/"
# TEST_DIR = UTIL_DIR + "vision/fer2013-min/test/"
# IMG_FORMAT = ".jpg"
# CHANNEL = 1

# DATA_DIR = USECASE_DIR + "Data/"
# MODELS_DIR = USECASE_DIR + "Models/"
# TRAIN_DUMP_FILE = "Train-fer2013.np"
# TEST_DUMP_FILE = "Test-fer2013.np"
# IMAGE_SIZE = 48
# EMOTIONS = {0: "neutral", 1: "happy", 2: "angry", 3: "perplex", 4: "sad"}
# DROP_OUT_VALUE = 0.4
# ACTIVATION = "relu"
# LR = 1e-3
# MODEL_NAME = "MinEmotionDetection-{}-{}_{}-{}.model".format(LR, '6-conv-basic', DROP_OUT_VALUE, int(time.time()))
# EPOCH = 500


# ### Insight

# In[4]:


def count_exp(path, label):
    
    dict = {}
    e_only_folder = re.compile("[∧a-zA-Z\-]")

    for expression in os.listdir(path):
        
        #omit other folders
        if not re.match(e_only_folder, expression):
            continue
        
        dir = path + expression
        dict[expression] = len(os.listdir(dir))
        
    df = pd.DataFrame(dict, index=[label])
    
    return df


# In[5]:


train_count = count_exp(TRAIN_DIR, 'train')
test_count = count_exp(TEST_DIR, 'test')
print(train_count)
print(test_count)


# In[6]:


# train_count.transpose().plot(kind='bar')


# ### Load a test image

# In[7]:


# file_path = TRAIN_DIR + "perplex/Training_MIP_1.jpg"
# print(file_path)
# image = cv.imread(file_path, cv.IMREAD_UNCHANGED)
# image.shape


# In[8]:


# plt.imshow(image)


# ### Label frames

# In[9]:


def label_image(category):
    
    label = [int(c == category) for c in EMOTIONS.values()]
    return label


# In[10]:


def get_images_list(dataPath):
    
    files = []
    e_only_folder = re.compile("[∧a-zA-Z\-]")
    
    #Iterate all emotions
    for emotion in os.listdir(dataPath):
        
        #omit other folders
        if not re.match(e_only_folder, emotion):
            continue
            
        for image_file in tqdm(os.listdir(dataPath + emotion)):
            
            #Omit other files
            if not image_file.endswith(IMG_FORMAT):
                continue
                
            image_path = os.path.join(dataPath, emotion, image_file)
            files.append([image_path, emotion])
    
    return files


# In[11]:


def preprocess(files, dumpFilePath):
    
    data = []

    #Iterate each image in it
    for image_path, category in tqdm(files):

        #Label identification
        label = label_image(category)

        #preprocessing
        image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
        #image_gs_resized = cv.resize(image_gs, (IMAGE_SIZE, IMAGE_SIZE))

        data.append([np.array(image), np.array(label)])
    
    #Save training data
    np.save(DATA_DIR + dumpFilePath, data)
    
    return data
        


# ### Training Data

# In[12]:


training_files = get_images_list(TRAIN_DIR)
shuffle(training_files)
train_data = preprocess(training_files, TRAIN_DUMP_FILE)


# In[13]:


print(len(train_data))


# ### Validation Data

# In[14]:


testing_files = get_images_list(TEST_DIR)
shuffle(testing_files)
test_data = preprocess(testing_files, TEST_DUMP_FILE)


# In[15]:


print(len(test_data))


# ### Creating CNN

# In[16]:


#get_ipython().run_line_magic('load_ext', 'tensorboard')

import tflearn
from sklearn.model_selection import train_test_split

from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.estimator import regression

tflearn.init_graph(num_cores=40,gpu_memory_fraction=GPU_MEM_LIMIT)
# In[17]:


#Input Layer
convnet = input_data(name="input", shape=[None, IMAGE_SIZE, IMAGE_SIZE, CHANNEL])

#Enabling Filters
convnet = conv_2d(convnet, 32, 5, activation = ACTIVATION)
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation = ACTIVATION)
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation = ACTIVATION)
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation = ACTIVATION)
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation = ACTIVATION)
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation = ACTIVATION)
convent = dropout(convnet, DROP_OUT_VALUE)
#Output Layer

convnet = fully_connected(convnet, len(EMOTIONS.keys()), activation = "softmax")
convnet = regression(convnet, optimizer = OPTIMIZER, learning_rate = LR, loss = "categorical_crossentropy", name = "targets")


# ### Creating model and spliting validation data

# In[18]:

model = tflearn.DNN(convnet, 
        tensorboard_dir="Logs/" + MODEL_NAME, 
        tensorboard_verbose = 3, 
        best_checkpoint_path = BEST_CHECKPOINT_PATH + '/', 
        best_val_accuracy = BEST_VAL_ACCURACY
        )

#Split train and test data
train = train_data
test = test_data

#Split for training and testing
train_x = np.array([index[0] for index in train]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, CHANNEL)

train_y = np.array([index[1] for index in train])

test_x = np.array([index[0] for index in test]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, CHANNEL)

test_y = np.array([index[1] for index in test])

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)


# ### Fit and save the model

# In[19]:
def fit(train_x, train_y, test_x, test_y, epoch, model_name):

    model.fit(
        train_x,
        train_y, 
        validation_set = (test_x, test_y),
        n_epoch = epoch,
        snapshot_step = 500,
        show_metric = True,
        run_id = model_name,
        snapshot_epoch = True
    )
start = time.time()
fit(train_x, train_y, test_x, test_y, EPOCH, MODEL_NAME)
# In[21]:


model.save(os.path.join(MODELS_DIR, MODEL_NAME, MODEL_NAME))
end = time.time()
print("Completed time: {0}".format(end - start))




