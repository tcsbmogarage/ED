import os
import re
from tqdm import tqdm
from random import shuffle
import numpy as np
import cv2 as cv

from Facial.Utils import Utils

class Data:

    __= None
    _emotions = None

    def __init__(self, emotions, config) -> None:
        self.__ = config
        self._emotions = emotions

    def __label_image(self, category):
    
        label = [int(c == category) for c in self._emotions]
        return label

    def __get_images_list(self, dataPath):
    
        files = []
        e_only_folder = re.compile("[∧a-zA-Z\-]")
        
        #Iterate all emotions
        for emotion in os.listdir(dataPath):
            
            #omit other folders
            if not re.match(e_only_folder, emotion):
                continue

            #omit other emotions
            if emotion not in self._emotions:
                continue
                
            for image_file in tqdm(os.listdir(dataPath + emotion)):
                
                #Omit other files
                if not image_file.endswith(self.__["Data"]["IMG_FORMAT"]):
                    continue
                    
                image_path = os.path.join(dataPath, emotion, image_file)
                files.append([image_path, emotion])

        shuffle(files)
        return files

    def __get_images_list_with_limit(self, dataPath, limit):

        files = []
        efiles = []
        e_only_folder = re.compile("[∧a-zA-Z\-]")
        
        #Iterate all emotions
        for emotion in os.listdir(dataPath):
            
            #omit other folders
            if not re.match(e_only_folder, emotion):
                continue

            #omit other emotions
            if emotion not in self._emotions:
                continue

            print(dataPath + emotion)

            for image_file in tqdm(os.listdir(dataPath + emotion)):
                
                #Omit other files
                if not image_file.endswith(self.__["Data"]["IMG_FORMAT"]):
                    continue
                    
                image_path = os.path.join(dataPath, emotion, image_file)
                efiles.append([image_path, emotion])
                
            shuffle(efiles)
            files = files + efiles[0:limit]
            efiles = []
            #debug
            #Utils.printPandasGroupBy(files, ["Path", "Emotion"], "Emotion")

        shuffle(files)
        return files

    def __preprocess(self, files):
    
        data = []

        #Iterate each image in it
        for image_path, category in tqdm(files):

            #Label identification
            label = self.__label_image(category)

            #preprocessing
            image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
            #image_gs_resized = cv.resize(image_gs, (IMAGE_SIZE, IMAGE_SIZE))

            data.append([np.array(image), np.array(label)])
        
        return data

    def getImagesListWithLimit(self, dataPath, limit):

        return self.__get_images_list_with_limit(dataPath, limit)

    def getImagesList(self, dataPath):

        return self.__get_images_list_with_limit(dataPath)

    def preProcess(self, files):

        return self.__preprocess(files)