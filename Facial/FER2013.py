from fer import FER
from mtcnn.mtcnn import MTCNN
import mediapipe

import cv2 as cv

class EmotionDetection(FER):

    _mtcnn = True
    _face_min_score = 0
    _emotion_model_hdf5 = None

    def __iniit__(self, mtcnn = True, image_size = 48, face_min_score = 0, emotion_model_hdf5 = None):
       
        self._mtcnn = mtcnn
        self._face_min_score = face_min_score
        self._emotion_model_hdf5 = emotion_model_hdf5
        self._image_size = image_size

        #Calling face detector
        if self._mtcnn:
            self._f_detector = MTCNN()
        else:
            self._f_detector = mediapipe.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=self._face_min_score)

        #Calling FER
        self._fer_detector = super.__init__(mtcnn = self._mtcnn, emotion_model = self._emotion_model_hdf5)

    def __detect_faces(self, frame):

        faces = None

        if self._mtcnn:
            faces = self._f_detector.detect_faces(frame)
        else:
            faces = self._f_detector.process(frame).detections

        return faces

    def __get_bbox(self, face, shape):

        feature = []
        if self._mtcnn:
            feature = face['box']
        else:
            (im_h, im_w, im_c) = shape
            coord = face.location_data.relative_bounding_box
            (x, y, w, h) = (int(coord.xmin * im_w), int(coord.ymin * im_h), int(coord.width * im_w), int(coord.height * im_h))

    def process(self, frame):

        c_faces = []
    
        #Convert into RGB
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        #Get faces using score
        faces = self._f_detector.__detect_faces(frame_rgb)
        
        if faces is not None:
            #Iterate bounding boxes
            for face in faces:
                
                if self._mtcnn:
                    #skip less confidence score faces
                    if face['confidence'] < self._face_min_score:
                        continue

                (x, y, w, h) = self.__get_bbox(face, frame_rgb.shape)
                
                #Extract face
                face = frame_rgb[y:y + h, x:x + w]
                
                #Get faces using FER
                faces = self._fer_detector.detect_emotions(face)

                if len(faces) > 0:
                    #Iterate bounding boxes
                    for face in faces:
                            
                        (x, y, w, h) = face['box']

                        #Extract face
                        face = face[y:y + h, x:x + w]

                        #Convert into gray
                        face_gray = cv.cvtColor(face, cv.COLOR_RGB2GRAY)

                        #resize the face
                        face_resized = cv.resize(face_gray, (self._image_size, self._image_size), interpolation = cv.INTER_AREA)   

                        #Add to the queue
                        c_faces.append((face_resized, x, y, w, h))
            
        return c_faces
