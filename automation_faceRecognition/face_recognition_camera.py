# MIT License
# Copyright (c) 2021 Loyio

import cv2
import time
import tensorflow as tf
from numpy import asarray, expand_dims
import sys
from api import *
from PIL import Image
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
#from matplotlib import pyplot

sys.path.append("..")
from mtcnn.mtcnn import MTCNN

detector = MTCNN()

capture = cv2.VideoCapture(0)
time.sleep(1)

while(True):
    ret, frame = capture.read()
    frame = cv2.flip(frame,1)
    frame = cv2.resize(frame, (600, 400))
    boxes = detector.detect_faces(frame)
    if boxes:
        box = boxes[0]['box']
        face_image = get_face_array(frame, box)
        face_image = face_image.astype('float32')
        face_sample = expand_dims(face_image, axis=0)
        face_sample = preprocess_input(face_sample, version=2)
        model = VGGFace(model='resnet50')
        yhat = model.predict(face_sample)
        results = decode_predictions(yhat)
        for result in results[0]:
            print("%s: %.3f%%"%(result[0], result[1]*100))
        conf = boxes[0]['confidence']
        x, y, w, h = box[0], box[1], box[2], box[3]
        if conf > 0.5:
            text= f"{conf*100:.2f}%"
            cv2.putText(frame, text, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)
            frame = merge_image(frame, face_image, 0 , 0)
    cv2.imshow("webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cv2.destroyAllWindows()
