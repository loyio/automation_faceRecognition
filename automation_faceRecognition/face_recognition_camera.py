# MIT License
# Copyright (c) 2021 Loyio

import cv2
import time
import tensorflow as tf
import numpy as np
from numpy import asarray, expand_dims
import sys
from api import *
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K

sys.path.append("..")

person_list = [f for f in os.listdir("Images_crop/") if not f.startswith('.')]
person_rep=dict()
for i,person in enumerate(person_list):
  person_rep[i]=person


vgg_face = vgg_face()

capture = cv2.VideoCapture(0)
time.sleep(1)

while(True):
    ret, frame = capture.read()
    frame = cv2.flip(frame,1)
    frame = cv2.resize(frame, (600, 400))
    box = get_face_frame_box(frame)
    if len(box) != 0:
        face_image = get_face_array(frame, box)
        face_image = face_image.astype('float32')
        face_sample = expand_dims(face_image, axis=0)
        face_sample = preprocess_input(face_sample)
        face_encode = vgg_face(face_sample)

        #make predictions
        embed = K.eval(face_encode)
        model = tf.keras.models.load_model('model/face_classifier_model.h5')
        person = model(embed)
        print("person probability: "+str(person))
        name=person_rep[np.argmax(person)]
        x, y, w, h = box[0], box[1], box[2], box[3]
        if np.max(person) > 0.5:
            print("recognition person: "+ name)
            text= f"{np.max(person)*100:.2f}%"
            cv2.putText(frame, name+text, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)
            # frame = merge_image(frame, face_image, 0 , 0)
    cv2.imshow("webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cv2.destroyAllWindows()
