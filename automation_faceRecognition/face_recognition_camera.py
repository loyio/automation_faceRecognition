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
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import decode_predictions
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D
from tensorflow.keras.layers import Dense,Dropout,Softmax,Flatten,Activation,BatchNormalization
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K

sys.path.append("..")
from mtcnn.mtcnn import MTCNN

person_list = [f for f in os.listdir("Images_crop/") if not f.startswith('.')]
person_rep=dict()
for i,person in enumerate(person_list):
  person_rep[i]=person

#Define VGG_FACE_MODEL architecture
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

# Load VGG Face model weights
model.load_weights('model/vgg_face_weights.h5')

# Remove Last Softmax layer and get model upto last flatten layer with outputs 2622 units
vgg_face=Model(inputs=model.layers[0].input,outputs=model.layers[-2].output)



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
        face_sample = preprocess_input(face_sample)
        face_encode = vgg_face(face_sample)

        #make predictions
        embed = K.eval(face_encode)
        model = tf.keras.models.load_model('model/face_classifier_model.h5')
        person = model.predict(embed)
        print("person probability: "+str(person))
        name=person_rep[np.argmax(person)]
        conf = boxes[0]['confidence']
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
