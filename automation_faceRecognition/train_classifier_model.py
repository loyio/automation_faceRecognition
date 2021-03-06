# MIT License
# Copyright (c) 2021 Loyio

import numpy as np
import os
from api import vgg_face, LossHistory 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Softmax,Flatten,Activation,BatchNormalization
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt 



vgg_face = vgg_face()

#Prepare Training Data
x_train=[]
y_train=[]
person_list = [f for f in os.listdir("Images_crop/") if not f.startswith('.')]
person_rep=dict()
for i,person in enumerate(person_list):
  person_rep[i]=person
  image_names = [f for f in os.listdir("Images_crop/"+person+"/") if not f.startswith('.')]
  for image_name in image_names:
    img=load_img('Images_crop/'+person+'/'+image_name,target_size=(224,224))
    img=img_to_array(img)
    img=np.expand_dims(img,axis=0)
    img=preprocess_input(img)
    img_encode=vgg_face(img)
    x_train.append(np.squeeze(K.eval(img_encode)).tolist())
    y_train.append(i)

print("persons :" + str(person_rep))

x_train=np.array(x_train)
y_train=np.array(y_train)

#Prepare Training Data
x_test=[]
y_test=[]
person_list = [f for f in os.listdir("Images_test_crop/") if not f.startswith('.')]
person_rep=dict()
for i,person in enumerate(person_list):
  person_rep[i]=person
  image_names = [f for f in os.listdir("Images_test_crop/"+person+"/") if not f.startswith('.')]
  for image_name in image_names:
    img=load_img('Images_test_crop/'+person+'/'+image_name,target_size=(224,224))
    img=img_to_array(img)
    img=np.expand_dims(img,axis=0)
    img=preprocess_input(img)
    img_encode=vgg_face(img)
    x_test.append(np.squeeze(K.eval(img_encode)).tolist())
    y_test.append(i)

x_test=np.array(x_test)
y_test=np.array(y_test)



# Save test and train data for later use
# np.save('model/train_data',x_train)
# np.save('model/train_labels',y_train)
# np.save('model/test_data',x_test)
# np.save('model/test_labels',y_test)

# Load saved data
# x_train=np.load('model/train_data.npy')
# y_train=np.load('model/train_labels.npy')
# x_test=np.load('model/test_data.npy')
# y_test=np.load('model/test_labels.npy')

print("y_train: "+ str(y_train))
print("y_test: "+ str(y_test))
# Softmax regressor to classify images based on encoding 
classifier_model=Sequential()
classifier_model.add(Dense(units=100,input_dim=x_train.shape[1],kernel_initializer='glorot_uniform'))
classifier_model.add(BatchNormalization())
classifier_model.add(Activation('tanh'))
classifier_model.add(Dropout(0.3))
classifier_model.add(Dense(units=7,kernel_initializer='glorot_uniform'))
classifier_model.add(BatchNormalization())
classifier_model.add(Activation('tanh'))
classifier_model.add(Dropout(0.2))
classifier_model.add(Dense(units=2,kernel_initializer='he_uniform'))
classifier_model.add(Activation('softmax'))
classifier_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer='nadam',metrics=['accuracy'])

history = LossHistory()

history_vgg = classifier_model.fit(x_train,y_train,epochs=100,validation_data=(x_test,y_test), callbacks=[history])

history.loss_plot('epoch')

# Save model for later use
tf.keras.models.save_model(classifier_model,'model/face_classifier_model.h5')

