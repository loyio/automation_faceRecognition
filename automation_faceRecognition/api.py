# MIT License
# Copyright (c) 2021 Loyio

from PIL import Image
from numpy import asarray
import matplotlib.pyplot
import os
import cv2
import sys

sys.path.append("..")
from mtcnn.mtcnn import MTCNN

# make directory
def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                   
		os.makedirs(path)
		print("create folder success!!!")
	else:
		print("create folder failed!!!")

# get face_frame parameter use MTCNN detect_faces
def get_face_frame_box(face_image):
    detector = MTCNN()
    results = detector.detect_faces(face_image)
    if results:
        return results[0]['box']
    else:
        return []

# get face rectangle from input image
def get_face_array(image_frame, face_frame_box, required_size=(224, 224)):
    x1, y1, width, height = face_frame_box
    x2, y2 = x1+width, y1+height
    face = image_frame[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


# merge image
def merge_image(back, front, x,y):
    if back.shape[2] == 3:
        back = cv2.cvtColor(back, cv2.COLOR_BGR2BGRA)
    if front.shape[2] == 3:
        front = cv2.cvtColor(front, cv2.COLOR_BGR2BGRA)

    bh,bw = back.shape[:2]
    fh,fw = front.shape[:2]
    x1, x2 = max(x, 0), min(x+fw, bw)
    y1, y2 = max(y, 0), min(y+fh, bh)
    front_cropped = front[y1-y:y2-y, x1-x:x2-x]
    back_cropped = back[y1:y2, x1:x2]

    alpha_front = front_cropped[:,:,3:4] / 255
    alpha_back = back_cropped[:,:,3:4] / 255
    
    result = back.copy()
    result[y1:y2, x1:x2, :3] = alpha_front * front_cropped[:,:,:3] + (1-alpha_front) * back_cropped[:,:,:3]
    result[y1:y2, x1:x2, 3:4] = (alpha_front + alpha_back) / (1 + alpha_front*alpha_back) * 255

    return result


    