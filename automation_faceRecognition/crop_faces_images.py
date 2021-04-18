import os
from api import get_face_frame_box, get_face_array, mkdir
import numpy as np
import cv2

person_list = [f for f in os.listdir("Images/") if not f.startswith('.')]

for person in person_list:
    mkdir("Images_crop/"+person)
    image_list = [f for f in os.listdir("Images/"+person+"/") if not f.startswith('.')]
    for image in image_list:
        img=cv2.imread('Images/'+person+"/"+image)
        frame_box = get_face_frame_box(img)
        if len(frame_box) != 0:
            crop_image = get_face_array(img, frame_box)
            cv2.imwrite('Images_crop/'+person+"/"+image, crop_image)