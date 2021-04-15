import cv2
import time
import tensorflow as tf
import sys
#import matplotlib.pyplot as plt

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
        conf = boxes[0]['confidence']
        x, y, w, h = box[0], box[1], box[2], box[3]
        if conf > 0.5:
            text= f"{conf*100:.2f}%"
            cv2.putText(frame, text, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)
    cv2.imshow("webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()