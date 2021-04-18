# MIT License
# Copyright (c) 2021 Loyio

import cv2
from api import mkdir
import sys

#url="rtsp://admin:admin@192.168.43.9:8554/live"

if len(sys.argv) < 2:
    print("Please specify a name for your photos!!!")
    exit(0)
else:
    person_name = str(sys.argv[1])
    # mkdir("Images/"+person_name)
    mkdir("Images_test/"+person_name)

cap = cv2.VideoCapture(0)
i = 0
while (1):
    ret, frame = cap.read()
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('s'):
        cv2.imwrite('./Images_test/'+person_name + '/'+ str(i) + '.jpg', frame)
        # cv2.imwrite('./Images/'+person_name + '/'+ str(i) + '.jpg', frame)
        i += 1
        print(str(i))
    cv2.imshow("capture", frame)
cap.release()
cv2.destroyAllWindows()