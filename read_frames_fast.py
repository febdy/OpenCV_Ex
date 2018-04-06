from thread_basic import FileVideoStream
import numpy as np
import cv2
import time

conda_path = 'D:/javaForever/util/opencv/sources/data/haarcascades'
upper_cascade = cv2.CascadeClassifier(conda_path + '/haarcascade_upperbody.xml')

path = 'C:/Users/BIT-USER/Desktop/python_workplace/5sec/ace.avi'
# cap = cv2.VideoCapture('C:/Users/BIT-USER/Desktop/python_workplace/5sec/ace.avi')
scaling_factor = 0.25

fvs = FileVideoStream(path).start()

time.sleep(1)

e1 = cv2.getTickCount()

while fvs.has_more():

    frame = fvs.read()

    if frame is None:
        break

    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # frame = rotate(frame, 90)

    upper_rects = upper_cascade.detectMultiScale(gray, 1.1, 1)

    for (x, y, w, h) in upper_rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Face Detector', frame)

    if cv2.waitKey(1) == 27:
        fvs.stop()


e2 = cv2.getTickCount()

print((e2 - e1) / cv2.getTickFrequency())

fvs.stop()
cv2.destroyAllWindows()
