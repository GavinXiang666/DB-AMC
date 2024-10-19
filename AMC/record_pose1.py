from camera_setting import *
import os,cv2
cap = cv2.VideoCapture("AMC/output/video/New Camera 0.avi")
ret,frame = cap.read()
cv2.imshow("right",frame)
cv2.waitKey(0)
cv2.imwrite("AMC/output/image/right.png",frame)