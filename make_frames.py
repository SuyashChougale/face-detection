import cv2
import os

cap = cv2.VideoCapture('Video.mp4')
framecount = 0
while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        break
    print(ret)
    framepath = os.path.join(r'Frames\\',r'frame_{}.jpg'.format(framecount))
    print(framepath)
    framecount+=1
    cv2.imwrite(framepath,frame)
cap.release()