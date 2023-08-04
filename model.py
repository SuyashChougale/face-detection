import cv2
import numpy as np
def detect_human(index):
    frame_path = r'Frames\frame_{}.jpg'.format(index)
    frame = cv2.imread(frame_path)
    height ,width ,_ = frame.shape

    model = cv2.dnn.readNetFromDarknet('yolov3.cfg','yolov3.weights')
    blob = cv2.dnn.blobFromImage(frame,1/255,(416, 416),swapRB=True,crop=False)
    model.setInput(blob)
    outputs = model.forward(model.getUnconnectedOutLayersNames())
    
    detected_persons = []
   
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            if class_id ==0:
                confidence = scores[class_id]
                if confidence>0.5:
                    box = detection[0:4]*np.array([width,height,width,height])
                    (center_x,center_y,boxwidth,boxheight) = box.astype("int")
                    x =int( center_x-boxwidth/2)
                    y =int( center_y-boxheight/2)
                    person_info = (x,y,x+boxwidth,y+boxheight)
                    detected_persons.append(person_info)
    return detected_persons

def detect_face(index):
    frame_path = r'Frames\frame_{}.jpg'.format(index)
    frame = cv2.imread(frame_path)
    gray_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = model.detectMultiScale(gray_image,1.1,5,minSize=(30,30))
    return faces
import make_frames
humans = []
faces = []
for i in range(make_frames.framecount):
    print(i)
    humans.extend(detect_human(i))
    faces.extend(detect_face(i))
    

                    
                    

                     


