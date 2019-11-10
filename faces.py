import numpy as np 
import cv2

#Lest use the camera
cap = cv2.VideoCapture(0)

#import the classifier data from cv2
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/haarcascade_smile.xml')

#start getting frames from the camera
while(True):
    #capture the frame
    ret, frame = cap.read()
   
    #convert it to a grayscale frame to work with it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x,y,w,h) in faces:
        print(x,y,w,h)
        #region of interest = roi (face detected)
        #save the roi to a file
        roi_gray = gray[y:y+h, x:x+w]
        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)

    #show the frame captured
    cv2.imshow('frame', frame)

    #stop the frame capture with q pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
