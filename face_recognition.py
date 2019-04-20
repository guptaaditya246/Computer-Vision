# -*- coding: utf-8 -*-

#importing Libraries
import cv2

#loading cascades
face_cascade = cv2.CascadeClassifier('./detector_architectures/haarcascade_frontalface_default.xml')
eye_cascade =  cv2.CascadeClassifier('./detector_architectures/haarcascade_eye.xml')

def detect_faces(gray_image, frame):
    #for detecting faces in frame
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.4, minNeighbors = 5)
    #detecting faces here
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (244, 0, 14), thickness = 2)
        #detecting eye inside the face
        gray_frame = gray_image[y:y+h, x:x+w]
        color_frame = frame[y:y+h, x:x+w]
        # we will detect eye in gray (haar cascade work on gray image) and draw on color
        eyes = eye_cascade.detectMultiScale(gray_frame, scaleFactor = 1.1, minNeighbors = 3)
        #extracting coordinates for rectangle across eyes
        for (x, y, w, h) in eyes:
            cv2.rectangle(color_frame, (x, y), (x+w, y+h), (173, 244, 66), thickness = 2)
            
    return frame


# o for PC cam and 1 for external webcam
video = cv2.VideoCapture(0)
while True:
    _, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    showcase = detect_faces(gray, frame)
    cv2.imshow('Video', showcase)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()
    


        
    
