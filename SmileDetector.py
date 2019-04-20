import cv2

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('./haarcascade_smile.xml')

def detect_smile(gray_image, frame):
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor = 1.3, minNeighbors = 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 244, 66), thickness = 2)
            
            #Since we need to detect the eyes on the detected face part so cropping that face part for color and gray image
            face_image_gray = gray_image[y:y+h, x:x+w]
            face_image_color = frame[y:y+h, x:x+w]
        
            eyes = eye_cascade.detectMultiScale(face_image_gray, scaleFactor = 1.1, minNeighbors = 22)
            for (x, y, w, h) in eyes:
                cv2.rectangle(face_image_color, (x, y), (x+w, y+h), (65, 80, 244), thickness = 2)

            #Now after eyes we need to find smile, so we will use smile cascade
            smile = smile_cascade.detectMultiScale(face_image_gray, scaleFactor = 1.9, minNeighbors = 15)
            for (x, y, w, h) in smile:
                cv2.rectangle(face_image_color, (x, y), (x+w, y+h), (0, 255, 123), thickness = 2)
        return frame

#capturing video throught pc web cam
video = cv2.VideoCapture(0)

while True:
    _, frame = video.read()
    gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    showcase = detect_smile(gray_image, frame)
    cv2.imshow('Video',showcase)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

        
