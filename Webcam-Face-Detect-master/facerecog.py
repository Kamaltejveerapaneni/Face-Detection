import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:\\Users\kamal\PycharmProjects\Webcam-Face-Detect-master\\train/train.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
# id counter initiated
id = 0
# id s for names
names_people = ['None', 'Kamal', 'yu', 'tejid', 'abc', 'W']
# NOw start intialization and then video capturing
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height
# Minimum window size to recog a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)
while True:
    ret, image = cam.read()
    image = cv2.flip(image, 1)  # horizontal
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray_scale,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence_level = recognizer.predict(gray_scale[y:y + h, x:x + w])
        # Checking if confidence level is less them 100
        if (confidence_level < 100):
            id = names_people[id]
            confidence_level = "  {0}%".format(round(100 - confidence_level))
        else:
            id = "unknown"
            confidence_level = "  {0}%".format(round(100 - confidence_level))

        cv2.putText(image, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(image, str(confidence_level), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', image)
    k = cv2.waitKey(10) & 0xff  # esc to exit
    if k == 27:
        break
# cleanung
print("\n Program ending and cleaning")
cam.release()
cv2.destroyAllWindows()