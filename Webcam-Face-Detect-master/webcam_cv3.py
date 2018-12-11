
#By team fast and Curious
#Libraries imported
import cv2
import os
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
face_detector = cv2.CascadeClassifier('C:\\Users\kamal\PycharmProjects\Webcam-Face-Detect-master\haarcascade_frontalface_default.xml')
# Entert one numeric id value for every face
face_id = input('\n Please Enter User Id ')
print("\n CApturing photo Look at Webcam")
# sampling individual person face count initializing
count = 0
while(True):
    ret, image = cam.read()
    image = cv2.flip(image, 1) # horizontal
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_scale, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        # Save the captured image into the photos folder
        cv2.imwrite("C:\\Users\kamal\PycharmProjects\Webcam-Face-Detect-master\photos/user." + str(face_id) + '.' + str(count) + ".jpg", gray_scale[y:y + h, x:x + w])
        cv2.imshow('image', image)
    k = cv2.waitKey(100) & 0xff # To exit press esc
    if k == 27:
        break
    elif count >= 35: # Here we consider 35 samples and stop the video
         break
# Do a bit of cleanup
print("\n Exit Program and Cleaning Up takes place")
cam.release()
cv2.destroyAllWindows()