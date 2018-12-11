#extra features like eyes and nose
import cv2


# Cascades Loading
face_cascade = cv2.CascadeClassifier('C:\\Users\kamal\PycharmProjects\Webcam-Face-Detect-master\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\kamal\PycharmProjects\Webcam-Face-Detect-master\haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('C:\\Users\kamal\PycharmProjects\Webcam-Face-Detect-master\haarcascade_smile.xml')
glasses_cascade = cv2.CascadeClassifier('C:\\Users\kamal\PycharmProjects\Webcam-Face-Detect-master\haarcascade_eye_tree_eyeglasses.xml')
nose_cascade = cv2.CascadeClassifier('C:\\Users\kamal\PycharmProjects\Webcam-Face-Detect-master\haarcascade_mcs_nose.xml')

# Defining a function that will do the detections
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        nose = nose_cascade.detectMultiScale(roi_gray, 1.4, 22)
        for (nx, ny, nw, nh) in nose:
            cv2.rectangle(roi_color, (nx, ny), (nx+nw, ny+nh), (255, 127, 0), 2)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
            print("smiled")
    return frame


# Face reco with webcam
video_capture = cv2.VideoCapture(0)
#For primary   Webcam Feed :- 0
#For secondary Webcam Feed :- 1
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # convert the image to grayscale, blur it, and detect edges
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()