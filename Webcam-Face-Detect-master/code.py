#face recognition
#step 1
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





#step2
# Face image dataset location
import cv2
import os
import numpy as np
path = 'C:\\Users\kamal\PycharmProjects\Webcam-Face-Detect-master\photos'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
# IMages and lbel data functions
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        from PIL import Image
        PIL_img = Image.open(imagePath).convert('L') # To change it to gray scale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
print ("\n Training faces yo please wait yo........")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
# store the model to train.yml
recognizer.write('C:\\Users\kamal\PycharmProjects\Webcam-Face-Detect-master\\train/train.yml') # recognizer.save()
# Show the no of faces trained and exit
print("\n  {0} Trained faces.Program Terminating".format(len(np.unique(ids))))


#step 3
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


#Face detection:emotion and gender:
import cv2

from cv2 import WINDOW_NORMAL
from face_detection import faces_find

#initializing escape key
ESC = 27

#initializing webcam
def begin_webcame(emotion_model, gender_model, win_size, win_name='Realtime', updatetime=50):
    cv2.namedWindow(win_name, WINDOW_NORMAL)
    if win_size:
        width, height = win_size
        cv2.resizeWindow(win_name, width, height)

    realtime_feed = cv2.VideoCapture(0)
    realtime_feed.set(3, width)
    realtime_feed.set(4, height)
    read_val, frame = realtime_feed.read()

    wait_time = 0 #delay bewteen the frames is zero
    init = True
    while read_val:
        read_val, frame = realtime_feed.read()
        for normal_face, (x, y, w, h) in faces_find(frame):  #finding the faces in faces_find function initiated in face_detection.py
          if init or wait_time == 0:
            init = False
            prediction_emotion = emotion_model.predict(normal_face)
            prediction_gender = gender_model.predict(normal_face)
          if (prediction_gender[0] == 0):
              cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2) #blue color for female
          else:
              cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2) #red color for male
          cv2.putText(frame, emotions[prediction_emotion[0]], (x, y - 10), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.5, (255, 0, 0), 2)
          cv2.putText(frame, gender[prediction_gender[0]], (10,20), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1,
                      (255, 0, 0), 2)

        wait_time += 1
        wait_time %= 20
        cv2.imshow(win_name, frame)
        key = cv2.waitKey(updatetime)
        if key == ESC:
            break

    cv2.destroyWindow(win_name)



if __name__ == '__main__':
    emotions = ["afraid", "angry", "disgusted", "happy", "neutral", "sad", "surprised"]
    gender   =  ["male","female"]
    # Loading the trained models of emotion
    fisher_face_emo = cv2.face.FisherFaceRecognizer_create()
    fisher_face_emo.read('C:\\Users\kamal\PycharmProjects\Webcam-Face-Detect-master\emotion_classifier_model.xml')

#loading the trained models of gender
    fisher_face_gen = cv2.face.FisherFaceRecognizer_create()
    fisher_face_gen.read('C:\\Users\kamal\PycharmProjects\Webcam-Face-Detect-master\gender_classifier_model.xml')

    # starting the model to predict
    choice = input("starting your webcam?(y/n) ")
    if (choice == 'y'):
        window_name = "Facifier Webcam (press ESC to exit)"
        begin_webcame(fisher_face_emo, fisher_face_gen, win_size=(1280, 720), win_name=window_name, updatetime=15)

    else:
        print("Invalid input, exiting program.")
#emotion training
import cv2
import glob
import numpy as np
import random

fisherface = cv2.face.FisherFaceRecognizer_create()

def getfile(emotion, training_size):
    #loading the dataset for training
    file = glob.glob("C:\\Users\kamal\PycharmProjects\Webcam-Face-Detect-master\data\\raw_emotion\\{0}\\*" .format(emotion))
    random.shuffle(file) #shuffling the data
    train = file[:int(len(file) * training_size)]
    predict = file[-int(len(file) * (1 - training_size)):]
    return train, predict

def make_sets(): #creating lists
    train_data = []
    train_labels = []
    predict_data = []
    predict_labels = []
    for emotion in emotions:
        training_set, prediction_set = getfile(emotion, 0.8) #getting first 80% of files

        for object in training_set:
            img = cv2.imread(object, 0) #reading the image
            face = cv2.resize(img, (350, 350)) #resizng all the images to same sizes
            train_data.append(face)
            train_labels.append(emotions.index(emotion))

        for object in prediction_set:
            object = cv2.imread(object, 0) #reading the image
            face1 = cv2.resize(object, (350, 350)) #resizing the images
            predict_data.append(face1)
            predict_labels.append(emotions.index(emotion))

    return train_data, train_labels, predict_data, predict_labels


def run_recognizer():
    data_training, labels_training, data_prediction, labels_prediction = make_sets()

    print("size of the training set is", len(labels_training), "images")
    fisherface.train(data_training, np.asarray(labels_training)) #training the data usimg the fishferface.train function

    print("size of the prediction set is:", len(labels_prediction), "images")
    positive = 0
    for idx, image in enumerate(data_prediction):
        if (fisherface.predict(image)[0] == labels_prediction[idx]):
            positive += 1

    percent = (positive * 100) / len(labels_prediction)

    return positive, percent

if __name__ == '__main__': #types of emotions
    emotions = ["afraid","angry","disgusted","happy","neutral","sad","surprised"]

    positive, percent = run_recognizer()
    print("handled ", positive, " data correctly")
    print("obtained", percent, " accuracy")

#Writing  the trained dataset
    fisherface.write('C:\\Users\kamal\PycharmProjects\Webcam-Face-Detect-master\emotion_classifier_model.xml')

#train gender
import cv2
import glob
import numpy as np
import random
#initilaizing fisherface recognizer

fisherface = cv2.face.FisherFaceRecognizer_create()

def getfiles(gender, training_size):
    #loading the dataset
    file = glob.glob("E:\\cropped_faces\\{0}\\*" .format(gender))
    random.shuffle(file)
    train = file[:int(len(file) * training_size)]
    predict = file[-int(len(file) * (1 - training_size)):]
    return train, predict

def make_sets(): #creating lists
    train_data = []
    train_labels = []
    predict_data = []
    predict_label = []
    for gender in genders:
        training_set, prediction_set = getfiles(gender, 0.8) #getting first 805 of files

        for object in training_set:
            img = cv2.imread(object, 0)#reading the object image
            face2 = cv2.resize(img, (350, 350)) #resizing the image

            train_data.append(face2)
            train_labels.append(genders.index(gender))

        for object in prediction_set:
            object = cv2.imread(object, 0) #reading the object
            face2 = cv2.resize(object, (350, 350)) #resizing the object

            predict_data.append(face2)
            predict_label.append(genders.index(gender))

    return train_data, train_labels, predict_data, predict_label


def run_recognizer():
    data_training, labels_training, data_prediction, labels_predictions = make_sets()

    print("size of the training set is", len(labels_training), "images")

#training the daraset
    fisherface.train(data_training, np.asarray(labels_training))


    positive = 0
    for id, img in enumerate(data_prediction):
        if (fisherface.predict(img)[0] == labels_predictions[id]):
            positive += 1

    percent = (positive * 100) / len(data_prediction)

    return positive, percent

if __name__ == '__main__':
    genders = ["female", "male"]

    positive, percent = run_recognizer()
    print("Processed ", positive, " data correctly")
    print("Got ", percent, " accuracy")

#writing the training data
    fisherface.write('D:\\models\gender_classifier_model.xml')

#extra

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