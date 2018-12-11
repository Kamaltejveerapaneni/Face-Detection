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

