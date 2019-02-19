import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #Path of dir
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_id = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            #print(label, path) # name of files and your root 
            if not label in label_id: #give an id for the each label
                label_id[label] = current_id
                current_id += 1
            id_ = label_id[label]
            #print(label_id)

            pil_image = Image.open(path).convert("L") # grayscale
            size= (550, 550)
            final_imagem = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_imagem, "uint8") # take each image and tranform it in a type of numpy array
            #print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            
            for(x,y,h,w) in faces:
                roi = image_array[y:y+h, x:x+w] # region of interest
                x_train.append(roi) # A numpy array with values
                y_labels.append(id_) # An array of id's root

with open('labels.pickle', 'wb') as f:
   pickle.dump(label_id, f)

recognizer.train(x_train, np.array(y_labels)) # convert our labels into dump arrays 
recognizer.save("trainner.yml")