# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 14:13:49 2020

@author: Bibek77
"""

import numpy as np
import cv2 as cv
import os

def faceDetection(test_img):
     gray_img= cv.cvtColor(test_img,cv.COLOR_BGR2GRAY)
     face_haar=cv.CascadeClassifier( r'D:\micro-expresion\Aplikasi Baru\Aplikasi2.2\classifiers\haarcascade_frontalface_alt.xml')
     faces=face_haar.detectMultiScale(gray_img, scaleFactor=1.3,minNeighbors=3)

     return faces,gray_img

#labels for

def labels_for_training_data(directory):
    faces=[]
    faceID=[]

    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("skipping system File")
                continue

            id=os.path.basename(path)
            img_path=os.path.join(path,filename)
            print("img_path",img_path)
            print("id",id)

            test_img = cv.imread(img_path)
            if test_img is None:
                print("Not Loaded Properly")
                continue


            faces_rect,gray_img=faceDetection(test_img)

            if len(faces_rect)!=1:
                continue
            (x,y,w,h)=faces_rect[0]
            roi_gray=gray_img[y:y+w,x:x+h]
            faces.append(roi_gray)
            faceID.append(int(id))
    return faces,faceID


  #Here training Classifier is called
def train_classifier(faces,faceID):
    face_recognizer=cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,np.array(faceID))
    return face_recognizer


#Drawing a Rectangle on the Face Function
def draw_rect(test_img,face):
    (x,y,w,h)=face
    cv.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),thickness=3)

#Putting text on images
def put_text(test_img,text,x,y):
    cv.putText(test_img,text,(x,y),cv.FONT_HERSHEY_DUPLEX,3,(255,0,0),6)
