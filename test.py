import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
from cvzone.ClassificationModule import Classifier
#import tensorflow

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/Keras_model.h5","Model/labels.txt")



offset=20
imgSize = 300

#when we click save button it saves the image in a folder defined
#COLLECTING IMAGES
folder = "C"
counter = 0
labels =["A","B","C"]


while True: 
    success,img = cap.read(0)
    imgOutput = img.copy()
    hands,img = detector.findHands(img)
    if hands:#cropping the image of hand
        hand=hands[0]
        x,y,w,h = hand['bbox']
        
        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255  #white image of 300*300
        
        imgCrop = img[y - offset: y + h + offset , x - offset:x + w +offset]  #cropping the hand
        
        #imgCropShape = imgCrop.shape 
        #imgWhite[0:imgCropShape[0],0:imgCropShape[1]] = imgCrop #overlaying image on top of white image
        
        aspectRatio = h/w
        
        if aspectRatio > 1:
            k = imgSize/h
            wcal = math.ceil(w*k)
            imgResize = cv2.resize(imgCrop,(wcal,imgSize))
            imgResizeShape = imgResize.shape 
            wGap = math.ceil((imgSize - wcal)/2) # gap to push forward to centre the image
            imgWhite[: , wGap:wcal + wGap] = imgResize # making height of hand image always 300 and altering the width accordingly
            prediction,index = classifier.getPrediction(imgWhite,draw = False)
            print(prediction,index)
        
        else:
            k = imgSize/w
            hcal = math.ceil(h*k)
            imgResize = cv2.resize(imgCrop,(imgSize,hcal))
            imgResizeShape = imgResize.shape 
            hGap = math.ceil((imgSize - hcal)/2) # gap to push forward to centre the image
            imgWhite[ hGap:hcal + hGap, : ] = imgResize # making width of hand image always 300 and altering the height accordingly
            prediction,index = classifier.getPrediction(imgWhite,draw=False)
            print(prediction,index)
        
        
        cv2.rectangle(imgOutput,(x-offset,y-offset-50),(x-offset+70,y-offset),(255,0,255),cv2.FILLED)
        cv2.putText(imgOutput,labels[index],(x,y-30),cv2.FONT_HERSHEY_COMPLEX,1.7,(255,255,255),2)   
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)
        
            
        #cv2.imshow("ImageCrop",imgCrop)
        cv2.imshow("ImageWhite",imgWhite)
        
        
    cv2.imshow("Image",imgOutput)
    cv2.waitKey(1)
    