import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset=20
imgSize = 300

#when we click save button it saves the image in a folder defined
#COLLECTING IMAGES
folder = "data/C"
counter = 0



while True: 
    success,img = cap.read(0)
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
        
        
        else:
            k = imgSize/w
            hcal = math.ceil(h*k)
            imgResize = cv2.resize(imgCrop,(imgSize,hcal))
            imgResizeShape = imgResize.shape 
            hGap = math.ceil((imgSize - hcal)/2) # gap to push forward to centre the image
            imgWhite[ hGap:hcal + hGap, : ] = imgResize # making width of hand image always 300 and altering the height accordingly
            
            
        #cv2.imshow("ImageCrop",imgCrop)
        cv2.imshow("ImageWhite",imgWhite)
        
        
    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/image_{time.time()}.jpg',imgWhite)
        print(counter)