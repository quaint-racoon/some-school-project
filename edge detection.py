# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 18:44:03 2023

@author: USER
"""

import cv2
 
def proccess(img):
# Display original image
    cv2.imshow('Original', img)
 
# Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
# Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
# Display Canny Edge Detection Image
    cv2.imshow('Canny Edge Detection', edges)
    
capture = cv2.VideoCapture(0)

while True:
    isTrue,frame = capture.read()
    proccess(frame)    
    if cv2.waitKey(20) & 0xFF==ord('q'):
        break
capture.release()
cv2.destroyAllWindows()