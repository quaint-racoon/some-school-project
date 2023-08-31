# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 18:41:24 2023

@author: USER
"""
def proccess(img):
# make image grey
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Blur the image for better edge detection
    img_blur = cv.GaussianBlur(img_gray, (7,7), 0) 
    return img_blur
def showsubtraction(img1,img2):
    
    img1 = proccess(img1)
    img2 = proccess(img2)
    cv.imshow("img1",img1)
    cv.imshow("img2",img2)
    subtracted = cv.subtract(img2, img1)
    cv.imshow("subtracted",subtracted)
    
# Canny Edge Detection
    edges = cv.Canny(image=subtracted, threshold1=50, threshold2=50) # Canny Edge Detection
    
# Display Canny Edge Detection Image
    cv.imshow('Canny Edge Detection', edges)
    cv.waitKey(0)

import cv2 as cv
img1 = None
img2 = None
  
capture = cv.VideoCapture(0)

while True:
    isTrue,frame = capture.read()
    
    
    
    cv.imshow('Video',frame)
    
    t = cv.waitKey(20)
    if t & 0xFF==ord('q'): 
        capture.release()
        break
    
    if t & 0xFF==ord(' '):
        if img2 is None:
            if img1 is not None:
                img2 = frame
                print("img2 is set")
                capture.release()
                cv.destroyAllWindows()
                showsubtraction(img1, img2)
        if img1 is None:
            img1 = frame
            print("img1 is set")
        if img1 is None :
            img1 = frame
            print("img1 is set")
        

cv.destroyAllWindows() 


