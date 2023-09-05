# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 18:41:24 2023

@author: USER
"""
bg = None
def proccess(img):
# make image grey
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Blur the image for better edge detection
    img_blur = cv.GaussianBlur(img_gray, (7,7), 0) 
    return img_blur
def showsubtraction(img1,img2):
    
    img1 = proccess(img1)
    img2 = proccess(img2)
    subtracted = cv.subtract(img2, img1)
# Canny Edge Detection
    edges = cv.Canny(image=subtracted, threshold1=50, threshold2=50) # Canny Edge Detection
    
# Display Canny Edge Detection Image
    # find Harris corners
    gray = np.float32(edges)
    dst = cv.cornerHarris(gray,2,3,0.1)
    dst = cv.dilate(dst,None)
    ret, dst = cv.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

    # Now draw them
    res = np.hstack((centroids,corners))
    res = np.int0(res)
    
    edges = cv.cvtColor(edges,cv.COLOR_GRAY2RGB)

    edges[res[:,1],res[:,0]]=[0,0,255]
    edges[res[:,3],res[:,2]] = [0,0,255]
    cv.imshow("corners",edges)

    
import cv2 as cv
import numpy as np
capture = cv.VideoCapture(0)

while True:
    isTrue,frame = capture.read()
    
    
    cv.imshow('Video',frame)
    
    t = cv.waitKey(20)
    if t & 0xFF==ord('q'): 
        capture.release()
        break
    
    if t & 0xFF==ord(' '):
        if bg is None:
            bg = frame
            print("img1 is set")
    if bg is not None:
        showsubtraction(frame, bg)

cv.destroyAllWindows() 

