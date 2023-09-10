# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 15:56:56 2023

@author: USER
"""

import cv2 as cv
import tensorflow as tf
keras = tf.keras 

# reading images

img = cv.imread('imgs/cat.jpg')

cv.imshow('cat', img)

cv.waitKey(0)

#resizing frames

def rescaleFrame(frame,scale = 0.75):
    # this method works for: images, videos, live videos
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# changing resolution

def changeRes(width,height):
    # this method works for live videos only
    capture.set(3,width)
    capture.set(4,height)

# reading videos

capture = cv.VideoCapture('vids/cat.mp4')

while True:
    isTrue,frame = capture.read()
    
    frame_resized = rescaleFrame(frame)
    
    cv.imshow('Video',frame)
    cv.imshow('Video resized',frame_resized)
    
    if cv.waitKey(20) & 0xFF==ord('q'):
        break
capture.release()
cv.destroyAllWindows()
