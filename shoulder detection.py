# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 20:48:54 2023

@author: USER
"""

# Importing OpenCV package
import cv2

# Loading the required haar-cascade xml classifier file
# haar_cascade = cv2.CascadeClassifier('xml/haarcascade_upperbody.xml')
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")

# Reading the image
capture = cv2.VideoCapture(0)

while True:
    isTrue,img = capture.read()
    # Converting image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

    # Applying the face detection method on the grayscale image
    faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9)
    
    # Iterating through rectangles of detected faces
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Detected faces', img)
    t = cv2.waitKey(20)
    if t & 0xFF==ord('q'):
        capture.release()
        break
cv2.destroyAllWindows() 
