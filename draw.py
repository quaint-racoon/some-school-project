# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:26:35 2023

@author: USER
"""

import cv2 as cv


def showText(img, width, height, text, start=[100,100]):
    cv.rectangle(img, (start[0]-(width/2),start[1]+(height/2)), (start[0]+(width/2),start[0]-(height/2)), (0,0,0), thickness=-1)
    cv.putText(img, text, (100,100), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0), 2)
    

img = cv.imread("imgs/cat.jpg")
showText(img,110,40,"test")
cv.imshow("img",img)
cv.waitKey(0)