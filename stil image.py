# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 18:12:12 2023

@author: USER
"""

import cv2

img = cv2.imread("img.jpg")
cv2.imshow("output image", img)

cv2.waitKey(0)