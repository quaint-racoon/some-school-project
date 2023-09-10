import cv2 as cv
import numpy as np

img = np.zeros((500,500,3), dtype=np.uint8)
dx = img.shape[0] 
dy = img.shape[1]
# img[0][0][0]=1
cv.imshow("img",img)

for i in range(dx):
    print(i,i)
    for j in range(i-1,-1,-1):
        print(j,i)
        print(i,j)
cv.waitKey(0)
