import cv2 as cv
from numpy import arctan,pi,mean,round
angles = []
startedaverage = True
a=None
b=None
text = ""
def findangle(a,b):
     if (b[0] - a[0]) == 0: return
     c = abs((arctan((b[1] - a[1]) / (b[0] - a[0]))* (180.0 / pi)))
     angles.append(c)
     if(startedaverage==False): return "angle : "+ str(round(c, decimals = 2)) + " || not calculating average"
     d = findavg()
     return "angle: "+str(round(c, decimals = 2)) + " || angle case: " +findscoliosis(c) + " || average: "+ str(round(d, decimals = 2))+" || average case:"+findscoliosis(d)
def findscoliosis(a):
     if 0 <= a <= 10:
         return "normal"
     if 10 <= a <= 25:
         return "significant intermediate scoliosis"
     if 25 <= a :
         return "severe scoliosis"
     
def findavg():
     return mean(angles)

# Create point matrix get coordinates of mouse click on image

counter = 0
def mousePoints(event,x,y,flags,params):
    global counter
    global a
    global b
    global text
    # Left button mouse click event opencv
    if event == cv.EVENT_LBUTTONDOWN:
        counter = counter + 1
        if(counter==1):
            a=(x,y)
        if(counter==2):
            counter=0
            b=(x,y)
            text=findangle(a,b)
 
# Read image

cap = cv.VideoCapture(0)
while True:
    
    hasFrame, img = cap.read()
    if not hasFrame:
        cv.waitKey()
        break
    # Showing original image
    
    cv.rectangle(img, (0,0), (img.shape[1],50), (0,0,0), -1)
    cv.line(img,a, b, (0, 255, 0), 3)
    cv.ellipse(img, a, (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
    cv.ellipse(img, b, (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
    cv.putText(img, str(text), (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    # Refreshing window all time
    btn = cv.waitKey(1)
    if(btn & 0xFF==ord('q')): break
    if(btn & 0xFF==ord(' ')):
        cv.setMouseCallback("Original Image ", mousePoints)
        cv.waitKey(0)
    cv.imshow("Original Image ", img)
cap.release()
cv.destroyAllWindows()
