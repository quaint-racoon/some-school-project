# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 17:21:24 2023

@author: USER
"""
# To use Inference Engine backend, specify location of plugins:
# export LD_LIBRARY_PATH=/opt/intel/deeplearning_deploymenttoolkit/deployment_tools/external/mklml_lnx/lib:$LD_LIBRARY_PATH
import cv2 as cv
from numpy import arctan,pi,mean,round
import argparse
angles = []
startedaverage = False
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

def nearestwhiteedge(a,b,edges):
    
    left = edges[(a[1]-50):a[1], (a[0]-50):a[0]]
    right = edges[(b[1]-50):b[1], b[0]:(b[0]+50)]
    
    for i in range(49, -1, -1):
        if((left is None) and (right is None)): break
        if(left is not None): 
            if(left[i,i]==255): points[idTo]=(a[0]+(i-49),a[1]+(i-49)) ; left = None
           
        if(right is not None): 
            if(right[i,(49-i)]==255): points[idFrom]=(b[0]+(49-i),b[1]+(i-49));  right = None

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

args = parser.parse_args()

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["LShoulder", "RShoulder"]]

inWidth = args.width
inHeight = args.height

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

cap = cv.VideoCapture(args.input if args.input else 0)

while True:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break
    edges = cv.Canny(frame,100,200) #canney edge detecton
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > args.thr else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        cv.rectangle(frame, (0,0), (frame.shape[1],50), (0,0,0), -1)
        
        if points[idFrom] and points[idTo]:
            nearestwhiteedge(points[idTo],points[idFrom], edges)
            # print(findangle(points[idFrom], points[idTo]))
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.putText(frame, str(findangle(points[idFrom], points[idTo])), (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    btn = cv.waitKey(1)
    if(btn & 0xFF==ord('q')): break
    if(btn & 0xFF==ord(' ')): startedaverage = True
    cv.imshow('Edges in the image', edges)
    cv.imshow('testing frame', frame)
cap.release()
cv.destroyAllWindows()
