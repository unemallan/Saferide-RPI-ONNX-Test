import numpy as np
import cv2
import os


# Load the calibration data
try:
    with np.load('calibration_data.npz') as data:
        mtx = data['mtx']
        dist = data['dist']
except FileNotFoundError:
    # File does not exist
    print("Calibration File does not exist please calibration camera and start this script again")
    exit()

width = 1920 # Camera Width
height = 1080 # Camera Height

newcameramatrix, _ = cv2.getOptimalCameraMatrix(
    mtx, dist, (width, height), 1, (width, height)
)

#Set Video Capture Properties
cap = cv2.VideoCapture(0)
cap.set(3,width) # set Width
cap.set(4,height) # set Height
 
while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, -1) # Flip camera vertically
    dst = cv2.undistort(frame, mtx, dist, None, newcameramatrix)
    
    cv2.imshow('frame', dst)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()