import cv2
import numpy as np

# Define the chessboard dimensions
CHECKERBOARD = (8,8)

# Termination criteria for calibration
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Start capturing video from the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', frame)
        cv2.waitKey(1000)  # Wait for 1 second

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Calibrate the camera once enough images are collected
if len(objpoints) > 0:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    np.savez('calibration_data.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print("Camera matrix:")
    print(mtx)
    print("Distortion coefficients:")
    print(dist)

cap.release()
cv2.destroyAllWindows()