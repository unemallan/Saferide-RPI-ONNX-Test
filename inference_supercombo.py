#%%
from PIL import Image
import numpy as np
import onnxruntime
import torch
import cv2
import time

# Load the calibration data
try:
    with np.load('calibration_data.npz') as data:
        mtx = data['mtx']
        dist = data['dist']
except FileNotFoundError:
    # File does not exist
    print("Calibration File does not exist please calibration camera and start this script again")
    exit()

width = 256
height = 512

newcameramatrix, _ = cv2.getOptimalCameraMatrix(
    mtx, dist, (width, height), 1, (width, height)
)


def preprocess_image(image, height, width, channels=3):
    image = image.resize((width, height), Image.LANCZOS)
    dst = cv2.undistort(frame, mtx, dist, None, newcameramatrix)
    image_data = np.asarray(dst).astype(np.float32)
    image_data = image_data.transpose([2, 0, 1]) # transpose to CHW
    mean = np.array([0.079, 0.05, 0]) + 0.406
    std = np.array([0.005, 0, 0.001]) + 0.224
    for channel in range(image_data.shape[0]):
        image_data[channel, :, :] = (image_data[channel, :, :] / 255 - mean[channel]) / std[channel]
    image_data = np.expand_dims(image_data, 0)
    return image_data

#%%

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def run_sample(session, image):
    output = session.run([], {'input':image})[0]
    output = output.flatten()
    output = softmax(output) # this is optional
    print (np.argsort(-output)[:5])
    

#%%
# create main function
if __name__ == "__main__":    
    # Create Inference Session
    session = onnxruntime.InferenceSession("supercombo.onnx")
    while True:
        start_time = time.time() # start time of the loop
        # get image from camera
        cap = cv2.VideoCapture(0)
        cap.set(3,width) # set Width
        cap.set(4,height) # set Height

        # capture image from camera
        ret, frame = cap.read()
        frame = cv2.flip(frame, -1) # Flip camera vertically
        processed_image = preprocess_image(frame, height, width, 3)

        run_sample(session, processed_image)
        print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop
        k = cv2.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit
            break
    
    cap.release()
    cv2.destroyAllWindows()


