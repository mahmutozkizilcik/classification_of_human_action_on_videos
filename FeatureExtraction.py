#!/usr/bin/env python
# coding: utf-8

# # AIN 313 : Assignment 4 - Feature Extraction

# Mahmut Özkızılcık  - 2220765019

# In this part, we utilized the OpenPose library (specifically the BODY_25 model) to extract 2D body keypoints from raw video frames. This approach avoids processing computationally expensive RGB data by focusing solely on skeletal dynamics.The processing pipeline implemented is as follows:
# 
# * Frame Capture: Each video frame was processed to detect human body joints.
# * Normalization: The extracted $(x, y)$ coordinates were normalized relative to the video resolution ($160\times120$). Specifically, x-coordinates were divided by 160 and y-coordinates by 120 to scale values between $[0, 1]$. This step is crucial for the stability and convergence speed of the subsequent deep learning models.
# * Storage: The resulting time-series data for each video was saved in compressed .npz format.

# Required Imports

# In[1]:


import sys
import os
import cv2
import numpy as np
import glob


# Loading OpenPose Library from computer files.

# In[ ]:


cwd = os.getcwd()
file_path = os.path.abspath(cwd)


# In[3]:


# Openpose files are located c because it said it is sensetive to blanks at folder files
OPENPOSE_BASE = r"C:\openpose"
os.chdir(OPENPOSE_BASE)

# Phthon api
sys.path.append(os.path.join(OPENPOSE_BASE, "bin", "python", "openpose", "Release"))
os.environ['PATH'] = os.environ['PATH'] + ';' + os.path.join(OPENPOSE_BASE, "bin") + ';'

import pyopenpose as op

# Parameters
params = {
    "model_folder": "models/",
    "model_pose": "BODY_25",
    "net_resolution": "-1x128",
    "display": 0,
    "render_pose": 0,
    "num_gpu_start": 0,
    "disable_multi_thread": False
}

print("OpenPose initializing...")
try:
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    print("Openpose is active")
except Exception as e:
    print(f"Error eccured {e}")


# OpenPose initialization

# In[4]:


os.chdir(file_path)
os.listdir()


# In[5]:


dataset_path = r"video_dataset"

actions = ["boxing","handclapping","handwaving","jogging","running","walking"]

for action in actions:

    path = os.path.join(dataset_path, action, "*.avi")
    videos = glob.glob(path)
    print(f"{action}: {len(videos)} videos found")



# In[6]:


def extract_and_save(input_root, output_root):
    if not os.path.exists(output_root): os.makedirs(output_root)

    for action in actions:
        videos = glob.glob(os.path.join(input_root, action, "*.avi"))
        for v_path in videos:
            cap = cv2.VideoCapture(v_path)
            all_frames_data = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # Send datum to OpenPose
                datum = op.Datum()
                datum.cvInputData = frame
                opWrapper.emplaceAndPop(op.VectorDatum([datum]))

                # (BODY-25)
                keypoints = datum.poseKeypoints # (1, 25, 3)
                if keypoints is not None:
                    # Normalization (x/160, y/120)
                    norm_kp = keypoints[0].copy()
                    norm_kp[:, 0] /= 160
                    norm_kp[:, 1] /= 120
                    all_frames_data.append(norm_kp)

            cap.release()

            # save as .npz
            v_name = os.path.basename(v_path).replace('.avi', '.npz')
            np.savez(os.path.join(output_root, f"{action}_{v_name}"),
                     data=np.array(all_frames_data),
                     label=action)
            print(f"Saved in: {action}_{v_name}")

# Running
extract_and_save(dataset_path, "extracted_features")


# Feature Extraction Part completed successfully and the rest of the assigment will be done in models.ipynb file.

# The proccesing interrupted sometimes and I had to restart it from the ones did not proccesed yet. So outputs of the cells does not match with the number of inputs. But It is done for all of the output. Here is the link for the .npz files:
# https://drive.google.com/file/d/1CJSfAfKf-7oFNLP99_0JM3MaIyXUdUBv/view?usp=sharing
# 
