# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 17:09:26 2026

@author: sanhi
"""

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix, view_points
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

DATAROOT = r"C:/Users/sanhi/Downloads/v1.0-mini"  

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=DATAROOT,
    verbose=True
)

print("Scenes:", len(nusc.scene))
print("Samples:", len(nusc.sample))

# Pick the first sample
sample = nusc.sample[20]

# Print all sensor keys for this sample
#print(sample['data'].keys())
#dict_keys(['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'LIDAR_TOP', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'])

#access camera data
cam_key = 'CAM_FRONT'  # or whatever key appeared in your print
cam_token = sample['data'][cam_key]
cam_data = nusc.get('sample_data', cam_token)
print(cam_data.keys())
#dict_keys(['token', 'sample_token', 'ego_pose_token', 'calibrated_sensor_token', 'timestamp', 'fileformat', 'is_key_frame', 'height', 'width', 'filename', 'prev', 'next', 'sensor_modality', 'channel'])
img_path = f"{DATAROOT}/{cam_data['filename']}"  # most recent devkit
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("Camera", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#access RADAR data
radar_key = 'RADAR_FRONT'
radar_token = sample['data'][radar_key]
radar_data = nusc.get('sample_data', radar_token)
#print(radar_data.keys())---same as camera
radar_file = f"{DATAROOT}/{radar_data['filename']}"
radar_pc = RadarPointCloud.from_file(radar_file)
# radar_pc.points is 18xN: multiple features per point
print("Radar points shape:", radar_pc.points.shape)#18x69
x = radar_pc.points[0]  # forward
y = radar_pc.points[1]  # left/right

points_ego = radar_pc.points[:3, :]  # x, y, z in ego frame

# ---Get camera calibration ---
cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
R_cam = Quaternion(cs['rotation']).rotation_matrix      # 3x3 rotation matrix
t_cam = np.array(cs['translation']).reshape(3, 1)       # 3x1 translation vector
K = np.array(cs['camera_intrinsic'])                    # 3x3 camera intrinsics

# --- 3️Transform radar points to camera frame ---
points_cam = R_cam.T @ (points_ego - t_cam)  # 3xN in camera frame

# --- 4️Project 3D points to 2D pixels ---
points_2d = view_points(points_cam, K, normalize=True)  # 3xN

# --- 5️Filter points behind the camera ---
mask = points_cam[2, :] > 0
points_2d = points_2d[:, mask]


#plot cam image with radar points
plt.imshow(img)
plt.scatter(points_2d[0, :], points_2d[1, :], s=5, c='r')
plt.axis('off')
plt.show()
##################################


