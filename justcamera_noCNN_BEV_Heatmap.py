# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 11:26:47 2026

@author: sanhi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 09:38:40 2026

@author: sanhi
"""
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from nuscenes.utils.data_classes import Box
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from nuscenes.utils.data_classes import LidarPointCloud
import os


#Get Sensor data
# Initialize nuScenes
DATAROOT = r"C:/Users/sanhi/Downloads/v1.0-mini"
nusc = NuScenes(version="v1.0-mini", dataroot=DATAROOT, verbose=True)
def process_camera_sample(nusc, sample_idx, grid_size=(128, 128), bev_range=200.0, sigma=3):
    """
    Process camera sample - returns (camera_bev, heatmap)
    """
    sample = nusc.sample[sample_idx]
    
    # Camera data
    cam_token = sample['data']['CAM_FRONT']
    cam_data = nusc.get('sample_data', cam_token)
    cam_calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    cam_intrinsic = np.array(cam_calib['camera_intrinsic'])
    
    # Transform: ego → camera
    ego_to_cam = transform_matrix(
        cam_calib['translation'], 
        Quaternion(cam_calib['rotation']), 
        inverse=True
    )
    
    # Load LIDAR
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    lidar_file = f"{nusc.dataroot}/{lidar_data['filename']}"
    lidar_pc = LidarPointCloud.from_file(lidar_file)
    
    # Transform lidar to ego
    cs = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    sensor_to_ego = transform_matrix(cs['translation'], Quaternion(cs['rotation']), inverse=False)
    lidar_pc.transform(sensor_to_ego)
    
    # Transform LiDAR from ego to camera
    lidar_points_ego = lidar_pc.points[:3, :]
    lidar_points_cam = ego_to_cam[:3, :3] @ lidar_points_ego + ego_to_cam[:3, 3:4]
    
    # Project to image
    depths = lidar_points_cam[2, :]
    points_2d = cam_intrinsic @ lidar_points_cam
    points_2d = points_2d[:2, :] / points_2d[2:3, :]
    
    # Filter valid points
    img_h, img_w = 900, 1600  # Camera image size
    mask = (depths > 0) & \
           (points_2d[0, :] >= 0) & (points_2d[0, :] < img_w) & \
           (points_2d[1, :] >= 0) & (points_2d[1, :] < img_h)
    
    valid_points = points_2d[:, mask]
    valid_depths = depths[mask]
    
    # Unproject to 3D camera frame
    u = valid_points[0, :]
    v = valid_points[1, :]
    Z = valid_depths
    
    fx = cam_intrinsic[0, 0]
    fy = cam_intrinsic[1, 1]
    cx = cam_intrinsic[0, 2]
    cy = cam_intrinsic[1, 2]
    
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    camera_3d_points = np.vstack([X, Y, Z])
    
    # Transform camera → ego
    cam_to_ego = transform_matrix(
        cam_calib['translation'], 
        Quaternion(cam_calib['rotation']), 
        inverse=False
    )
    camera_points_ego = cam_to_ego[:3, :3] @ camera_3d_points + cam_to_ego[:3, 3:4]
    
    # Create BEV map
    x = camera_points_ego[0, :]
    y = camera_points_ego[1, :]
    
    bev_map = np.zeros(grid_size, dtype=np.float32)
    grid_x = ((x + bev_range/2) / bev_range * grid_size[0]).astype(int)
    grid_y = ((y + bev_range/2) / bev_range * grid_size[1]).astype(int)
    
    for i in range(len(grid_x)):
        gx, gy = grid_x[i], grid_y[i]
        if 0 <= gx < grid_size[0] and 0 <= gy < grid_size[1]:
            bev_map[gy, gx] = 1.0
    
    # --- HEATMAP (EXACT SAME AS RADAR) ---
    ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
    heatmap = np.zeros(grid_size, dtype=np.float32)
    
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)
        
        bx, by = box.center[0], box.center[1]#this is the center of the boxes
        
        if abs(bx) >= bev_range/2 or abs(by) >= bev_range/2 or bx < 0:
            continue
        
        px = int((bx + bev_range/2) / bev_range * grid_size[0])
        py = int((by + bev_range/2) / bev_range * grid_size[1])
        
        for i in range(max(0, px-10), min(grid_size[0], px+10)):
            for j in range(max(0, py-10), min(grid_size[1], py+10)):
                dist = (i - px)**2 + (j - py)**2
                heatmap[j, i] = max(heatmap[j, i], np.exp(-dist / (2 * sigma**2)))
    
    return bev_map, heatmap

#VISUALIZE
sample_idx = 1
bev, heat = process_camera_sample(nusc, sample_idx)
print(f"BEV shape: {bev.shape}")
print(f"Heatmap shape: {heat.shape}")

# Get sample and ego_pose for visualization
sample = nusc.sample[sample_idx]
cam_token = sample['data']['CAM_FRONT']
cam_data = nusc.get('sample_data', cam_token)
ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])  # ADD THIS LINE

bev_range = 200.0
grid_size = (128, 128)

bev_boxes = []
for ann_token in sample['anns']:
    ann = nusc.get('sample_annotation', ann_token)
    box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']), name=ann['category_name'])
    box.translate(-np.array(ego_pose['translation']))
    box.rotate(Quaternion(ego_pose['rotation']).inverse) 
    
    x, y, z = box.center
    w, l, h = box.wlh
    
    bev_boxes.append({
        'class': box.name,
        'x': x,
        'y': y,
        'w': w,
        'l': l
    })

# BEV+ Heatmap+Annotation boxes visualization
plt.figure(figsize=(14, 6))
# Plot 1: BEV with boxes
plt.subplot(1, 2, 1)
plt.imshow(bev.T, cmap='hot', origin='lower', extent=[-bev_range/2, bev_range/2, -bev_range/2, bev_range/2])
for box in bev_boxes:
    if abs(box['x']) < bev_range/2 and abs(box['y']) < bev_range/2:
        rect = plt.Rectangle(
            (box['y'] - box['w']/2, box['x'] - box['l']/2),
            box['w'], box['l'],
            fill=False, edgecolor='cyan', linewidth=2
        )
        plt.gca().add_patch(rect)
plt.xlabel("Y (lateral)")
plt.ylabel("X (forward)")
plt.title("BEV + Boxes")

# Plot 2: Heatmap with boxes
plt.subplot(1, 2, 2)
plt.imshow(heat.T, cmap='hot', origin='lower', extent=[-bev_range/2, bev_range/2, -bev_range/2, bev_range/2])
for box in bev_boxes:
    if abs(box['x']) < bev_range/2 and abs(box['y']) < bev_range/2:
        rect = plt.Rectangle(
            (box['y'] - box['w']/2, box['x'] - box['l']/2),
            box['w'], box['l'],
            fill=False, edgecolor='cyan', linewidth=2
        )
        plt.gca().add_patch(rect)
plt.xlabel("Y (lateral)")
plt.ylabel("X (forward)")
plt.title("Heatmap + Boxes")

#center points from heatmap
# Find max intensity points in ground truth heatmap
from scipy.ndimage import maximum_filter

# Find peaks
local_max = maximum_filter(heat, size=5) == heat
peaks = (heat > 0.3) & local_max
peak_rows, peak_cols = np.where(peaks)

# Plot them  using pixel coordinates converted to the extent range
fig = plt.gcf()
ax = fig.axes[1]  # Heatmap subplot

for i in range(len(peak_rows)):
    row = peak_rows[i]
    col = peak_cols[i]
    
    lateral_m = (row / 128) * 200 - 100
    forward_m = (col / 128) * 200 - 100
    
    ax.plot(lateral_m, forward_m, 'go', markersize=10, markeredgecolor='white', markeredgewidth=2)
plt.show()