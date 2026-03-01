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
import os
#get Radar data
# Initialize nuScenes
#mini
DATAROOT = r"C:/Users/sanhi/Downloads/v1.0-mini"
nusc = NuScenes(version="v1.0-mini", dataroot=DATAROOT, verbose=True)

#notmini
#DATAROOT = r"C:\Users\sanhi\Downloads"
#nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=True)

#wrapping in a func
def process_one_sample(nusc, sample_idx, grid_size=(128, 128), bev_range=200.0, sigma=3):
    """
    Process one nuScenes sample and return BEV map + heatmap.
    
    Args:
        nusc: NuScenes object
        sample_idx: Index into nusc.sample
        grid_size: (height, width) of output grids
        bev_range: Range in meters (e.g., 200 = -100 to +100)
        sigma: Gaussian sigma for heatmap
    
    Returns:
        bev_map: numpy array (128, 128) with radar points
        heatmap: numpy array (128, 128) with object centers
    """
    # Get the sample
    sample = nusc.sample[sample_idx]
    
    # --- RADAR BEV ---
    radar_key = 'RADAR_FRONT'
    radar_token = sample['data'][radar_key]
    radar_data = nusc.get('sample_data', radar_token)
    
    radar_file = f"{nusc.dataroot}/{radar_data['filename']}"
    radar_pc = RadarPointCloud.from_file(radar_file)
    
    # Transform to ego frame
    cs = nusc.get('calibrated_sensor', radar_data['calibrated_sensor_token'])
    sensor_to_ego = transform_matrix(cs['translation'], Quaternion(cs['rotation']), inverse=False)
    radar_pc.transform(sensor_to_ego)
    
    # Extract x, y
    x = radar_pc.points[0, :]  # forward
    y = radar_pc.points[1, :]  # lateral
    
    # Create BEV map
    bev_map = np.zeros(grid_size, dtype=np.float32)
    grid_x = ((x + bev_range/2) / bev_range * grid_size[0]).astype(int)
    grid_y = ((y + bev_range/2) / bev_range * grid_size[1]).astype(int)
    
    for i in range(len(grid_x)):
        gx, gy = grid_x[i], grid_y[i]
        if 0 <= gx < grid_size[0] and 0 <= gy < grid_size[1]:
            bev_map[gy, gx] = 1.0
    
    # --- HEATMAP ---
    ego_pose = nusc.get('ego_pose', radar_data['ego_pose_token'])
    heatmap = np.zeros(grid_size, dtype=np.float32)
    
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        
        # Create box and transform to ego
        box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)
        
        bx, by = box.center[0], box.center[1]
        
        # Skip if out of range
        if abs(bx) >= bev_range/2 or abs(by) >= bev_range/2 or bx < 0:
            continue
        
        # Convert to pixel coordinates
        px = int((bx + bev_range/2) / bev_range * grid_size[0])
        py = int((by + bev_range/2) / bev_range * grid_size[1])
        
        # Draw Gaussian
        for i in range(max(0, px-10), min(grid_size[0], px+10)):
            for j in range(max(0, py-10), min(grid_size[1], py+10)):
                dist = (i - px)**2 + (j - py)**2
                heatmap[j, i] = max(heatmap[j, i], np.exp(-dist / (2 * sigma**2)))
    
    return bev_map, heatmap


#VISUALIZE
sample_idx = 1
bev, heat = process_one_sample(nusc, sample_idx)
print(f"BEV shape: {bev.shape}")
print(f"Heatmap shape: {heat.shape}")

# Get sample and ego_pose for visualization
sample = nusc.sample[sample_idx]
radar_token = sample['data']['RADAR_FRONT']
radar_data = nusc.get('sample_data', radar_token)
ego_pose = nusc.get('ego_pose', radar_data['ego_pose_token'])  # ADD THIS LINE

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

# Your exact visualization
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

plt.tight_layout()
plt.show()