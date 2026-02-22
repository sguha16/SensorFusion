# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 14:11:51 2026

@author: sanhi
"""
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from preprocess_data import NuScenesPreprocessor
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box


# ============================================
# EXAMPLE USAGE
# ============================================

# Initialize nuScenes
DATAROOT = r"C:/Users/sanhi/Downloads/v1.0-mini"
nusc = NuScenes(version="v1.0-mini", dataroot=DATAROOT, verbose=True)

# Initialize preprocessor
preprocessor = NuScenesPreprocessor(
    image_size=(640, 480),  # Resize all images to this size
    max_radar_range=100.0,   # Max radar range in meters
    max_radar_velocity=100.0  # Max velocity in m/s
)

# Get a sample
sample = nusc.sample[1]

# --- CAMERA ---
cam_key = 'CAM_FRONT'
cam_token = sample['data'][cam_key]
cam_data = nusc.get('sample_data', cam_token)

img_path = f"{DATAROOT}/{cam_data['filename']}"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --- RADAR ---
radar_key = 'RADAR_FRONT'
radar_token = sample['data'][radar_key]
radar_data = nusc.get('sample_data', radar_token)
radar_file = f"{DATAROOT}/{radar_data['filename']}"
radar_pc = RadarPointCloud.from_file(radar_file)

# Create BEV map
#bev_tensor = preprocessor.radar_to_bev(radar_pc.points, grid_size=(128, 128), bev_range=50.0)
bev_range=100.0
bev_tensor,x,y = preprocessor.radar_to_bev(radar_pc, radar_data, nusc, grid_size=(128, 128), bev_range=100)

#Annotation
# annotation boxes
# Get ego pose-position of ego vehicle in global cooord
ego_pose = nusc.get('ego_pose', radar_data['ego_pose_token'])
# Global -> Ego transform (inverse=True means global to ego)
global_to_ego = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=True)

# Get boxes and transform
bev_boxes = []
for ann_token in sample['anns']:
    ann = nusc.get('sample_annotation', ann_token)
    
    # Create Box object (it's in global frame)
    box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']), name=ann['category_name'])
    
    # Transform to ego frame
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
    


# --- RADAR-CAMERA PROJECTION (Optional) ---
# points_2d, depths = preprocessor.project_radar_to_camera(radar_pc.points, cam_data, nusc)

# print("\n✅ Radar-camera projection:")
# print(f"   2D points shape: {points_2d.shape}")  # (N, 2)
# print(f"   Depths shape: {depths.shape}")  # (N,)

# ============================================
# VISUALIZE
# ============================================
#image just for ref in cam coordinates
plt.figure(figsize=(20, 20))
plt.imshow(img)
plt.title("CAM_FRONT")
plt.axis('off')
plt.show()

#radar scatter plot with points in ego coordinates
plt.figure(figsize=(20, 20))
plt.scatter(y, x, c='white', s=10)
plt.gca().set_facecolor('black')
plt.xlabel("Y (lateral)")
plt.ylabel("X (forward)")
plt.title("Radar Points (Ego Frame)")
plt.axis('equal')
plt.show()

#Radar scatter plot in ego frame with annotation boxes
plt.figure(figsize=(10, 10))
plt.scatter(y, x, c='white', s=10, label='Radar')
# Boxes
for box in bev_boxes:
    rect = plt.Rectangle(
        (box['y'] - box['w']/2, box['x'] - box['l']/2),
        box['w'], box['l'],
        fill=False, edgecolor='cyan', linewidth=2
    )
    plt.gca().add_patch(rect)

plt.gca().set_facecolor('black')
plt.xlabel("Y (lateral)")
plt.ylabel("X (forward)")
plt.title("Radar Points + Boxes (Ego Frame)")
plt.axis('equal')
plt.legend()
plt.show()

#BEV with boxes
plt.figure(figsize=(10, 10))
plt.imshow(bev_tensor.T, cmap='hot', origin='lower', extent=[-bev_range/2, bev_range/2, -bev_range/2, bev_range/2])

# Filter boxes to same range as BEV
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
plt.title("BEV + Boxes (Ego Frame)")
plt.show()
print(f"✅ BEV tensor: {bev_tensor.shape}")  # (1, 128, 128)
plt.figure(figsize=(20, 20))
plt.imshow(bev_tensor.T, cmap='hot', origin='lower', extent=[-50,50,-50,50])
plt.title("BEV (Radar in Ego Frame)")
plt.xlabel("X (forward)")
plt.ylabel("Y (lateral)")
plt.show()

#HEATMAP
# Create target heatmap from boxes
sigma = 3
heatmap = np.zeros(grid_size, dtype=np.float32)

for box in bev_boxes:
    if abs(box['x']) >= bev_range/2 or abs(box['y']) >= bev_range/2:
        continue
    
    px = int((box['x'] + bev_range/2) / bev_range * grid_size[0])
    py = int((box['y'] + bev_range/2) / bev_range * grid_size[1])
    
    for i in range(max(0, px-10), min(grid_size[0], px+10)):
        for j in range(max(0, py-10), min(grid_size[1], py+10)):
            dist = (i - px)**2 + (j - py)**2
            heatmap[i, j] = max(heatmap[i, j], np.exp(-dist / (2 * sigma**2)))

# Plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(bev_map.T, cmap='hot', origin='lower')
plt.title("Input: BEV")

plt.subplot(1, 2, 2)
plt.imshow(heatmap.T, cmap='hot', origin='lower')
plt.title("Target: Heatmap")
plt.show()
        