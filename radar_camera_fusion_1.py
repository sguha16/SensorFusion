# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 22:33:15 2026

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

#get Radar data
# Initialize nuScenes
DATAROOT = r"C:/Users/sanhi/Downloads/v1.0-mini"
nusc = NuScenes(version="v1.0-mini", dataroot=DATAROOT, verbose=True)
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
        
        bx, by = box.center[0], box.center[1]
        
        if abs(bx) >= bev_range/2 or abs(by) >= bev_range/2 or bx < 0:
            continue
        
        px = int((bx + bev_range/2) / bev_range * grid_size[0])
        py = int((by + bev_range/2) / bev_range * grid_size[1])
        
        for i in range(max(0, px-10), min(grid_size[0], px+10)):
            for j in range(max(0, py-10), min(grid_size[1], py+10)):
                dist = (i - px)**2 + (j - py)**2
                heatmap[j, i] = max(heatmap[j, i], np.exp(-dist / (2 * sigma**2)))
    
    return bev_map, heatmap

#CNN part
# Get train/val split (reuse same random_state for consistency)
all_indices = list(range(len(nusc.sample)))
train_indices, val_indices = train_test_split(all_indices, test_size=0.2, random_state=42)

# Collect paired data
print("Processing training data...")
X_train_radar = []
X_train_camera = []
y_train = []

for idx in train_indices:
    radar_bev, heat_radar = process_one_sample(nusc, idx)
    camera_bev, heat_camera = process_camera_sample(nusc, idx)
    
    X_train_radar.append(radar_bev)
    X_train_camera.append(camera_bev)
    y_train.append(heat_radar)  # Use either, they're the same

X_train_radar = np.array(X_train_radar)
X_train_camera = np.array(X_train_camera)
y_train = np.array(y_train)

# Same for validation
print("Processing validation data...")
X_val_radar = []
X_val_camera = []
y_val = []

for idx in val_indices:
    radar_bev, heat_radar = process_one_sample(nusc, idx)
    camera_bev, heat_camera = process_camera_sample(nusc, idx)
    
    X_val_radar.append(radar_bev)
    X_val_camera.append(camera_bev)
    y_val.append(heat_radar)

X_val_radar = np.array(X_val_radar)
X_val_camera = np.array(X_val_camera)
y_val = np.array(y_val)

print(f"\nCollected shapes:")
print(f"Radar train: {X_train_radar.shape}")
print(f"Camera train: {X_train_camera.shape}")
print(f"Heatmap train: {y_train.shape}")

# Add channel dimensions
X_train_radar = X_train_radar[:, np.newaxis, :, :]
X_train_camera = X_train_camera[:, np.newaxis, :, :]
y_train = y_train[:, np.newaxis, :, :]

X_val_radar = X_val_radar[:, np.newaxis, :, :]
X_val_camera = X_val_camera[:, np.newaxis, :, :]
y_val = y_val[:, np.newaxis, :, :]

#DATALOADER
# Convert to tensors
X_train_radar_t = torch.FloatTensor(X_train_radar)
X_train_camera_t = torch.FloatTensor(X_train_camera)
y_train_t = torch.FloatTensor(y_train)
X_val_radar_t = torch.FloatTensor(X_val_radar)
X_val_camera_t = torch.FloatTensor(X_val_camera)
y_val_t = torch.FloatTensor(y_val)
# Create dataset that pairs both inputs with target
train_dataset = torch.utils.data.TensorDataset(X_train_radar_t, X_train_camera_t, y_train_t)
val_dataset = torch.utils.data.TensorDataset(X_val_radar_t, X_val_camera_t, y_val_t)
# DataLoaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print("\nDataLoaders ready:")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# Test the loader
for radar_batch, camera_batch, heat_batch in train_loader:
    print(f"Batch shapes: radar={radar_batch.shape}, camera={camera_batch.shape}, heat={heat_batch.shape}")
    break
# Fusion CNN
class FusionCNN(nn.Module):
    def __init__(self):
        super(FusionCNN, self).__init__()
        
        # Radar encoder
        self.radar_conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.radar_conv2 = nn.Conv2d(16, 32, 3, padding=1)
        
        # Camera encoder
        self.camera_conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.camera_conv2 = nn.Conv2d(16, 32, 3, padding=1)
        
        # Shared decoder (after concat: 64 channels)
        self.decoder_conv1 = nn.Conv2d(64, 32, 3, padding=1)
        self.decoder_conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.decoder_conv3 = nn.Conv2d(16, 1, 3, padding=1)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, radar, camera,return_features=False):
        # Radar branch
        r = self.relu(self.radar_conv1(radar))
        r = self.pool(r)  # 64x64
        r = self.relu(self.radar_conv2(r))
        r_features = self.pool(r)  # SAVE: (B, 32, 32, 32)

        
        # Camera branch
        c = self.relu(self.camera_conv1(camera))
        c = self.pool(c)  # 64x64
        c = self.relu(self.camera_conv2(c))
        c_features = self.pool(c)  # SAVE: (B, 32, 32, 32)

        # Concat
        fused = torch.cat([r_features, c_features], dim=1)  # 64 channels
        
        # Decoder
        x = self.relu(self.decoder_conv1(fused))
        x = self.upsample(x)  # 64x64
        x = self.relu(self.decoder_conv2(x))
        x = self.upsample(x)  # 128x128
        x = self.decoder_conv3(x)
        
        if return_features:
            return x, r_features, c_features
        
        return x

# Training
model = FusionCNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs=50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for radar_batch, camera_batch, heat_batch in train_loader:
        radar_batch = radar_batch.to(device)
        camera_batch = camera_batch.to(device)
        heat_batch = heat_batch.to(device)
        
        outputs = model(radar_batch, camera_batch)
        loss = criterion(outputs, heat_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for radar_batch, camera_batch, heat_batch in val_loader:
            radar_batch = radar_batch.to(device)
            camera_batch = camera_batch.to(device)
            heat_batch = heat_batch.to(device)
            
            outputs = model(radar_batch, camera_batch)
            loss = criterion(outputs, heat_batch)
            val_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/50], Train: {train_loss/len(train_loader):.4f}, Val: {val_loss/len(val_loader):.4f}')

print("Fusion training complete!")

# Visualize what CNN learnt
model.eval()
i = 0  # first validation sample
with torch.no_grad():
    radar_sample = X_val_radar_t[i:i+1].to(device)
    camera_sample = X_val_camera_t[i:i+1].to(device)
    heat_true = y_val_t[i:i+1].to(device)
    
    heat_pred = model(radar_sample, camera_sample)
    
    # Move to CPU for plotting
    radar_np = radar_sample.cpu().numpy()[0, 0]
    camera_np = camera_sample.cpu().numpy()[0, 0]
    heat_true_np = heat_true.cpu().numpy()[0, 0]
    heat_pred_np = heat_pred.cpu().numpy()[0, 0]

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

axes[0, 0].imshow(radar_np.T, cmap='hot', origin='lower')
axes[0, 0].set_title('Input: Radar BEV')

axes[0, 1].imshow(camera_np.T, cmap='hot', origin='lower')
axes[0, 1].set_title('Input: Camera BEV')

axes[1, 0].imshow(heat_true_np.T, cmap='hot', origin='lower')
axes[1, 0].set_title('Ground Truth Heatmap')

axes[1, 1].imshow(heat_pred_np.T, cmap='hot', origin='lower')
axes[1, 1].set_title('Predicted Heatmap (Fusion)')

plt.tight_layout()
plt.savefig('results/BEV_Heatmaps', dpi=150, bbox_inches='tight')

plt.show()

# Feature visualization
model.eval()
i = 0  # first validation sample

with torch.no_grad():
    radar_sample = X_val_radar_t[i:i+1].to(device)
    camera_sample = X_val_camera_t[i:i+1].to(device)
    
    # Get features
    output, r_feat, c_feat = model(radar_sample, camera_sample, return_features=True)
    
    # Move to CPU
    r_feat = r_feat.cpu().numpy()[0]  # (32, 32, 32)
    c_feat = c_feat.cpu().numpy()[0]  # (32, 32, 32)

# Plot first 8 channels of each
fig, axes = plt.subplots(2, 8, figsize=(16, 4))

for i in range(8):
    # Radar features
    axes[0, i].imshow(r_feat[i], cmap='viridis')
    axes[0, i].set_title(f'Radar-{i}', fontsize=10)
    axes[0, i].axis('off')
    
    # Camera features
    axes[1, i].imshow(c_feat[i], cmap='viridis')
    axes[1, i].set_title(f'Camera-{i}', fontsize=10)
    axes[1, i].axis('off')

plt.suptitle('Feature Maps Before Fusion (First 8 of 32 channels)', fontsize=14)
plt.tight_layout()
# SAVE Results
os.makedirs('results', exist_ok=True)
plt.savefig('results/Feature_maps_before_fusion_first8.png', dpi=150, bbox_inches='tight')
plt.show()

# Show which channels have highest activation
r_mean = r_feat.mean(axis=(1, 2))  # Average activation per channel
c_mean = c_feat.mean(axis=(1, 2))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.bar(range(32), r_mean)
plt.title('Radar Channel Activations')
plt.xlabel('Channel')
plt.ylabel('Mean Activation')

plt.subplot(1, 2, 2)
plt.bar(range(32), c_mean)
plt.title('Camera Channel Activations')
plt.tight_layout()
# SAVE Results
os.makedirs('results', exist_ok=True)
plt.savefig('results/fusion_simple_activations.png', dpi=150, bbox_inches='tight')

plt.show()  # THEN SHOW

# Save model and results
torch.save(model.state_dict(), 'results/fusion_simple_model.pth')

with open('results/fusion_simple_results.txt', 'w') as f:
    f.write("Simple Fusion Model Results\n")
    f.write("="*40 + "\n")
    f.write(f"Final Train Loss: {train_loss/len(train_loader):.6f}\n")
    f.write(f"Final Val Loss: {val_loss/len(val_loader):.6f}\n")
    f.write(f"Total Epochs: {num_epochs}\n")

print("✓ Simple fusion saved!")
plt.show()