# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 22:41:40 2026

@author: sanhi
"""

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
import random

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

# Collect paired data for training
print("Processing training data...")
X_train_radar = []
X_train_camera = []
y_train = []

for idx in train_indices:
    radar_bev, heat_radar = process_one_sample(nusc, idx)
    camera_bev, heat_camera = process_camera_sample(nusc, idx)
    
    #section to add noise randomly to camera--simulate bad weather
    # 50% chance: add noise to camera
    if random.random() < 0.2:
        noise = np.random.randn(*camera_bev.shape) * 15.0  # Gaussian noise
        camera_bev = np.clip(camera_bev + noise, 0, 1)
    
    X_train_radar.append(radar_bev)
    X_train_camera.append(camera_bev)
    y_train.append(heat_radar)  # Use either, they're the same

X_train_radar = np.array(X_train_radar)
X_train_camera = np.array(X_train_camera)
y_train = np.array(y_train)

# Collecting paired data for validation
print("Processing validation data...")
X_val_radar = []
X_val_camera = []
y_val = []

for idx in val_indices:
    radar_bev, heat_radar = process_one_sample(nusc, idx)
    camera_bev, heat_camera = process_camera_sample(nusc, idx)
    
    #section to add noise randomly to camera--simulate bad weather
    # 50% chance: add noise to camera
    if random.random() < 0.2:
        noise = np.random.randn(*camera_bev.shape) * 15.0  # Gaussian noise
        camera_bev = np.clip(camera_bev + noise, 0, 1)
    
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

#Attention mechanism--weights dynamically learned per channel
class ChannelAttention(nn.Module):
    def __init__(self, channels=64):
        super(ChannelAttention, self).__init__()
        # Global pooling - compress spatial info
        self.gap = nn.AdaptiveAvgPool2d(1)  # (B, 64, 32, 32) → (B, 64, 1, 1)
        
        # Learn importance
        self.fc1 = nn.Linear(channels, channels // 4)  # 64 → 16
        self.fc2 = nn.Linear(channels // 4, channels)  # 16 → 64
        self.sigmoid = nn.Sigmoid()  # Output weights between 0-1
    
    def forward(self, x):
        # x: (B, 64, 32, 32)
        b, c, _, _ = x.size()
        
        # Squeeze spatial dimensions
        y = self.gap(x).view(b, c)  # (B, 64)
        
        # Learn weights
        y = torch.relu(self.fc1(y))  # (B, 16)
        y = self.sigmoid(self.fc2(y))  # (B, 64) - weights between 0-1
        
        # Reshape for broadcasting
        y = y.view(b, c, 1, 1)  # (B, 64, 1, 1)
        
        return x * y  # Apply weights to input
    
    def get_weights(self, x):
        """Extract just the attention weights without applying them"""
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = torch.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        return y  # (B, 64) - the weights
    
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
        
        # ADDITION:
        self.attention = ChannelAttention(channels=64)
    
    def forward(self, radar, camera,return_features=False, return_attention=False):
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
        
        # ADD ATTENTION: save weights BEFORE applying
        if return_attention:
            att_weights = self.attention.get_weights(fused)  # Get weights
        fused = self.attention(fused)  # Apply channel attention

        # Decoder
        x = self.relu(self.decoder_conv1(fused))
        x = self.upsample(x)  # 64x64
        x = self.relu(self.decoder_conv2(x))
        x = self.upsample(x)  # 128x128
        x = self.decoder_conv3(x)
        
        # RETURN LOGIC
        if return_attention and return_features:
            return x, r_features, c_features, att_weights
        elif return_attention:
            return x, att_weights
        elif return_features:
            return x, r_features, c_features
        else:
            return x
        
        

# Training
model = FusionCNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
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
def make_decision(heatmap, attention_weights):
    """
    Make driving decision based on heatmap and sensor confidence.
    
    Args:
        heatmap: (128, 128) predicted object heatmap
        attention_weights: (64,) attention weights for channels
    
    Returns:
        decision dict with action, confidence, reason
    """
    # Extract radar vs camera confidence
    radar_conf = attention_weights[:32].mean().item()
    camera_conf = attention_weights[32:].mean().item()
    
    # Determine which sensor is more reliable
    if camera_conf > radar_conf * 1.2:  # Camera significantly better
        sensor_trust = "CAMERA"
        brake_threshold = 0.7  # Less conservative
    elif radar_conf > camera_conf * 1.2:  # Radar significantly better
        sensor_trust = "RADAR"
        brake_threshold = 0.5  # More conservative (bad weather likely)
    else:
        sensor_trust = "BOTH"
        brake_threshold = 0.6
    
    # Analyze heatmap
    max_intensity = heatmap.max()
    
    # Find strongest detection location (in BEV coordinates)
    max_idx = heatmap.argmax()
    row = max_idx // 128
    col = max_idx % 128
    
    # Convert to meters (assuming 200m range, 128x128 grid)
    # Center is at (64, 64)
    forward_dist = (64 - row) * (200 / 128)  # meters forward
    lateral_dist = (col - 64) * (200 / 128)  # meters lateral
    
    # Decision logic
    if max_intensity > brake_threshold and forward_dist > 1 and forward_dist < 20:
        if forward_dist < 10:
            action = "HARD_BRAKE"
            intensity = 100
        else:
            action = "SOFT_BRAKE"
            intensity = int((20 - forward_dist) / 20 * 100)
        
        reason = f"Obstacle at {forward_dist:.1f}m, trust={sensor_trust}"
    
    elif max_intensity > brake_threshold * 0.7 and forward_dist > 1 and forward_dist < 30:
        action = "SLOW_DOWN"
        intensity = 50
        reason = f"Caution: possible obstacle at {forward_dist:.1f}m"
    
    else:
        action = "CONTINUE"
        intensity = 0
        reason = "Path clear"
    
    return {
        'action': action,
        'intensity': intensity,
        'reason': reason,
        'sensor_trust': sensor_trust,
        'radar_conf': radar_conf,
        'camera_conf': camera_conf,
        'obstacle_distance': forward_dist,
        'max_heatmap': max_intensity
    }

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
plt.show()

#TEST DECISION OF AGENT
import os
import matplotlib.pyplot as plt

# Create results directory
os.makedirs('results', exist_ok=True)

print("\n=== CREATING VISUALIZATIONS ===")

# Create comparison visualization: 3 samples, each showing clean vs noisy
fig = plt.figure(figsize=(16, 12))

model.eval()

for sample_idx in range(3):
    # Get CLEAN samples
    radar_sample = X_val_radar_t[sample_idx:sample_idx+1].to(device)
    camera_sample = X_val_camera_t[sample_idx:sample_idx+1].to(device)
    
    # === CLEAN CONDITION ===
    with torch.no_grad():
        pred_clean, att_clean = model(radar_sample, camera_sample, return_attention=True)
    
    radar_np = radar_sample.cpu().numpy()[0, 0]
    camera_clean_np = camera_sample.cpu().numpy()[0, 0]
    heatmap_clean_np = pred_clean.cpu().numpy()[0, 0]
    weights_clean = att_clean.cpu().numpy()[0]
    
    decision_clean = make_decision(heatmap_clean_np, weights_clean)
    
    # === NOISY CONDITION (same sample, add noise) ===
    noise = torch.randn_like(camera_sample) * 5.0  # Same std as training
    camera_noisy = torch.clamp(camera_sample + noise, 0, 1)
    
    with torch.no_grad():
        pred_noisy, att_noisy = model(radar_sample, camera_noisy, return_attention=True)
    
    camera_noisy_np = camera_noisy.cpu().numpy()[0, 0]
    heatmap_noisy_np = pred_noisy.cpu().numpy()[0, 0]
    weights_noisy = att_noisy.cpu().numpy()[0]
    
    decision_noisy = make_decision(heatmap_noisy_np, weights_noisy)
    
    # === PLOT: 2 rows (clean, noisy) × 4 cols (radar, camera, heatmap, decision) ===
    base_row = sample_idx * 2
    
    # --- ROW 1: CLEAN ---
    # Radar
    ax = plt.subplot(6, 4, base_row*4 + 1)
    ax.imshow(radar_np.T, cmap='hot', origin='lower')
    ax.set_title(f'Sample {sample_idx} - CLEAN\nRadar BEV', fontsize=9)
    ax.axis('off')
    
    # Camera
    ax = plt.subplot(6, 4, base_row*4 + 2)
    ax.imshow(camera_clean_np.T, cmap='hot', origin='lower')
    ax.set_title('Camera BEV', fontsize=9)
    ax.axis('off')
    
    # Heatmap
    ax = plt.subplot(6, 4, base_row*4 + 3)
    ax.imshow(heatmap_clean_np.T, cmap='jet', origin='lower')
    ax.set_title('Fusion Heatmap', fontsize=9)
    ax.axis('off')
    
    # Decision
    ax = plt.subplot(6, 4, base_row*4 + 4)
    ax.imshow(heatmap_clean_np.T, cmap='jet', origin='lower', alpha=0.3)
    decision_text = f"{decision_clean['action']}\n{decision_clean['intensity']}%\n"
    decision_text += f"{decision_clean['obstacle_distance']:.1f}m\n"
    decision_text += f"Trust: {decision_clean['sensor_trust']}\n"
    decision_text += f"R:{decision_clean['radar_conf']:.2f}\nC:{decision_clean['camera_conf']:.2f}"
    ax.text(64, 64, decision_text, fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax.set_title('Decision', fontsize=9)
    ax.axis('off')
    
    # --- ROW 2: NOISY ---
    # Radar (same)
    ax = plt.subplot(6, 4, (base_row+1)*4 + 1)
    ax.imshow(radar_np.T, cmap='hot', origin='lower')
    ax.set_title(f'Sample {sample_idx} - NOISY\nRadar BEV', fontsize=9)
    ax.axis('off')
    
    # Camera (noisy)
    ax = plt.subplot(6, 4, (base_row+1)*4 + 2)
    ax.imshow(camera_noisy_np.T, cmap='hot', origin='lower')
    ax.set_title('Camera BEV (degraded)', fontsize=9)
    ax.axis('off')
    
    # Heatmap
    ax = plt.subplot(6, 4, (base_row+1)*4 + 3)
    ax.imshow(heatmap_noisy_np.T, cmap='jet', origin='lower')
    ax.set_title('Fusion Heatmap', fontsize=9)
    ax.axis('off')
    
    # Decision
    ax = plt.subplot(6, 4, (base_row+1)*4 + 4)
    ax.imshow(heatmap_noisy_np.T, cmap='jet', origin='lower', alpha=0.3)
    decision_text = f"{decision_noisy['action']}\n{decision_noisy['intensity']}%\n"
    decision_text += f"{decision_noisy['obstacle_distance']:.1f}m\n"
    decision_text += f"Trust: {decision_noisy['sensor_trust']}\n"
    decision_text += f"R:{decision_noisy['radar_conf']:.2f}\nC:{decision_noisy['camera_conf']:.2f}"
    ax.text(64, 64, decision_text, fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax.set_title('Decision', fontsize=9)
    ax.axis('off')

plt.suptitle('Attention-Based Fusion: Clean vs Degraded Camera Comparison', fontsize=14)
plt.tight_layout()
plt.savefig('results/clean_vs_noisy_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Saved clean_vs_noisy_comparison.png")

# Also save attention weights comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Clean sample attention
with torch.no_grad():
    _, att_clean = model(X_val_radar_t[0:1].to(device), X_val_camera_t[0:1].to(device), return_attention=True)
    weights_clean = att_clean.cpu().numpy()[0]

axes[0].bar(range(32), weights_clean[:32], color='steelblue', alpha=0.7, label='Radar')
axes[0].bar(range(32), weights_clean[32:], color='coral', alpha=0.7, label='Camera')
axes[0].set_title('Attention Weights (Clean Camera)')
axes[0].set_xlabel('Channel')
axes[0].set_ylabel('Weight')
axes[0].legend()
axes[0].set_ylim([0, 1])

# Noisy sample attention
noise = torch.randn_like(X_val_camera_t[0:1]) * 15.0
camera_noisy = torch.clamp(X_val_camera_t[0:1] + noise, 0, 1).to(device)

with torch.no_grad():
    _, att_noisy = model(X_val_radar_t[0:1].to(device), camera_noisy, return_attention=True)
    weights_noisy = att_noisy.cpu().numpy()[0]

axes[1].bar(range(32), weights_noisy[:32], color='steelblue', alpha=0.7, label='Radar')
axes[1].bar(range(32), weights_noisy[32:], color='coral', alpha=0.7, label='Camera')
axes[1].set_title('Attention Weights (Noisy Camera)')
axes[1].set_xlabel('Channel')
axes[1].set_ylabel('Weight')
axes[1].legend()
axes[1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig('results/attention_weights_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Saved attention_weights_comparison.png")

print("\n=== DONE ===")