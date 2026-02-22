# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 10:34:42 2026

@author: sanhi
"""

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
camera_quality_train = []  # NEW: track camera quality labels

for idx in train_indices:
    radar_bev, heat_radar = process_one_sample(nusc, idx)
    camera_bev, heat_camera = process_camera_sample(nusc, idx)
    
    #section to add noise randomly to camera--simulate bad weather
    # 20% chance: add noise to camera
    # if random.random() < 0.2:
    #     noise = np.random.randn(*camera_bev.shape) * 15.0  # Gaussian noise
    #     camera_bev = np.clip(camera_bev + noise, 0, 1)
    #     camera_quality = 0.0  # NEW: label as degraded
    # else:
    #     camera_quality = 1.0  # NEW: label as clean
    
    #Making quality labels more variable--->
    rand = random.random()

    if rand < 0.15:  # 15% - severely degraded
        noise = np.random.randn(*camera_bev.shape) * 15.0
        camera_bev = np.clip(camera_bev + noise, 0, 1)
        camera_quality = 0.0
    elif rand < 0.30:  # 15% - moderately degraded
        noise = np.random.randn(*camera_bev.shape) * 8.0
        camera_bev = np.clip(camera_bev + noise, 0, 1)
        camera_quality = 0.5
    elif rand < 0.45:  # 15% - slightly degraded
        noise = np.random.randn(*camera_bev.shape) * 3.0
        camera_bev = np.clip(camera_bev + noise, 0, 1)
        camera_quality = 0.8
    else:  # 55% - clean
        camera_quality = 1.0
    
    X_train_radar.append(radar_bev)
    X_train_camera.append(camera_bev)
    y_train.append(heat_radar)  # Use either, they're the same
    camera_quality_train.append(camera_quality)  # NEW: save label


X_train_radar = np.array(X_train_radar)
X_train_camera = np.array(X_train_camera)
y_train = np.array(y_train)
camera_quality_train = np.array(camera_quality_train)  # NEW

# Collecting paired data for validation
print("Processing validation data...")
X_val_radar = []
X_val_camera = []
y_val = []
camera_quality_val = []  # NEW: track camera quality labels


for idx in val_indices:
    radar_bev, heat_radar = process_one_sample(nusc, idx)
    camera_bev, heat_camera = process_camera_sample(nusc, idx)
    
    #section to add noise randomly to camera--simulate bad weather
    # 20% chance: add noise to camera
    # if random.random() < 0.2:
    #     noise = np.random.randn(*camera_bev.shape) * 15.0  # Gaussian noise
    #     camera_bev = np.clip(camera_bev + noise, 0, 1)
    #     camera_quality = 0.0  # NEW: degraded
    # else:
    #     camera_quality = 1.0  # NEW: clean
    
    #Making quality labels more variable--->
    rand = random.random()

    if rand < 0.15:  # 15% - severely degraded
        noise = np.random.randn(*camera_bev.shape) * 15.0
        camera_bev = np.clip(camera_bev + noise, 0, 1)
        camera_quality = 0.0
    elif rand < 0.30:  # 15% - moderately degraded
        noise = np.random.randn(*camera_bev.shape) * 8.0
        camera_bev = np.clip(camera_bev + noise, 0, 1)
        camera_quality = 0.5
    elif rand < 0.45:  # 15% - slightly degraded
        noise = np.random.randn(*camera_bev.shape) * 3.0
        camera_bev = np.clip(camera_bev + noise, 0, 1)
        camera_quality = 0.8
    else:  # 55% - clean
        camera_quality = 1.0
        

    
    X_val_radar.append(radar_bev)
    X_val_camera.append(camera_bev)
    y_val.append(heat_radar)
    camera_quality_val.append(camera_quality)  # NEW: save label


X_val_radar = np.array(X_val_radar)
X_val_camera = np.array(X_val_camera)
y_val = np.array(y_val)
camera_quality_val = np.array(camera_quality_val)  # NEW

print(f"\nCollected shapes:")
print(f"Radar train: {X_train_radar.shape}")
print(f"Camera train: {X_train_camera.shape}")
print(f"Heatmap train: {y_train.shape}")
print(f"camera quality train: {camera_quality_train.shape}")

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
camera_quality_train_t = torch.FloatTensor(camera_quality_train)  # NEW

X_val_radar_t = torch.FloatTensor(X_val_radar)
X_val_camera_t = torch.FloatTensor(X_val_camera)
y_val_t = torch.FloatTensor(y_val)
camera_quality_val_t = torch.FloatTensor(camera_quality_val)  # NEW

# Create dataset that pairs both inputs with target
#train_dataset = torch.utils.data.TensorDataset(X_train_radar_t, X_train_camera_t, y_train_t)
#val_dataset = torch.utils.data.TensorDataset(X_val_radar_t, X_val_camera_t, y_val_t)
# NEW: Include camera quality labels in dataset
train_dataset = torch.utils.data.TensorDataset(
    X_train_radar_t, 
    X_train_camera_t, 
    y_train_t,
    camera_quality_train_t  # ADD THIS
)

val_dataset = torch.utils.data.TensorDataset(
    X_val_radar_t, 
    X_val_camera_t, 
    y_val_t,
    camera_quality_val_t  # ADD THIS
)

# DataLoaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print("\nDataLoaders ready:")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# Test the loader
#for radar_batch, camera_batch, heat_batch in train_loader:
    #print(f"Batch shapes: radar={radar_batch.shape}, camera={camera_batch.shape}, heat={heat_batch.shape}")
    #break
for radar_batch, camera_batch, heat_batch, quality_batch in train_loader:  # ADD quality_batch
    print(f"Batch shapes: radar={radar_batch.shape}, camera={camera_batch.shape}, heat={heat_batch.shape}, quality={quality_batch.shape}")
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
    
class SensorQualityPredictor(nn.Module):
    """
    Predicts sensor quality (0=degraded, 1=clean) from BEV input.
    """
    def __init__(self):
        super(SensorQualityPredictor, self).__init__()
        # Small CNN to assess input quality
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2)   # 128->64
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2)  # 64->32
        self.pool = nn.AdaptiveAvgPool2d(1)  # 32x32 -> 1x1
        self.fc = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (B, 1, 128, 128)
        x = torch.relu(self.conv1(x))  # (B, 8, 64, 64)
        x = torch.relu(self.conv2(x))  # (B, 16, 32, 32)
        x = self.pool(x)               # (B, 16, 1, 1)
        x = x.view(x.size(0), -1)      # (B, 16)
        quality = self.sigmoid(self.fc(x))  # (B, 1) -> values 0-1
        return quality  # Keep as (B, 1) - don't squeeze
    
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
        
        # NEW: Add quality predictor for camera
        self.camera_quality_predictor = SensorQualityPredictor()
    
    def forward(self, radar, camera,return_features=False, return_attention=False):
        # NEW: Predict camera quality
        camera_quality_pred = self.camera_quality_predictor(camera)  # (B,)
        
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
            return x, r_features, c_features, att_weights, camera_quality_pred
        elif return_attention:
            return x, att_weights, camera_quality_pred
        elif return_features:
            return x, r_features, c_features, camera_quality_pred
        else:
            return x, camera_quality_pred
        
        

# Training
model = FusionCNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training with quality prediction
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion_heatmap = nn.MSELoss()  # For heatmap prediction
criterion_quality = nn.BCELoss()  # For quality prediction (0 or 1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_heatmap_loss = 0.0
    train_quality_loss = 0.0
    
    # Unpack 4 items: radar, camera, heatmap, quality
    for radar_batch, camera_batch, heat_batch, quality_batch in train_loader:
        radar_batch = radar_batch.to(device)
        camera_batch = camera_batch.to(device)
        heat_batch = heat_batch.to(device)
        quality_batch = quality_batch.to(device).unsqueeze(1)  # (B,) -> (B, 1)

        
        # Forward pass - now returns (heatmap, quality)
        heatmap_pred, quality_pred = model(radar_batch, camera_batch)
        
        # Two losses
        loss_heatmap = criterion_heatmap(heatmap_pred, heat_batch)
        loss_quality = criterion_quality(quality_pred, quality_batch)
        
        # Combined loss (weight quality less, it's auxiliary task)
        loss = loss_heatmap + 0.1 * loss_quality
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_heatmap_loss += loss_heatmap.item()
        train_quality_loss += loss_quality.item()
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_heatmap_loss = 0.0
    val_quality_loss = 0.0
    
    with torch.no_grad():
        for radar_batch, camera_batch, heat_batch, quality_batch in val_loader:
            radar_batch = radar_batch.to(device)
            camera_batch = camera_batch.to(device)
            heat_batch = heat_batch.to(device)
            quality_batch = quality_batch.to(device).unsqueeze(1)  # (B,) -> (B, 1)

            
            heatmap_pred, quality_pred = model(radar_batch, camera_batch)
            
            loss_heatmap = criterion_heatmap(heatmap_pred, heat_batch)
            loss_quality = criterion_quality(quality_pred, quality_batch)
            loss = loss_heatmap + 0.1 * loss_quality
            
            val_loss += loss.item()
            val_heatmap_loss += loss_heatmap.item()
            val_quality_loss += loss_quality.item()
    
    # Print every 5 epochs
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train: {train_loss/len(train_loader):.4f} '
              f'(heat={train_heatmap_loss/len(train_loader):.4f}, '
              f'qual={train_quality_loss/len(train_loader):.4f}), '
              f'Val: {val_loss/len(val_loader):.4f}')

print("Training complete!")
# Save model
torch.save(model.state_dict(), 'fusion_model_with_quality.pth')
print("✓ Model saved")

print("Fusion training complete and model saved!")


# === STEP 4: TEST QUALITY PREDICTOR ===
print("\n=== TESTING QUALITY PREDICTOR ===")
model.eval()

with torch.no_grad():
    # Test on clean sample
    clean_pred, clean_quality = model(X_val_radar_t[0:1].to(device), X_val_camera_t[0:1].to(device))
    print(f"Clean camera quality prediction: {clean_quality.item():.3f} (should be ~1.0)")
    
    # Test on noisy sample
    noise = torch.randn_like(X_val_camera_t[0:1]) * 15.0
    camera_noisy = torch.clamp(X_val_camera_t[0:1] + noise, 0, 1).to(device)
    noisy_pred, noisy_quality = model(X_val_radar_t[0:1].to(device), camera_noisy)
    print(f"Noisy camera quality prediction: {noisy_quality.item():.3f} (should be ~0.0)")

print("\n" + "="*50)

def make_decision(heatmap, camera_quality_pred):
    """
    Make driving decision based on heatmap and predicted camera quality.
    
    Args:
        heatmap: (128, 128) predicted object heatmap
        camera_quality_pred: float (0-1), predicted camera quality
    
    Returns:
        decision dict with action, confidence, reason
    """
    # Determine sensor trust based on quality
    if camera_quality_pred < 0.5:  # Camera is degraded
        sensor_trust = "RADAR"
        brake_threshold = 0.5  # More conservative
    elif camera_quality_pred > 0.7:  # Camera is good
        sensor_trust = "CAMERA"
        brake_threshold = 0.7  # Less conservative
    else:  # Uncertain quality
        sensor_trust = "BOTH"
        brake_threshold = 0.6
    
    # Analyze heatmap
    max_intensity = heatmap.max()
    
    # Find strongest detection location
    max_idx = heatmap.argmax()
    row = max_idx // 128
    col = max_idx % 128
    
    # Convert to meters
    forward_dist = (64 - row) * (200 / 128)
    lateral_dist = (col - 64) * (200 / 128)
    
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
        'camera_quality': camera_quality_pred,
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
    
    heat_pred, _ = model(radar_sample, camera_sample)

    
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
    #output, r_feat, c_feat = model(radar_sample, camera_sample, return_features=True)
    output, r_feat, c_feat, quality = model(radar_sample, camera_sample, return_features=True)
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
    # Get FRESH clean sample (reprocess without pre-added noise)
    val_idx = val_indices[sample_idx]
    radar_bev, _ = process_one_sample(nusc, val_idx)
    camera_bev, heat = process_camera_sample(nusc, val_idx)
    
    # Convert to tensors
    radar_tensor = torch.FloatTensor(radar_bev[np.newaxis, np.newaxis, :, :]).to(device)
    camera_clean_tensor = torch.FloatTensor(camera_bev[np.newaxis, np.newaxis, :, :]).to(device)
    
    # === CLEAN CONDITION ===
    with torch.no_grad():
        pred_clean, att_clean, quality_clean = model(radar_tensor, camera_clean_tensor, return_attention=True)
    
    radar_np = radar_bev
    camera_clean_np = camera_bev
    heatmap_clean_np = pred_clean.cpu().numpy()[0, 0]
    decision_clean = make_decision(heatmap_clean_np, quality_clean.item())
    
    # === NOISY CONDITION (add noise to clean sample) ===
    noise = torch.randn_like(camera_clean_tensor) * 15.0
    camera_noisy_tensor = torch.clamp(camera_clean_tensor + noise, 0, 1)
    
    with torch.no_grad():
        pred_noisy, att_noisy, quality_noisy = model(radar_tensor, camera_noisy_tensor, return_attention=True)

    camera_noisy_np = camera_noisy_tensor.cpu().numpy()[0, 0]
    heatmap_noisy_np = pred_noisy.cpu().numpy()[0, 0]
    decision_noisy = make_decision(heatmap_noisy_np, quality_noisy.item())
    
    # === PLOT: 2 rows (clean, noisy) × 4 cols (radar, camera, heatmap, decision) ===
    base_row = sample_idx * 2
    
    # --- ROW 1: CLEAN ---
    ax = plt.subplot(6, 4, base_row*4 + 1)
    ax.imshow(radar_np.T, cmap='hot', origin='lower')
    ax.set_title(f'Sample {sample_idx}\nClean\nRadar BEV', fontsize=9)
    ax.axis('off')
    
    ax = plt.subplot(6, 4, base_row*4 + 2)
    ax.imshow(camera_clean_np.T, cmap='hot', origin='lower')
    ax.set_title(f'Camera BEV\nQuality: {quality_clean.item():.2f}', fontsize=9)
    ax.axis('off')
    
    ax = plt.subplot(6, 4, base_row*4 + 3)
    ax.imshow(heatmap_clean_np.T, cmap='jet', origin='lower')
    ax.set_title('Fusion Heatmap', fontsize=9)
    ax.axis('off')
    
    ax = plt.subplot(6, 4, base_row*4 + 4)
    ax.imshow(heatmap_clean_np.T, cmap='jet', origin='lower', alpha=0.3)
    decision_text = f"{decision_clean['action']}\n{decision_clean['intensity']}%\n"
    decision_text += f"{decision_clean['obstacle_distance']:.1f}m\n"
    decision_text += f"Trust: {decision_clean['sensor_trust']}\n"
    decision_text += f"Quality: {decision_clean['camera_quality']:.2f}"
    ax.text(64, 64, decision_text, fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax.set_title('Decision', fontsize=9)
    ax.axis('off')
    
    # --- ROW 2: NOISY ---
    ax = plt.subplot(6, 4, (base_row+1)*4 + 1)
    ax.imshow(radar_np.T, cmap='hot', origin='lower')
    ax.set_title(f'Sample {sample_idx}\nDegraded\nRadar BEV', fontsize=9)
    ax.axis('off')
    
    ax = plt.subplot(6, 4, (base_row+1)*4 + 2)
    ax.imshow(camera_noisy_np.T, cmap='hot', origin='lower')
    ax.set_title(f'Camera BEV (noise std=15)\nQuality: {quality_noisy.item():.2f}', fontsize=9)
    ax.axis('off')
    
    ax = plt.subplot(6, 4, (base_row+1)*4 + 3)
    ax.imshow(heatmap_noisy_np.T, cmap='jet', origin='lower')
    ax.set_title('Fusion Heatmap', fontsize=9)
    ax.axis('off')
    
    ax = plt.subplot(6, 4, (base_row+1)*4 + 4)
    ax.imshow(heatmap_noisy_np.T, cmap='jet', origin='lower', alpha=0.3)
    decision_text = f"{decision_noisy['action']}\n{decision_noisy['intensity']}%\n"
    decision_text += f"{decision_noisy['obstacle_distance']:.1f}m\n"
    decision_text += f"Trust: {decision_noisy['sensor_trust']}\n"
    decision_text += f"Quality: {decision_noisy['camera_quality']:.2f}"
    ax.text(64, 64, decision_text, fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax.set_title('Decision', fontsize=9)
    ax.axis('off')

plt.suptitle('Quality-Aware Fusion: Clean vs Degraded Camera Comparison', fontsize=14)
plt.tight_layout()
plt.savefig('results/clean_vs_noisy_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()

print("✓ Saved clean_vs_noisy_comparison.png")

# Quality prediction comparison (also use fresh samples)
val_idx_viz = val_indices[0]
_, camera_clean_viz = process_camera_sample(nusc, val_idx_viz)
camera_clean_viz_tensor = torch.FloatTensor(camera_clean_viz[np.newaxis, np.newaxis, :, :]).to(device)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Clean sample
radar_viz, _ = process_one_sample(nusc, val_idx_viz)
radar_viz_tensor = torch.FloatTensor(radar_viz[np.newaxis, np.newaxis, :, :]).to(device)

with torch.no_grad():
    _, _, quality_clean_viz = model(radar_viz_tensor, camera_clean_viz_tensor, return_attention=True)

axes[0].bar(['Camera Quality'], [quality_clean_viz.item()], color='green', alpha=0.7)
axes[0].set_ylim([0, 1])
axes[0].set_title('Clean Camera')
axes[0].set_ylabel('Predicted Quality')
axes[0].axhline(y=0.5, color='r', linestyle='--', label='Threshold')
axes[0].legend()

# Noisy sample
noise = torch.randn_like(camera_clean_viz_tensor) * 15.0
camera_noisy_viz = torch.clamp(camera_clean_viz_tensor + noise, 0, 1)

with torch.no_grad():
    _, _, quality_noisy_viz = model(radar_viz_tensor, camera_noisy_viz, return_attention=True)

axes[1].bar(['Camera Quality'], [quality_noisy_viz.item()], color='red', alpha=0.7)
axes[1].set_ylim([0, 1])
axes[1].set_title('Degraded Camera (noise std=15)')
axes[1].set_ylabel('Predicted Quality')
axes[1].axhline(y=0.5, color='r', linestyle='--', label='Threshold')
axes[1].legend()

plt.tight_layout()
plt.savefig('results/quality_prediction_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Saved quality_prediction_comparison.png")
print("\n=== DONE ===")

# Test on controlled degradation levels
print("\n=== QUALITY PREDICTION ACROSS DEGRADATION LEVELS ===")

# Get one fresh clean sample
val_idx = val_indices[0]
radar_bev, _ = process_one_sample(nusc, val_idx)
camera_bev, _ = process_camera_sample(nusc, val_idx)

radar_tensor = torch.FloatTensor(radar_bev[np.newaxis, np.newaxis, :, :]).to(device)
camera_tensor = torch.FloatTensor(camera_bev[np.newaxis, np.newaxis, :, :]).to(device)

noise_levels = [0, 1, 3, 5, 8, 10, 15, 20, 25, 30]  # Increasing degradation
qualities = []
trusts = []

model.eval()
for noise_std in noise_levels:
    if noise_std == 0:
        camera_input = camera_tensor
    else:
        noise = torch.randn_like(camera_tensor) * noise_std
        camera_input = torch.clamp(camera_tensor + noise, 0, 1)
    
    with torch.no_grad():
        _, _, quality = model(radar_tensor, camera_input, return_attention=True)
    
    q = quality.item()
    qualities.append(q)
    
    # Determine trust
    if q < 0.5:
        trust = "RADAR"
    elif q > 0.7:
        trust = "CAMERA"
    else:
        trust = "BOTH"
    trusts.append(trust)
    
    print(f"Noise std={noise_std:2d} → Quality={q:.3f} → Trust: {trust}")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(noise_levels, qualities, 'o-', linewidth=2, markersize=8)
plt.axhline(y=0.7, color='g', linestyle='--', label='CAMERA threshold')
plt.axhline(y=0.5, color='r', linestyle='--', label='RADAR threshold')
plt.xlabel('Noise Standard Deviation', fontsize=12)
plt.ylabel('Predicted Quality', fontsize=12)
plt.title('Camera Quality Prediction vs Degradation Level', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('results/quality_vs_noise_spectrum.png', dpi=150)
plt.close()

print("✓ Saved quality_vs_noise_spectrum.png")