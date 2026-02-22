# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 12:29:20 2026

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

#PREP FOR CNN
#SPLIT DATA
all_indices = list(range(len(nusc.sample)))
train_indices, val_indices = train_test_split(all_indices, test_size=0.2, random_state=42)
print(f"Train: {len(train_indices)}")#323
print(f"Val: {len(val_indices)}")#81

# Collect training data-x=bev, y= heat, 323,128,128 train, 81,128,128 val
print("Processing training data...")
X_train = []
y_train = []

for idx in train_indices:
    bev, heat = process_camera_sample(nusc, idx)
    X_train.append(bev)
    y_train.append(heat)

X_train = np.array(X_train)
y_train = np.array(y_train)


# Collect validation data
print("Processing validation data...")
X_val = []
y_val = []

for idx in val_indices:
    bev, heat = process_camera_sample(nusc, idx)
    X_val.append(bev)
    y_val.append(heat)

X_val = np.array(X_val)
y_val = np.array(y_val)

# Add channel dimension for CNN (expects 4D: batch, channel height, width) for pytorch
X_train = X_train[:, np.newaxis, :, :]  # (N, 1, 128, 128)
y_train = y_train[:, np.newaxis, :, :]  # (N, 1, 128, 128)
X_val = X_val[:, np.newaxis, :, :]
y_val = y_val[:, np.newaxis, :, :]

print("\nFinal shapes:")
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"y_val: {y_val.shape}")

#DATALOADER
# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val)
# Create datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
# Create dataloaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#for each epoch CNN trains on batch size iteratively
#first 8->CNN calculates loss, changes weights--go to next 8 till
#323 samples are covered. Repeat for every epoch

print("\nDataLoaders ready:")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# Test the loader
for bev_batch, heat_batch in train_loader:
    print(f"Batch shapes: {bev_batch.shape}, {heat_batch.shape}")
    break  # just test first batch
    
#FEED TO CNN: TRAIN PREDICT HEATMAP FROM BEV points
class CameraCNN(nn.Module):
    def __init__(self):
        super(CameraCNN, self).__init__()
        
        # Encoder (downsample)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Decoder (upsample)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        # Encoder
        x = self.relu(self.conv1(x))  # (B, 16, 128, 128)
        x = self.pool(x)               # (B, 16, 64, 64)
        
        x = self.relu(self.conv2(x))  # (B, 32, 64, 64)
        x = self.pool(x)               # (B, 32, 32, 32)
        
        x = self.relu(self.conv3(x))  # (B, 64, 32, 32)
        
        # Decoder
        x = self.upsample(x)           # (B, 64, 64, 64)
        x = self.relu(self.conv4(x))  # (B, 32, 64, 64)
        
        x = self.upsample(x)           # (B, 32, 128, 128)
        x = self.relu(self.conv5(x))  # (B, 16, 128, 128)
        
        x = self.conv6(x)              # (B, 1, 128, 128)
        return x

# Create model
model = CameraCNN()
print(model)
#TrainingLoop

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    #bev_batch is X train accessed by dataloader
    for bev_batch, heat_batch in train_loader:
        bev_batch = bev_batch.to(device)
        heat_batch = heat_batch.to(device)
        
        # Forward pass
        outputs = model(bev_batch)
        loss = criterion(outputs, heat_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for bev_batch, heat_batch in val_loader:
            bev_batch = bev_batch.to(device)
            heat_batch = heat_batch.to(device)
            outputs = model(bev_batch)
            loss = criterion(outputs, heat_batch)
            val_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

print("Training complete!")

#Visualize what CNN learnt
# Visualize predictions
model.eval()
with torch.no_grad():
    # Pick first validation sample
    i=41
    sample_bev = X_val_tensor[i:i+1].to(device)  # (1, 1, 128, 128)
    sample_heat_true = y_val_tensor[i:i+1].to(device)
    
    # Predict
    sample_heat_pred = model(sample_bev)
    
    # Move to CPU for plotting
    bev_np = sample_bev.cpu().numpy()[0, 0]  # (128, 128)
    heat_true_np = sample_heat_true.cpu().numpy()[0, 0]
    heat_pred_np = sample_heat_pred.cpu().numpy()[0, 0]

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(bev_np.T, cmap='hot', origin='lower')
axes[0].set_title('Input: BEV Camera')

axes[1].imshow(heat_true_np.T, cmap='hot', origin='lower')
axes[1].set_title('Ground Truth Heatmap')

axes[2].imshow(heat_pred_np.T, cmap='hot', origin='lower')
axes[2].set_title('Predicted Heatmap')

plt.tight_layout()
plt.savefig('results/camera_only_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
#large loss gap between training and val loss= overfitting

os.makedirs('results', exist_ok=True)

# Save model
torch.save(model.state_dict(), 'results/camera_only_model.pth')
print("✓ Model saved: results/camera_only_model.pth")

# Save results
with open('results/camera_only_results.txt', 'w') as f:
    f.write(f"Camera-Only Model Results\n")
    f.write(f"="*40 + "\n")
    f.write(f"Final Train Loss: {train_loss/len(train_loader):.6f}\n")
    f.write(f"Final Val Loss: {val_loss/len(val_loader):.6f}\n")
    f.write(f"Total Epochs: {num_epochs}\n")

# Save visualization
print("✓ Saved: results/camera_only_visualization.png")

plt.show()