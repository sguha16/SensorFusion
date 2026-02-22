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
    bev, heat = process_one_sample(nusc, idx)
    X_train.append(bev)
    y_train.append(heat)

X_train = np.array(X_train)
y_train = np.array(y_train)


# Collect validation data
print("Processing validation data...")
X_val = []
y_val = []

for idx in val_indices:
    bev, heat = process_one_sample(nusc, idx)
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
class RadarCNN(nn.Module):
    def __init__(self):
        super(RadarCNN, self).__init__()
        
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
model = RadarCNN()
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
axes[0].set_title('Input: BEV Radar')

axes[1].imshow(heat_true_np.T, cmap='hot', origin='lower')
axes[1].set_title('Ground Truth Heatmap')

axes[2].imshow(heat_pred_np.T, cmap='hot', origin='lower')
axes[2].set_title('Predicted Heatmap')

plt.tight_layout()
plt.savefig('results/radar_only_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
#large loss gap between training and val loss= overfitting

# Add AFTER plt.show()

os.makedirs('results', exist_ok=True)

torch.save(model.state_dict(), 'results/radar_only_model.pth')

with open('results/radar_only_results.txt', 'w') as f:
    f.write(f"Radar-Only Model Results\n")
    f.write(f"="*40 + "\n")
    f.write(f"Final Train Loss: {train_loss/len(train_loader):.6f}\n")
    f.write(f"Final Val Loss: {val_loss/len(val_loader):.6f}\n")

print("✓ All saved!")