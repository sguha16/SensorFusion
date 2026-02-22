# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 12:25:57 2026

@author: sanhi
"""

# create_gif_visualization.py
# Standalone script to create LinkedIn GIF

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
import os
from nuscenes.utils.data_classes import Box


# ============================================================================
# ARCHITECTURE DEFINITIONS (copy from your working code)
# ============================================================================

class SensorQualityPredictor(nn.Module):
    def __init__(self):
        super(SensorQualityPredictor, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        quality = self.sigmoid(self.fc(x))
        return quality

class ChannelAttention(nn.Module):
    def __init__(self, channels=64):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // 4)
        self.fc2 = nn.Linear(channels // 4, channels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = torch.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1)
        return x * y
    
    def get_weights(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = torch.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        return y

class FusionCNN(nn.Module):
    def __init__(self):
        super(FusionCNN, self).__init__()
        
        self.radar_conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.radar_conv2 = nn.Conv2d(16, 32, 3, padding=1)
        
        self.camera_conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.camera_conv2 = nn.Conv2d(16, 32, 3, padding=1)
        
        self.decoder_conv1 = nn.Conv2d(64, 32, 3, padding=1)
        self.decoder_conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.decoder_conv3 = nn.Conv2d(16, 1, 3, padding=1)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.attention = ChannelAttention(channels=64)
        self.camera_quality_predictor = SensorQualityPredictor()
    
    def forward(self, radar, camera, return_features=False, return_attention=False):
        camera_quality_pred = self.camera_quality_predictor(camera)
        
        r = self.relu(self.radar_conv1(radar))
        r = self.pool(r)
        r = self.relu(self.radar_conv2(r))
        r_features = self.pool(r)
        
        c = self.relu(self.camera_conv1(camera))
        c = self.pool(c)
        c = self.relu(self.camera_conv2(c))
        c_features = self.pool(c)
        
        fused = torch.cat([r_features, c_features], dim=1)
        
        if return_attention:
            att_weights = self.attention.get_weights(fused)
        
        fused = self.attention(fused)
        
        x = self.relu(self.decoder_conv1(fused))
        x = self.upsample(x)
        x = self.relu(self.decoder_conv2(x))
        x = self.upsample(x)
        x = self.decoder_conv3(x)
        
        if return_attention and return_features:
            return x, r_features, c_features, att_weights, camera_quality_pred
        elif return_attention:
            return x, att_weights, camera_quality_pred
        elif return_features:
            return x, r_features, c_features, camera_quality_pred
        else:
            return x, camera_quality_pred

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def process_one_sample(nusc, idx):
    sample = nusc.sample[idx]
    radar_token = sample['data']['RADAR_FRONT']
    radar_data = nusc.get('sample_data', radar_token)
    
    radar_file = f"{nusc.dataroot}/{radar_data['filename']}"
    radar_pc = RadarPointCloud.from_file(radar_file)
    
    cs = nusc.get('calibrated_sensor', radar_data['calibrated_sensor_token'])
    sensor_to_ego = transform_matrix(cs['translation'], Quaternion(cs['rotation']), inverse=False)
    radar_pc.transform(sensor_to_ego)
    
    x = radar_pc.points[0, :]
    y = radar_pc.points[1, :]
    
    grid_size = (128, 128)
    bev_range = 200.0
    bev_map = np.zeros(grid_size, dtype=np.float32)
    
    grid_x = ((x + bev_range/2) / bev_range * grid_size[0]).astype(int)
    grid_y = ((y + bev_range/2) / bev_range * grid_size[1]).astype(int)
    
    for i in range(len(grid_x)):
        gx, gy = grid_x[i], grid_y[i]
        if 0 <= gx < grid_size[0] and 0 <= gy < grid_size[1]:
            bev_map[gy, gx] = 1.0
    
    ego_pose = nusc.get('ego_pose', radar_data['ego_pose_token'])
    global_to_ego = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=True)
    
    sigma = 3
    heatmap = np.zeros(grid_size, dtype=np.float32)
    
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        from nuscenes.utils.data_classes import Box
        box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)
        
        bx, by, bz = box.center
        
        if abs(bx) >= bev_range/2 or abs(by) >= bev_range/2:
            continue
        
        px = int((bx + bev_range/2) / bev_range * grid_size[0])
        py = int((by + bev_range/2) / bev_range * grid_size[1])
        
        for i in range(max(0, px-10), min(grid_size[0], px+10)):
            for j in range(max(0, py-10), min(grid_size[1], py+10)):
                dist = (i - px)**2 + (j - py)**2
                heatmap[j, i] = max(heatmap[j, i], np.exp(-dist / (2 * sigma**2)))
    
    return bev_map, heatmap

def process_camera_sample(nusc, idx):
    sample = nusc.sample[idx]
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    
    from nuscenes.utils.data_classes import LidarPointCloud
    lidar_file = f"{nusc.dataroot}/{lidar_data['filename']}"
    lidar_pc = LidarPointCloud.from_file(lidar_file)
    
    cs = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    sensor_to_ego = transform_matrix(cs['translation'], Quaternion(cs['rotation']), inverse=False)
    lidar_pc.transform(sensor_to_ego)
    
    x = lidar_pc.points[0, :]
    y = lidar_pc.points[1, :]
    
    grid_size = (128, 128)
    bev_range = 200.0
    bev_map = np.zeros(grid_size, dtype=np.float32)
    
    grid_x = ((x + bev_range/2) / bev_range * grid_size[0]).astype(int)
    grid_y = ((y + bev_range/2) / bev_range * grid_size[1]).astype(int)
    
    for i in range(len(grid_x)):
        gx, gy = grid_x[i], grid_y[i]
        if 0 <= gx < grid_size[0] and 0 <= gy < grid_size[1]:
            bev_map[gy, gx] = 1.0
    
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    
    sigma = 3
    heatmap = np.zeros(grid_size, dtype=np.float32)
    
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)
        
        bx, by, bz = box.center
        
        if abs(bx) >= bev_range/2 or abs(by) >= bev_range/2:
            continue
        
        px = int((bx + bev_range/2) / bev_range * grid_size[0])
        py = int((by + bev_range/2) / bev_range * grid_size[1])
        
        for i in range(max(0, px-10), min(grid_size[0], px+10)):
            for j in range(max(0, py-10), min(grid_size[1], py+10)):
                dist = (i - px)**2 + (j - py)**2
                heatmap[j, i] = max(heatmap[j, i], np.exp(-dist / (2 * sigma**2)))
    
    return bev_map, heatmap

def make_decision(heatmap, camera_quality_pred):
    if camera_quality_pred < 0.5:
        sensor_trust = "RADAR"
        brake_threshold = 0.5
    elif camera_quality_pred > 0.7:
        sensor_trust = "CAMERA"
        brake_threshold = 0.7
    else:
        sensor_trust = "BOTH"
        brake_threshold = 0.6
    
    max_intensity = heatmap.max()
    max_idx = heatmap.argmax()
    row = max_idx // 128
    col = max_idx % 128
    
    forward_dist = (64 - row) * (200 / 128)
    
    if max_intensity > brake_threshold and forward_dist > 1 and forward_dist < 20:
        if forward_dist < 10:
            action = "HARD_BRAKE"
            intensity = 100
        else:
            action = "SOFT_BRAKE"
            intensity = int((20 - forward_dist) / 20 * 100)
        reason = f"Obstacle at {forward_dist:.1f}m"
    elif max_intensity > brake_threshold * 0.7 and forward_dist > 1 and forward_dist < 30:
        action = "SLOW_DOWN"
        intensity = 50
        reason = "Caution: possible obstacle"
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
        'obstacle_distance': forward_dist
    }
def get_annotation_boxes(nusc, sample_idx, grid_size=(128, 128), bev_range=200.0):
    """Get annotation boxes for visualization"""
    sample = nusc.sample[sample_idx]
    
    # Get ego pose
    radar_token = sample['data']['RADAR_FRONT']
    radar_data = nusc.get('sample_data', radar_token)
    ego_pose = nusc.get('ego_pose', radar_data['ego_pose_token'])
    
    boxes = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)
        
        x, y, z = box.center
        w, l, h = box.wlh
        
        boxes.append({'x': x, 'y': y, 'w': w, 'l': l, 'class': box.name})
    
    return boxes


# ============================================================================
# MAIN GIF GENERATION
# ============================================================================

print("Initializing...")

# Load nuScenes
DATAROOT = r"C:/Users/sanhi/Downloads/v1.0-mini"
nusc = NuScenes(version="v1.0-mini", dataroot=DATAROOT, verbose=True)

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FusionCNN().to(device)
model.load_state_dict(torch.load('fusion_model_with_quality.pth', map_location=device))
model.eval()

# Validation indices
from sklearn.model_selection import train_test_split
all_indices = list(range(len(nusc.sample)))
train_indices, val_indices = train_test_split(all_indices, test_size=0.2, random_state=42)

# Create results directory
os.makedirs('results', exist_ok=True)

print("Generating frames...")

# Get samples from ONE scene (sequential)
scene = nusc.scene[0]
scene_samples = []
sample_token = scene['first_sample_token']
while sample_token:
    sample = nusc.get('sample', sample_token)
    scene_samples.append(sample)
    sample_token = sample['next']

num_frames = min(10, len(scene_samples))
frames = []

for frame_idx in range(num_frames):
    sample = scene_samples[frame_idx]
    sample_idx = nusc.sample.index(sample)  # Convert to index for process functions
    
    # Get fresh sample
    radar_bev, _ = process_one_sample(nusc, sample_idx)
    camera_bev, _ = process_camera_sample(nusc, sample_idx)
    
    # Get actual camera image
    sample = nusc.sample[sample_idx]
    cam_token = sample['data']['CAM_FRONT']
    cam_data = nusc.get('sample_data', cam_token)
    cam_img_path = f"{nusc.dataroot}/{cam_data['filename']}"
    cam_img = Image.open(cam_img_path).resize((400, 225))
    #cam_img = project_boxes_to_camera(nusc, sample_idx).resize((400, 225))
    
    # Convert to tensors
    radar_t = torch.FloatTensor(radar_bev[np.newaxis, np.newaxis, :, :]).to(device)
    camera_t = torch.FloatTensor(camera_bev[np.newaxis, np.newaxis, :, :]).to(device)
    
    # Create 3 quality levels
    camera_clean = camera_t
    cam_img_clean = cam_img.copy()
    
    noise_medium = torch.randn_like(camera_t) * 5.0
    camera_medium = torch.clamp(camera_t + noise_medium, 0, 1)
    cam_img_medium = np.array(cam_img)
    cam_img_medium = np.clip(cam_img_medium + np.random.randn(*cam_img_medium.shape) * 30, 0, 255).astype(np.uint8)
    cam_img_medium = Image.fromarray(cam_img_medium)
    
    noise_heavy = torch.randn_like(camera_t) * 20.0
    camera_heavy = torch.clamp(camera_t + noise_heavy, 0, 1)
    cam_img_heavy = np.array(cam_img)
    cam_img_heavy = np.clip(cam_img_heavy + np.random.randn(*cam_img_heavy.shape) * 80, 0, 255).astype(np.uint8)
    cam_img_heavy = Image.fromarray(cam_img_heavy)
    
    # Get predictions
    with torch.no_grad():
        pred_clean, _, qual_clean = model(radar_t, camera_clean, return_attention=True)
        pred_medium, _, qual_medium = model(radar_t, camera_medium, return_attention=True)
        pred_heavy, _, qual_heavy = model(radar_t, camera_heavy, return_attention=True)
    
    dec_clean = make_decision(pred_clean.cpu().numpy()[0, 0], qual_clean.item())
    dec_medium = make_decision(pred_medium.cpu().numpy()[0, 0], qual_medium.item())
    dec_heavy = make_decision(pred_heavy.cpu().numpy()[0, 0], qual_heavy.item())
    
    # Create 3x3 grid
    fig = plt.figure(figsize=(18, 12))
    
    # Row 1: Camera images
    ax1 = plt.subplot(3, 3, 1)
    ax1.imshow(cam_img_clean)
    ax1.set_title(f'Clean Camera\nQuality: {qual_clean.item():.2f}', fontsize=14, weight='bold', color='green')
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.imshow(cam_img_medium)
    ax2.set_title(f'Medium Degraded\nQuality: {qual_medium.item():.2f}', fontsize=14, weight='bold', color='orange')
    ax2.axis('off')
    
    ax3 = plt.subplot(3, 3, 3)
    ax3.imshow(cam_img_heavy)
    ax3.set_title(f'Heavy Degraded\nQuality: {qual_heavy.item():.2f}', fontsize=14, weight='bold', color='red')
    ax3.axis('off')
    
    bev_range = 200.0

    # Row 2: BEV Heatmaps
    boxes = get_annotation_boxes(nusc, sample_idx)
    
    ax4 = plt.subplot(3, 3, 4)
    ax4.imshow(pred_clean.cpu().numpy()[0, 0].T, cmap='jet', origin='lower', 
               extent=[-bev_range/2, bev_range/2, -bev_range/2, bev_range/2])  # ADD EXTENT
    for box in boxes:
        if abs(box['x']) < bev_range/2 and abs(box['y']) < bev_range/2:
            rect = plt.Rectangle((box['y'] - box['w']/2, box['x'] - box['l']/2),
                                 box['w'], box['l'], fill=False, 
                                 edgecolor='magenta', linewidth=2)
            ax4.add_patch(rect)
    ax4.set_title('BEV Detection', fontsize=12)
    ax4.axis('off')
    
    # Same for ax5 and ax6
    ax5 = plt.subplot(3, 3, 5)
    ax5.imshow(pred_medium.cpu().numpy()[0, 0].T, cmap='jet', origin='lower',
               extent=[-bev_range/2, bev_range/2, -bev_range/2, bev_range/2])  # ADD THIS
    for box in boxes:
        if abs(box['x']) < bev_range/2 and abs(box['y']) < bev_range/2:
            rect = plt.Rectangle((box['y'] - box['w']/2, box['x'] - box['l']/2),
                                 box['w'], box['l'], fill=False, 
                                 edgecolor='magenta', linewidth=2)
            ax5.add_patch(rect)
    ax5.set_title('BEV Detection', fontsize=12)
    ax5.axis('off')
    
    ax6 = plt.subplot(3, 3, 6)
    ax6.imshow(pred_heavy.cpu().numpy()[0, 0].T, cmap='jet', origin='lower',
               extent=[-bev_range/2, bev_range/2, -bev_range/2, bev_range/2])  # ADD THIS
    for box in boxes:
        if abs(box['x']) < bev_range/2 and abs(box['y']) < bev_range/2:
            rect = plt.Rectangle((box['y'] - box['w']/2, box['x'] - box['l']/2),
                                 box['w'], box['l'], fill=False, 
                                 edgecolor='magenta', linewidth=2)
            ax6.add_patch(rect)
    ax6.set_title('BEV Detection', fontsize=12)
    ax6.axis('off')
    
    
    
    # Row 3: Decisions
    ax7 = plt.subplot(3, 3, 7)
    ax7.text(0.5, 0.5, 
             f"{dec_clean['action']}\n{dec_clean['intensity']}%\n\n"
             f"Trust: {dec_clean['sensor_trust']}\n"
             f"Dist: {dec_clean['obstacle_distance']:.1f}m",
             ha='center', va='center', fontsize=16, weight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, edgecolor='green', linewidth=3))
    ax7.set_xlim(0, 1)
    ax7.set_ylim(0, 1)
    ax7.axis('off')
    
    ax8 = plt.subplot(3, 3, 8)
    ax8.text(0.5, 0.5,
             f"{dec_medium['action']}\n{dec_medium['intensity']}%\n\n"
             f"Trust: {dec_medium['sensor_trust']}\n"
             f"Dist: {dec_medium['obstacle_distance']:.1f}m",
             ha='center', va='center', fontsize=16, weight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9, edgecolor='orange', linewidth=3))
    ax8.set_xlim(0, 1)
    ax8.set_ylim(0, 1)
    ax8.axis('off')
    
    ax9 = plt.subplot(3, 3, 9)
    ax9.text(0.5, 0.5,
             f"{dec_heavy['action']}\n{dec_heavy['intensity']}%\n\n"
             f"Trust: {dec_heavy['sensor_trust']}\n"
             f"Dist: {dec_heavy['obstacle_distance']:.1f}m",
             ha='center', va='center', fontsize=16, weight='bold',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.9, edgecolor='red', linewidth=3))
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    ax9.axis('off')
    
    plt.suptitle(f'Adaptive Sensor Fusion - Driving Scenario {frame_idx+1}/{num_frames}', 
                 fontsize=18, weight='bold')
    plt.tight_layout()
    
    # Save frame
    frame_path = f'results/temp_frame_{frame_idx:02d}.png'
    plt.savefig(frame_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.savefig(f'results/frame_{frame_idx:02d}_KEEP.png', dpi=150, bbox_inches='tight', facecolor='white')  # ADD THIS
    plt.close()
    
    frames.append(Image.open(frame_path))
    print(f"✓ Frame {frame_idx+1}/{num_frames} complete")

# Create GIF
print("\nCreating GIF...")
frames[0].save(
    'results/adaptive_fusion_driving.gif',
    save_all=True,
    append_images=frames[1:],
    duration=1000,
    loop=0
)

# Cleanup temp files
for i in range(num_frames):
    os.remove(f'results/temp_frame_{i:02d}.png')

print("\n✅ SUCCESS! Saved: results/adaptive_fusion_driving.gif")
print("Ready for LinkedIn! 🚀")