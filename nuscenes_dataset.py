# -*- coding: utf-8 -*-
"""
NuScenes Dataset for Sensor Fusion
Step 3.1 — Wraps your preprocessor into a proper PyTorch Dataset
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from preprocess_data import NuScenesPreprocessor


class NuScenesFusionDataset(Dataset):
    """
    PyTorch Dataset for camera + radar fusion.
    
    Returns:
        camera_tensor: (3, H, W) normalized image
        radar_tensor: (N, 5) normalized radar points [x, y, z, vx, vy]
        bev_tensor: (1, 128, 128) bird's eye view occupancy grid
    """
    
    def __init__(self, nusc, sample_indices, dataroot, preprocessor=None):
        """
        Args:
            nusc: NuScenes object (already loaded)
            sample_indices: list of sample indices to use (train or val split)
            dataroot: path to nuScenes data
            preprocessor: NuScenesPreprocessor instance (or creates default)
        """
        self.nusc = nusc
        self.sample_indices = sample_indices
        self.dataroot = dataroot
        
        # Use provided preprocessor or create default
        self.preprocessor = preprocessor or NuScenesPreprocessor(
            image_size=(640, 480),
            max_radar_range=50.0,
            max_radar_velocity=10.0
        )
        
        # Which sensors to use
        self.cam_key = 'CAM_FRONT'
        self.radar_key = 'RADAR_FRONT'
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        """
        Load and preprocess one sample.
        
        Returns:
            dict with keys: 'camera', 'radar', 'bev', 'sample_token'
        """
        # Get the actual sample index
        sample_idx = self.sample_indices[idx]
        sample = self.nusc.sample[sample_idx]
        
        # --- CAMERA ---
        cam_token = sample['data'][self.cam_key]
        cam_data = self.nusc.get('sample_data', cam_token)
        img_path = f"{self.dataroot}/{cam_data['filename']}"
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        camera_tensor = self.preprocessor.preprocess_image(img)
        
        # --- RADAR ---
        radar_token = sample['data'][self.radar_key]
        radar_data = self.nusc.get('sample_data', radar_token)
        radar_path = f"{self.dataroot}/{radar_data['filename']}"
        
        radar_pc = RadarPointCloud.from_file(radar_path)
        radar_tensor = self.preprocessor.preprocess_radar(radar_pc.points)
        
        # --- BEV ---
        bev_tensor = self.preprocessor.radar_to_bev(radar_pc.points)
        
        return {
            'camera': camera_tensor,      # (3, 480, 640)
            'radar': radar_tensor,        # (N, 5) — N varies per sample!
            'bev': bev_tensor,            # (1, 128, 128)
            'sample_token': sample['token']
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable-length radar points.
    
    Cameras and BEVs stack normally.
    Radar points are kept as a list (since N varies per sample).
    """
    cameras = torch.stack([item['camera'] for item in batch])
    bevs = torch.stack([item['bev'] for item in batch])
    radars = [item['radar'] for item in batch]  # List of (N_i, 5) tensors
    tokens = [item['sample_token'] for item in batch]
    
    return {
        'camera': cameras,    # (B, 3, H, W)
        'radar': radars,      # List of B tensors, each (N_i, 5)
        'bev': bevs,          # (B, 1, 128, 128)
        'sample_token': tokens
    }


# ============================================
# TEST IT
# ============================================

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    
    # --- Setup (same as your main_run.py) ---
    DATAROOT = r"C:/Users/sanhi/Downloads/v1.0-mini"
    nusc = NuScenes(version="v1.0-mini", dataroot=DATAROOT, verbose=True)
    
    # Split samples
    all_samples = list(range(len(nusc.sample)))
    train_indices, val_indices = train_test_split(
        all_samples, test_size=0.2, random_state=42
    )
    
    # --- Create datasets ---
    train_dataset = NuScenesFusionDataset(nusc, train_indices, DATAROOT)
    val_dataset = NuScenesFusionDataset(nusc, val_indices, DATAROOT)
    
    print(f"✅ Train dataset: {len(train_dataset)} samples")
    print(f"✅ Val dataset: {len(val_dataset)} samples")
    
    # --- Test single item ---
    sample = train_dataset[0]
    print(f"\n📦 Single sample:")
    print(f"   Camera: {sample['camera'].shape}")
    print(f"   Radar: {sample['radar'].shape}")
    print(f"   BEV: {sample['bev'].shape}")
    
    # --- Test DataLoader ---
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    batch = next(iter(train_loader))
    print(f"\n📦 Batch (batch_size=4):")
    print(f"   Camera batch: {batch['camera'].shape}")
    print(f"   BEV batch: {batch['bev'].shape}")
    print(f"   Radar: list of {len(batch['radar'])} tensors")
    print(f"   Radar shapes: {[r.shape for r in batch['radar']]}")
    
    print("\n✅ Step 3.1 complete! Your data pipeline is ready.")
