# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 14:09:42 2026

@author: sanhi
"""

import torch
from torchvision import transforms
import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix

# ============================================
# PREPROCESSING FUNCTIONS
# ============================================

class NuScenesPreprocessor:
    """
    Handles preprocessing for nuScenes camera + radar data.
    Makes everything model-ready: normalized, resized, tensorized.
    """
    
    def __init__(self, image_size=(640, 480), max_radar_range=50.0, max_radar_velocity=10.0):
        """
        Args:
            image_size: (width, height) to resize all images to
            max_radar_range: maximum radar range in meters (for normalization)
            max_radar_velocity: maximum radar velocity in m/s (for normalization)
        """
        self.image_size = image_size
        self.max_radar_range = max_radar_range
        self.max_radar_velocity = max_radar_velocity
        
        # Image normalization: standard ImageNet stats
        # These work well for transfer learning with pretrained models
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            transforms.Resize(image_size),  # Resize to fixed size
            transforms.ToTensor(),  # Convert to tensor [0, 1], shape [C, H, W]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]     # ImageNet std
            )
        ])
    
    def preprocess_image(self, img):
        """
        Preprocess a camera image.
        
        Args:
            img: numpy array (H, W, 3) in RGB format
            
        Returns:
            torch.Tensor: shape (3, H, W), normalized
        """
        # Apply transforms: resize → to tensor → normalize
        img_tensor = self.image_transform(img)
        return img_tensor
    
    def preprocess_radar(self, radar_points):
        """
        Preprocess radar points: normalize coordinates and velocities.
        
        Args:
            radar_points: numpy array shape (18, N) - raw radar point cloud
            
        Returns:
            torch.Tensor: shape (N, 5) - [x, y, z, vx, vy] normalized
        """
        # Extract relevant features
        # radar_points[0:3] = x, y, z (position in ego frame)
        # radar_points[8:10] = vx, vy (velocity in m/s)
        
        x = radar_points[0, :]  # forward
        y = radar_points[1, :]  # lateral
        z = radar_points[2, :]  # vertical
        vx = radar_points[8, :]  # velocity forward
        vy = radar_points[9, :]  # velocity lateral
        
        # Stack into (N, 5) array
        points = np.stack([x, y, z, vx, vy], axis=1)  # shape: (N, 5)
        
        # Normalize positions to [-1, 1] based on max range
        points[:, 0:3] = points[:, 0:3] / self.max_radar_range
        
        # Normalize velocities to [-1, 1] based on max velocity
        points[:, 3:5] = points[:, 3:5] / self.max_radar_velocity
        
        # Clip outliers (in case some points exceed expected range)
        points = np.clip(points, -1.0, 1.0)
        
        # Convert to torch tensor
        radar_tensor = torch.tensor(points, dtype=torch.float32)
        
        return radar_tensor
    
    def project_radar_to_camera(self, radar_points, cam_data, nusc):
        """
        Project radar points onto camera image.
        Returns 2D pixel coordinates + depth.
        
        Args:
            radar_points: numpy array (18, N)
            cam_data: camera sample_data dict from nuScenes
            nusc: NuScenes object
            
        Returns:
            points_2d: numpy array (N, 2) - pixel coordinates [u, v]
            depths: numpy array (N,) - distance from camera
        """
        # Get camera calibration
        cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        R_cam = Quaternion(cs['rotation']).rotation_matrix
        t_cam = np.array(cs['translation']).reshape(3, 1)
        K = np.array(cs['camera_intrinsic'])
        
        # Transform to camera frame
        points_ego = radar_points[:3, :]
        points_cam = R_cam.T @ (points_ego - t_cam)
        
        # Project to 2D
        points_2d_homog = view_points(points_cam, K, normalize=True)  # (3, N)
        
        # Filter points behind camera
        mask = points_cam[2, :] > 0
        points_2d = points_2d_homog[:2, mask].T  # (N, 2) [u, v]
        depths = points_cam[2, mask]  # (N,)
    
        return points_2d, depths
    
    def radar_to_bev(self, radar_pc, radar_data, nusc, grid_size=(128, 128), bev_range=50.0):
        """
        Convert radar points to a Bird's Eye View (BEV) occupancy grid.
        
        Args:
            radar_points: numpy array (18, N) - raw radar data (in sensor frame)
            radar_data: sample_data dict for the radar
            nusc: NuScenes object
            grid_size: (width, height) of BEV grid in pixels
            bev_range: range in meters (e.g., 50m = -25m to +25m in each direction)
            
        Returns:
            bev_tensor: torch.Tensor shape (1, H, W) - BEV occupancy map
        """
        # Step 1: Transform sensor frame -> ego frame
        # Sensor -> Ego
        cs = nusc.get('calibrated_sensor', radar_data['calibrated_sensor_token'])
        sensor_to_ego = transform_matrix(cs['translation'], Quaternion(cs['rotation']), inverse=False)
        radar_pc.transform(sensor_to_ego)  # now in ego frame
        
        x = radar_pc.points[0, :]  # forward
        y = radar_pc.points[1, :]  # lateral
        # Create BEV grid
        grid_size = (128, 128)
        bev_map = np.zeros(grid_size, dtype=np.float32)
        # Convert meters to pixels
        # ego frame: x from -25 to +25, y from -25 to +25
        grid_x = ((x + bev_range/2) / bev_range * grid_size[0]).astype(int)
        grid_y = ((y + bev_range/2) / bev_range * grid_size[1]).astype(int)
        #projection to grid
        # Fill grid
        for i in range(len(grid_x)):
            gx, gy = grid_x[i], grid_y[i]
            if 0 <= gx < grid_size[0] and 0 <= gy < grid_size[1]:
                bev_map[gy, gx] = 1.0
        
    
        return torch.tensor(bev_map, dtype=torch.float32).unsqueeze(0),x,y