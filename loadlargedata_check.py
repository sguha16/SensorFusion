# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 21:32:20 2026

@author: sanhi
"""

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud
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

# Initialize nuScenes with trainval
DATAROOT = r"C:\Users\sanhi\Downloads"
nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=True)

print(f"Loaded {len(nusc.scene)} scenes")
print(f"Total samples: {len(nusc.sample)}")