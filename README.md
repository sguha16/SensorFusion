# Quality-Aware Radar-Camera Fusion for ADAS

An implementation of adaptive sensor fusion that dynamically adjusts trust between radar and camera sensors based on real-time camera quality assessment.

## Overview

This project explores quality-aware multi-modal sensor fusion for autonomous driving perception. Objectives:

1. **Predict camera quality in real-time** (0.0 = heavily degraded, 1.0 = clean)
2. **Adapt sensor trust dynamically** (CAMERA → BOTH → RADAR as quality degrades)
3. **Make driving decisions** based on closest detected obstacle
4. **Demonstrate graceful degradation** when camera input is compromised

Implemented using the nuScenes mini dataset (404 samples).
---

## System Architecture

### 1. Input Processing

**Radar (Front Sensor Only):**
- Raw point cloud from RADAR_FRONT
- Transformed to ego coordinate frame
- Projected to 128×128 Bird's Eye View (BEV) grid
- Range: ±100m lateral/forward
- **Limitation:** Sparse coverage (only front radar used for PoC)

**Camera (Simulated from LiDAR):**
- LiDAR points projected through camera FOV
- Back-projected to ego frame
- Creates dense 128×128 BEV representation
- **Note:** Camera BEV is created from LiDAR points in the Camera FoV, to simulate depth-augmented camera perception

**Noise Augmentation (Training):**
- Clean: quality = 1.0, no noise
- Slight: quality = 0.8, std = 3.0
- Medium: quality = 0.5, std = 8.0
- Heavy: quality = 0.0, std = 15.0

### 2. Neural Network Architecture

**Dual-Task CNN with Channel Attention:**

Input: Radar BEV (128×128×1) + Camera BEV (128×128×1)
                                            │
                    ┌─────────────────┴─────────────────┐
                    │                                              │
         ┌──────────▼──────────┐           ┌───────────▼──────────┐
         │  Radar Encoder      		  │           │  Camera Encoder             │
         │   Conv → Pool       		  │           │   Conv → Pool               │
         │   Conv → Pool       		  │           │   Conv → Pool               │
         └──────────┬──────────┘           └───────────┬──────────┘
                    │                                  │
                    │                     ┌────────────┴────────────┐
                    │                     │                                 │
                    └──────────┬──────────┘              ┌──────────▼──────────┐
                                                       		│  Quality Predictor  	    │
                    ┌──────────▼──────────┐              │   Conv → Conv       		│
                    │   Concatenation     		 │              │   Pool → FC         		│
                    │    (64 channels)   		 │              │   Sigmoid         		│
                    └──────────┬──────────┘              └──────────┬──────────┘
                                  │                                    		   │
                    ┌──────────▼──────────┐                         	   │
                    │  Channel Attention  		 │                         	   │
                    │  (learns to weight  		 │                         	   │
                    │   sensor features)  		 │                         	   │
                    └──────────┬──────────┘                         	   │
                                  │                                    		   │
                    ┌──────────▼──────────┐                         	   │
                    │   Fusion Decoder    		 │                         	   │
                    │ Upsample → Upsample 		 │                         	   │
                    └──────────┬──────────┘                         	   │
                               	  │                                  	       │
                    ┌──────────▼──────────┐              ┌──────────▼──────────┐
                    │   Output Heatmap    		 │              │  Quality Score      		 │
                    │     (128×128)       		 │              │      (0-1)          		 │
                    └─────────────────────┘              └─────────────────────┘

**Components:**

1. **Radar Encoder:** Conv(16) → Pool → Conv(32) → Pool
2. **Camera Encoder:** Conv(16) → Pool → Conv(32) → Pool  
3. **Channel Attention:** Learns to weight the 64 fused channels adaptively
4. **Decoder:** Conv(32) → Upsample → Conv(16) → Upsample → Conv(1)
5. **Quality Predictor:** Lightweight CNN → Sigmoid (predicts camera quality)

**Loss Function:**
```python
total_loss = heatmap_loss + quality_loss
heatmap_loss = MSE(predicted_heatmap, ground_truth_heatmap)
quality_loss = MSE(predicted_quality, actual_quality_label)
```

### 3. Object Detection (Peak Detection)

**Problem:** Heatmaps are continuous probability distributions, not discrete objects.

**Solution:** Find local maxima (peaks) in heatmap:

```python
from scipy.ndimage import maximum_filter

# Find peaks: pixels that are local maxima in 5×5 neighborhood
local_max = maximum_filter(heatmap, size=5) == heatmap
peaks = (heatmap > threshold) & local_max

# Each peak = one detected object
# Extract: location (row, col), intensity, distance
```

**Key Parameters:**
- Threshold varies with quality:
  - Quality > 0.7 (trust camera): threshold = 0.4
  - Quality < 0.5 (trust radar): threshold = 0.3
  - Quality 0.5-0.7 (trust both): threshold = 0.35

**Distance Calculation:**
```python
# BEV grid: row 64 = ego position
# Smaller row = further forward, larger row = behind
forward_distance = (col - 64) * (200 / 128)  # meters
lateral_position = (row - 64) * (200 / 128)  # meters
```

**Handling Multiple Peaks:**
- Sort by (distance, -intensity): closest first, then highest intensity
- Ensures closest high-confidence object is selected for decision

### 4. Decision Making

**Trust Assignment:**
```python
if camera_quality < 0.5:
    sensor_trust = "RADAR"
elif camera_quality > 0.7:
    sensor_trust = "CAMERA"
else:
    sensor_trust = "BOTH"
```

**Driving Logic:**
```python
closest_object = find_closest_peak(heatmap, quality)

if closest_object.distance < 5m and intensity > threshold:
    action = "HARD_BRAKE" (100%)
elif closest_object.distance < 15m and intensity > threshold * 0.8:
    action = "SOFT_BRAKE" (scaled by distance)
else:
    action = "CONTINUE" (0%)
```

**Visualization:**
- Green circles: all detected objects (peaks)
- Magenta circle with black border: closest object (used for decision)

---

## Training Details

**Dataset:** nuScenes v1.0-mini
- Total samples: 404
- Train: 323 (80%)
- Validation: 81 (20%)
- Random split, seed=42

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Epochs: 50
- Batch size: 8
- Device: CUDA (if available)

**Data Augmentation:**
- 15% heavily degraded (quality=0.0, noise std=15)
- 15% moderately degraded (quality=0.5, noise std=8)  
- 15% slightly degraded (quality=0.8, noise std=3)
- 55% clean (quality=1.0, no noise)

**Results:**
- Final training loss: ~0.02-0.03
- Final validation loss: ~0.03-0.04
- Quality prediction: High accuracy (1.0 for clean, <0.2 for heavy noise)

---

## Project Structure

```
SensorFusion/
├── radar_camera_fusion_attentionmechanism_channelquality.py
│   └── Main training script with quality prediction
│
├── gif_radarfusion_channelqual_withGT.py
│   └── GIF generation for visualization
│
├── justcamera.py
│   └── Camera-only baseline
│
├── justradar.py
│   └── Radar-only baseline
│
├── results/
│   ├── CameraOnly 
│   ├── RadarOnly
│   ├── Fusion_AttentionwithChannelQuality
│
└── README.md
```

---

## How to Run

### Prerequisites

```bash
# Core dependencies
pip install torch torchvision numpy matplotlib scipy
pip install scikit-learn pillow

# nuScenes SDK
pip install nuscenes-devkit

# Download nuScenes mini dataset
# https://www.nuscenes.org/nuscenes#download
# Extract to: C:/Users/sanhi/Downloads/v1.0-mini (or update DATAROOT)
```

### Training

```python
python radar_camera_fusion_attentionmechanism_channelquality.py
```

**Outputs:**
- `fusion_model_with_quality.pth` - trained model weights
- Training/validation loss curves (printed)
- Feature visualizations
- Quality prediction accuracy plots

### Generate GIF Visualization

```python
python gif_radarfusion_channelqual_withGT.py
```

**Outputs:**
- `results/adaptive_fusion_driving.gif` (20 frames)
- Individual frames saved as `frame_XX_KEEP.png`

**GIF Structure (4 rows × 3 columns):**
- Row 1: Camera images (clean, medium degraded, heavy degraded)
- Row 2: BEV detection heatmaps with peak markers
- Row 3: Driving decisions (action, intensity, trust, distance)
- Row 4: Ground truth BEVs (Camera LiDAR, Radar)

### Baseline Comparisons

```python
# Camera-only
python justcamera.py

# Radar-only  
python justradar.py
```

---

## Key Results

### Quality Prediction Performance

| Noise Level (std) | Predicted Quality | Sensor Trust |
|-------------------|-------------------|--------------|
| 0                 | 1.000            | CAMERA       |
| 1                 | 0.998            | CAMERA       |
| 3                 | 0.787            | CAMERA       |
| 5                 | 0.456            | RADAR        |
| 8                 | 0.305            | RADAR        |
| 15                | 0.190            | RADAR        |
| 20                | 0.163            | RADAR        |
| 30                | 0.109            | RADAR        |

**Observation:** Clear threshold behavior around std=5, quality switches from CAMERA trust to RADAR trust.

### Example Scenarios (from GIF)

**Frame 1:**
- Clean: Trust CAMERA, obstacle 12.5m → SOFT_BRAKE 16%
- Medium: Trust BOTH, obstacle 6.2m → SOFT_BRAKE 58%
- Heavy: Trust RADAR, obstacle 12.5m → SOFT_BRAKE 16%

**Frame 8:**
- Clean: Trust CAMERA, obstacle 15.6m → CONTINUE
- Medium: Trust BOTH, obstacle 4.7m → HARD_BRAKE 100%
- Heavy: Trust RADAR, obstacle 4.7m → HARD_BRAKE 100%

**Frame 11:**
- Clean: Trust CAMERA, obstacle 20.3m → CONTINUE
- Medium: Trust BOTH, obstacle 14.1m → SOFT_BRAKE 6%
- Heavy: Trust RADAR, obstacle 17.2m → CONTINUE

---

## Known Limitations

### 1. Heatmap Prediction Robustness

**Issue:** Heavily degraded camera inputs (quality < 0.2) can produce structurally different heatmaps, not just noisier versions of clean predictions.

**Impact:** 
- False positive detections at incorrect distances
- Example: Clean camera detects truck at 10m, degraded detects phantom obstacle at 4.7m
- Causes inconsistent decisions for noisier scenarios

**Root Cause:**
- Limited training data (323 samples)
- Model learned structurally different heatmaps when the camera quality degraded

**Mitigation Strategies (Future Work):**
- Temporal filtering: track objects across frames to filter false positives
- Consistency loss: penalize heatmap structure changes between quality levels
- More training data (full nuScenes dataset: 28k samples)

### 2. Peak Detection Fragmentation

**Issue:** Large objects can produce multiple local maxima, fragmenting one object into several detections.

**Example:** A truck creates a broad high-intensity blob, but peak detection finds 3-5 separate peaks within it.

**Impact:**
- "Closest object" might be an edge peak, not the true object center
- Distance estimates can be off by 2-3 meters

**Better Approach:**
- Connected component labeling would treat the entire truck as ONE object

### 3. Sparse Radar Coverage

**Issue:** Only RADAR_FRONT sensor used; nuScenes has 5 radars total.

**Impact:**
- Large gaps in radar BEV (visible in ground truth plots)
- Real production systems fuse all 5 radars for 360° coverage

**Future Work:** Integrate RADAR_FRONT_LEFT, RADAR_FRONT_RIGHT, RADAR_BACK_LEFT, RADAR_BACK_RIGHT

### 4. Camera Proxy from LiDAR

**Issue:** Using LiDAR points projected through camera as "camera BEV" instead of actual image-based perception.

**Why:** Simplifies pipeline to focus on fusion logic, not computer vision.

**Real System:** Would use camera image → semantic segmentation → BEV projection.

### 5. Simple Decision Logic

**Current:** Distance-based thresholds (<5m, <15m) with linear intensity scaling.

**Production Would Have:**
- Time-to-collision (TTC) calculations
- Lateral position consideration (center lane vs. shoulder)
- Vehicle dynamics (speed, acceleration)
- Multi-object tracking with Kalman filters
- Trajectory prediction

### 6. No Temporal Filtering

**Issue:** Each frame processed independently; no memory of previous detections.

**Impact:** Object positions can "jump" between frames.

**Solution:** Implement object tracking (e.g., SORT, DeepSORT) to maintain object IDs and smooth trajectories.

---

## What Works Well

- **Quality prediction:** Accurately assesses camera degradation (0.0-1.0 scale)

- **Trust switching:** Logic correctly shifts reliance (CAMERA → BOTH → RADAR)

- **Visualization:** Clear demonstration of full pipeline (camera → heatmap → peaks → decision)

- **Graceful degradation:** System remains functional even with heavily corrupted camera input

- **Multi-task learning:** Single network handles both perception (heatmap) and quality assessment

---

## Design Decisions & Rationale

### Why Channel Attention?

**Problem:** How to weight radar vs. camera features during fusion?

**Solution:** Channel attention learns to dynamically emphasize certain feature channels based on input quality.

**Effect:** When camera quality is high, attention weights camera features more; when low, radar features dominate.

### Why Heatmaps Instead of Bounding Boxes?

**Reason:** 
1. Simpler ground truth generation (Gaussian blobs at annotation centers)
2. Naturally handles uncertainty (soft probabilities vs. hard boxes)
3. Works well with BEV representation
4. Easier to train with limited data

**Tradeoff:** Less precise than object detection, but sufficient for proof-of-concept.

### Why Peak Detection?

**Problem:** Heatmaps are continuous; need discrete object locations for decisions.

**Alternatives Considered:**
- Global maximum: Only finds one object (misses multi-object scenes)
- Simple thresholding: Doesn't separate nearby objects
- YOLO/Faster R-CNN: Requires bounding box labels, more complex

**Chosen:** Local maxima detection - simple, finds multiple objects, no additional labels needed.

### Why Quality-Based Thresholds?

**Insight:** Lower camera quality → trust radar more → use lower threshold for detections.

**Logic:**
- Clean camera (Q>0.7): High threshold (0.4) → only accept confident detections
- Degraded camera (Q<0.5): Low threshold (0.3) → accept weaker radar detections

This makes the system more conservative when sensors are unreliable.

---

## Comparison to Related Work

### BEVFusion (Liu et al., 2023)
- **Similarity:** Both use BEV representation for multi-modal fusion
- **Difference:** BEVFusion uses transformers; this uses CNNs with channel attention. BEVFusion doesn't adapt to sensor quality.

### CenterNet (Zhou et al., 2019)
- **Similarity:** Heatmap-based object detection with peak finding
- **Difference:** CenterNet is single-modality (image); this is multi-modal fusion with quality awareness

**Key Contribution:** Quality-aware dynamic trust switching for multi-modal fusion with different degradation levels.

---

## Future Work

### Model & Architecture
- **Integrate all 5 radars** for full 360° coverage (currently only front radar)
- **Replace LiDAR proxy with actual camera perception** (semantic segmentation → BEV)
- **Add consistency regularization** in training to enforce similar heatmap structures across quality levels
- **Expand to full nuScenes dataset** (28k samples vs current 404) for better generalization
- **Implement blob detection** instead of peak detection to handle large objects better

### Perception & Tracking
- **Add temporal filtering** (Kalman filter, SORT, or DeepSORT) for object tracking across frames
- **Trajectory prediction** to anticipate where objects will be in 1-2 seconds
- **Multi-object tracking** with consistent IDs to filter false positives

### Advanced Directions
- **Certifiable AI for ADAS** with explainable fallback logic (ISO 26262 compliance)
- **Curriculum learning approach** (train radar backbone first, add camera later)
- **Real-world sensor degradation testing** (sun glare, rain, fog, dirt on lens)

---
