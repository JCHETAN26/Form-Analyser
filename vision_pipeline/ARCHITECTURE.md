# Computer Vision Pipeline Plan

## Goal
Convert raw exercise videos into high-quality, smoothed, normalized skeleton trajectories for the Modeling team.

## Architecture

### 1. Ingestion Layer
- **Input:** MP4/Mov files.
- **Preprocessing:** Resizing, Frame Extraction.
- **Tech:** OpenCV (`cv2`)

### 2. Pose Estimation Core
- **Primary Engine:** MMPose (RTMPose-Large for accuracy).
- **Fallback/Prototype:** MediaPipe (Google).
- **Output:** 17 Keypoints (COCO format) or 133 Keypoints (WholeBody).

### 3. Signal Processing (The "Secret Sauce")
Raw keypoints are shaky. We must apply:
- **Confidence Gating:** Drop points with low prediction confidence.
- **Smoothing Filters:** Savitzky-Golay filter or OneEuro filter to remove jitter.
- **Normalization:** Scale coordinates relative to user height (Torso Length) to perform "Camera Normalization".

### 4. Output Layer
- **Format:** JSON / Numpy (.npy)
- **Structure:**
  ```json
  {
    "video_id": "squat_001",
    "fps": 30,
    "trajectory": [
      [x1, y1, c1, x2, y2, c2, ...], // Frame 1
      [x1, y1, c1, x2, y2, c2, ...]  // Frame 2
    ]
  }
  ```

## Implementation Steps

1. [ ] **Environment Setup**: Install Python 3.9+, PyTorch, OpenMIM, MMPose.
2. [ ] **Prototype**: Script to read video -> Draw Skeleton -> Save Video.
3. [ ] **Refinement**: Add Smoothing (SciPy).
4. [ ] **Pipeline**: Batch processing script for the dataset.
