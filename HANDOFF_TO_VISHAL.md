# Vision Pipeline → Modeling Handoff Guide

## 📦 What You're Getting

Hi Vishal! Here's the output from the Computer Vision pipeline. This README explains exactly what data you're receiving and how to use it for your models.

---

## 📊 Output Format: JSON Structure

When you run the vision pipeline on a video, you'll get a `.json` file with this structure:

```json
{
  "video_id": "pullup.mp4",
  "frame_count": 150,
  "raw_keypoints": [...],
  "smoothed_keypoints": [...],
  "normalized_keypoints": [...],
  "scores": [...]
}
```

### Field Descriptions

| Field | Shape | Description | Use This For |
|-------|-------|-------------|--------------|
| `video_id` | String | Original video filename | Metadata tracking |
| `frame_count` | Integer | Total frames processed | Sequence length |
| **`smoothed_keypoints`** | `[T, 17, 2]` | **Smoothed X,Y coordinates** | **Primary input for your model** |
| `normalized_keypoints` | `[T, 17, 2]` | Scale-invariant coordinates | Advanced: Person-height normalization |
| `raw_keypoints` | `[T, 17, 2]` | Direct MMPose output | Debugging only |
| `scores` | `[T, 17]` | Confidence per keypoint | Filter low-confidence frames |

**Legend:**
- `T` = Number of frames (varies per video)
- `17` = COCO keypoints (see below)
- `2` = (X, Y) pixel coordinates

---

## 🦴 Keypoint Indices (COCO Format)

```
0:  Nose
1:  Left Eye
2:  Right Eye
3:  Left Ear
4:  Right Ear
5:  Left Shoulder
6:  Right Shoulder
7:  Left Elbow
8:  Right Elbow
9:  Left Wrist
10: Right Wrist
11: Left Hip
12: Right Hip
13: Left Knee
14: Right Knee
15: Left Ankle
16: Right Ankle
```

**Example:** To get the **left elbow position** at frame 10:
```python
import json
data = json.load(open('pullup_analysis.json'))
left_elbow = data['smoothed_keypoints'][10][7]  # [x, y]
```

---

## 🧮 Signal Processing Applied

The CV pipeline applies these transformations (in order):

### 1. **Savitzky-Golay Smoothing**
- **Why:** Camera jitter causes raw coordinates to "shake."
- **Settings:** `window_length=5`, `polyorder=2`
- **Effect:** Removes high-frequency noise while preserving motion.

### 2. **Normalization (Optional)**
- **Why:** Makes the model scale-invariant (e.g., works for tall/short people).
- **Method:** 
  - Calculates torso length = distance(mid_shoulder, mid_hip)
  - Scales all points by `1/torso_length`
  - Centers origin at `mid_hip`
- **Use Case:** If training on mixed-height subjects.

**Recommendation:** Start with `smoothed_keypoints`. Use `normalized_keypoints` only if you notice the model is biased toward certain body types.

---

## 🚀 How to Run the Pipeline Yourself

### Step 1: Clone the Repo
```bash
git clone https://github.com/JCHETAN26/Form-Analyser.git
cd Form-Analyser
```

### Step 2: Setup Environment
```bash
conda create -n mmpose_env python=3.8 -y
conda activate mmpose_env
pip install torch torchvision
pip install openmim
mim install mmengine "mmcv>=2.0.0" "mmdet>=3.0.0" "mmpose>=1.0.0"
pip install scipy opencv-python numpy
```

### Step 3: Process a Video
```bash
python vision_pipeline/src/video_processor.py \
  --input path/to/your_video.mp4 \
  --output your_video_analysis.json
```

### Step 4: (Optional) Batch Process
```bash
python vision_pipeline/src/batch_runner.py \
  --input_dir /path/to/video/folder \
  --output_dir /path/to/save/jsons
```

---

## 📁 Files to Review

| File | Purpose |
|------|---------|
| `vision_pipeline/src/video_processor.py` | Core pipeline logic |
| `vision_pipeline/README.md` | Pipeline usage instructions |
| `videopose3d2d.py` | Your original 3D script (restored) |
| `pullup_analysis.json` | **Example output** (once generated) |

---

## 🤝 Data Contract Agreement

To avoid integration headaches later, let's agree on:

1. **Input to your model:** `smoothed_keypoints` array
2. **Shape:** `[num_frames, 17, 2]`
3. **Data type:** `float32`
4. **Coordinate system:** Pixel space (unless you prefer normalized)

If you need a different format (e.g., `.npy` instead of `.json`, or flattened vectors), let me know and I can adjust the pipeline output.

---

## 🐛 Troubleshooting

### Q: What if some frames have missing detections?
**A:** Check the `scores` array. Any keypoint with `score < 0.3` is unreliable. The pipeline currently fills gaps with interpolation, but you can add custom logic.

### Q: Can I get 3D coordinates instead of 2D?
**A:** Yes! Use `videopose3d2d.py` as a starting point. It converts the 2D output into VideoPose3D format (`.npz`).

### Q: The coordinates look "shaky" even after smoothing.
**A:** Try increasing the smoothing window in `video_processor.py`:
```python
smoothed_keypoints = self.smooth_signal(keypoints, window_length=11)
```

---

## 📞 Contact

If you run into issues or need a different data format, ping me on Slack or create a GitHub Issue on the repo.

**Happy Modeling!**  
— Chetan (CV Lead)
