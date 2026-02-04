# Computer Vision Pipeline Instructions

## 📦 Setup
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   # You might need to install mmcv/mmpose using 'mim' for better luck:
   pip install openmim
   mim install mmengine "mmcv>=2.0.0" "mmdet>=3.0.0" "mmpose>=1.0.0"
   ```

## 🚀 Converting Video to Signal
This pipeline is designed to extract **clean, smoothed, normalized** skeleton data from videos.

### Single Video
Use this to test on one file.
```bash
python src/video_processor.py --input path/to/video.mp4 --output result.json
```
*   **Check:** Open `result.json` to see `keypoints` (raw) vs `smoothed_keypoints`.

### Batch Processing (Whole Dataset)
Use this to process a folder of videos.
```bash
python src/batch_runner.py --input_dir /path/to/videos --output_dir /path/to/save/json
```

## 📊 Data Format (The Handoff)
The output JSON contains:
- `video_id`: Filename
- `smoothed_keypoints`: **[Important]** This is what Vishal should use for training.
- `normalized_keypoints`: **[Important]** Use this if valid multi-person scale invariance is needed.
- `scores`: Confidence scores (0 to 1) for each keypoint.

## 🛠 Features
- **Smoothing:** Applies Savitzky-Golay filter to remove camera jitter.
- **Normalization:** Scales data so 1 unit = Torso Length.
- **Occlusion Handling:** Uses RTMPose-Large (SOTA) for robust detection.
