# Once u clone the repo videopose3d add this in commons visualization.py ( This probably breaks the visualize)

# def get_resolution(filename):
#     command = [
#         'ffprobe',
#         '-v', 'error',
#         '-select_streams', 'v:0',
#         '-show_entries', 'stream=width,height',
#         '-of', 'csv=p=0',
#         filename
#     ]

#     with sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE) as pipe:
#         for line in pipe.stdout:
#             parts = line.decode().strip().split(',')
#             if len(parts) >= 2:
#                 width = int(parts[0])
#                 height = int(parts[1])
#                 return width, height
#     raise RuntimeError(f"Could not determine resolution for video: {filename}")

            
# import subprocess as sp

# def get_fps(filename):
#     command = [
#         'ffprobe',
#         '-v', 'error',
#         '-select_streams', 'v:0',
#         '-show_entries', 'stream=r_frame_rate',
#         '-of', 'csv=p=0',
#         filename
#     ]
#     with sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE) as pipe:
#         for line in pipe.stdout:
#             value = line.decode().strip()
#             value = value.replace(',', '').strip()
#             if '/' in value:
#                 a, b = value.split('/')
#                 return float(a) / float(b)
#             else:
#                 return float(value)
#     raise RuntimeError(f"Could not determine FPS for video: {filename}")

import os
import sys
import numpy as np
import subprocess
from scipy.signal import savgol_filter
import cv2

sys.modules['sitecustomize'] = None

from mmpose.apis import MMPoseInferencer

# ---------------- CONFIG ----------------
VIDEO_PATH = "data/pull_ups.mp4"
VIDEOPOSE_ROOT = "VideoPose3D"
CHECKPOINT_FILENAME = "pretrained_h36m_detectron_coco.bin"

OUTPUT_DIR = "output"
DATASET_NAME = "custom_video"
NUM_JOINTS = 17

# ---------------- RUN 2D POSE ----------------
print(f"Running 2D pose estimation on {VIDEO_PATH}...")

inferencer = MMPoseInferencer(
    pose2d='td-hm_hrnet-w32_8xb64-210e_coco-256x192',
    device='cuda:0'
)

result_generator = inferencer(VIDEO_PATH, return_vis=False)

keypoints_2d = []

for result in result_generator:

    frame_preds = result.get('predictions', [])

    if len(frame_preds) == 0 or len(frame_preds[0]) == 0:
        keypoints_2d.append(
            np.full((NUM_JOINTS, 2), np.nan, dtype=np.float32)
        )
        continue

    persons = frame_preds[0]

    best_kp = None
    best_score = -np.inf

    for person in persons:
        kp = np.array(person['keypoints'], dtype=np.float32)
        scores = np.array(person['keypoint_scores'], dtype=np.float32)

        score = np.nanmean(scores)

        if score > best_score:
            best_score = score
            best_kp = kp[:, :2]

    keypoints_2d.append(best_kp[:NUM_JOINTS])

keypoints_2d = np.array(keypoints_2d, dtype=np.float32)
print(f"Extracted 2D keypoints for {keypoints_2d.shape[0]} frames.")

# ---------------- TEMPORAL CLEANING ----------------

for j in range(NUM_JOINTS):
    for d in range(2):
        col = keypoints_2d[:, j, d]
        if np.isnan(col).any():
            valid = ~np.isnan(col)
            if valid.sum() < 2:
                continue
            col[~valid] = np.interp(
                np.flatnonzero(~valid),
                np.flatnonzero(valid),
                col[valid]
            )

if keypoints_2d.shape[0] >= 7:
    keypoints_2d = savgol_filter(
        keypoints_2d,
        window_length=7,
        polyorder=3,
        axis=0
    )

# ---------------- SAVE FOR VIDEOPOSE3D ----------------

# 🔥 Get video resolution + fps
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

video_name = os.path.basename(VIDEO_PATH)

coco_metadata = {
    'layout_name': 'coco',
    'num_joints': NUM_JOINTS,
    'keypoints_symmetry': [
        [1, 3, 5, 7, 9, 11, 13, 15],
        [2, 4, 6, 8, 10, 12, 14, 16]
    ],
    'video_metadata': {
        video_name: {
            'w': width,
            'h': height,
            'fps': fps
        }
    }
}

positions_2d = {
    video_name: {
        'custom': [keypoints_2d]
    }
}

npz_path = os.path.join(
    VIDEOPOSE_ROOT,
    "data",
    f"data_2d_custom_{DATASET_NAME}.npz"
)

print(f"Saving 2D keypoints to {npz_path}")

np.savez_compressed(
    npz_path,
    positions_2d=positions_2d,
    metadata=coco_metadata
)

# ---------------- RUN VIDEOPOSE3D ----------------

print("Running VideoPose3D inference...")

output_video_path = os.path.abspath(
    os.path.join(OUTPUT_DIR, "output_3d.mp4")
)

input_video_abs_path = os.path.abspath(VIDEO_PATH)

cmd = [
    sys.executable, "run.py",
    "-d", "custom",
    "-k", DATASET_NAME,
    "-arc", "3,3,3,3,3",
    "-c", "checkpoints",
    "--evaluate", "../pretrained_h36m_detectron_coco.bin",
    "--render",
    "--viz-subject", video_name,
    "--viz-action", "custom",
    "--viz-video", input_video_abs_path,
    "--viz-output", output_video_path,
    "--viz-size", "6"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

subprocess.run(cmd, cwd=VIDEOPOSE_ROOT, check=True)

print(f"\nSuccess! 3D output saved to: {output_video_path}")
