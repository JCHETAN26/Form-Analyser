
import json
import cv2
import numpy as np
import argparse
import os

# Skeleton connections for COCO format (17 keypoints)
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), 
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), 
    (1, 3), (2, 4), (3, 5), (4, 6)
]

def visualize_pose(video_path, json_path, output_path):
    # 1. Load data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    keypoints_seq = np.array(data['smoothed_keypoints'])
    
    # 2. Setup video reader & writer
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"🎥 Generating visualization: {output_path}...")
    
    frame_idx = 0
    while cap.isOpened() and frame_idx < len(keypoints_seq):
        ret, frame = cap.read()
        if not ret: break
        
        kps = keypoints_seq[frame_idx]
        
        # Draw skeleton lines
        for pair in SKELETON:
            p1 = tuple(kps[pair[0]].astype(int))
            p2 = tuple(kps[pair[1]].astype(int))
            # Only draw if both points are non-zero (simple check for missing data)
            if p1 != (0,0) and p2 != (0,0):
                cv2.line(frame, p1, p2, (0, 255, 0), 2)
        
        # Draw keypoint dots
        for i, kp in enumerate(kps):
            pt = tuple(kp.astype(int))
            if pt != (0,0):
                cv2.circle(frame, pt, 4, (0, 0, 255), -1)
                # Optional: Label the index for debugging
                # cv2.putText(frame, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  Processed {frame_idx} frames...")

    cap.release()
    out.release()
    print(f"✅ Visualization saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", "-v", required=True, help="Original video file")
    parser.add_argument("--json", "-j", required=True, help="Processed JSON file")
    parser.add_argument("--output", "-o", default="video_viz.mp4", help="Output file name")
    
    args = parser.parse_args()
    visualize_pose(args.video, args.json, args.output)
