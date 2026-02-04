
import os
import json
import logging
import argparse
import numpy as np
import cv2
from scipy.signal import savgol_filter
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOPoseExtractor:
    def __init__(self, model_variant='yolov8n-pose.pt', device='cpu'):
        """
        Initialize the YOLOv8 Pose model.
        Args:
            model_variant (str): 'yolov8n-pose.pt' (fast), 'yolov8s-pose.pt', or 'yolov8x-pose.pt' (accurate).
            device (str): 'cuda', 'cpu', or 'mps' (for Mac).
        """
        logger.info(f"Initializing YOLO Pose (model={model_variant}, device={device})...")
        self.model = YOLO(model_variant)
        self.device = device

    def smooth_signal(self, keypoints, window_length=5, polyorder=2):
        """Applies Savitzky-Golay filter to smooth trajectories."""
        if len(keypoints) < window_length:
            return keypoints
            
        smoothed_keypoints = np.zeros_like(keypoints)
        num_points = keypoints.shape[1]
        
        for i in range(num_points):
            smoothed_keypoints[:, i, 0] = savgol_filter(keypoints[:, i, 0], window_length, polyorder)
            smoothed_keypoints[:, i, 1] = savgol_filter(keypoints[:, i, 1], window_length, polyorder)
            
        return smoothed_keypoints

    def normalize_signal(self, keypoints):
        """Normalizes coordinates based on Torso Length."""
        normalized_keypoints = np.zeros_like(keypoints)
        
        for f in range(len(keypoints)):
            frame_kps = keypoints[f]
            # COCO Indices: 5,6 (shoulders), 11,12 (hips)
            l_shoulder, r_shoulder = frame_kps[5], frame_kps[6]
            l_hip, r_hip = frame_kps[11], frame_kps[12]
            
            mid_shoulder = (l_shoulder + r_shoulder) / 2
            mid_hip = (l_hip + r_hip) / 2
            torso_len = np.linalg.norm(mid_shoulder - mid_hip)
            
            scale = 1.0 if torso_len < 1e-3 else 1.0 / torso_len
            centered = frame_kps - mid_hip
            normalized_keypoints[f] = centered * scale
            
        return normalized_keypoints

    def process_video(self, video_path, output_path=None):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video {video_path} not found.")
            
        logger.info(f"Processing video: {video_path}")
        
        # Run YOLO inference
        # stream=True allows processing long videos frame by frame
        results = self.model(video_path, stream=True, device=self.device, verbose=False)
        
        raw_keypoints = []
        scores = []
        
        for result in results:
            if result.keypoints is not None and len(result.keypoints.data) > 0:
                # Get the first person detected [1, 17, 2]
                kp_data = result.keypoints.data[0].cpu().numpy()
                raw_keypoints.append(kp_data[:, :2])
                # Confidence scores
                scores.append(result.keypoints.conf[0].cpu().numpy())
            else:
                # Fallback for empty frame
                raw_keypoints.append(np.zeros((17, 2)))
                scores.append(np.zeros(17))

        raw_keypoints = np.array(raw_keypoints)
        scores = np.array(scores)
        
        logger.info(f"Frames processed: {len(raw_keypoints)}")
        
        # Pipeline Steps
        smoothed_keypoints = self.smooth_signal(raw_keypoints)
        normalized_keypoints = self.normalize_signal(smoothed_keypoints)
        
        data_packet = {
            "video_id": os.path.basename(video_path),
            "frame_count": len(raw_keypoints),
            "raw_keypoints": raw_keypoints.tolist(),
            "smoothed_keypoints": smoothed_keypoints.tolist(),
            "normalized_keypoints": normalized_keypoints.tolist(),
            "scores": scores.tolist()
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(data_packet, f)
            logger.info(f"Saved processed data to {output_path}")
            
        return data_packet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Pose Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input video")
    parser.add_argument("--output", "-o", required=True, help="Output JSON")
    parser.add_argument("--model", "-m", default="yolov8s-pose.pt", help="YOLO model variant")
    
    args = parser.parse_args()
    
    # Use 'mps' for Mac M1/M2, 'cuda' for PC, 'cpu' for default
    device = 'cpu'
    
    extractor = YOLOPoseExtractor(model_variant=args.model, device=device)
    extractor.process_video(args.input, args.output)
