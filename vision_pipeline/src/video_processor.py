
import os
import json
import logging
import argparse
import numpy as np
import cv2
from scipy.signal import savgol_filter
from mmpose.apis import MMPoseInferencer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PoseExtractor:
    def __init__(self, mode='human', device='cpu'):
        """
        Initialize the MMPose Inferencer.
        Args:
            mode (str): 'human' uses RTMPose-Large by default.
            device (str): 'cuda' or 'cpu'.
        """
        logger.info(f"Initializing MMPoseInferencer (mode={mode}, device={device})...")
        self.inferencer = MMPoseInferencer(mode, device=device)

    def smooth_signal(self, keypoints, window_length=5, polyorder=2):
        """
        Applies Savitzky-Golay filter to smooth the keypoint trajectories.
        Keypoints shape: [Frames, Num_Points, 2]
        """
        logger.info("Applying Savitzky-Golay smoothing...")
        
        # We need at least 'window_length' frames to smooth
        if len(keypoints) < window_length:
            logger.warning(f"Not enough frames to smooth (got {len(keypoints)}, need {window_length}). Returning raw.")
            return keypoints
            
        smoothed_keypoints = np.zeros_like(keypoints)
        num_points = keypoints.shape[1]
        
        for i in range(num_points):
            # Smooth X coordinate
            smoothed_keypoints[:, i, 0] = savgol_filter(keypoints[:, i, 0], window_length, polyorder)
            # Smooth Y coordinate
            smoothed_keypoints[:, i, 1] = savgol_filter(keypoints[:, i, 1], window_length, polyorder)
            
        return smoothed_keypoints

    def normalize_signal(self, keypoints):
        """
        Normalizes coordinates based on Torso Length (Hip-to-Shoulder).
        Assuming COCO format:
        - Left Shoulder: 5
        - Right Shoulder: 6
        - Left Hip: 11
        - Right Hip: 12
        """
        logger.info("Normalizing signal based on torso length...")
        normalized_keypoints = np.zeros_like(keypoints)
        
        # Calculate Torso Length for every frame
        # Mid-Shoulder = (LeftShoulder + RightShoulder) / 2
        # Mid-Hip = (LeftHip + RightHip) / 2
        # Torso Length = Dist(Mid-Shoulder, Mid-Hip)
        
        for f in range(len(keypoints)):
            frame_kps = keypoints[f]
            
            # Check if we have confidence (sometimes points are 0,0)
            # If standard COCO, indices are stable.
            l_shoulder = frame_kps[5]
            r_shoulder = frame_kps[6]
            l_hip = frame_kps[11]
            r_hip = frame_kps[12]
            
            # Calculate Midpoints
            mid_shoulder = (l_shoulder + r_shoulder) / 2
            mid_hip = (l_hip + r_hip) / 2
            
            # Calculate Torso Length (Euclidean distance)
            torso_len = np.linalg.norm(mid_shoulder - mid_hip)
            
            # If torso length is near zero (failed detection), avoid div/0
            if torso_len < 1e-3:
                scale = 1.0
            else:
                scale = 1.0 / torso_len
            
            # Normalize: Scale everything relative to torso
            # Optional: Center the hip at (0,0)? For now, just scaling.
            # Using Mid-Hip as origin for centering could also be useful.
            
            # Implementation: Center at Mid-Hip, then Scale.
            centered = frame_kps - mid_hip
            normalized_keypoints[f] = centered * scale
            
        return normalized_keypoints

    def process_video(self, video_path, output_path=None, visualize=False):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video {video_path} not found.")
            
        logger.info(f"Processing video: {video_path}")
        
        # Generators allow processing long videos without OOM
        result_generator = self.inferencer(video_path, return_vis=visualize)
        
        raw_keypoints = []
        scores = []
        
        for result in result_generator:
            preds = result['predictions']
            if preds and len(preds) > 0:
                # Take the first detected person (Index 0)
                # Structure: [{'keypoints': [[x,y], ...], 'keypoint_scores': [...]}]
                raw_keypoints.append(preds[0]['keypoints'])
                scores.append(preds[0]['keypoint_scores'])
            else:
                # Fallback for empty frame
                # Assume COCO 17 points
                raw_keypoints.append(np.zeros((17, 2)))
                scores.append(np.zeros(17))

        raw_keypoints = np.array(raw_keypoints) # Shape: (Frames, 17, 2)
        scores = np.array(scores)
        
        # Pipeline Steps
        logger.info(f"Raw data shape: {raw_keypoints.shape}")
        
        # 1. Smoothing
        smoothed_keypoints = self.smooth_signal(raw_keypoints)
        
        # 2. Normalization
        normalized_keypoints = self.normalize_signal(smoothed_keypoints)
        
        # 3. Serialization
        data_packet = {
            "video_id": os.path.basename(video_path),
            "frame_count": len(raw_keypoints),
            "raw_keypoints": raw_keypoints.tolist(),
            "smoothed_keypoints": smoothed_keypoints.tolist(), # Optional: keep raw for debug
            "normalized_keypoints": normalized_keypoints.tolist(),
            "scores": scores.tolist()
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(data_packet, f)
            logger.info(f"Saved processed data to {output_path}")
            
        return data_packet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fitness-AQA Vision Pipeline Processor")
    parser.add_argument("--input", "-i", required=True, help="Path to input video")
    parser.add_argument("--output", "-o", required=True, help="Path to output JSON")
    parser.add_argument("--visualize", "-v", action="store_true", help="Generate visualization video (slow)")
    
    args = parser.parse_args()
    
    extractor = PoseExtractor()
    extractor.process_video(args.input, args.output, args.visualize)
