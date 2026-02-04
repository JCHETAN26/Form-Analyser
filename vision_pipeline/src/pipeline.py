import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer
import shutil
import os

class VisionPipeline:
    def __init__(self, output_dir="output_data"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Load the SOTA model (RTMPose-Large)
        print("Loading AI Models...")
        self.inferencer = MMPoseInferencer('human')
    
    def process_video(self, video_path):
        """
        Runs the full pipeline on a single video.
        """
        print(f"Processing: {video_path}")
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 1. Run Inference (Video -> Raw Keypoints)
        # returns variable is a generator
        result_generator = self.inferencer(video_path, return_vis=True)
        
        all_keypoints = []
        all_scores = []
        
        # 2. Extract Data Frame-by-Frame
        for i, result in enumerate(result_generator):
            # result is a dictionary containing 'predictions' and 'visualization'
            predictions = result['predictions']
            
            # Assuming single person for now (index 0)
            if predictions and len(predictions) > 0:
                keypoints = predictions[0]['keypoints'] # List of [x, y] coordinates
                scores = predictions[0]['keypoint_scores']
                
                all_keypoints.append(keypoints)
                all_scores.append(scores)
            else:
                # Handle missing detection (interpolation needed later)
                all_keypoints.append([]) 
                all_scores.append([])

        # 3. Save Raw Signal (The "handoff" to Data Team)
        save_path = os.path.join(self.output_dir, f"{video_name}_raw.npy")
        np.save(save_path, {"keypoints": all_keypoints, "scores": all_scores})
        print(f"✅ Extracted signals saved to: {save_path}")
        
        return save_path

if __name__ == "__main__":
    # Example Usage
    pipeline = VisionPipeline()
    # pipeline.process_video("sample_squat.mp4")
    print("Pipeline initialized. Ready to process.")
