
import os
import argparse
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from video_processor import PoseExtractor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_single_video(extractor, video_path, output_dir, visualize=False):
    """
    Wrapper to process a single video and save it to the output directory.
    """
    try:
        video_name = os.path.basename(video_path)
        base_name = os.path.splitext(video_name)[0]
        output_path = os.path.join(output_dir, f"{base_name}.json")
        
        if os.path.exists(output_path):
            logger.info(f"Skipping {video_name}, output already exists.")
            return
            
        extractor.process_video(video_path, output_path, visualize)
        logger.info(f"Successfully processed {video_name}")
        
    except Exception as e:
        logger.error(f"Failed to process {video_path}: {e}")

def batch_process(input_dir, output_dir, extensions=['.mp4', '.mov', '.avi'], visualize=False, workers=1):
    """
    Scans input_dir for videos and processes them.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        
    logger.info(f"Found {len(video_files)} videos in {input_dir}")
    
    # Initialize Extractor (Done once to load model)
    # Note: MMPose might not be thread-safe for inference if using GPU. 
    # If using CPU, threading might work but multiprocessing is safer.
    # For simplicity, we initialize one extractor and run sequentially or use Threading with caution.
    # RTMPose is fast, so sequential might be fine for small batches.
    
    extractor = PoseExtractor()
    
    if workers > 1:
        # Warning: MMPose on CUDA isn't thread-safe usually. 
        logger.warning("Using multiple workers. Ensure your environment supports threaded inference.")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_single_video, extractor, v, output_dir, visualize) for v in video_files]
            for future in as_completed(futures):
                pass
    else:
        for video_path in video_files:
            process_single_video(extractor, video_path, output_dir, visualize)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Fitness-AQA Vision Pipeline")
    parser.add_argument("--input_dir", "-i", required=True, help="Directory containing video files")
    parser.add_argument("--output_dir", "-o", required=True, help="Directory to save JSON output")
    parser.add_argument("--visualize", "-v", action="store_true", help="Generate visualization videos")
    parser.add_argument("--workers", "-w", type=int, default=1, help="Number of parallel workers (default: 1)")
    
    args = parser.parse_args()
    
    batch_process(args.input_dir, args.output_dir, visualize=args.visualize, workers=args.workers)
