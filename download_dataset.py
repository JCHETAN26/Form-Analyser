#!/usr/bin/env python3
"""
Download a curated subset of Fitness-AQA videos for testing.
Focuses on videos with labeled form errors for validation.
"""

import json
import os
import subprocess
from pathlib import Path

# Configuration
DATASET_DIR = Path("Fitness-AQA-Datasets/Squat/Labeled_Dataset")
DOWNLOAD_DIR = Path("Fitness-AQA-Datasets/downloaded_videos")
DOWNLOAD_DIR.mkdir(exist_ok=True, parents=True)

# Load error labels
with open(DATASET_DIR / "Labels/error_knees_inward.json") as f:
    knees_inward = json.load(f)

with open(DATASET_DIR / "Shallow_Squat_Error_Dataset/labels_shallow_depth.json") as f:
    shallow_depth = json.load(f)

# Select videos with clear errors (non-empty timestamp ranges)
def has_errors(video_id, error_dict):
    """Check if video has labeled error timestamps"""
    if video_id not in error_dict:
        return False
    val = error_dict[video_id]
    # Handle both list and int types
    if isinstance(val, list):
        return len(val) > 0
    return val > 0  # If it's an int/score

# Curate a test set
test_videos = []

# Get 5 videos with knees inward errors
for vid_id, timestamps in list(knees_inward.items())[:100]:
    if has_errors(vid_id, knees_inward):
        test_videos.append({
            'id': vid_id,
            'error_type': 'knees_inward',
            'timestamps': timestamps
        })
        if len([v for v in test_videos if v['error_type'] == 'knees_inward']) >= 5:
            break

# Get 5 videos with shallow depth errors  
for vid_id, timestamps in list(shallow_depth.items())[:100]:
    if has_errors(vid_id, shallow_depth) and vid_id not in [v['id'] for v in test_videos]:
        test_videos.append({
            'id': vid_id,
            'error_type': 'shallow_depth',
            'timestamps': timestamps
        })
        if len([v for v in test_videos if v['error_type'] == 'shallow_depth']) >= 5:
            break

print(f"📋 Selected {len(test_videos)} videos for download:\n")
for v in test_videos:
    print(f"  {v['id']} - {v['error_type']} - Errors at: {v['timestamps']}")

# Download function
def download_video(video_info):
    """Download video from YouTube using yt-dlp"""
    vid_id = video_info['id']
    youtube_id = vid_id.split('_')[0]  # Extract YouTube ID (before underscore)
    clip_num = vid_id.split('_')[1] if '_' in vid_id else '1'
    
    output_path = DOWNLOAD_DIR / f"{vid_id}.mp4"
    
    if output_path.exists():
        print(f"✅ Already downloaded: {vid_id}")
        return True
    
    # yt-dlp command
    cmd = [
        'yt-dlp',
        '-f', 'best[height<=720]',  # Max 720p to save space
        '-o', str(output_path),
        f'https://www.youtube.com/watch?v={youtube_id}'
    ]
    
    try:
        print(f"⬇️  Downloading {vid_id}...")
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ Downloaded: {vid_id}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download {vid_id}: {e}")
        return False
    except FileNotFoundError:
        print("❌ yt-dlp not found. Install with: brew install yt-dlp")
        return False

# Check if yt-dlp is installed
try:
    subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
except FileNotFoundError:
    print("\n🚨 yt-dlp is not installed!")
    print("Install it with: brew install yt-dlp")
    print("Or: pip install yt-dlp")
    exit(1)

# Download all selected videos
print(f"\n🚀 Starting downloads to {DOWNLOAD_DIR}...\n")
success_count = 0
for video in test_videos:
    if download_video(video):
        success_count += 1

print(f"\n✅ Downloaded {success_count}/{len(test_videos)} videos successfully!")
print(f"📁 Videos saved to: {DOWNLOAD_DIR.absolute()}")

# Save metadata
metadata_path = DOWNLOAD_DIR / "test_set_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(test_videos, f, indent=2)
print(f"📝 Metadata saved to: {metadata_path}")
