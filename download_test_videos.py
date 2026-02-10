#!/usr/bin/env python3
"""
Download high-quality squat form videos from YouTube for testing.
These are publicly available fitness tutorial videos.
"""

import subprocess
from pathlib import Path

DOWNLOAD_DIR = Path("test_videos")
DOWNLOAD_DIR.mkdir(exist_ok=True)

# Curated list of high-quality squat videos with known characteristics
test_videos = [
    {
        'url': 'https://www.youtube.com/watch?v=ultWZbUMPL8',
        'name': 'perfect_form_squat',
        'description': 'Perfect form demonstration (720p+)',
        'expected': 'Good depth, neutral spine, knees tracking toes'
    },
    {
        'url': 'https://www.youtube.com/watch?v=YaXPRqUwItQ',
        'name': 'common_mistakes',
        'description': 'Common squat mistakes tutorial',
        'expected': 'Shows knees caving, shallow depth, forward lean'
    },
    {
        'url': 'https://www.youtube.com/watch?v=gcNh17Ckjgg',
        'name': 'form_breakdown',
        'description': 'Squat form analysis',
        'expected': 'Multiple angles, detailed breakdown'
    }
]

def download_video(video_info):
    """Download video clip using yt-dlp"""
    output_path = DOWNLOAD_DIR / f"{video_info['name']}.mp4"
    
    if output_path.exists():
        print(f"✅ Already exists: {video_info['name']}")
        return True
    
    cmd = [
        'yt-dlp',
        '-f', 'best[height<=720][ext=mp4]',
        '--download-sections', '*0-30',  # First 30 seconds only
        '-o', str(output_path),
        video_info['url']
    ]
    
    try:
        print(f"⬇️  Downloading: {video_info['description']}...")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Downloaded: {video_info['name']}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e.stderr}")
        return False

print("🎥 Downloading test videos for form analysis validation...\n")

for video in test_videos:
    print(f"\n📹 {video['description']}")
    print(f"   Expected: {video['expected']}")
    download_video(video)

print(f"\n✅ Videos saved to: {DOWNLOAD_DIR.absolute()}")
print("\n🚀 Next step: Run your analyzer on these videos to validate metrics!")
