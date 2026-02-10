#!/usr/bin/env python3
"""
Download pull-up test videos for validation.
"""

import subprocess
from pathlib import Path

DOWNLOAD_DIR = Path("test_videos")
DOWNLOAD_DIR.mkdir(exist_ok=True)

# Pull-up specific videos
pullup_videos = [
    {
        'url': 'https://www.youtube.com/watch?v=eGo4IYlbE5g',
        'name': 'perfect_pullup',
        'description': 'Perfect form pull-up tutorial',
        'expected': 'Full ROM, controlled tempo, no kipping'
    },
    {
        'url': 'https://www.youtube.com/watch?v=mRznU6pzez0',
        'name': 'pullup_mistakes',
        'description': 'Common pull-up form errors',
        'expected': 'Partial ROM, kipping, asymmetry'
    },
    {
        'url': 'https://www.youtube.com/watch?v=tB3X4TjTIes',
        'name': 'pullup_variations',
        'description': 'Pull-up technique analysis',
        'expected': 'Multiple angles and variations'
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
        '--download-sections', '*0-30',  # First 30 seconds
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

print("🏋️ Downloading pull-up test videos...\n")

for video in pullup_videos:
    print(f"\n📹 {video['description']}")
    print(f"   Expected: {video['expected']}")
    download_video(video)

print(f"\n✅ Videos saved to: {DOWNLOAD_DIR.absolute()}")
print("\n🚀 Next: Update batch_analyze.py to include these videos!")
