#!/usr/bin/env python3
"""
Batch process test videos through the full Fitness-AQA pipeline.
Extracts 2D pose → Analyzes form → Generates reports
"""

import subprocess
import json
from pathlib import Path
import sys

# Directories
TEST_DIR = Path("test_videos")
OUTPUT_DIR = Path("test_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Test videos
videos = [
    # Squats
    {"file": "perfect_form_squat.mp4", "type": "squat", "expected": "Good form"},
    {"file": "common_mistakes.mp4", "type": "squat", "expected": "Form errors"},
    {"file": "form_breakdown.mp4", "type": "squat", "expected": "Mixed quality"},
    
    # Pull-ups
    {"file": "perfect_pullup.mp4", "type": "pullup", "expected": "Perfect ROM"},
    {"file": "pullup_mistakes.mp4", "type": "pullup", "expected": "Partial ROM, kipping"},
    {"file": "pullup_variations.mp4", "type": "pullup", "expected": "Multiple variations"}
]

def run_yolo_extraction(video_path, output_json):
    """Step 1: Extract 2D keypoints using YOLO"""
    print(f"  [1/3] Extracting 2D keypoints...")
    
    # Use the YOLO processor with correct arguments
    cmd = [
        sys.executable,
        "vision_pipeline/src/video_processor_yolo.py",
        "--input", str(video_path),
        "--output", str(output_json)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"  ✅ 2D extraction complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ YOLO extraction failed: {e.stderr}")
        return False

def run_form_analysis(video_path, json_path, output_video, exercise_type):
    """Step 2: Run biomechanical analysis"""
    print(f"  [2/3] Running form analysis...")
    
    cmd = [
        sys.executable,
        "vision_pipeline/src/analyze_form.py",
        "--video", str(video_path),
        "--json", str(json_path),
        "--type", exercise_type,
        "--output", str(output_video)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"  ✅ Analysis complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Analysis failed: {e.stderr}")
        return False

def summarize_results(report_path, tokens_path):
    """Step 3: Print summary"""
    print(f"  [3/3] Generating summary...")
    
    # Load report
    with open(report_path) as f:
        report = json.load(f)
    
    # Load tokens
    with open(tokens_path) as f:
        tokens = f.read()
    
    print(f"\n  📊 RESULTS:")
    print(f"     Reps: {report['rep_count']}")
    print(f"     Avg Jerk: {report['avg_jerk']:.2f}")
    print(f"     Avg Symmetry: {report['avg_symmetry']:.2f}")
    print(f"\n  🤖 FitnessGPT Tokens:")
    for line in tokens.split('\n')[1:6]:  # First 5 lines
        print(f"     {line}")
    
    return report

# Process all videos
print("🚀 Starting batch analysis pipeline...\n")
results = []

for video_info in videos:
    video_path = TEST_DIR / video_info['file']
    video_name = video_path.stem
    
    print(f"{'='*60}")
    print(f"📹 Processing: {video_name}")
    print(f"   Expected: {video_info['expected']}")
    print(f"{'='*60}\n")
    
    # Define output paths
    json_path = OUTPUT_DIR / f"{video_name}_keypoints.json"
    output_video = OUTPUT_DIR / f"{video_name}_analysis.mp4"
    report_path = OUTPUT_DIR / f"{video_name}_analysis_report.json"
    tokens_path = OUTPUT_DIR / f"{video_name}_analysis_tokens.txt"
    
    # Run pipeline
    if not run_yolo_extraction(video_path, json_path):
        continue
    
    if not run_form_analysis(video_path, json_path, output_video, video_info['type']):
        continue
    
    # Summarize
    report = summarize_results(report_path, tokens_path)
    results.append({
        'video': video_name,
        'expected': video_info['expected'],
        'reps': report['rep_count'],
        'jerk': report['avg_jerk'],
        'symmetry': report['avg_symmetry']
    })
    
    print(f"\n✅ {video_name} complete!\n")

# Final comparison
print(f"\n{'='*60}")
print("📊 FINAL COMPARISON")
print(f"{'='*60}\n")

print(f"{'Video':<25} {'Expected':<20} {'Reps':<6} {'Jerk':<12} {'Symmetry'}")
print("-" * 75)
for r in results:
    jerk_status = "STABLE" if r['jerk'] < 50000 else "SHAKY"
    print(f"{r['video']:<25} {r['expected']:<20} {r['reps']:<6} {r['jerk']:<8.0f} ({jerk_status[:3]})  {r['symmetry']:.2f}")

print(f"\n✅ All results saved to: {OUTPUT_DIR.absolute()}")
print(f"📹 Watch analyzed videos to see overlays!")
