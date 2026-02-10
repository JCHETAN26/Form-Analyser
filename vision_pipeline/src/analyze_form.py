import json
import os
import argparse
import sys

# Ensure we can find the src modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analyzer import FormAnalyzer
from visualizer import visualize_pose
from tokenizer import SignalTokenizer

def main():
    parser = argparse.ArgumentParser(description="Fitness Form Analysis Pipeline")
    parser.add_argument("--video", "-v", required=True, help="Path to input video")
    parser.add_argument("--json", "-j", required=True, help="Path to processed keypoints JSON")
    parser.add_argument("--type", "-t", default="pullup", choices=["pullup", "squat"], help="Exercise type")
    parser.add_argument("--output", "-o", default="analysis_output.mp4", help="Path to output video")
    
    args = parser.parse_args()
    
    # 1. Load the keypoints
    print(f"📂 Loading data from {args.json}...")
    with open(args.json, 'r') as f:
        data = json.load(f)
    
    # Check if we have smoothed_keypoints
    if 'smoothed_keypoints' in data:
        keypoints_seq = data['smoothed_keypoints']
    elif 'keypoints' in data:
        keypoints_seq = data['keypoints']
    else:
        print("❌ Error: JSON must contain 'smoothed_keypoints' or 'keypoints'")
        return

    import numpy as np
    keypoints_seq = np.array(keypoints_seq)
    
    # 2. Run Analysis
    print(f"🧠 Analyzing {args.type} form...")
    analyzer = FormAnalyzer()
    analysis_results = analyzer.process_sequence(keypoints_seq, exercise_type=args.type)
    
    # Generate Tokens for GPT
    tokenizer = SignalTokenizer()
    gpt_summary = tokenizer.tokenize_analysis(analysis_results)
    
    print(f"📊 Summary: Detected {analysis_results['rep_count']} repetitions.")
    print("\n--- FitnessGPT Tokens ---")
    print(gpt_summary)
    print("--------------------------\n")
    
    # 3. Generate Visualization
    print("🎨 Rendering visualization with analytics...")
    visualize_pose(args.video, args.json, args.output, analysis_data=analysis_results)
    
    # 4. Save analysis report
    report_path = args.output.replace(".mp4", "_report.json")
    with open(report_path, 'w') as f:
        json.dump(analysis_results, f, indent=4)
        
    token_path = args.output.replace(".mp4", "_tokens.txt")
    with open(token_path, 'w') as f:
        f.write(gpt_summary)
        
    print(f"📝 Full report saved to {report_path}")
    print(f"📝 GPT Tokens saved to {token_path}")

if __name__ == "__main__":
    main()
