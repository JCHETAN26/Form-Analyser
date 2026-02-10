#!/usr/bin/env python3
"""
Integration Script: CV Pipeline → FitnessGPT Training Data

Converts your biomechanical analysis outputs into Vishal's training format.
Takes your test_results/ JSON files and generates training data for FitnessGPT.

Author: Chetan (CV Lead) + Vishal (Modeling Lead)
"""

import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "FitnessAQA_analysis"))

from FitnessAQA_analysis.time_series_encoder import TimeSeriesEncoder

def convert_cv_output_to_training_data(results_dir="test_results", output_dir="FitnessAQA_analysis/outputs"):
    """
    Convert CV pipeline outputs to FitnessGPT training format.
    
    Input: Your *_keypoints.json and *_report.json files
    Output: Alpaca-format training data for LLM fine-tuning
    """
    
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    encoder = TimeSeriesEncoder()
    print(f"🤖 Initialized TimeSeriesEncoder (vocab size: {encoder.vocab_size})")
    
    training_examples = []
    
    # Process all keypoint files
    keypoint_files = list(results_path.glob("*_keypoints.json"))
    print(f"\n📊 Found {len(keypoint_files)} videos to process\n")
    
    for kp_file in keypoint_files:
        video_name = kp_file.stem.replace("_keypoints", "")
        report_file = results_path / f"{video_name}_analysis_report.json"
        tokens_file = results_path / f"{video_name}_analysis_tokens.txt"
        
        print(f"Processing: {video_name}")
        
        # Load your CV outputs
        with open(kp_file) as f:
            kp_data = json.load(f)
        
        if not report_file.exists():
            print(f"  ⚠️  No report file found, skipping")
            continue
            
        with open(report_file) as f:
            report_data = json.load(f)
        
        # Load your tokens (if available)
        your_tokens = ""
        if tokens_file.exists():
            with open(tokens_file) as f:
                your_tokens = f.read()
        
        # Encode using Vishal's encoder
        try:
            encoded = encoder.encode_from_json(str(kp_file), stride=5)
            
            # Build instruction-response pair for training
            instruction = f"Analyze this {video_name.split('_')[0]} exercise based on the biomechanical data."
            
            # Input: Vishal's encoded tokens
            input_data = f"""
BIOMECHANICAL ANALYSIS:
{encoded['summary_tokens']}

QUALITY ASSESSMENT:
{encoded['quality_tokens']}

YOUR ANALYSIS:
{your_tokens}
""".strip()
            
            # Output: Expert coaching feedback (you'll need to add this)
            # For now, we'll generate based on your metrics
            output_feedback = generate_coaching_feedback(report_data, video_name)
            
            training_examples.append({
                "instruction": instruction,
                "input": input_data,
                "output": output_feedback,
                "system": "You are FitnessGPT, an expert biomechanics coach. Provide precise, actionable feedback based on pose estimation data."
            })
            
            print(f"  ✅ Encoded successfully")
            print(f"     - Token count: {encoded['metadata']['token_count']}")
            print(f"     - Reps detected: {len(encoded['features']['reps'])}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    # Save training data in Alpaca format
    train_file = output_path / "train_fitness_alpaca.json"
    with open(train_file, 'w') as f:
        json.dump(training_examples, f, indent=2)
    
    print(f"\n✅ Generated {len(training_examples)} training examples")
    print(f"📁 Saved to: {train_file}")
    
    # Save vocabulary
    vocab_file = output_path / "vocab_integrated.json"
    encoder.save_vocab(str(vocab_file))
    print(f"📚 Vocabulary saved to: {vocab_file}")
    
    return training_examples

def generate_coaching_feedback(report_data, video_name):
    """Generate expert coaching feedback based on your analysis."""
    
    reps = report_data['rep_count']
    jerk = report_data['avg_jerk']
    symmetry = report_data['avg_symmetry']
    
    feedback = []
    
    # Rep count feedback
    feedback.append(f"**Repetitions:** Detected {reps} reps.")
    
    # Stability feedback
    if jerk < 50000:
        feedback.append("**Stability:** Excellent control! Your movement is smooth and controlled.")
    elif jerk < 500000:
        feedback.append(f"**Stability:** Moderate shakiness detected (jerk: {jerk:.0f}). Focus on slower, more controlled movements.")
    else:
        feedback.append(f"**Stability:** High instability detected (jerk: {jerk:.0f}). You're fighting the weight. Reduce load and focus on form.")
    
    # Symmetry feedback
    if symmetry > 0.9:
        feedback.append(f"**Balance:** Excellent symmetry ({symmetry:.2f}). Left and right sides are well-balanced.")
    elif symmetry > 0.8:
        feedback.append(f"**Balance:** Good symmetry ({symmetry:.2f}), but slight imbalance detected. Check your form in a mirror.")
    else:
        feedback.append(f"**Balance:** Asymmetry detected ({symmetry:.2f}). This could lead to injury. Focus on engaging both sides equally.")
    
    # Exercise-specific feedback
    if 'squat' in video_name.lower():
        feedback.append("\n**Squat-Specific Tips:**")
        feedback.append("- Keep knees tracking over toes")
        feedback.append("- Maintain neutral spine throughout")
        feedback.append("- Aim for depth below parallel")
    elif 'pullup' in video_name.lower():
        feedback.append("\n**Pull-up-Specific Tips:**")
        feedback.append("- Full extension at bottom (dead hang)")
        feedback.append("- Chin over bar at top")
        feedback.append("- Avoid kipping unless training for CrossFit")
    
    return "\n".join(feedback)

if __name__ == "__main__":
    print("="*60)
    print("  CV Pipeline → FitnessGPT Integration")
    print("="*60)
    
    examples = convert_cv_output_to_training_data()
    
    print("\n" + "="*60)
    print("  Sample Training Example")
    print("="*60)
    if examples:
        sample = examples[0]
        print(f"\n**Instruction:**\n{sample['instruction']}")
        print(f"\n**Input (first 500 chars):**\n{sample['input'][:500]}...")
        print(f"\n**Output:**\n{sample['output']}")
    
    print("\n🎯 Ready for Vishal to train FitnessGPT!")
