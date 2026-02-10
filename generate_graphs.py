#!/usr/bin/env python3
"""
Generate biomechanical graphs for Vishal.
Shows jerk, symmetry, and phase over time.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_biomechanics(report_path, output_path):
    """Generate comprehensive biomechanical graphs"""
    
    # Load data
    with open(report_path) as f:
        data = json.load(f)
    
    frames = [f['frame_idx'] for f in data['frame_analysis']]
    jerk = [f['jerk'] for f in data['frame_analysis']]
    symmetry = [f['symmetry_score'] for f in data['frame_analysis']]
    phases = [f['phase'] for f in data['frame_analysis']]
    
    # Convert phases to numeric for plotting
    phase_map = {'eccentric': 1, 'concentric': 2, 'isometric': 0}
    phase_numeric = [phase_map.get(p, 0) for p in phases]
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f'Biomechanical Analysis - {Path(report_path).stem}', fontsize=16, fontweight='bold')
    
    # Plot 1: Jerk (Smoothness)
    ax1.plot(frames, jerk, color='#FF6B6B', linewidth=1.5, alpha=0.8)
    ax1.axhline(y=data['avg_jerk'], color='red', linestyle='--', label=f"Avg: {data['avg_jerk']:.0f}")
    ax1.fill_between(frames, jerk, alpha=0.3, color='#FF6B6B')
    ax1.set_ylabel('Jerk (Smoothness)', fontsize=12, fontweight='bold')
    ax1.set_title('Movement Stability (Lower = Better)', fontsize=11)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(frames))
    
    # Plot 2: Symmetry
    ax2.plot(frames, symmetry, color='#4ECDC4', linewidth=1.5, alpha=0.8)
    ax2.axhline(y=data['avg_symmetry'], color='blue', linestyle='--', label=f"Avg: {data['avg_symmetry']:.2f}")
    ax2.axhline(y=0.9, color='green', linestyle=':', alpha=0.5, label='Good (>0.9)')
    ax2.axhline(y=0.8, color='orange', linestyle=':', alpha=0.5, label='Fair (>0.8)')
    ax2.fill_between(frames, symmetry, alpha=0.3, color='#4ECDC4')
    ax2.set_ylabel('Symmetry Score', fontsize=12, fontweight='bold')
    ax2.set_title('Left/Right Balance (Higher = Better)', fontsize=11)
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(frames))
    
    # Plot 3: Movement Phases
    ax3.fill_between(frames, phase_numeric, alpha=0.6, 
                     color=['#95E1D3' if p == 0 else '#F38181' if p == 1 else '#AA96DA' for p in phase_numeric])
    ax3.set_ylabel('Phase', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Frame Number', fontsize=12, fontweight='bold')
    ax3.set_title('Movement Phases', fontsize=11)
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(['Isometric', 'Eccentric', 'Concentric'])
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.set_xlim(0, max(frames))
    
    # Add rep markers
    for peak in data['rep_peaks']:
        ax1.axvline(x=peak, color='black', linestyle=':', alpha=0.4, linewidth=1)
        ax2.axvline(x=peak, color='black', linestyle=':', alpha=0.4, linewidth=1)
        ax3.axvline(x=peak, color='black', linestyle=':', alpha=0.4, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved graph: {output_path}")
    plt.close()

# Generate graphs for all videos
results_dir = Path("test_results")
graphs_dir = Path("test_results/graphs")
graphs_dir.mkdir(exist_ok=True)

print("📊 Generating biomechanical graphs...\n")

for report_file in results_dir.glob("*_report.json"):
    video_name = report_file.stem.replace("_analysis_report", "")
    output_path = graphs_dir / f"{video_name}_graph.png"
    
    print(f"Processing: {video_name}")
    plot_biomechanics(report_file, output_path)

print(f"\n✅ All graphs saved to: {graphs_dir.absolute()}")
print("\n📸 Send these to Vishal!")
