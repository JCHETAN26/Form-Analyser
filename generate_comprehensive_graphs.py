#!/usr/bin/env python3
"""
Generate Vishal-style comprehensive biomechanical analysis graphs.
Creates 4-panel visualization: Vertical Pull, Horizontal Swing, Velocity, Jerk
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def create_comprehensive_graph(report_path, keypoints_path, output_path):
    """Generate 4-panel biomechanical analysis graph"""
    
    # Load data
    with open(report_path) as f:
        report = json.load(f)
    
    with open(keypoints_path) as f:
        kp_data = json.load(f)
    
    # Extract data
    frames = [f['frame_idx'] for f in report['frame_analysis']]
    jerk = [f['jerk'] for f in report['frame_analysis']]
    symmetry = [f['symmetry_score'] for f in report['frame_analysis']]
    phases = [f['phase'] for f in report['frame_analysis']]
    
    # Load keypoints for displacement calculation
    keypoints = np.array(kp_data['smoothed_keypoints'])
    
    # Calculate vertical displacement (shoulder center - wrist center)
    shoulder_center = (keypoints[:, 5, :] + keypoints[:, 6, :]) / 2
    wrist_center = (keypoints[:, 9, :] + keypoints[:, 10, :]) / 2
    hip_center = (keypoints[:, 11, :] + keypoints[:, 12, :]) / 2
    
    # Vertical pull (normalized)
    vert_disp = shoulder_center[:, 1] - wrist_center[:, 1]
    vert_disp_norm = (vert_disp - np.min(vert_disp)) / (np.max(vert_disp) - np.min(vert_disp) + 1e-8)
    
    # Horizontal swing (wasted energy)
    horiz_disp = hip_center[:, 0]
    horiz_disp_norm = horiz_disp - np.mean(horiz_disp)
    
    # Velocity components
    vert_vel = np.gradient(vert_disp_norm)
    horiz_vel = np.gradient(horiz_disp_norm)
    
    # Smooth jerk for visualization
    from scipy.signal import savgol_filter
    jerk_smooth = savgol_filter(jerk, min(51, len(jerk)), 3)
    
    # Calculate efficiency
    vert_energy = np.sum(vert_vel ** 2)
    horiz_energy = np.sum(horiz_vel ** 2)
    jerk_energy = np.sum(np.array(jerk) ** 2)
    
    total_wasted = horiz_energy + 0.1 * jerk_energy
    efficiency_db = 10 * np.log10(vert_energy / (total_wasted + 1e-8)) if total_wasted > 0 else 0
    efficiency_pct = (vert_energy / (vert_energy + total_wasted)) * 100 if (vert_energy + total_wasted) > 0 else 0
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('black')
    
    title = f"Pull-Up Biomechanics | Efficiency: {efficiency_db:.1f} dB ({efficiency_pct:.1f}%)"
    fig.suptitle(title, fontsize=16, fontweight='bold', color='white', y=0.98)
    
    # Panel 1: Vertical Pull (Useful Work)
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_facecolor('black')
    ax1.plot(frames, vert_disp_norm, color='#00FF00', linewidth=2, alpha=0.9)
    ax1.fill_between(frames, vert_disp_norm, alpha=0.3, color='#00FF00')
    ax1.set_ylabel('Displacement (body units)', fontsize=11, color='white', fontweight='bold')
    ax1.set_title('Vertical Pull (Useful Work)', fontsize=12, color='white', fontweight='bold')
    ax1.grid(True, alpha=0.2, color='white')
    ax1.tick_params(colors='white')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['top'].set_color('white')
    ax1.spines['right'].set_color('white')
    
    # Panel 2: Horizontal Swing (Wasted Energy)
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_facecolor('black')
    ax2.plot(frames, horiz_disp_norm, color='#FF4444', linewidth=2, alpha=0.9)
    ax2.fill_between(frames, horiz_disp_norm, alpha=0.3, color='#FF4444')
    ax2.set_ylabel('Displacement (body units)', fontsize=11, color='white', fontweight='bold')
    ax2.set_title('Horizontal Swing (Wasted Energy)', fontsize=12, color='white', fontweight='bold')
    ax2.grid(True, alpha=0.2, color='white')
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['top'].set_color('white')
    ax2.spines['right'].set_color('white')
    
    # Panel 3: Velocity Components
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_facecolor('black')
    ax3.plot(frames, vert_vel, color='#00FFFF', linewidth=2, alpha=0.9, label='Vertical Velocity')
    ax3.plot(frames, horiz_vel, color='#FF00FF', linewidth=2, alpha=0.9, label='Horizontal Velocity')
    ax3.axhline(y=0, color='white', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Velocity Components', fontsize=11, color='white', fontweight='bold')
    ax3.set_ylabel('Velocity (body units/s)', fontsize=11, color='white', fontweight='bold')
    ax3.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')
    ax3.grid(True, alpha=0.2, color='white')
    ax3.tick_params(colors='white')
    ax3.spines['bottom'].set_color('white')
    ax3.spines['left'].set_color('white')
    ax3.spines['top'].set_color('white')
    ax3.spines['right'].set_color('white')
    
    # Panel 4: Jerk (Motion Smoothness)
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_facecolor('black')
    ax4.plot(frames, jerk_smooth, color='#FFFF00', linewidth=2, alpha=0.9)
    ax4.fill_between(frames, jerk_smooth, alpha=0.3, color='#FFFF00')
    ax4.set_xlabel('Frame', fontsize=11, color='white', fontweight='bold')
    ax4.set_ylabel('Jerk (body units/s³)', fontsize=11, color='white', fontweight='bold')
    ax4.set_title('Jerk (Motion Smoothness)', fontsize=12, color='white', fontweight='bold')
    ax4.grid(True, alpha=0.2, color='white')
    ax4.tick_params(colors='white')
    ax4.spines['bottom'].set_color('white')
    ax4.spines['left'].set_color('white')
    ax4.spines['top'].set_color('white')
    ax4.spines['right'].set_color('white')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
    print(f"✅ Saved comprehensive graph: {output_path}")
    plt.close()
    
    return {
        'efficiency_db': efficiency_db,
        'efficiency_pct': efficiency_pct,
        'vert_energy': vert_energy,
        'horiz_energy': horiz_energy,
        'jerk_energy': jerk_energy
    }

# Generate graphs for all videos
results_dir = Path("test_results")
graphs_dir = Path("test_results/comprehensive_graphs")
graphs_dir.mkdir(exist_ok=True)

print("📊 Generating Vishal-style comprehensive graphs...\n")

all_metrics = []

for report_file in results_dir.glob("*_report.json"):
    video_name = report_file.stem.replace("_analysis_report", "")
    keypoints_file = results_dir / f"{video_name}_keypoints.json"
    
    if not keypoints_file.exists():
        print(f"⚠️  Skipping {video_name} - no keypoints file")
        continue
    
    output_path = graphs_dir / f"{video_name}_comprehensive.png"
    
    print(f"Processing: {video_name}")
    metrics = create_comprehensive_graph(report_file, keypoints_file, output_path)
    
    all_metrics.append({
        'video': video_name,
        **metrics
    })
    
    print(f"  Efficiency: {metrics['efficiency_db']:.1f} dB ({metrics['efficiency_pct']:.1f}%)")
    print()

# Create comparison summary
print("="*70)
print("📊 EFFICIENCY COMPARISON")
print("="*70)
print(f"\n{'Video':<25} {'Efficiency (dB)':<18} {'Efficiency (%)':<15} {'Quality'}")
print("-" * 75)

for m in sorted(all_metrics, key=lambda x: x['efficiency_pct'], reverse=True):
    quality = "EXCELLENT" if m['efficiency_pct'] > 50 else "GOOD" if m['efficiency_pct'] > 20 else "NEEDS WORK"
    print(f"{m['video']:<25} {m['efficiency_db']:<18.1f} {m['efficiency_pct']:<15.1f} {quality}")

print(f"\n✅ All graphs saved to: {graphs_dir.absolute()}")
print(f"\n🎯 Compare with Vishal's graph to see if yours is better!")
