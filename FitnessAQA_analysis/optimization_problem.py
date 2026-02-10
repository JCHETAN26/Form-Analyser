import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Load data
with open('./data/pullup_analysis.json', 'r') as f:
    data = json.load(f)

raw_kps = np.array(data['raw_keypoints'])

# Joint indices (COCO format)
L_SHO, R_SHO = 5, 6
L_HIP, R_HIP = 11, 12
L_WRI, R_WRI = 9, 10

def get_center(idx1, idx2):
    return (raw_kps[:, idx1, :] + raw_kps[:, idx2, :]) / 2

shoulder_center = get_center(L_SHO, R_SHO)
hip_center = get_center(L_HIP, R_HIP)
wrist_center = get_center(L_WRI, R_WRI)

# Normalization
torso_lengths = np.linalg.norm(hip_center - shoulder_center, axis=1)
scale_factor = np.median(torso_lengths)

# === OPTIMIZATION MODEL ===
# We want to maximize: Useful Work / Total Energy Expenditure

dt = 1/30.0  # 30 fps
window = 15
if len(shoulder_center) < window: window = 5

# 1. USEFUL WORK: Vertical displacement of center of mass (approximated by hip center)
# Work = m*g*Δh, but we can ignore constants since we're computing a ratio
vertical_displacement = shoulder_center[:, 1] - wrist_center[:, 1]
vertical_displacement_normalized = (vertical_displacement - np.min(vertical_displacement)) / scale_factor

# Total useful displacement (full range of motion)
useful_work = np.max(vertical_displacement_normalized) - np.min(vertical_displacement_normalized)

# 2. WASTED ENERGY: Horizontal motion + Jerk
# Horizontal kinetic energy (proportional to velocity²)
hip_x_smooth = savgol_filter(hip_center[:, 0], window, 3)
hip_vx = savgol_filter(hip_center[:, 0], window, 3, deriv=1, delta=dt) / scale_factor

# Vertical jerk (indicates jerky, inefficient motion)
hip_vy = savgol_filter(vertical_displacement, window, 3, deriv=1, delta=dt) / scale_factor
hip_jerk_y = savgol_filter(vertical_displacement, window, 3, deriv=3, delta=dt) / (scale_factor * dt**2)

# Energy metrics (time-integrated)
horizontal_energy = np.sum(hip_vx**2) * dt  # ∫v_x² dt
jerk_energy = np.sum(hip_jerk_y**2) * dt    # ∫jerk² dt (smoothness penalty)

# 3. EFFICIENCY SCORE
# Higher = better (more ROM per unit of wasted energy)
total_wasted = horizontal_energy + 0.1 * jerk_energy  # Weight jerk less
if total_wasted < 1e-6: total_wasted = 1e-6

efficiency_score = useful_work / total_wasted

# Convert to dB-like scale for readability
efficiency_db = 10 * np.log10(efficiency_score)

# === ALTERNATIVE: MECHANICAL EFFICIENCY ===
# Also compute traditional mechanical efficiency
vertical_energy = np.sum(hip_vy**2) * dt
if vertical_energy + total_wasted < 1e-6:
    mechanical_efficiency = 0
else:
    mechanical_efficiency = vertical_energy / (vertical_energy + total_wasted) * 100

print(f"=== PULL-UP ANALYSIS ===")
print(f"Useful Work (ROM): {useful_work:.3f} body units")
print(f"Horizontal Energy: {horizontal_energy:.3f}")
print(f"Jerk Energy: {jerk_energy:.3f}")
print(f"Efficiency Score: {efficiency_db:.2f} dB")
print(f"Mechanical Efficiency: {mechanical_efficiency:.1f}%")

# === VISUALIZATION ===
plt.style.use('dark_background')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

frames = np.arange(len(vertical_displacement_normalized))

# 1. Vertical Pull (Signal)
ax1.plot(frames, vertical_displacement_normalized, color='#00FF00', linewidth=2)
ax1.set_title('Vertical Pull (Useful Work)', fontweight='bold')
ax1.set_ylabel('Displacement (body units)')
ax1.grid(alpha=0.3)

# 2. Horizontal Deviation (Noise)
ax2.plot(frames, hip_x_smooth / scale_factor, color='#FF4500', linewidth=2)
ax2.set_title('Horizontal Swing (Wasted Energy)', fontweight='bold')
ax2.set_ylabel('Displacement (body units)')
ax2.grid(alpha=0.3)

# 3. Velocity Profile
ax3.plot(frames, hip_vy, color='#00FFFF', linewidth=1.5, label='Vertical Velocity')
ax3.plot(frames, hip_vx, color='#FF69B4', linewidth=1.5, label='Horizontal Velocity')
ax3.set_title('Velocity Components', fontweight='bold')
ax3.set_ylabel('Velocity (body units/s)')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Jerk (Smoothness)
ax4.plot(frames, np.abs(hip_jerk_y), color='#FFD700', linewidth=1.5)
ax4.set_title('Jerk (Motion Smoothness)', fontweight='bold')
ax4.set_ylabel('|Jerk| (body units/s³)')
ax4.set_xlabel('Frame')
ax4.grid(alpha=0.3)

plt.suptitle(f'Pull-Up Biomechanics | Efficiency: {efficiency_db:.1f} dB ({mechanical_efficiency:.1f}%)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('./outputs/pullup_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nVisualization saved to outputs folder")