import numpy as np
from scipy.signal import find_peaks, savgol_filter

class FormAnalyzer:
    def __init__(self):
        # COCO Mapping (Standard 17 joints)
        self.joints = {
            'nose': 0,
            'l_eye': 1, 'r_eye': 2,
            'l_ear': 3, 'r_ear': 4,
            'l_shoulder': 5, 'r_shoulder': 6,
            'l_elbow': 7, 'r_elbow': 8,
            'l_wrist': 9, 'r_wrist': 10,
            'l_hip': 11, 'r_hip': 12,
            'l_knee': 13, 'r_knee': 14,
            'l_ankle': 15, 'r_ankle': 16
        }
    
    def calculate_angle(self, a, b, c):
        """
        Calculates the angle at point b given points a and c.
        Points should be [x, y] or [x, y, z].
        Returns NaN-safe angle or 0.0 if calculation fails.
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b
        
        # Check for zero-length vectors (missing keypoints)
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 0.0  # Return neutral angle for missing data

        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        # Clip to avoid numerical errors outside [-1, 1]
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        angle = np.arccos(cosine_angle)
        angle_deg = np.degrees(angle)
        
        # Final NaN check
        return 0.0 if np.isnan(angle_deg) else angle_deg

    def count_reps(self, trajectory, prometheus_threshold=0.5):
        """
        Counts repetitions based on a 1D trajectory (e.g., wrist Y-coordinate).
        Uses peak detection to identify full movement cycles.
        """
        if len(trajectory) < 10:
            return 0, []

        # Smooth the signal for better peak detection
        smoothed = savgol_filter(trajectory, window_length=min(15, len(trajectory)-1 if len(trajectory)%2==0 else len(trajectory)), polyorder=3)
        
        # Normalize
        norm_traj = (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed) + 1e-6)
        
        # Find peaks (top of movement) and valleys (bottom of movement)
        peaks, _ = find_peaks(norm_traj, height=0.7, distance=20)
        
        return len(peaks), peaks

    def calculate_jerk(self, trajectory, dt=1/30):
        """
        Calculates the jerk (3rd derivative of position).
        High jerk = shaky/unstable movement.
        """
        if len(trajectory) < 4:
            return np.zeros_like(trajectory)
        
        # Velocity -> Acceleration -> Jerk
        velocity = np.diff(trajectory) / dt
        acceleration = np.diff(velocity) / dt
        jerk = np.diff(acceleration) / dt
        
        # Pad to match original length
        return np.pad(jerk, (3, 0), mode='edge')

    def calculate_symmetry(self, left_angle, right_angle):
        """
        Calculates a symmetry index (0 to 1, where 1 is perfect symmetry).
        """
        if left_angle == 0 and right_angle == 0:
            return 1.0
        return 1.0 - (abs(left_angle - right_angle) / max(left_angle, right_angle, 1e-6))

    def detect_phases(self, trajectory):
        """
        Segments a trajectory into Concentric, Eccentric, and Isometric phases.
        Assumes Y-coordinate (pixels) increases as weight goes down.
        """
        velocity = np.gradient(trajectory)
        phases = []
        for v in velocity:
            if abs(v) < 0.5:
                phases.append("isometric")
            elif v > 0:
                phases.append("eccentric") # Moving down (increasing Y)
            else:
                phases.append("concentric") # Moving up (decreasing Y)
        return phases

    def analyze_squat(self, keypoints_frame):
        """
        Specific heuristics for Squat form.
        """
        l_hip, l_knee, l_ankle = keypoints_frame[self.joints['l_hip']], keypoints_frame[self.joints['l_knee']], keypoints_frame[self.joints['l_ankle']]
        r_hip, r_knee, r_ankle = keypoints_frame[self.joints['r_hip']], keypoints_frame[self.joints['r_knee']], keypoints_frame[self.joints['r_ankle']]
        
        l_knee_angle = self.calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = self.calculate_angle(r_hip, r_knee, r_ankle)
        
        symmetry = self.calculate_symmetry(l_knee_angle, r_knee_angle)
        
        feedback = "Perfect"
        if l_knee_angle > 100: feedback = "Go Deeper"
        elif l_knee_angle < 60: feedback = "Too Deep/Lose Tension"
            
        return {
            'knee_angle': l_knee_angle,
            'symmetry_score': symmetry,
            'feedback': feedback
        }

    def analyze_pullup(self, keypoints_frame):
        """
        Specific heuristics for Pullup form.
        """
        l_shoulder, l_elbow, l_wrist = keypoints_frame[self.joints['l_shoulder']], keypoints_frame[self.joints['l_elbow']], keypoints_frame[self.joints['l_wrist']]
        r_shoulder, r_elbow, r_wrist = keypoints_frame[self.joints['r_shoulder']], keypoints_frame[self.joints['r_elbow']], keypoints_frame[self.joints['r_wrist']]
        
        l_elbow_angle = self.calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = self.calculate_angle(r_shoulder, r_elbow, r_wrist)
        
        symmetry = self.calculate_symmetry(l_elbow_angle, r_elbow_angle)
        
        feedback = "Good"
        if l_elbow_angle > 60: feedback = "Full Pull Required"
            
        return {
            'elbow_angle': l_elbow_angle,
            'symmetry_score': symmetry,
            'feedback': feedback
        }

    def process_sequence(self, keypoints_seq, exercise_type='squat'):
        """
        Processes a sequence of frames.
        """
        # 1. Trajectory & Rep Counting
        if exercise_type == 'squat':
            traj = keypoints_seq[:, self.joints['l_hip'], 1]
        else:
            traj = keypoints_seq[:, self.joints['l_wrist'], 1]
            
        rep_count, rep_frames = self.count_reps(traj)
        
        # 2. Advanced Biometrics
        jerk_signal = self.calculate_jerk(traj)
        phases = self.detect_phases(traj)
        
        # 3. Frame-by-frame analysis
        results = []
        for i in range(len(keypoints_seq)):
            if exercise_type == 'squat':
                analysis = self.analyze_squat(keypoints_seq[i])
            else:
                analysis = self.analyze_pullup(keypoints_seq[i])
                
            analysis['frame_idx'] = i
            analysis['jerk'] = float(jerk_signal[i])
            analysis['phase'] = phases[i]
            results.append(analysis)
            
        return {
            'rep_count': rep_count,
            'frame_analysis': results,
            'rep_peaks': rep_frames.tolist(),
            'avg_jerk': float(np.mean(np.abs(jerk_signal))),
            'avg_symmetry': float(np.mean([r['symmetry_score'] for r in results]))
        }
