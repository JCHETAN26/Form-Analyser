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
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        # Clip to avoid numerical errors outside [-1, 1]
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

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
        valleys, _ = find_peaks(-norm_traj, height=-0.3, distance=20)
        
        return len(peaks), peaks

    def analyze_squat(self, keypoints_frame):
        """
        Specific heuristics for Squat form.
        Expected keypoints_frame: (17, 2) or (17, 3)
        """
        l_hip = keypoints_frame[self.joints['l_hip']]
        l_knee = keypoints_frame[self.joints['l_knee']]
        l_ankle = keypoints_frame[self.joints['l_ankle']]
        
        knee_angle = self.calculate_angle(l_hip, l_knee, l_ankle)
        
        feedback = "Perfect"
        if knee_angle > 100:
            feedback = "Go Deeper"
        elif knee_angle < 60:
            feedback = "Too Deep/Lose Tension"
            
        return {
            'knee_angle': knee_angle,
            'feedback': feedback
        }

    def analyze_pullup(self, keypoints_frame):
        """
        Specific heuristics for Pullup form.
        """
        l_shoulder = keypoints_frame[self.joints['l_shoulder']]
        l_elbow = keypoints_frame[self.joints['l_elbow']]
        l_wrist = keypoints_frame[self.joints['l_wrist']]
        
        elbow_angle = self.calculate_angle(l_shoulder, l_elbow, l_wrist)
        
        # Vertical tracking for rep quality
        # In pullups, we want the wrist to get close to the shoulder height or below it (relative)
        feedback = "Good"
        if elbow_angle > 60: # Rough heuristic for chin-over-bar
            feedback = "Full Pull Required"
            
        return {
            'elbow_angle': elbow_angle,
            'feedback': feedback
        }

    def process_sequence(self, keypoints_seq, exercise_type='squat'):
        """
        Processes a sequence of frames.
        keypoints_seq: (N, 17, 2) or (N, 17, 3)
        """
        results = []
        
        # 1. Calculate trajectory for rep counting
        # For squats/pullups, Y-coordinate of a key joint (like hip for squat or wrist for pullup)
        if exercise_type == 'squat':
            traj = keypoints_seq[:, self.joints['l_hip'], 1]
        else:
            traj = keypoints_seq[:, self.joints['l_wrist'], 1]
            
        rep_count, rep_frames = self.count_reps(traj)
        
        # 2. Frame-by-frame analysis
        for i in range(len(keypoints_seq)):
            if exercise_type == 'squat':
                analysis = self.analyze_squat(keypoints_seq[i])
            else:
                analysis = self.analyze_pullup(keypoints_seq[i])
                
            analysis['frame_idx'] = i
            results.append(analysis)
            
        return {
            'rep_count': rep_count,
            'frame_analysis': results,
            'rep_peaks': rep_frames.tolist()
        }
