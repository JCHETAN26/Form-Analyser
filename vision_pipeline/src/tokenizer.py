import numpy as np
import json

class SignalTokenizer:
    """
    Converts raw biomechanical signals into discrete tokens for LLM context.
    """
    def __init__(self):
        # Bins for tokenization
        self.jerk_thresholds = {
            'JERK_STABLE': (0, 50),
            'JERK_MODERATE': (50, 200),
            'JERK_SHAKY': (200, float('inf'))
        }
        self.symmetry_thresholds = {
            'SYMMETRIC': (0.9, 1.0),
            'SLIGHT_ASYMMETRY': (0.8, 0.9),
            'HIGH_ASYMMETRY': (0.0, 0.8)
        }

    def get_token(self, value, thresholds):
        for token, (low, high) in thresholds.items():
            if low <= value < high:
                return token
        return "UNKNOWN"

    def tokenize_analysis(self, analysis_results):
        """
        Converts full analysis results into a structured prompt-friendly string.
        """
        rep_count = analysis_results['rep_count']
        avg_jerk = analysis_results['avg_jerk']
        avg_symmetry = analysis_results['avg_symmetry']
        
        jerk_token = self.get_token(avg_jerk, self.jerk_thresholds)
        sym_token = self.get_token(avg_symmetry, self.symmetry_thresholds)
        
        # Phase distribution
        phases = [f['phase'] for f in analysis_results['frame_analysis']]
        eccentric_count = phases.count('eccentric')
        concentric_count = phases.count('concentric')
        
        # Evidence Snippets (Example: find frames with high jerk)
        shaky_frames = [f['frame_idx'] for f in analysis_results['frame_analysis'] if f['jerk'] > 200]
        evidence = ""
        if shaky_frames:
            # Group into ranges (simplified)
            evidence = f" High jerk detected at frame range [{shaky_frames[0]}-{shaky_frames[-1]}]."

        summary = (
            f"EXERCISE_SUMMARY:\n"
            f"- REPS_DETECTED: {rep_count}\n"
            f"- STABILITY: {jerk_token} (Avg Jerk: {avg_jerk:.2f})\n"
            f"- SYMMETRY: {sym_token} (Score: {avg_symmetry:.2f})\n"
            f"- TEMPO: Eccentric ({eccentric_count} frames), Concentric ({concentric_count} frames)\n"
            f"- EVIDENCE:{evidence}"
        )
        
        return summary

if __name__ == "__main__":
    # Test with dummy data
    tokenizer = SignalTokenizer()
    dummy_data = {
        'rep_count': 5,
        'avg_jerk': 120.5,
        'avg_symmetry': 0.85,
        'frame_analysis': [{'phase': 'eccentric', 'jerk': 50, 'frame_idx': 0}, {'phase': 'concentric', 'jerk': 300, 'frame_idx': 100}],
    }
    print(tokenizer.tokenize_analysis(dummy_data))
