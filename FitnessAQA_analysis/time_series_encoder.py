"""
Time Series Encoder for Fitness Pose Data → LLM Token Sequences

Converts raw COCO-17 keypoint trajectories into structured text tokens
that a GPT-class model can learn to interpret and reason about.

Encoding Strategy:
  1. Quantize continuous (x,y) values into discrete bins (vocabulary tokens)
  2. Compute derived biomechanical features (angles, velocities, jerk)
  3. Segment into movement phases (eccentric/concentric/hold)
  4. Serialize everything into a structured text format for instruction tuning

Author: FitnessAQA Capstone
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from scipy.signal import savgol_filter, find_peaks


# ─────────────────────── COCO-17 Joint Mapping ───────────────────────
COCO_JOINTS = {
    0: "nose", 1: "l_eye", 2: "r_eye", 3: "l_ear", 4: "r_ear",
    5: "l_shoulder", 6: "r_shoulder", 7: "l_elbow", 8: "r_elbow",
    9: "l_wrist", 10: "r_wrist", 11: "l_hip", 12: "r_hip",
    13: "l_knee", 14: "r_knee", 15: "l_ankle", 16: "r_ankle"
}

# Key joint groups for biomechanical analysis
JOINT_GROUPS = {
    "upper_body": [5, 6, 7, 8, 9, 10],
    "lower_body": [11, 12, 13, 14, 15, 16],
    "core": [5, 6, 11, 12],
    "arms": [5, 7, 9, 6, 8, 10],  # shoulder-elbow-wrist chains
}

# Angle triplets: (point_a, vertex, point_c)
ANGLE_DEFINITIONS = {
    "l_elbow": (5, 7, 9),    # shoulder-elbow-wrist
    "r_elbow": (6, 8, 10),
    "l_shoulder": (7, 5, 11),  # elbow-shoulder-hip
    "r_shoulder": (8, 6, 12),
    "l_knee": (11, 13, 15),   # hip-knee-ankle
    "r_knee": (12, 14, 16),
    "l_hip": (5, 11, 13),     # shoulder-hip-knee
    "r_hip": (6, 12, 14),
    "torso": (5, 11, 13),     # body alignment
}


@dataclass
class EncoderConfig:
    """Configuration for the time series encoder."""
    # Quantization
    n_spatial_bins: int = 64         # Number of bins for x,y coordinates
    n_angle_bins: int = 36           # Number of bins for angles (0-180°)
    n_velocity_bins: int = 32        # Number of bins for velocity values
    
    # Temporal
    fps: int = 30
    window_size: int = 15            # Savgol filter window
    poly_order: int = 3
    
    # Phase detection
    phase_min_frames: int = 5        # Minimum frames per movement phase
    
    # Token vocabulary
    spatial_prefix: str = "S"        # Spatial bin token prefix
    angle_prefix: str = "A"          # Angle bin token prefix
    velocity_prefix: str = "V"       # Velocity bin token prefix
    phase_tokens: Dict[str, str] = field(default_factory=lambda: {
        "concentric": "<CONC>",      # Muscle shortening (pulling up)
        "eccentric": "<ECC>",        # Muscle lengthening (lowering down)
        "isometric": "<ISO>",        # Static hold
        "transition": "<TRANS>",     # Between phases
    })
    temporal_tokens: Dict[str, str] = field(default_factory=lambda: {
        "frame_start": "<F>",
        "frame_end": "</F>",
        "rep_start": "<REP>",
        "rep_end": "</REP>",
        "sequence_start": "<SEQ>",
        "sequence_end": "</SEQ>",
    })


class TimeSeriesEncoder:
    """
    Encodes pose keypoint time series into structured text tokens
    for LLM fine-tuning.
    """
    
    def __init__(self, config: Optional[EncoderConfig] = None):
        self.config = config or EncoderConfig()
        self._build_vocabulary()
    
    def _build_vocabulary(self):
        """Builds the complete token vocabulary."""
        self.vocab = {}
        idx = 0
        
        # Special tokens
        special = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        for tok in special:
            self.vocab[tok] = idx
            idx += 1
        
        # Temporal tokens
        for tok in self.config.temporal_tokens.values():
            self.vocab[tok] = idx
            idx += 1
        
        # Phase tokens
        for tok in self.config.phase_tokens.values():
            self.vocab[tok] = idx
            idx += 1
        
        # Joint name tokens
        for name in COCO_JOINTS.values():
            tok = f"<{name}>"
            self.vocab[tok] = idx
            idx += 1
        
        # Spatial quantization tokens
        for i in range(self.config.n_spatial_bins):
            self.vocab[f"{self.config.spatial_prefix}{i}"] = idx
            idx += 1
        
        # Angle quantization tokens
        for i in range(self.config.n_angle_bins):
            self.vocab[f"{self.config.angle_prefix}{i}"] = idx
            idx += 1
        
        # Velocity quantization tokens
        for i in range(self.config.n_velocity_bins):
            self.vocab[f"{self.config.velocity_prefix}{i}"] = idx
            idx += 1
        
        # Feature descriptor tokens
        descriptors = [
            "<ROM_LOW>", "<ROM_MED>", "<ROM_HIGH>", "<ROM_FULL>",
            "<SMOOTH>", "<JERKY>", "<MODERATE_SMOOTH>",
            "<STABLE>", "<UNSTABLE>", "<MODERATE_STABLE>",
            "<FAST>", "<SLOW>", "<MODERATE_SPEED>",
            "<GOOD_FORM>", "<FAIR_FORM>", "<POOR_FORM>",
            "<SYMMETRIC>", "<ASYMMETRIC>",
        ]
        for tok in descriptors:
            self.vocab[tok] = idx
            idx += 1
        
        self.vocab_size = idx
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
    
    # ───────────────── Core Math Utilities ─────────────────
    
    @staticmethod
    def compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Compute angle at vertex b (in degrees). Vectorized over frames."""
        ba = a - b
        bc = c - b
        cos_angle = np.sum(ba * bc, axis=-1) / (
            np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8
        )
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    
    def _smooth(self, signal: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay smoothing."""
        win = min(self.config.window_size, len(signal))
        if win % 2 == 0:
            win -= 1
        if win < self.config.poly_order + 2:
            return signal
        return savgol_filter(signal, win, self.config.poly_order)
    
    def _derivative(self, signal: np.ndarray, order: int = 1) -> np.ndarray:
        """Compute smoothed derivative."""
        win = min(self.config.window_size, len(signal))
        if win % 2 == 0:
            win -= 1
        if win < self.config.poly_order + 2:
            return np.gradient(signal)
        dt = 1.0 / self.config.fps
        return savgol_filter(signal, win, self.config.poly_order, deriv=order, delta=dt)
    
    def _quantize(self, values: np.ndarray, n_bins: int,
                  vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
        """Quantize continuous values into discrete bin indices."""
        if vmin is None:
            vmin = np.min(values)
        if vmax is None:
            vmax = np.max(values)
        
        span = vmax - vmin
        if span < 1e-8:
            return np.zeros_like(values, dtype=int)
        
        normalized = (values - vmin) / span
        bins = np.clip((normalized * (n_bins - 1)).astype(int), 0, n_bins - 1)
        return bins
    
    # ───────────────── Feature Extraction ─────────────────
    
    def extract_features(self, keypoints: np.ndarray,
                         scores: Optional[np.ndarray] = None) -> Dict:
        """
        Extract biomechanical features from keypoint sequence.
        
        Args:
            keypoints: (T, 17, 2) array of keypoint coordinates
            scores: (T, 17) optional confidence scores
            
        Returns:
            Dictionary of computed features
        """
        T = keypoints.shape[0]
        features = {}
        
        # ── Normalization ──
        shoulder_center = (keypoints[:, 5, :] + keypoints[:, 6, :]) / 2
        hip_center = (keypoints[:, 11, :] + keypoints[:, 12, :]) / 2
        wrist_center = (keypoints[:, 9, :] + keypoints[:, 10, :]) / 2
        
        torso_lengths = np.linalg.norm(hip_center - shoulder_center, axis=1)
        scale = np.median(torso_lengths)
        if scale < 1e-6:
            scale = 1.0
        
        # Normalize keypoints by torso length
        norm_kps = keypoints.copy()
        origin = hip_center[:, np.newaxis, :]  # Use hip as origin
        norm_kps = (norm_kps - origin) / scale
        
        features["normalized_keypoints"] = norm_kps
        features["scale_factor"] = scale
        
        # ── Joint Angles ──
        angles = {}
        for name, (a, b, c) in ANGLE_DEFINITIONS.items():
            angles[name] = self.compute_angle(
                keypoints[:, a, :], keypoints[:, b, :], keypoints[:, c, :]
            )
        features["angles"] = angles
        
        # ── Vertical Displacement (key signal for pull-ups) ──
        vert_disp = shoulder_center[:, 1] - wrist_center[:, 1]
        vert_disp_norm = (vert_disp - np.min(vert_disp)) / (scale + 1e-8)
        features["vertical_displacement"] = vert_disp_norm
        
        # ── Velocities ──
        velocities = {}
        for name, angle_series in angles.items():
            velocities[f"{name}_angular_vel"] = self._derivative(angle_series)
        
        vert_vel = self._derivative(vert_disp_norm)
        features["vertical_velocity"] = vert_vel
        velocities["vertical"] = vert_vel
        
        horiz_vel = self._derivative(hip_center[:, 0] / scale)
        features["horizontal_velocity"] = horiz_vel
        velocities["horizontal"] = horiz_vel
        
        features["velocities"] = velocities
        
        # ── Jerk (smoothness metric) ──
        vert_jerk = self._derivative(vert_disp_norm, order=3)
        features["vertical_jerk"] = vert_jerk
        
        # ── Range of Motion ──
        rom = {}
        for name, angle_series in angles.items():
            rom[name] = float(np.max(angle_series) - np.min(angle_series))
        features["range_of_motion"] = rom
        
        # ── Symmetry (left vs right) ──
        symmetry = {}
        for side_pair in [("l_elbow", "r_elbow"), ("l_shoulder", "r_shoulder"),
                          ("l_knee", "r_knee"), ("l_hip", "r_hip")]:
            l_name, r_name = side_pair
            if l_name in angles and r_name in angles:
                diff = np.mean(np.abs(angles[l_name] - angles[r_name]))
                symmetry[f"{l_name[2:]}_symmetry"] = float(diff)
        features["symmetry"] = symmetry
        
        # ── Phase Detection ──
        features["phases"] = self._detect_phases(vert_disp_norm)
        
        # ── Rep Detection ──
        features["reps"] = self._detect_reps(vert_disp_norm)
        
        # ── Summary Metrics ──
        dt = 1.0 / self.config.fps
        horiz_energy = float(np.sum(horiz_vel ** 2) * dt)
        jerk_energy = float(np.sum(vert_jerk ** 2) * dt)
        rom_total = float(np.max(vert_disp_norm) - np.min(vert_disp_norm))
        
        total_wasted = horiz_energy + 0.1 * jerk_energy
        if total_wasted < 1e-6:
            total_wasted = 1e-6
        
        vert_energy = float(np.sum(vert_vel ** 2) * dt)
        mech_eff = vert_energy / (vert_energy + total_wasted) * 100 if (vert_energy + total_wasted) > 1e-6 else 0
        
        features["summary"] = {
            "useful_work_rom": rom_total,
            "horizontal_energy": horiz_energy,
            "jerk_energy": jerk_energy,
            "efficiency_db": float(10 * np.log10(rom_total / total_wasted)) if rom_total > 0 else -999,
            "mechanical_efficiency_pct": mech_eff,
            "total_frames": T,
            "duration_sec": T / self.config.fps,
        }
        
        if scores is not None:
            features["mean_confidence"] = float(np.mean(scores))
            features["min_confidence"] = float(np.min(scores))
        
        return features
    
    def _detect_phases(self, vertical_signal: np.ndarray) -> List[Dict]:
        """Detect movement phases from vertical displacement signal."""
        smoothed = self._smooth(vertical_signal)
        velocity = self._derivative(smoothed)
        
        phases = []
        current_phase = None
        phase_start = 0
        
        threshold = np.std(velocity) * 0.15
        
        for i in range(len(velocity)):
            if velocity[i] > threshold:
                phase = "concentric"   # Moving up
            elif velocity[i] < -threshold:
                phase = "eccentric"    # Moving down
            else:
                phase = "isometric"    # Holding
            
            if phase != current_phase:
                if current_phase is not None and (i - phase_start) >= self.config.phase_min_frames:
                    phases.append({
                        "type": current_phase,
                        "start_frame": int(phase_start),
                        "end_frame": int(i - 1),
                        "duration_frames": int(i - phase_start),
                    })
                current_phase = phase
                phase_start = i
        
        # Final phase
        if current_phase and (len(velocity) - phase_start) >= self.config.phase_min_frames:
            phases.append({
                "type": current_phase,
                "start_frame": int(phase_start),
                "end_frame": int(len(velocity) - 1),
                "duration_frames": int(len(velocity) - phase_start),
            })
        
        return phases
    
    def _detect_reps(self, vertical_signal: np.ndarray) -> List[Dict]:
        """Detect repetitions from vertical signal."""
        smoothed = self._smooth(vertical_signal)
        norm = (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed) + 1e-8)
        
        peaks, _ = find_peaks(norm, height=0.6, distance=20)
        valleys, _ = find_peaks(-norm, height=-0.4, distance=20)
        
        reps = []
        for i, peak in enumerate(peaks):
            # Find surrounding valleys
            prev_valleys = valleys[valleys < peak]
            next_valleys = valleys[valleys > peak]
            
            start = int(prev_valleys[-1]) if len(prev_valleys) > 0 else 0
            end = int(next_valleys[0]) if len(next_valleys) > 0 else len(vertical_signal) - 1
            
            rep_rom = float(norm[peak] - min(norm[start], norm[min(end, len(norm) - 1)]))
            
            reps.append({
                "rep_number": i + 1,
                "peak_frame": int(peak),
                "start_frame": start,
                "end_frame": end,
                "peak_height": float(norm[peak]),
                "rom": rep_rom,
            })
        
        return reps
    
    # ───────────────── Tokenization ─────────────────
    
    def encode_frame(self, norm_kps_frame: np.ndarray,
                     angles_frame: Dict[str, float],
                     phase: str = "transition") -> str:
        """Encode a single frame into text tokens."""
        tokens = [self.config.temporal_tokens["frame_start"]]
        
        # Phase token
        tokens.append(self.config.phase_tokens.get(phase, "<TRANS>"))
        
        # Key joint positions (quantized) — only biomechanically important joints
        important_joints = [5, 6, 7, 8, 9, 10, 11, 12]  # shoulders through hips
        for j_idx in important_joints:
            j_name = COCO_JOINTS[j_idx]
            x_bin = self._quantize(
                np.array([norm_kps_frame[j_idx, 0]]),
                self.config.n_spatial_bins, vmin=-2.0, vmax=2.0
            )[0]
            y_bin = self._quantize(
                np.array([norm_kps_frame[j_idx, 1]]),
                self.config.n_spatial_bins, vmin=-2.0, vmax=2.0
            )[0]
            tokens.append(f"<{j_name}>")
            tokens.append(f"{self.config.spatial_prefix}{x_bin}")
            tokens.append(f"{self.config.spatial_prefix}{y_bin}")
        
        # Key angles (quantized)
        for a_name, a_val in angles_frame.items():
            a_bin = self._quantize(
                np.array([a_val]), self.config.n_angle_bins, vmin=0, vmax=180
            )[0]
            tokens.append(f"{self.config.angle_prefix}{a_bin}")
        
        tokens.append(self.config.temporal_tokens["frame_end"])
        return " ".join(tokens)
    
    def encode_sequence(self, keypoints: np.ndarray,
                        scores: Optional[np.ndarray] = None,
                        stride: int = 3) -> Dict:
        """
        Encode full keypoint sequence into structured text.
        
        Args:
            keypoints: (T, 17, 2) raw keypoints
            scores: (T, 17) confidence scores (optional)
            stride: frame subsampling stride (reduces token count)
            
        Returns:
            Dict with encoded tokens, features, and metadata
        """
        features = self.extract_features(keypoints, scores)
        norm_kps = features["normalized_keypoints"]
        angles = features["angles"]
        phases = features["phases"]
        
        T = keypoints.shape[0]
        
        # Build phase lookup (frame → phase type)
        phase_lookup = ["transition"] * T
        for p in phases:
            for f in range(p["start_frame"], min(p["end_frame"] + 1, T)):
                phase_lookup[f] = p["type"]
        
        # ── Encode frame-level tokens ──
        frame_tokens = []
        sampled_frames = list(range(0, T, stride))
        
        for fi in sampled_frames:
            angles_frame = {name: vals[fi] for name, vals in angles.items()}
            frame_str = self.encode_frame(norm_kps[fi], angles_frame, phase_lookup[fi])
            frame_tokens.append(frame_str)
        
        # ── Build full sequence string ──
        seq_tokens = [self.config.temporal_tokens["sequence_start"]]
        
        # Rep-aware encoding
        reps = features["reps"]
        if reps:
            rep_boundaries = {}
            for rep in reps:
                for f in range(rep["start_frame"], rep["end_frame"] + 1):
                    rep_boundaries[f] = rep["rep_number"]
            
            current_rep = None
            for i, fi in enumerate(sampled_frames):
                rep_num = rep_boundaries.get(fi)
                if rep_num != current_rep:
                    if current_rep is not None:
                        seq_tokens.append(self.config.temporal_tokens["rep_end"])
                    if rep_num is not None:
                        seq_tokens.append(self.config.temporal_tokens["rep_start"])
                    current_rep = rep_num
                seq_tokens.append(frame_tokens[i])
            
            if current_rep is not None:
                seq_tokens.append(self.config.temporal_tokens["rep_end"])
        else:
            seq_tokens.extend(frame_tokens)
        
        seq_tokens.append(self.config.temporal_tokens["sequence_end"])
        
        full_text = " ".join(seq_tokens)
        
        # ── Build compact summary encoding ──
        summary_text = self._encode_summary(features)
        
        # ── Build descriptive quality tokens ──
        quality_tokens = self._encode_quality(features)
        
        return {
            "frame_tokens": full_text,
            "summary_tokens": summary_text,
            "quality_tokens": quality_tokens,
            "features": features,
            "metadata": {
                "total_frames": T,
                "sampled_frames": len(sampled_frames),
                "stride": stride,
                "vocab_size": self.vocab_size,
                "token_count": len(full_text.split()),
            }
        }
    
    def _encode_summary(self, features: Dict) -> str:
        """Encode high-level summary as structured text."""
        s = features["summary"]
        reps = features["reps"]
        rom = features["range_of_motion"]
        sym = features["symmetry"]
        
        lines = [
            f"exercise_type: pullup",
            f"duration_sec: {s['duration_sec']:.1f}",
            f"total_frames: {s['total_frames']}",
            f"rep_count: {len(reps)}",
            f"useful_work_rom: {s['useful_work_rom']:.3f}",
            f"horizontal_energy: {s['horizontal_energy']:.4f}",
            f"jerk_energy: {s['jerk_energy']:.4f}",
            f"efficiency_db: {s['efficiency_db']:.2f}",
            f"mechanical_efficiency_pct: {s['mechanical_efficiency_pct']:.1f}",
        ]
        
        # Per-rep metrics
        for rep in reps:
            lines.append(
                f"rep_{rep['rep_number']}_rom: {rep['rom']:.3f} "
                f"peak_height: {rep['peak_height']:.3f}"
            )
        
        # ROM per joint
        for name, val in rom.items():
            lines.append(f"rom_{name}: {val:.1f}deg")
        
        # Symmetry
        for name, val in sym.items():
            lines.append(f"sym_{name}: {val:.1f}deg")
        
        # Phase breakdown
        phases = features["phases"]
        phase_durations = {}
        for p in phases:
            phase_durations.setdefault(p["type"], 0)
            phase_durations[p["type"]] += p["duration_frames"]
        for ptype, dur in phase_durations.items():
            lines.append(f"phase_{ptype}_frames: {dur}")
        
        return "\n".join(lines)
    
    def _encode_quality(self, features: Dict) -> str:
        """Encode quality assessment tokens."""
        s = features["summary"]
        tokens = []
        
        # ROM quality
        rom = s["useful_work_rom"]
        if rom > 0.8:
            tokens.append("<ROM_FULL>")
        elif rom > 0.6:
            tokens.append("<ROM_HIGH>")
        elif rom > 0.3:
            tokens.append("<ROM_MED>")
        else:
            tokens.append("<ROM_LOW>")
        
        # Smoothness quality
        jerk = s["jerk_energy"]
        if jerk < 0.5:
            tokens.append("<SMOOTH>")
        elif jerk < 2.0:
            tokens.append("<MODERATE_SMOOTH>")
        else:
            tokens.append("<JERKY>")
        
        # Stability quality (horizontal energy)
        horiz = s["horizontal_energy"]
        if horiz < 0.1:
            tokens.append("<STABLE>")
        elif horiz < 0.5:
            tokens.append("<MODERATE_STABLE>")
        else:
            tokens.append("<UNSTABLE>")
        
        # Overall form
        eff = s["mechanical_efficiency_pct"]
        if eff > 80:
            tokens.append("<GOOD_FORM>")
        elif eff > 50:
            tokens.append("<FAIR_FORM>")
        else:
            tokens.append("<POOR_FORM>")
        
        # Symmetry
        sym_vals = list(features["symmetry"].values())
        if sym_vals:
            avg_asym = np.mean(sym_vals)
            tokens.append("<SYMMETRIC>" if avg_asym < 10 else "<ASYMMETRIC>")
        
        return " ".join(tokens)
    
    # ───────────────── I/O ─────────────────
    
    def encode_from_json(self, json_path: str, stride: int = 3) -> Dict:
        """Load a pullup_analysis.json and encode it."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        keypoints = np.array(data["raw_keypoints"])  # (T, 17, 2)
        scores = np.array(data.get("scores", None))
        if scores is not None and scores.ndim == 0:
            scores = None
        
        result = self.encode_sequence(keypoints, scores, stride=stride)
        result["metadata"]["video_id"] = data.get("video_id", "unknown")
        return result
    
    def get_vocab(self) -> Dict[str, int]:
        """Return the full vocabulary mapping."""
        return self.vocab.copy()
    
    def save_vocab(self, path: str):
        """Save vocabulary to JSON."""
        with open(path, 'w') as f:
            json.dump(self.vocab, f, indent=2)


# ───────────────── Standalone Usage ─────────────────

if __name__ == "__main__":
    print("=== Time Series Encoder Test ===\n")
    
    encoder = TimeSeriesEncoder()
    print(f"Vocabulary size: {encoder.vocab_size}")
    
    result = encoder.encode_from_json("./data/pullup_analysis.json", stride=5)
    
    print(f"\nMetadata: {json.dumps(result['metadata'], indent=2)}")
    print(f"\n--- Summary Tokens ---\n{result['summary_tokens']}")
    print(f"\n--- Quality Tokens ---\n{result['quality_tokens']}")
    print(f"\n--- Frame Tokens (first 500 chars) ---\n{result['frame_tokens'][:500]}...")
    
    # Save vocab
    encoder.save_vocab("./outputs/vocab.json")
    print("\nVocab saved to outputs/vocab.json")
