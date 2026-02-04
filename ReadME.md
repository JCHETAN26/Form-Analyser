# Core Brainstorm: 2D → 3D AQA Modelling (Harmonic Motion) - with end recommendation systems

This README is **only the core algorithm exploration**: scrape a *small subset* of videos, extract 2D pose, lift to 3D, canonicalize, build time-series waves, and decide what harmonic model + metrics best separate **good vs bad form**.

We are **not** doing LoRA or full end-to-end product here.

---

## 0) Core Goal (One Sentence)

Given a short exercise clip, produce a **rep-wise waveform + physics metrics** that quantify motion quality (efficiency, cheating, fatigue) using **2D→3D pose + harmonic modeling**.

---

## I) Data: Minimal Scrape + Pose Capture

### A. Scrape (subset only)
**Questions**
- What keywords yield the cleanest side-view pull-up/row clips?
- What constraints reduce junk?
  - duration (10–30s)
  - single person
  - gym lighting
  - side view bias
- How do we avoid saving full videos (keep only clips)?

**Implementation anchor**
- Use `yt-dlp` for downloading and optional sectioning.  
- If clipping by timestamps is needed, `yt-dlp` supports downloading sections (e.g., start/end). 

**Artifacts**
- `data/raw/*.mp4`
- `data/meta.csv` (id, url, fps, duration, exercise, view_guess)

---

### B. 2D Pose Capture (MediaPipe baseline)
**Why**
- fast, easy, stable for prototypes
- returns landmarks per frame and also provides 3D “world” outputs (useful sanity check) 
**Questions**
- Which landmark subset is sufficient for pull-ups/rows?
- What confidence threshold invalidates a frame?
- Do we track only one person (largest bbox) to avoid multi-person errors?

**Artifacts**
- `data/pose2d/*.json` with per-frame joint coords + confidences

---

## II) 2D → 3D Pose Lifting (Monocular)

### A. PandaPose as reference (design inspiration)
PandaPose addresses common monocular lifting issues:
- 2D error propagation
- self-occlusion ambiguity
by **propagating 2D pose priors into a 3D anchor space**, using joint-wise anchors + depth-aware lifting + anchor-feature interaction, then predicting joints via ensemble offsets. 
**Core Questions**
- Do we need PandaPose-level complexity, or can we start with a simpler lift?
- What is the minimum viable 3D for AQA:
  - relative joint geometry only?
  - or true metric depth?

**Decision (prototype)**
- Start with **relative 3D** (scale-normalized) → enough for wave shape + phase-space.
- Use PandaPose ideas mainly for:
  - robustness to 2D noise
  - occlusion handling framing

---

## III) Canonicalization (Normalization + Rotate Same Direction)

### A. Normalization (required)
**Questions**
- Root joint: hip-center or torso-center?
- Scale: divide by torso length (shoulder-center → hip-center)?
- Do we normalize per clip or per rep?

**Artifact**
- `pose_norm[t, j, 3]` in canonical units

### B. Rotate to same direction (required)
**Questions**
- Define canonical axes:
  - Y axis: shoulder→hip (gravity-ish in body frame)
  - Movement plane: PCA on (wrist/hip) trajectory
- How do we handle mirrored views (left/right)? Flip?

**Output**
- all clips mapped into the same “side-view body frame” so signal/noise are comparable.

---

## IV) Time-Series Wave Signals Over Time

### A. Choose the primary waveform
**Candidates**
- wrist vertical displacement y(t)
- elbow flexion angle θ(t)
- shoulder elevation

**Questions**
- Which has the cleanest periodicity across subjects?
- Which is least sensitive to camera framing?

### B. Rep segmentation (missing in your notes, but mandatory)
**Questions**
- How do we detect reps?
  - velocity zero-crossings
  - peak/trough detection
- What is a valid rep?
  - minimum ROM
  - min duration

**Artifacts**
- `reps = [(t_start, t_end), ...]`

---

## V) Harmonic Wave Function (The Heart)

### A. What wave family are we fitting?
**Candidate models**
1) Simple harmonic: x(t)=A sin(ωt+φ)
2) Damped harmonic: x(t)=A e^{-βt} sin(ωt+φ)
3) Forced oscillator (cheating): harmonic + impulse term

**Questions**
- Is a single rep best modeled as simple harmonic?
- Is an entire set better modeled as damped (fatigue)?
- Does cheating show up as:
  - phase discontinuities
  - impulsive acceleration spikes
  - higher-frequency energy?

### B. What is being optimized?
**Options**
- Fit residual energy: minimize ||x - x_hat||^2
- Penalize jerk spikes
- Penalize frequency drift across reps

**Output**
- per rep: {A, ω, φ, residual_energy, drift}

---

## VI) Metrics (Define or Search)

### A. Core metrics (must-have)
**Questions**
- Efficiency metric:
  - Signal vs Noise power ratio (SNR-like)
- Noise sources:
  - torso swing (angle velocity)
  - horizontal wrist drift
  - jerk spikes (impulses)
- Fatigue:
  - increasing residual energy across reps
  - increasing high-frequency power (FFT)

### B. “Form improvement” metrics (optional, risky)
These are tempting but often become garbage if measured poorly.

**Questions**
- Do we collect heart rate as:
  - user input?
  - wearable sync?
  - (not inferred from video—too unreliable)
- BMI / muscle mass:
  - is it used only for normalization groups?
  - or does it actually improve metric interpretability?
- Injuries:
  - do we treat as “risk flags” instead of predictive features?

**Rule**
If a variable cannot be measured reliably, it must not be a core metric.

---

## VII) 2D vs 3D Analysis (What changes when going 3D?)

**Key Questions**
- Which metrics are stable in 2D alone?
  - vertical wrist displacement (yes)
  - jerk spikes (yes, but noisier)
- Which require 3D?
  - out-of-plane torso rotation
  - scapular cheating proxies
- When 3D confidence is low, do we fallback to 2D automatically?

**Missing**
- A confidence gate that decides: {use 3D} vs {fallback 2D}

---

## VIII) Minimal Experiment Plan (subset-driven)

1) Scrape ~20 clips (10 strict, 10 cheating)
2) Extract 2D pose with MediaPipe 
3) Lift to relative 3D (prototype; PandaPose-inspired robustness framing) 
4) Normalize + rotate to canonical frame
5) Segment reps
6) Fit harmonic model per rep
7) Compute metrics + compare strict vs cheating separation

**Deliverables**
- `wave_plots/` (per clip)
- `metrics.csv` (per rep)
- One figure: strict vs cheating metric separation

---

## What We Were Missing (Critical)
- Rep segmentation definition + validity criteria
- Confidence propagation (pose quality → metric reliability)
- Explicit 2D fallback when 3D is uncertain
- A single chosen primary waveform (y(t) or θ(t)) to avoid metric explosion

---
