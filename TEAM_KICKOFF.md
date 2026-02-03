# Team Kickoff & Execution Plan

## 🔗 The Dependency Chain (The "Critical Path")
In an AI project like this, work is sequential. You cannot train a model without data, and you cannot analyze performance without a model.

**The Flow:**
`Raw Video (Smit)` → **`Pose Signals (YOU)`** → `Model Training (Vishal)` → `System Integration (Zhan)` → `Analysis (Lochan)`

### 🚨 Crucial Insight
**You (CV Lead) and Smit (Data Lead) are the bottleneck for the first 2 weeks.**
Vishal (Modeling) cannot do anything meaningful until you give him "numbers" (coordinates) to train on. Zhan (Integration) cannot build the app until he knows what the inputs/outputs look like.

---

## 📅 Week 1: Unblocking the Team (The "Hello World" Phase)

### 1. The "Data Contract" (All Leads)
**Goal:** Agree on the file formats so everyone can work in parallel.
*   **Action:** Hold a 30-min meeting to decide:
    *   **Input:** What video format? (MP4, 30fps, 720p?)
    *   **Output:** What does your JSON look like?
        *   *Suggestion:* Use a standard dictionary: `{"frame": 1, "keypoints": [[x,y,c], ...], "score": null}`.
    *   **Keypoints:** 17 points (COCO) or 133 (WholeBody)? (Start with 17).

### 2. Individual Tasks (Parallel Execution)

#### 🟢 **Smit (Data Lead)**
*   **Task:** Download the `Fitness-AQA` dataset (fill the form in README).
*   **Blocker Removal:** If access takes time, record 5 videos of *himself* doing squats/pushups.
*   **Deliverable:** A folder `raw_data/` with 10 sample videos.

#### 🔵 **You (CV Lead)**
*   **Task:** Build the `prototype_processor`.
*   **Action:** Don't wait for the full dataset. Use a webcam video.
*   **Deliverable:** A script `video_to_pose.py` that takes a video and saves a `.npy` or `.json` file.
    *   *Why?* Hand this file to Vishal immediately so he can write his data loaders.

#### 🟣 **Vishal (Modeling Lead)**
*   **Task:** Build the "Dummy Model".
*   **Action:** Write a simple script that takes a random list of 17 keypoints and outputs a random "score" (0-100).
*   **Why?** This tests the *interface* between your code and his code.

#### 🟠 **Zhan (Architecture)**
*   **Task:** repo setup.
*   **Action:** Set up the GitHub repo, folder structure `backend/`, `frontend/`, `models/`, and CI/CD (GitHub Actions).

#### 🟡 **Lochan (Analysis)**
*   **Task:** Define the "Metrics of Success".
*   **Action:** Read the `Fitness-AQA` paper. List out exactly what makes a squat "good" vs "bad" (e.g., "Depth < 90 degrees"). Give this rule list to You and Vishal.

---

## 🏃 Sprint 1 Workflow (The "Trace Bullet")

Do not try to build the perfect system. Build a **"Trace Bullet"** — a gun that fires a bullet through the whole system to prove all parts connect.

1.  **Smit** puts `test_squat.mp4` in a folder.
2.  **You** run `python process.py test_squat.mp4` → produces `test_squat.json`.
3.  **Vishal** runs `python score.py test_squat.json` → produces `Score: 85`.
4.  **Zhan** wraps this in a bash script `run_pipeline.sh`.

**Once this works, the project is "Started".**
