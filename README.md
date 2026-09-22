# BJJ Positions

Recognising Brazilian Jiu-Jitsu positions in sparring footage, with the long-term goal of
turning my own rolls into training statistics: where I spend my time, which positions I get
passed from, how often I reach the back.

This is a learning project (I am working through Karpathy's *Neural Networks: Zero to Hero*)
and a portfolio piece. The README doubles as a **lab notebook**: every decision, result and
failure is logged below.

## Data

[Brazilian Jiu-Jitsu Positions Dataset](https://vicos.si/resources/jiujitsu/) by the ViCoS Lab
(University of Ljubljana), licensed CC BY-NC-SA 4.0. According to the authors it was recorded
with 3 smartphone cameras over 6 sparring sequences; poses were detected automatically and
manually verified, positions were labelled manually. The data is **not** included in this
repository (size). Download it yourself and place the annotation file at:

```
data/raw/annotations.json
```

What the file contains (measured, see lab notebook 2026-09-21):

| | |
|---|---|
| Frames | 120,279 |
| Classes | 18 (8 positions × athlete perspective 1/2, plus `standing`, `5050_guard`) |
| Per athlete | 17 COCO keypoints as (x, y, confidence) |
| Missing athlete | 36.5 % of frames have only one of the two poses |
| Segments | 16 camera views, inferred from frame-counter resets (no explicit video column) |
| Sequences | 6 sparring sequences, recovered by grouping segments (see lab notebook) |
| Images | not used; keypoints only |

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # Windows; use .venv/bin/python on Linux/macOS
.venv/Scripts/python -m pytest
```

PyTorch will be installed from the CUDA wheel index once training starts (milestone 1, step 5).

## Frontend

A static dashboard (plain HTML/CSS/JS, no build step) lives in `frontend/`. It shows the
dataset statistics and animates real skeleton sequences per position. `frontend/data/` holds
derived dataset excerpts and is therefore under the dataset's CC BY-NC-SA 4.0 licence.

```bash
.venv/Scripts/python scripts/export_frontend_data.py   # writes frontend/data/*.json
.venv/Scripts/python scripts/serve_frontend.py        # no-cache dev server
```

Then open <http://localhost:8000>.

## Backend plan (not started)

The frontend already has an upload section. It only previews the video locally and calls a
backend on the same machine once one exists. Planned design:

- **Local only.** A FastAPI app on `localhost` serves `frontend/` and `/api`. Videos are stored
  under `data/uploads/` (git-ignored) and never leave the machine. Confidentiality comes from
  the architecture, not from a promise.
- **API contract** (already used by `frontend/js/upload.js`):
  `GET /api/health`, `POST /api/analyses` (multipart `video`) -> `202 {"id"}`,
  later `GET /api/analyses/{id}` for status and results.
- **Pipeline** (runs as a background job): pretrained pose estimator producing the same
  17 COCO keypoints as the dataset -> track both athletes and let the user pick which one is
  them -> position classifier from M1/M2 -> temporal smoothing -> statistics JSON.
- **Coaching chat** via an LLM API (e.g. Claude). It receives only the aggregated statistics,
  never video or keypoints. The API key lives in an environment variable on the backend,
  never in the frontend.
- **Order:** the pipeline needs M1 (classifier), M2 (time) and M3 (pose estimation on own
  footage) first. A backend skeleton (health check, upload, job status) can start after M1.

## Project layout

```
src/bjj/        library code (data loading, later: features, models, evaluation)
tests/          pytest tests
scripts/        one-off entry points (exports, training runs)
results/        committed experiment outputs (CSV/JSON/figures)
frontend/       static dashboard
data/           raw data, git-ignored
```

## Roadmap

- **M1 — Keypoint baseline:** single-frame classifier on keypoints, honest video-level split,
  baselines, MLP in PyTorch, confusion matrix, split and normalisation experiments.
- **M2 — Time:** use neighbouring frames (smoothing, temporal model).
- **M3 — My own footage:** pose estimation on my videos, identifying which athlete is me.
- **M4 — Training statistics:** time per position, transitions, where I get passed.

## Lab notebook

### 2026-09-22 — Phase 0: off-the-shelf pose estimation on sparring clips

Setup: `scripts/pose_spike.py`, rtmlib "balanced" (YOLOX-m detector + RTMPose-m, top-down),
CPU, frames downscaled to 1080 px wide, sampled at 12.5 fps. Input: two 11–17 s stock clips
(handheld, close-up, portrait 4K, kids in dark gis on a dark mat) — **not** my own footage.

| Clip | frames | exactly 2 people | 1 person | 3+ | mean kp conf | s/frame (CPU) |
|---|---|---|---|---|---|---|
| 7988417 (11 s) | 137 | 27 % | 72 % | 1 % | 0.49 | 0.145 |
| 7988994 (17 s) | 209 | 36 % | 61 % | 2 % | 0.46 | 0.141 |

- **Prediction before the run:** ~70 % of frames with two people; problems in clinches,
  scrambles and unstructured exchanges. Result: far below the guess; the qualitative
  prediction (entangled phases fail) was right.
- **Two causes mixed together.** (1) Footage: the camera follows the action closely, so often
  only one athlete is fully in frame — "1 person" is then correct. (2) Real model failure:
  with both athletes clearly visible but entangled on the ground, the detector returns a
  single box, and some skeletons mix limbs of both athletes.
- **Consequence:** these clips cannot tell us how well the pipeline works on my own training
  footage. Needed next: a clip recorded like the dataset (fixed camera, whole mat, both
  athletes fully visible), then compare top-down vs. bottom-up (RTMO) on it.

### 2026-09-21 — Project setup, first look at the data

- **Decision: keypoints only, no images.** Only `annotations.json` is available locally, and a
  keypoint MLP is the natural next step after micrograd. Rejected: CNN on frames (needs the
  images, much bigger jump).
- **Finding: frames are heavily correlated.** Consecutive frames almost always share a label
  (only 35 label changes between directly consecutive frames). A random frame-level split would
  put near-identical neighbours into train and test and inflate accuracy.
- **Decision: split by recording, not by frame.** There is no video column, but the frame
  counter resets 15 times, giving 16 segments of 5–9k frames (`bjj.data.infer_video_ids`).
- **Open problem: segments are camera views, not independent sparring sessions.** The authors
  filmed 6 sparring sequences with 3 cameras. Some segments are the same moment from another
  angle (two segments have exactly 8,609 frames each). A split by segment would still leak.
  The split must group segments by sparring sequence.
- **Finding: the 16 segments form 6 sequences — and each sequence drilled only a few positions.**
  Adjacent segments have cosine similarity of their class mix of either >= 0.9 or <= 0.01, so
  grouping them is unambiguous (`bjj.stats.group_segments`): 00–02, 03–05, 06–08, 09–10, 11–13,
  14–15 (two sequences have only two camera views). But the grouping reveals a bigger problem:
  sequence 1 is almost only open/closed guard, sequence 6 only back, etc. A split by sequence
  would remove whole classes from training. **Open decision for step 4:** e.g. hold out one
  camera view per sequence (tests viewpoint generalisation, but the same moments are in train),
  or hold out contiguous time blocks within each sequence across all its cameras (honest in
  time, keeps all classes). To be discussed before any split is implemented.
- **Upload section added to the frontend** without a backend: honest status instead of fake
  results. Verified that clicking "Start analysis" only issues `GET /api/health` and does not
  transmit the video when no backend answers.
- **Frontend palette** (athlete 1 gi white, athlete 2 belt red, symmetric gray) validated with a
  colour-vision-deficiency check: worst adjacent dE 8.8 (protan, gray vs red), target >= 8.
  Muted label gray raised to #80848b for >= 4.5:1 contrast.
- **Finding: confidence values go up to 1.45**, so the "confidence" is not a clean probability.
- **Decision: raw data stays out of git** (size > GitHub's 100 MB limit; licence requires attribution).
