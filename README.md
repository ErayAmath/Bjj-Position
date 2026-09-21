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
| Missing athlete | 36 % of frames have only one of the two poses |
| Segments | 16, inferred from frame-counter resets (no explicit video column) |
| Images | not used; keypoints only |

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # Windows; use .venv/bin/python on Linux/macOS
.venv/Scripts/python -m pytest
```

PyTorch will be installed from the CUDA wheel index once training starts (milestone 1, step 5).

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
  The split must group segments by sparring sequence; how to recover that grouping is part of
  step 1.
- **Finding: confidence values go up to 1.45**, so the "confidence" is not a clean probability.
- **Decision: raw data stays out of git** (size > GitHub's 100 MB limit; licence requires attribution).
