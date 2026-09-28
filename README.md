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

**Own footage stays private:** `export_round_to_frontend.py` writes to the git-ignored
`frontend/data/private/`, and the dashboard prefers that file over the committed demo round.
Only `--demo` touches the public one.

**Live: https://erayamath.github.io/Bjj-Position/** — published from `frontend/` by
`.github/workflows/pages.yml` on every push to `main`. The upload section is inert there: the
analysis pipeline runs locally, so the public page reports that no backend is reachable.

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

## Pipeline

```
frames ──► pose estimation ──► tracking ──► features ──► MLP ──► Viterbi ──► timeline
           bjj.pipeline.pose   bjj.pipeline bjj.features bjj.train bjj.temporal
                               .tracking
```

| Stage | What it does | Why it is built that way |
|---|---|---|
| Pose | Top-down (YOLOX + RTMPose) **and** bottom-up (RTMO), results merged | The two models fail on different frames; the union found both athletes in 74.6 % of frames vs 61.7 % for the default |
| Tracking | Hungarian nearest-pose assignment, plus gap filling | The pose model returns people in arbitrary order; the classifier and any statistic about *me* need a stable identity |
| Features | Owner task A normalisation + athlete-relation geometry + neighbouring frames | Raw pixel coordinates do not transfer to another camera (10.7 % vs 68.1 %) |
| Model | MLP (PyTorch), athlete-swap augmentation, camera-holdout split | Swap augmentation stops the model keying on annotation order; the split stops leakage between camera views of the same moment |
| Time | Transition matrix from the dataset + Viterbi | Positions last seconds and not every transition is possible, so the most likely *sequence* beats per-frame guesses |

```bash
# train (needs owner tasks A and B)
.venv/Scripts/python scripts/train_position_model.py --epochs 40

# analyse a round
.venv/Scripts/python scripts/analyze_round.py data/videos/roll.mp4 --fps 10
.venv/Scripts/python scripts/export_round_to_frontend.py data/analyses/roll.json
.venv/Scripts/python scripts/serve_frontend.py            # add --lan to reach it from a phone

# measure the whole pipeline against the dataset labels
.venv/Scripts/python scripts/precompute_poses.py --frames data/raw/round_images     --manifest data/raw/round_manifest.json --name round_images --fps 25
.venv/Scripts/python scripts/eval_round.py --manifest data/raw/round_manifest.json --name round_images
```

## Licences

| Part | Licence | Consequence |
|---|---|---|
| Code in this repository | MIT (see `LICENSE`) | free to use |
| `frontend/data/*.json` (derived from the dataset) | CC BY-NC-SA 4.0 | attribution, non-commercial, share-alike |
| ViCoS BJJ dataset (not included) | CC BY-NC-SA 4.0 | non-commercial |
| YOLOX detector weights (Human-Art) | CC BY-NC-SA 4.0 | non-commercial |
| RTMPose / RTMO weights (body7 mix) | mixed, check per source dataset | verify before any product use |

**This project is therefore research/portfolio only.** A commercial product would need its own
recorded and labelled data plus models trained on permissively licensed sources. Noted here so
the constraint is not discovered late.

## Project layout

```
src/bjj/        library code: data, stats, features, train, temporal, pose_eval, pipeline/
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

### 2026-09-26 — Milestone 1 done: a working end-to-end MVP

Owner tasks A (`normalize_pose`) and B (`train_step`) are implemented, the MLP is trained, and
the whole pipeline runs from frames to a position timeline (`scripts/analyze_round.py`).

**Classifier** (MLP, 936 features incl. ±2 s context, athlete-swap and noise augmentation,
40 epochs, 59 s on an RTX 4070)

| set | what it measures | accuracy |
|---|---|---|
| validation (time blocks inside training cameras) | new moments, known camera | 79.2 % |
| test (one untouched camera per sequence) | known moment, new angle | 90.4 % |

**The test number is optimistic, and that is the main lesson of this milestone.** The held-out
camera films the *same* sparring at the *same* second as the training cameras, so the model only
has to recognise a known moment from a new angle.

An attempt to measure real generalisation by holding out a whole sparring sequence collapsed to
0.4–27 % accuracy — but that measurement is invalid: each position class lives almost entirely
in one sequence (holding out sequence 3 leaves 0 training frames for mount1 and side_control1).
**This dataset cannot answer "does it work on a new roll with new people".** Only own footage can.

**End to end** (`scripts/eval_round.py`, 2000-frame round from the held-out camera, real
predicted poses): **70.5 % over 18 classes, 84.5 % over the 10 base positions.** The gap between
the two says the model usually gets the position right but confuses *which athlete* holds it —
in the product that is solved by the user pointing at themselves once.

Viterbi smoothing barely moves accuracy (+0.4 points) but cuts the flicker from 208 position
changes to 44 (ground truth: 25). Gap filling did not help here: the pose ensemble already found
both athletes in 92 % of this round's frames.

Example output (`data/analyses/round_images.json`, visible in the dashboard): 80 s round,
65.5 % standing, 20.9 % takedowns, the rest guard, side control, mount and back.

### 2026-09-24 — Improving recognition: pose stage and a first end-to-end number

Test data: 358 dataset images sampled evenly over classes and sequences, plus one contiguous
400-frame clip (side control). Downloaded via HTTP range requests from the 10 GB archive
(`scripts/fetch_dataset_images.py`, 60 MB instead of 10 GB). All numbers: results/summary.csv.

**Pose stage** (`scripts/eval_pose_on_dataset.py`, metric: both athletes found / PCK@0.2)

| configuration | both found | PCK@0.2 | s/image |
|---|---|---|---|
| top-down YOLOX-m + RTMPose-m (default) | 61.7 % | 88.7 % | 0.11 |
| RTMO bottom-up, score_thr 0.1 | 69.5 % | 85.6 % | 0.09 |
| **ensemble of both** | **74.6 %** | 86.8 % | 0.21 |
| ensemble with the large models | 75.4 % | 88.6 % | 0.57 |

- **Detector thresholds are a dead end**: the shipped ONNX detector has NMS baked in, so
  `score_thr`/`nms_thr` in rtmlib are ignored (verified: 0.05 and 0.7 give identical boxes).
- **Failures are real, not a metric artefact**: of 358 images, top-down produced fewer than two
  people in 133; only 2 predictions were rejected as too far from the annotation.
- Per class: standing 100 %, open guard ~95 %, but back2 45 %, mount1 60 %, side control1 60 %.
  Entangled positions are where it breaks — as predicted.
- **Temporal repair** (`bjj.pose_fill`, 400-frame clip): both found 37.9 % -> 67.5 % (gap 5) ->
  81.5 % (gap 10), while PCK drops 77.0 % -> 68.9 %: borrowed poses are stale.

**Classifier baseline** (`scripts/train_baseline.py`, logistic regression, athlete-swap
augmentation, held-out camera view per sequence)

| setup | accuracy (18 classes) |
|---|---|
| majority class | 12.8 % |
| raw keypoints, **random frame split** (leaky control) | 81.7 % |
| raw keypoints, honest camera holdout | 10.7 % |
| **normalised keypoints**, honest camera holdout | 68.1 % |

- The leaky split would have reported 81.7 % for a model that transfers *nothing* to a new
  camera (10.7 %). This is the clearest possible demonstration of why the split matters.
- Normalisation is worth ~57 points. It is owner task A (`bjj.features.normalize_pose`); the
  68.1 % above come from a throwaway preview implementation, not from committed library code.

**End to end** (images -> ensemble pose -> normalised baseline classifier, held-out camera):
43.2 % over 18 classes, 49.2 % over the 10 base positions, 55.1 % on the frames where both
athletes were found. On the side-control clip, per-frame 28.0 %, 31.8 % with a majority filter
over ±12 frames.

- Pose noise costs ~25 points versus the same classifier on annotated keypoints (68.1 %).
- Both stages need work; the pose stage is the bigger problem in entangled positions.

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
