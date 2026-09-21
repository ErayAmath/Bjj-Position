"""Loading the ViCoS BJJ annotations into NumPy arrays.

The raw file is a JSON list with one entry per frame:
    {"position": str, "image": str, "frame": int,
     "pose1": [[x, y, conf] * 17], "pose2": [[x, y, conf] * 17]}
Either pose may be missing when the pose estimator lost an athlete.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NUM_KEYPOINTS = 17  # COCO keypoint layout

# COCO skeleton edges (index pairs), used for drawing.
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # head
    (5, 6), (5, 11), (6, 12), (11, 12),        # torso
    (5, 7), (7, 9), (6, 8), (8, 10),           # arms
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
]


@dataclass
class Annotations:
    poses: np.ndarray      # (N, 2, 17, 3) float32: athlete, keypoint, (x, y, conf)
    present: np.ndarray    # (N, 2) bool: was the athlete detected?
    labels: np.ndarray     # (N,) int64 index into `classes`
    classes: list[str]     # sorted class names
    frames: np.ndarray     # (N,) int64 frame number within its video
    video_ids: np.ndarray  # (N,) int64, inferred by `infer_video_ids`

    def __len__(self) -> int:
        return len(self.labels)


def infer_video_ids(frames: np.ndarray) -> np.ndarray:
    """Assign a video id to every frame.

    The dataset has no explicit video column. Entries are stored in order and the
    frame counter restarts for every video, so a *decrease* in the frame number
    marks a new video. Forward gaps (frames skipped inside a video) do not.
    """
    frames = np.asarray(frames)
    if frames.size == 0:
        return np.zeros(0, dtype=np.int64)
    starts_new_video = np.diff(frames) < 0
    return np.concatenate([[0], np.cumsum(starts_new_video)]).astype(np.int64)


def load_annotations(path: str | Path) -> Annotations:
    """Parse the raw JSON file. Missing poses become zeros with present=False."""
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)

    n = len(entries)
    poses = np.zeros((n, 2, NUM_KEYPOINTS, 3), dtype=np.float32)
    present = np.zeros((n, 2), dtype=bool)
    for i, entry in enumerate(entries):
        for athlete, key in enumerate(("pose1", "pose2")):
            if key in entry:
                poses[i, athlete] = entry[key]
                present[i, athlete] = True

    classes = sorted({e["position"] for e in entries})
    class_index = {name: i for i, name in enumerate(classes)}
    labels = np.array([class_index[e["position"]] for e in entries], dtype=np.int64)
    frames = np.array([e["frame"] for e in entries], dtype=np.int64)

    return Annotations(poses, present, labels, classes, frames, infer_video_ids(frames))


def contiguous_runs(frames: np.ndarray, labels: np.ndarray) -> list[tuple[int, int]]:
    """Split the frame sequence into runs of consecutive frames with one label.

    Returns half-open index ranges [start, end). A run breaks when the frame number
    does not increase by exactly 1 or when the label changes.
    """
    runs = []
    start = 0
    for i in range(1, len(frames) + 1):
        if i == len(frames) or frames[i] != frames[i - 1] + 1 or labels[i] != labels[i - 1]:
            runs.append((start, i))
            start = i
    return runs
