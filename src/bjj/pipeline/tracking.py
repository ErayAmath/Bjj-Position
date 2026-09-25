"""Keep the two athletes apart over time.

The pose model returns people in arbitrary order, so "person 0" in frame 10 may be the other
athlete in frame 11. The classifier however is trained on a fixed order (athlete 1, athlete 2),
and any statistic about *me* needs a stable identity.

Approach: nearest-pose assignment between consecutive frames (Hungarian algorithm on mean
keypoint distance, scaled by body size). A track survives short gaps by keeping its last pose,
so a frame where an athlete was lost does not break the identity.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from bjj.pose_eval import mean_distance, person_scale

NUM_TRACKS = 2


def _cost(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two poses, relative to body size (0 = identical)."""
    visible = (a[:, 2] >= 0.3) & (b[:, 2] >= 0.3)
    scale = max(person_scale(a, a[:, 2] >= 0.3), person_scale(b, b[:, 2] >= 0.3))
    if not visible.any() or scale == 0:
        return np.inf
    return mean_distance(a, b, visible) / scale


def track_athletes(frames: list[np.ndarray], max_cost: float = 0.6, max_age: int = 15):
    """Assign the detections of every frame to two stable tracks.

    frames: per frame an (P, 17, 3) array (P may be 0, 1 or 2).
    max_cost: how far a pose may move between frames, relative to body size, to stay the
        same athlete. 0.6 is generous, because grapplers move fast and poses are noisy.
    max_age: how many frames a track keeps its last pose while unseen before it may be
        re-used for a different person (15 frames ~ 0.6 s at 25 fps).

    Returns:
        poses:   (T, 2, 17, 3) poses per track and frame (zeros where the track was unseen)
        present: (T, 2) bool, True where the track was actually detected in that frame
    """
    n = len(frames)
    poses = np.zeros((n, NUM_TRACKS, 17, 3))
    present = np.zeros((n, NUM_TRACKS), dtype=bool)
    last_pose: list[np.ndarray | None] = [None] * NUM_TRACKS
    last_seen = [-10**9] * NUM_TRACKS

    for t, detections in enumerate(frames):
        detections = np.asarray(detections)
        if not len(detections):
            continue

        active = [i for i in range(NUM_TRACKS) if last_pose[i] is not None and t - last_seen[i] <= max_age]
        assigned: dict[int, int] = {}
        if active:
            cost = np.array([[_cost(last_pose[i], d) for d in detections] for i in active])
            finite = np.where(np.isfinite(cost), cost, 1e9)
            rows, cols = linear_sum_assignment(finite)
            assigned = {active[r]: c for r, c in zip(rows, cols) if cost[r, c] <= max_cost}

        used = set(assigned.values())
        free_tracks = [i for i in range(NUM_TRACKS) if i not in assigned]
        for d in range(len(detections)):
            if d in used:
                continue
            if free_tracks:                      # new athlete -> take an empty/stale track
                assigned[free_tracks.pop(0)] = d
                used.add(d)

        for track, d in assigned.items():
            poses[t, track] = detections[d]
            present[t, track] = True
            last_pose[track] = detections[d]
            last_seen[track] = t

    return poses, present


def fill_track_gaps(poses: np.ndarray, present: np.ndarray, max_gap: int = 10):
    """Carry the last known pose of a track forward across short gaps.

    A missed detection is almost always a model failure, not the athlete leaving. Filling gaps
    raises the share of frames where both athletes are available (measured: 37.9 % -> 81.5 %),
    at the cost of a slightly stale pose.

    Returns a copy of `poses` and a mask of frames that are usable (detected or filled).
    """
    filled = poses.copy()
    usable = present.copy()
    for track in range(poses.shape[1]):
        last_index = None
        for t in range(len(poses)):
            if present[t, track]:
                last_index = t
            elif last_index is not None and t - last_index <= max_gap:
                filled[t, track] = poses[last_index, track]
                usable[t, track] = True
    return filled, usable
