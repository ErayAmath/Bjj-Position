"""Compare predicted poses against the dataset's annotated poses.

Two questions, two metrics:
  1. Were both athletes found?      -> matching predictions to ground truth (`match_poses`)
  2. Are the joints in the right place? -> PCK, the share of keypoints close enough (`pck`)

PCK ("percentage of correct keypoints") counts a joint as correct when it is within
`threshold * scale` pixels of the annotated joint, where `scale` is the size of the annotated
person. Scaling by person size makes the metric independent of how big the athlete appears.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

MIN_VISIBLE_CONFIDENCE = 0.3


def person_scale(keypoints: np.ndarray, visible: np.ndarray) -> float:
    """Size of an annotated person: diagonal of the box around its visible keypoints."""
    if visible.sum() < 2:
        return 0.0
    points = keypoints[visible, :2]
    extent = points.max(axis=0) - points.min(axis=0)
    return float(np.hypot(*extent))


def mean_distance(gt: np.ndarray, pred: np.ndarray, visible: np.ndarray) -> float:
    """Mean pixel distance over the annotated person's visible keypoints."""
    if not visible.any():
        return np.inf
    return float(np.linalg.norm(gt[visible, :2] - pred[visible, :2], axis=1).mean())


def match_poses(gt_poses: np.ndarray, pred_poses: np.ndarray, max_distance_ratio: float = 0.5):
    """Assign each annotated athlete at most one prediction (globally best assignment).

    gt_poses:   (G, 17, 3) annotated [x, y, conf]
    pred_poses: (P, 17, 3) predicted [x, y, conf]
    A pair only counts as matched when the mean keypoint distance is below
    `max_distance_ratio * person_scale`, so a far-away person is not force-matched.

    Returns (matches, costs): matches[g] is the prediction index for athlete g or -1.
    """
    matches = np.full(len(gt_poses), -1, dtype=int)
    costs = np.full(len(gt_poses), np.inf)
    if len(gt_poses) == 0 or len(pred_poses) == 0:
        return matches, costs

    visible = gt_poses[..., 2] >= MIN_VISIBLE_CONFIDENCE
    cost = np.array([[mean_distance(g, p, v) for p in pred_poses]
                     for g, v in zip(gt_poses, visible)])
    # Hungarian algorithm: the assignment with the lowest total cost, rather than greedily
    # taking the best pair first (which can strand the second athlete with a bad partner).
    finite = np.where(np.isfinite(cost), cost, 1e9)
    rows, cols = linear_sum_assignment(finite)
    for g, p in zip(rows, cols):
        scale = person_scale(gt_poses[g], visible[g])
        if scale > 0 and cost[g, p] <= max_distance_ratio * scale:
            matches[g], costs[g] = p, cost[g, p]
    return matches, costs


def pck(gt: np.ndarray, pred: np.ndarray, threshold: float = 0.2) -> tuple[int, int]:
    """(correct, total) keypoints of one matched athlete, at `threshold * person_scale`."""
    visible = gt[:, 2] >= MIN_VISIBLE_CONFIDENCE
    scale = person_scale(gt, visible)
    if scale == 0 or not visible.any():
        return 0, 0
    distances = np.linalg.norm(gt[visible, :2] - pred[visible, :2], axis=1)
    return int((distances <= threshold * scale).sum()), int(visible.sum())
