"""Pose estimation for BJJ: two models, because they fail on different frames.

Measured on 358 dataset images (results/pose_benchmark.csv), share of frames where both
athletes were found:

    top-down (YOLOX + RTMPose)   61.7 %
    bottom-up (RTMO)             69.5 %
    union of both                74.6 %

Top-down first runs a person detector; entangled athletes often end up in a single box, so one
of them is lost. RTMO finds keypoints directly and does not have that failure mode, but it
misses people the detector finds. The union is the best of both.
"""

from __future__ import annotations

import numpy as np

from bjj.pose_eval import mean_distance, person_scale

NUM_KEYPOINTS = 17
EMPTY = np.zeros((0, NUM_KEYPOINTS, 3))


def merge_poses(primary: np.ndarray, extra: np.ndarray, min_distance_ratio: float = 0.3) -> np.ndarray:
    """Union of two pose sets: keep `primary`, add poses from `extra` that are new people."""
    merged = list(primary)
    for candidate in extra:
        visible = candidate[:, 2] >= 0.3
        scale = person_scale(candidate, visible)
        if scale == 0:
            continue
        if not any(mean_distance(candidate, kept, visible) < min_distance_ratio * scale for kept in merged):
            merged.append(candidate)
    return np.array(merged) if merged else EMPTY.copy()


def keep_most_confident(poses: np.ndarray, limit: int = 2) -> np.ndarray:
    """Keep the `limit` most confident people (spectators score lower than the athletes)."""
    if len(poses) <= limit:
        return poses
    order = np.argsort(poses[..., 2].mean(axis=1))[::-1]
    return poses[order[:limit]]


class PoseEstimator:
    """Callable: image -> (P, 17, 3) poses as [x, y, confidence]."""

    def __init__(self, mode: str = "balanced", device: str = "cpu", rtmo_score_thr: float = 0.1,
                 ensemble: bool = True, max_people: int = 2):
        from rtmlib import Body

        self.ensemble = ensemble
        self.max_people = max_people
        body = Body(mode=mode, backend="onnxruntime", device=device)
        self.detector, self.top_down = body.det_model, body.pose_model
        self.bottom_up = None
        if ensemble:
            rtmo = Body(pose="rtmo", mode=mode, backend="onnxruntime", device=device).pose_model
            rtmo.score_thr = rtmo_score_thr
            self.bottom_up = rtmo

    def _top_down(self, image: np.ndarray) -> np.ndarray:
        boxes = self.detector(image)
        if not len(boxes):
            return EMPTY.copy()
        keypoints, scores = self.top_down(image, bboxes=boxes)
        return np.concatenate([keypoints, scores[..., None]], axis=-1)

    def _bottom_up(self, image: np.ndarray) -> np.ndarray:
        keypoints, scores = self.bottom_up(image)
        if not len(keypoints):
            return EMPTY.copy()
        return np.concatenate([keypoints, scores[..., None]], axis=-1)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        poses = self._top_down(image)
        if self.ensemble:
            poses = merge_poses(poses, self._bottom_up(image))
        return keep_most_confident(poses, self.max_people)
