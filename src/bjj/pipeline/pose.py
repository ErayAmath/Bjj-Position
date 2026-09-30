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
    """Keep the `limit` most confident people."""
    if len(poses) <= limit:
        return poses
    order = np.argsort(poses[..., 2].mean(axis=1))[::-1]
    return poses[order[:limit]]


def _person_geometry(pose: np.ndarray) -> tuple[np.ndarray, float]:
    """(centre, size) of one detected person, from its reliable keypoints."""
    visible = pose[:, 2] >= 0.3
    if visible.sum() < 2:
        return np.zeros(2), 0.0
    points = pose[visible, :2]
    return points.mean(axis=0), float(np.hypot(*(points.max(axis=0) - points.min(axis=0))))


def select_rolling_pair(poses: np.ndarray) -> np.ndarray:
    """Pick the two people who are actually rolling, out of everyone in frame.

    Confidence is the wrong criterion in a busy gym: a bystander standing in full view scores
    higher than two entangled grapplers, so the pipeline ended up tracking spectators.

    Two properties separate the pair from everyone else:
      * they are close to the camera, so they are LARGE in the image,
      * they are grappling, so they are CLOSE TO EACH OTHER relative to their own size.

    The score multiplies both, which keeps a big pair that briefly separates while rejecting a
    distant pair that happens to stand next to each other.
    """
    if len(poses) <= 2:
        return poses
    geometry = [_person_geometry(p) for p in poses]
    best, best_score = None, -np.inf
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            (centre_i, size_i), (centre_j, size_j) = geometry[i], geometry[j]
            if size_i == 0 or size_j == 0:
                continue
            distance = float(np.linalg.norm(centre_i - centre_j))
            closeness = 1.0 / (1.0 + distance / (0.5 * (size_i + size_j)))
            score = (size_i + size_j) * closeness
            if score > best_score:
                best, best_score = (i, j), score
    return poses[list(best)] if best else keep_most_confident(poses, 2)


class PoseEstimator:
    """Callable: image -> (P, 17, 3) poses as [x, y, confidence]."""

    def __init__(self, mode: str = "balanced", device: str = "cpu", rtmo_score_thr: float = 0.1,
                 ensemble: bool = True, max_people: int = 2, pick_rolling_pair: bool = True):
        from rtmlib import Body

        self.ensemble = ensemble
        self.max_people = max_people
        self.pick_rolling_pair = pick_rolling_pair
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
        if self.pick_rolling_pair and self.max_people == 2:
            return select_rolling_pair(poses)
        return keep_most_confident(poses, self.max_people)
