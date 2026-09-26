"""Turning raw keypoints into features a model can learn from.

Measured motivation (results/summary.csv): with raw pixel coordinates the baseline reaches
10.7 % accuracy on a held-out camera — barely above guessing. A quick normalisation preview
reached 68.1 %. Normalisation is the single biggest lever in the whole project.
"""

from __future__ import annotations

import numpy as np

# COCO keypoint indices used for the reference frame.
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12

MIN_KEYPOINT_CONFIDENCE = 0.1   # below this a keypoint carries no usable position
MIN_SIZE = 1e-6                 # never divide by (almost) zero


def normalize_pose(poses: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Make poses independent of where in the image the athletes are and how big they appear.

    Args:
        poses:   (N, 2, 17, 3) float array of [x, y, confidence] per athlete and keypoint.
        present: (N, 2) bool array — False where the athlete was not detected at all.

    Returns:
        (N, 2, 17, 3) float array. The confidence column stays untouched; only x and y change:
          * both athletes of a frame are shifted by the SAME offset, and scaled by the SAME
            factor, so their relative position to each other survives (that is what tells
            mount apart from being mounted),
          * the result is centred on the hip midpoint and scaled so that a typical torso has
            length 1,
          * rows of athletes with present == False stay all zeros.

    Edge cases to handle:
        * an athlete whose hips or shoulders were not detected (confidence ~ 0),
        * a torso length of 0 (never divide by zero),
        * frames where only one athlete is present — the reference frame then comes from that
          athlete alone.
    """
    normalised = np.asarray(poses, dtype=np.float64).copy()
    xy = normalised[..., :2]                     # a view: writing to it writes into `normalised`
    confidence = normalised[..., 2]

    reliable = (confidence >= MIN_KEYPOINT_CONFIDENCE) & present[..., None]   # (N, 2, 17)
    hips_reliable = reliable[:, :, [LEFT_HIP, RIGHT_HIP]].all(axis=2)         # (N, 2)
    torso_reliable = hips_reliable & reliable[:, :, [LEFT_SHOULDER, RIGHT_SHOULDER]].all(axis=2)

    hip_centre = xy[:, :, [LEFT_HIP, RIGHT_HIP]].mean(axis=2)                 # (N, 2, 2)
    shoulder_centre = xy[:, :, [LEFT_SHOULDER, RIGHT_SHOULDER]].mean(axis=2)

    # Fallback when the hips were not detected: the centre of whatever keypoints are reliable.
    weight = reliable[..., None]
    reliable_count = reliable.sum(axis=2)                                     # (N, 2)
    keypoint_centre = (xy * weight).sum(axis=2) / np.maximum(reliable_count, 1)[..., None]
    centre = np.where(hips_reliable[..., None], hip_centre, keypoint_centre)  # (N, 2, 2)
    has_centre = present & (hips_reliable | (reliable_count > 0))

    # Fallback when the torso is unusable: the spread of the reliable keypoints around the centre.
    distance = np.linalg.norm(xy - centre[:, :, None, :], axis=-1)            # (N, 2, 17)
    spread = np.sqrt((distance**2 * reliable).sum(axis=2) / np.maximum(reliable_count, 1))
    torso = np.linalg.norm(shoulder_centre - hip_centre, axis=-1)             # (N, 2)
    size = np.where(torso_reliable, torso, 2 * spread)
    has_size = has_centre & (size > MIN_SIZE)

    # One reference point and ONE scale per frame, averaged over the usable athletes: both
    # athletes must be moved and scaled together, otherwise "who is on top" is lost.
    frame_centre = (centre * has_centre[..., None]).sum(axis=1) / np.maximum(
        has_centre.sum(axis=1), 1)[:, None]                                   # (N, 2)
    frame_size = (size * has_size).sum(axis=1) / np.maximum(has_size.sum(axis=1), 1)
    frame_size = np.where(has_size.any(axis=1) & (frame_size > MIN_SIZE), frame_size, 1.0)

    xy -= frame_centre[:, None, None, :]
    xy /= frame_size[:, None, None, None]
    normalised[~present] = 0.0                   # athletes that were never detected stay zero
    return normalised


def pose_features(poses: np.ndarray, present: np.ndarray) -> np.ndarray:
    """(N, 2, 17, 3) -> (N, 104): normalised skeletons flattened, plus two presence flags.

    The presence flags matter: an athlete that was never detected is all zeros, and without a
    flag the model cannot tell "missing" from "exactly at the reference point".
    """
    normalised = normalize_pose(poses, present)
    return np.concatenate([normalised.reshape(len(poses), -1), present.astype(np.float32)], axis=1)


def pair_features(normalised: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Geometric relations BETWEEN the two athletes, computed from normalised poses.

    The flat keypoint list already contains this information implicitly, but a linear or small
    model has to discover it. Handing it over directly is cheap and usually worth several
    accuracy points: who is above whom is what separates mount from being mounted.

    Returns (N, 8): hip-to-hip vector (2) and distance (1), which athlete is higher (1), torso
    angle of each athlete (2), their difference (1), and the shoulder height gap (1).
    """
    hips = normalised[:, :, [LEFT_HIP, RIGHT_HIP], :2].mean(axis=2)        # (N, 2, 2)
    shoulders = normalised[:, :, [LEFT_SHOULDER, RIGHT_SHOULDER], :2].mean(axis=2)
    torso = shoulders - hips                                               # (N, 2, 2)

    delta = hips[:, 1] - hips[:, 0]                                        # athlete 2 relative to 1
    distance = np.linalg.norm(delta, axis=1, keepdims=True)
    # Image coordinates: y grows downwards, so a smaller y means higher up.
    above = np.sign(hips[:, 0, 1] - hips[:, 1, 1])[:, None]
    angles = np.arctan2(torso[..., 1], torso[..., 0])                      # (N, 2)
    angle_difference = np.arctan2(np.sin(angles[:, 1] - angles[:, 0]),
                                  np.cos(angles[:, 1] - angles[:, 0]))[:, None]
    shoulder_gap = (shoulders[:, 1, 1] - shoulders[:, 0, 1])[:, None]

    out = np.concatenate([delta, distance, above, angles, angle_difference, shoulder_gap], axis=1)
    # Relations are meaningless when an athlete is missing.
    return np.where(present.all(axis=1)[:, None], out, 0.0)


def frame_features(poses: np.ndarray, present: np.ndarray, with_pairs: bool = True) -> np.ndarray:
    """Per-frame feature vector: normalised keypoints + presence flags (+ pair geometry)."""
    normalised = normalize_pose(poses, present)
    parts = [normalised.reshape(len(poses), -1), present.astype(np.float32)]
    if with_pairs:
        parts.append(pair_features(normalised, present))
    return np.concatenate(parts, axis=1).astype(np.float32)


def add_context(X: np.ndarray, groups: np.ndarray, offsets: tuple[int, ...] = (-6, -2, 2, 6)) -> np.ndarray:
    """Append the features of neighbouring frames (same video only).

    A single frame is ambiguous; half a second of context tells a static position apart from a
    transition. Frames near a video boundary repeat the closest frame inside the same video, so
    no information leaks across a cut.
    """
    n = len(X)
    index = np.arange(n)
    parts = [X]
    for offset in offsets:
        shifted = np.clip(index + offset, 0, n - 1)
        same_video = groups[shifted] == groups
        shifted = np.where(same_video, shifted, index)
        parts.append(X[shifted])
    return np.concatenate(parts, axis=1)


# Context is specified in SECONDS, not frames: the dataset runs at 25 fps while a user video
# may be analysed at 10 fps, and the model must see the same time span in both cases.
DEFAULT_CONTEXT_SECONDS = (-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0)


def context_offsets(context_seconds, fps: float) -> tuple[int, ...]:
    """Seconds -> frame offsets at this frame rate, dropping duplicates and 0.

    Rounds away from zero: Python's round() would turn 0.5 into 0 (banker's rounding) and
    silently drop an offset at low frame rates.
    """
    offsets = {int(np.sign(seconds) * int(abs(seconds) * fps + 0.5)) for seconds in context_seconds}
    return tuple(sorted(offset for offset in offsets if offset != 0))


def build_features(poses: np.ndarray, present: np.ndarray, groups: np.ndarray, fps: float,
                   with_pairs: bool = False,
                   context_seconds=DEFAULT_CONTEXT_SECONDS) -> np.ndarray:
    """The single entry point used by both training and inference.

    Training and inference must build features identically, so the settings are stored with the
    model and passed back in here. `groups` marks video boundaries (all one video at inference),
    `fps` is the frame rate of THIS data.

    with_pairs defaults to False: measured on the held-out camera, the relation features made no
    difference (74.4 % without, 74.1 % with) — the network learns those relations by itself.
    """
    X = frame_features(poses, present, with_pairs)
    offsets = context_offsets(context_seconds, fps) if context_seconds else ()
    return add_context(X, groups, offsets) if offsets else X
