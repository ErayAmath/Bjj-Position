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
    # TODO(human): implement the normalisation described above.
    #
    # Suggested order:
    #   1. per frame, compute the reference point (hip midpoint) from the present athletes
    #   2. per frame, compute a scale (e.g. shoulder-to-hip distance) from the present athletes
    #   3. subtract the reference point from x/y, divide by the scale
    #   4. zero out athletes that are not present, keep the confidence column as it is
    #
    # Run `python -m pytest tests/test_features.py` until everything passes.
    raise NotImplementedError("owner task A")


def pose_features(poses: np.ndarray, present: np.ndarray) -> np.ndarray:
    """(N, 2, 17, 3) -> (N, 104): normalised skeletons flattened, plus two presence flags.

    The presence flags matter: an athlete that was never detected is all zeros, and without a
    flag the model cannot tell "missing" from "exactly at the reference point".
    """
    normalised = normalize_pose(poses, present)
    return np.concatenate([normalised.reshape(len(poses), -1), present.astype(np.float32)], axis=1)
