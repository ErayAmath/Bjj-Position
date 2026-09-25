"""Make the classifier robust to the pose estimator's mistakes.

The model is trained on the dataset's verified keypoints but runs on keypoints predicted from
video, which are noisy and sometimes missing. Training on deliberately degraded copies closes
part of that gap.

Measured end to end on a held-out 2000-frame round (results/round_eval.csv):
    trained on clean poses          62.6 % (18 classes), 66.1 % (10 positions)
    trained with jitter + dropouts  64.6 % (18 classes), 69.4 % (10 positions)
Too much noise destroys the signal instead: 10 % jitter collapsed the model to 12.8 %.
"""

from __future__ import annotations

import numpy as np

LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12


def torso_size(poses: np.ndarray) -> np.ndarray:
    """(N, 2) shoulder-to-hip distance per athlete: the natural unit for "how far is far"."""
    shoulders = poses[:, :, [LEFT_SHOULDER, RIGHT_SHOULDER], :2]
    hips = poses[:, :, [LEFT_HIP, RIGHT_HIP], :2]
    return np.linalg.norm(shoulders - hips, axis=-1).mean(axis=2)


def degrade(poses: np.ndarray, present: np.ndarray, jitter: float = 0.06,
            drop_keypoint: float = 0.05, drop_athlete: float = 0.08,
            rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return a noisy copy of (poses, present), imitating pose-estimation errors.

    jitter:        keypoint displacement as a fraction of torso size (gaussian)
    drop_keypoint: probability that a single keypoint is lost (set to zero)
    drop_athlete:  probability that a whole athlete is lost in a frame; never both at once,
                   because a frame without any athlete carries no information
    """
    rng = rng or np.random.default_rng()
    out = poses.copy()
    if jitter:
        scale = (jitter * torso_size(poses))[..., None, None]
        out[..., :2] += rng.normal(0.0, 1.0, out[..., :2].shape) * scale
    if drop_keypoint:
        out[rng.random(out.shape[:3]) < drop_keypoint] = 0.0

    still_present = present.copy()
    if drop_athlete:
        lost = (rng.random(present.shape) < drop_athlete) & present.all(axis=1)[:, None]
        lost[:, 1] &= ~lost[:, 0]          # keep at least one athlete
        still_present = present & ~lost
    out[~still_present] = 0.0
    return out, still_present
