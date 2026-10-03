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


def random_viewpoint(poses: np.ndarray, present: np.ndarray, max_rotation: float = 15.0,
                     min_squeeze: float = 0.55, max_shear: float = 0.25,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Simulate filming the same position from a different camera position.

    Measured motivation: in the dataset's open guard the two athletes' hips are 2.45 torso
    lengths apart and the legs project at 1.54 torso lengths. In footage filmed from one end of
    the mat the same position gives 1.27 and 1.00 — which is what the dataset's *turtle* looks
    like, and turtle is exactly what the model predicted.

    A real viewpoint change is a projection, not an affine map, but squeezing one axis plus a
    small rotation and shear reproduces its main effect: foreshortening along the camera axis.
    The same transform is applied to BOTH athletes of a frame, so their relation survives.
    """
    rng = rng or np.random.default_rng()
    out = poses.copy()
    n = len(poses)

    angle = np.radians(rng.uniform(-max_rotation, max_rotation, n))
    squeeze = rng.uniform(min_squeeze, 1.0, n)
    squeeze_axis = np.radians(rng.uniform(0, 180, n))       # direction that gets compressed
    shear = rng.uniform(-max_shear, max_shear, n)

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    cos_s, sin_s = np.cos(squeeze_axis), np.sin(squeeze_axis)

    # squeeze along an arbitrary axis = rotate into that axis, scale y, rotate back
    for i in range(n):
        rotate_in = np.array([[cos_s[i], sin_s[i]], [-sin_s[i], cos_s[i]]])
        scale = np.array([[1.0, 0.0], [0.0, squeeze[i]]])
        rotate_back = rotate_in.T
        view = rotate_back @ scale @ rotate_in
        camera = np.array([[cos_a[i], -sin_a[i]], [sin_a[i], cos_a[i]]]) @ np.array([[1.0, shear[i]], [0.0, 1.0]])
        out[i, :, :, :2] = poses[i, :, :, :2] @ (camera @ view).T

    out[~present] = 0.0
    return out
