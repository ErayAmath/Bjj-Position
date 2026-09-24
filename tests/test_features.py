"""Tests for owner task A (`normalize_pose`). They fail until the function is implemented."""

import numpy as np
import pytest

from bjj.features import normalize_pose


def make_frame(offset=(0.0, 0.0), scale=1.0, second_athlete=True):
    """One frame with an upright athlete; optionally a second athlete 2 units to the right."""
    body = np.zeros((17, 3))
    body[5] = [-0.5, -1.0, 1.0]   # left shoulder
    body[6] = [0.5, -1.0, 1.0]    # right shoulder
    body[11] = [-0.5, 0.0, 1.0]   # left hip
    body[12] = [0.5, 0.0, 1.0]    # right hip
    body[0] = [0.0, -1.6, 1.0]    # nose
    frame = np.zeros((2, 17, 3))
    frame[0] = body
    if second_athlete:
        frame[1] = body.copy()
        frame[1, :, 0] += 2.0
    frame[:, :, :2] = frame[:, :, :2] * scale + np.array(offset)
    present = np.array([True, second_athlete])
    frame[~present] = 0
    return frame[None], present[None]


def test_same_pose_at_different_image_positions_gives_the_same_features():
    a, present = make_frame(offset=(100.0, 50.0))
    b, _ = make_frame(offset=(900.0, 700.0))
    np.testing.assert_allclose(normalize_pose(a, present), normalize_pose(b, present), atol=1e-6)


def test_same_pose_at_different_sizes_gives_the_same_features():
    near, present = make_frame(scale=200.0)
    far, _ = make_frame(scale=40.0)
    np.testing.assert_allclose(normalize_pose(near, present), normalize_pose(far, present), atol=1e-6)


def test_relative_position_of_the_two_athletes_is_kept():
    frame, present = make_frame(offset=(300.0, 200.0), scale=100.0)
    out = normalize_pose(frame, present)
    # athlete 2 stands to the right of athlete 1 — that must still be true after normalising
    assert out[0, 1, 11, 0] > out[0, 0, 11, 0]


def test_confidence_column_is_untouched():
    frame, present = make_frame(scale=80.0)
    frame[0, 0, 7, 2] = 0.42
    out = normalize_pose(frame, present)
    np.testing.assert_allclose(out[..., 2], frame[..., 2])


def test_missing_athlete_stays_zero_and_does_not_break_the_frame():
    frame, present = make_frame(offset=(500.0, 500.0), scale=120.0, second_athlete=False)
    out = normalize_pose(frame, present)
    assert np.all(out[0, 1] == 0)
    assert np.isfinite(out[0, 0]).all()


def test_zero_sized_athlete_does_not_produce_nan_or_inf():
    frame = np.zeros((1, 2, 17, 3))
    frame[0, :, :, 2] = 1.0  # confident, but all keypoints at the same spot -> scale 0
    present = np.array([[True, True]])
    assert np.isfinite(normalize_pose(frame, present)).all()


@pytest.mark.parametrize("n", [1, 5])
def test_shape_is_preserved(n):
    frames = np.tile(make_frame(scale=50.0)[0], (n, 1, 1, 1))
    present = np.ones((n, 2), bool)
    assert normalize_pose(frames, present).shape == (n, 2, 17, 3)
