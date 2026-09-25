"""Tests for the feature helpers that sit around owner task A."""

import numpy as np

from bjj.features import add_context, pair_features


def athlete(x: float, y: float) -> np.ndarray:
    """Upright athlete: shoulders above hips, centred at (x, y)."""
    person = np.zeros((17, 3))
    person[5] = [x - 0.5, y - 1.0, 1.0]   # left shoulder
    person[6] = [x + 0.5, y - 1.0, 1.0]   # right shoulder
    person[11] = [x - 0.5, y, 1.0]        # left hip
    person[12] = [x + 0.5, y, 1.0]        # right hip
    return person


def frame(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.stack([a, b])[None]


def test_pair_features_measure_the_gap_between_the_athletes():
    poses = frame(athlete(0, 0), athlete(3, 0))
    out = pair_features(poses, np.array([[True, True]]))
    np.testing.assert_allclose(out[0, :2], [3.0, 0.0])   # athlete 2 is 3 units to the right
    np.testing.assert_allclose(out[0, 2], 3.0)           # distance


def test_pair_features_know_who_is_higher_up():
    # y grows downwards: athlete 2 at y = -2 is ABOVE athlete 1 at y = 0
    out = pair_features(frame(athlete(0, 0), athlete(0, -2)), np.array([[True, True]]))
    assert out[0, 3] == 1.0
    flipped = pair_features(frame(athlete(0, -2), athlete(0, 0)), np.array([[True, True]]))
    assert flipped[0, 3] == -1.0


def test_pair_features_are_zero_when_an_athlete_is_missing():
    poses = frame(athlete(0, 0), np.zeros((17, 3)))
    out = pair_features(poses, np.array([[True, False]]))
    np.testing.assert_array_equal(out[0], np.zeros(8))


def test_add_context_appends_neighbouring_frames():
    X = np.arange(8).reshape(4, 2).astype(float)
    groups = np.zeros(4, dtype=int)
    out = add_context(X, groups, offsets=(-1, 1))
    assert out.shape == (4, 6)
    np.testing.assert_array_equal(out[1], [2, 3, 0, 1, 4, 5])


def test_add_context_does_not_cross_a_video_boundary():
    X = np.arange(8).reshape(4, 2).astype(float)
    groups = np.array([0, 0, 1, 1])
    out = add_context(X, groups, offsets=(1,))
    np.testing.assert_array_equal(out[1], [2, 3, 2, 3])   # would be frame 2 -> repeats itself
    np.testing.assert_array_equal(out[2], [4, 5, 6, 7])   # inside video 1 the neighbour is used


def test_add_context_clamps_at_the_ends():
    X = np.arange(6).reshape(3, 2).astype(float)
    groups = np.zeros(3, dtype=int)
    out = add_context(X, groups, offsets=(-2,))
    np.testing.assert_array_equal(out[0], [0, 1, 0, 1])


def test_context_offsets_scale_with_frame_rate():
    from bjj.features import context_offsets
    assert context_offsets((-1.0, -0.5, 0.5, 1.0), 25) == (-25, -13, 13, 25)
    assert context_offsets((-1.0, -0.5, 0.5, 1.0), 10) == (-10, -5, 5, 10)


def test_context_offsets_drop_duplicates_and_zero():
    from bjj.features import context_offsets
    # at 2 fps, 0.1 s rounds to 0 and must be dropped; 0.25 s rounds away from zero to +-1
    assert context_offsets((-1.0, -0.25, -0.1, 0.1, 0.25, 1.0), 2) == (-2, -1, 1, 2)
