import numpy as np

from bjj.pipeline.tracking import fill_track_gaps, track_athletes


def pose(x: float, y: float, spread: float = 100.0) -> np.ndarray:
    offsets = np.linspace(0, spread, 17)
    return np.column_stack([x + offsets, y + offsets, np.ones(17)])


def test_identity_survives_when_the_detection_order_flips():
    a, b = pose(0, 0), pose(400, 0)
    frames = [np.stack([a, b]), np.stack([b, a]), np.stack([a, b])]  # order swapped in frame 1
    poses, present = track_athletes(frames)
    assert present.all()
    # track 0 must stay with athlete A in every frame
    for t in range(3):
        assert poses[t, 0, 0, 0] == a[0, 0]
        assert poses[t, 1, 0, 0] == b[0, 0]


def test_tracks_follow_movement():
    frames = [np.stack([pose(i * 10, 0), pose(400 - i * 5, 0)]) for i in range(10)]
    poses, present = track_athletes(frames)
    assert present.all()
    assert poses[-1, 0, 0, 0] == 90     # moved right
    assert poses[-1, 1, 0, 0] == 355    # moved left


def test_missing_detection_marks_track_absent():
    a, b = pose(0, 0), pose(400, 0)
    frames = [np.stack([a, b]), np.stack([a]), np.stack([a, b])]
    _, present = track_athletes(frames)
    np.testing.assert_array_equal(present[:, 0], [True, True, True])
    np.testing.assert_array_equal(present[:, 1], [True, False, True])


def test_fill_track_gaps_carries_the_last_pose_forward():
    a, b = pose(0, 0), pose(400, 0)
    frames = [np.stack([a, b]), np.stack([a]), np.stack([a, b])]
    poses, present = track_athletes(frames)
    filled, usable = fill_track_gaps(poses, present, max_gap=2)
    assert usable.all()
    np.testing.assert_array_equal(filled[1, 1], b)


def test_fill_track_gaps_respects_max_gap():
    a, b = pose(0, 0), pose(400, 0)
    frames = [np.stack([a, b])] + [np.stack([a])] * 4 + [np.stack([a, b])]
    poses, present = track_athletes(frames)
    _, usable = fill_track_gaps(poses, present, max_gap=2)
    np.testing.assert_array_equal(usable[:, 1], [True, True, True, False, False, True])


def test_empty_frames_do_not_crash():
    poses, present = track_athletes([np.zeros((0, 17, 3)), np.zeros((0, 17, 3))])
    assert poses.shape == (2, 2, 17, 3)
    assert not present.any()
