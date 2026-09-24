import numpy as np

from bjj.pose_fill import fill_missing_people


def pose(x: float, y: float, spread: float = 100.0) -> np.ndarray:
    offsets = np.linspace(0, spread, 17)
    return np.column_stack([x + offsets, y + offsets, np.ones(17)])


def test_fills_a_single_dropped_frame_from_its_neighbour():
    a, b = pose(0, 0), pose(400, 400)
    frames = [np.stack([a, b]), np.stack([a]), np.stack([a, b])]
    repaired, borrowed = fill_missing_people(frames)
    assert [len(f) for f in repaired] == [2, 2, 2]
    assert borrowed == 1


def test_does_not_duplicate_a_person_already_present():
    a = pose(0, 0)
    frames = [np.stack([a]), np.stack([a]), np.stack([a])]
    repaired, borrowed = fill_missing_people(frames)
    assert [len(f) for f in repaired] == [1, 1, 1]
    assert borrowed == 0


def test_respects_max_gap():
    a, b = pose(0, 0), pose(400, 400)
    frames = [np.stack([a, b])] + [np.stack([a])] * 4 + [np.stack([a, b])]
    repaired, _ = fill_missing_people(frames, max_gap=1)
    # only the frames directly next to a complete frame can be repaired
    assert [len(f) for f in repaired] == [2, 2, 1, 1, 2, 2]


def test_empty_frames_are_left_alone_when_no_neighbour_helps():
    frames = [np.zeros((0, 17, 3)), np.zeros((0, 17, 3))]
    repaired, borrowed = fill_missing_people(frames)
    assert [len(f) for f in repaired] == [0, 0]
    assert borrowed == 0
