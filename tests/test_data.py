import json

import numpy as np

from bjj.data import contiguous_runs, infer_video_ids, load_annotations


def _pose(value: float) -> list[list[float]]:
    return [[value, value, 1.0]] * 17


def test_infer_video_ids_splits_on_frame_reset():
    frames = np.array([1, 2, 3, 5, 6, 1, 2, 9, 1])
    np.testing.assert_array_equal(infer_video_ids(frames), [0, 0, 0, 0, 0, 1, 1, 1, 2])


def test_infer_video_ids_empty():
    assert infer_video_ids(np.array([])).shape == (0,)


def test_contiguous_runs_break_on_gap_and_label_change():
    frames = np.array([1, 2, 3, 5, 6, 7])
    labels = np.array([0, 0, 1, 1, 1, 1])
    assert contiguous_runs(frames, labels) == [(0, 2), (2, 3), (3, 6)]


def test_load_annotations_handles_missing_pose(tmp_path):
    entries = [
        {"position": "mount1", "image": "1", "frame": 1, "pose1": _pose(1), "pose2": _pose(2)},
        {"position": "standing", "image": "2", "frame": 2, "pose1": _pose(3)},
        {"position": "mount1", "image": "3", "frame": 1, "pose2": _pose(4)},
    ]
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(entries))

    ann = load_annotations(path)

    assert len(ann) == 3
    assert ann.poses.shape == (3, 2, 17, 3)
    assert ann.classes == ["mount1", "standing"]
    np.testing.assert_array_equal(ann.labels, [0, 1, 0])
    np.testing.assert_array_equal(ann.present, [[True, True], [True, False], [False, True]])
    assert ann.poses[1, 1].sum() == 0  # missing athlete is zero-filled
    np.testing.assert_array_equal(ann.video_ids, [0, 0, 1])
