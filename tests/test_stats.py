import numpy as np
import pytest

from bjj.data import Annotations
from bjj.stats import (
    class_counts,
    group_segments,
    pick_sample_run,
    segment_class_matrix,
    split_class_name,
)


def _ann(labels, frames, present=None, video_ids=None):
    n = len(labels)
    return Annotations(
        poses=np.zeros((n, 2, 17, 3), dtype=np.float32),
        present=np.ones((n, 2), bool) if present is None else np.array(present),
        labels=np.array(labels),
        classes=["a", "b"],
        frames=np.array(frames),
        video_ids=np.zeros(n, np.int64) if video_ids is None else np.array(video_ids),
    )


@pytest.mark.parametrize(
    "name, expected",
    [("mount1", ("mount", 1)), ("side_control2", ("side_control", 2)),
     ("standing", ("standing", None)), ("5050_guard", ("5050_guard", None))],
)
def test_split_class_name(name, expected):
    assert split_class_name(name) == expected


def test_counts_and_segment_matrix():
    ann = _ann([0, 0, 1, 1, 1], [1, 2, 3, 1, 2], video_ids=[0, 0, 0, 1, 1])
    assert class_counts(ann) == {"a": 2, "b": 3}
    np.testing.assert_array_equal(segment_class_matrix(ann), [[2, 1], [0, 2]])


def test_group_segments_joins_similar_neighbours():
    matrix = np.array([
        [10, 10, 0], [20, 19, 0],   # same mix -> one sequence
        [0, 0, 5],                  # different mix -> new sequence
        [10, 10, 0],                # same mix as the first, but not adjacent -> new sequence
    ])
    np.testing.assert_array_equal(group_segments(matrix), [0, 0, 1, 2])


def test_pick_sample_run_prefers_both_athletes_and_centres():
    # run 1: label 0, frames 1-4, one athlete missing throughout
    # run 2: label 0, frames 10-15, both present -> preferred, cropped to the middle 4
    labels = [0] * 4 + [0] * 6
    frames = [1, 2, 3, 4] + list(range(10, 16))
    present = [[True, False]] * 4 + [[True, True]] * 6
    ann = _ann(labels, frames, present)
    assert pick_sample_run(ann, label=0, length=4) == (5, 9)
    assert pick_sample_run(ann, label=1, length=4) is None
