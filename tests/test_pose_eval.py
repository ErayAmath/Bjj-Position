import numpy as np

from bjj.pose_eval import match_poses, pck, person_scale


def pose(x: float, y: float, conf: float = 1.0, spread: float = 100.0) -> np.ndarray:
    """A fake person: 17 keypoints spread over a `spread` x `spread` box at (x, y)."""
    offsets = np.linspace(0, spread, 17)
    return np.column_stack([x + offsets, y + offsets, np.full(17, conf)])


def test_person_scale_is_the_box_diagonal():
    p = pose(0, 0, spread=30)
    assert person_scale(p, p[:, 2] >= 0.3) == np.hypot(30, 30)


def test_match_poses_assigns_globally_best_pairs():
    gt = np.stack([pose(0, 0), pose(500, 500)])
    pred = np.stack([pose(505, 505), pose(3, 3)])  # reversed order, both close
    matches, _ = match_poses(gt, pred)
    np.testing.assert_array_equal(matches, [1, 0])


def test_match_poses_rejects_far_predictions():
    gt = np.stack([pose(0, 0)])
    pred = np.stack([pose(5000, 5000)])
    matches, _ = match_poses(gt, pred)
    np.testing.assert_array_equal(matches, [-1])


def test_match_poses_handles_missing_predictions():
    gt = np.stack([pose(0, 0), pose(500, 500)])
    matches, _ = match_poses(gt, np.zeros((0, 17, 3)))
    np.testing.assert_array_equal(matches, [-1, -1])


def test_pck_counts_keypoints_within_threshold():
    gt = pose(0, 0, spread=100)          # scale = hypot(100, 100) ~ 141.4
    pred = gt.copy()
    pred[:5, 0] += 200                   # 5 joints far off, 12 exact
    correct, total = pck(gt, pred, threshold=0.2)
    assert (correct, total) == (12, 17)


def test_pck_ignores_invisible_keypoints():
    gt = pose(0, 0)
    gt[:7, 2] = 0.0                      # annotated as unreliable
    correct, total = pck(gt, gt.copy())
    assert (correct, total) == (10, 10)
