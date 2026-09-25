import numpy as np

from bjj.augment import degrade, torso_size


def make_poses(n: int = 50) -> tuple[np.ndarray, np.ndarray]:
    poses = np.zeros((n, 2, 17, 3))
    for a in range(2):
        poses[:, a, 5] = [0 + 100 * a, 0, 1.0]     # shoulders
        poses[:, a, 6] = [20 + 100 * a, 0, 1.0]
        poses[:, a, 11] = [0 + 100 * a, 50, 1.0]   # hips
        poses[:, a, 12] = [20 + 100 * a, 50, 1.0]
    return poses, np.ones((n, 2), bool)


def test_torso_size_measures_shoulder_to_hip():
    poses, _ = make_poses(1)
    np.testing.assert_allclose(torso_size(poses), 50.0)


def test_jitter_moves_keypoints_but_keeps_the_scale_sane():
    poses, present = make_poses()
    noisy, _ = degrade(poses, present, jitter=0.06, drop_keypoint=0, drop_athlete=0,
                       rng=np.random.default_rng(0))
    shift = np.linalg.norm(noisy[..., :2] - poses[..., :2], axis=-1)
    assert shift.mean() > 0
    assert shift.mean() < 0.06 * 50 * 2        # roughly within the requested magnitude


def test_confidence_column_is_not_jittered():
    poses, present = make_poses()
    noisy, _ = degrade(poses, present, jitter=0.06, drop_keypoint=0, drop_athlete=0,
                       rng=np.random.default_rng(0))
    np.testing.assert_array_equal(noisy[..., 2], poses[..., 2])


def test_dropping_keypoints_zeroes_some_of_them():
    poses, present = make_poses(200)
    noisy, _ = degrade(poses, present, jitter=0, drop_keypoint=0.3, drop_athlete=0,
                       rng=np.random.default_rng(0))
    had_value = (poses != 0).any(axis=-1)                    # the fixture only fills 4 joints
    lost = had_value & (noisy == 0).all(axis=-1)
    assert 0.2 < lost.sum() / had_value.sum() < 0.4          # ~30 % of the real joints


def test_never_drops_both_athletes():
    poses, present = make_poses(500)
    _, still = degrade(poses, present, jitter=0, drop_keypoint=0, drop_athlete=0.9,
                       rng=np.random.default_rng(0))
    assert still.any(axis=1).all()
    assert not still.all(axis=1).all()          # but it does drop one of them


def test_dropped_athlete_is_zeroed():
    poses, present = make_poses(200)
    noisy, still = degrade(poses, present, jitter=0, drop_keypoint=0, drop_athlete=0.5,
                           rng=np.random.default_rng(1))
    assert (noisy[~still] == 0).all()


def test_input_is_not_modified_in_place():
    poses, present = make_poses()
    before = poses.copy()
    degrade(poses, present, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(poses, before)
