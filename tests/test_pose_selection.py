import numpy as np

from bjj.pipeline.pose import select_rolling_pair


def person(x: float, y: float, size: float) -> np.ndarray:
    """A person of roughly `size` pixels, centred at (x, y)."""
    offsets = np.linspace(-size / 2, size / 2, 17)
    return np.column_stack([x + offsets, y + offsets, np.ones(17)])


def test_picks_the_big_close_pair_over_confident_bystanders():
    rolling = [person(500, 500, 300), person(560, 520, 280)]     # large, overlapping
    bystanders = [person(100, 100, 60), person(200, 100, 60)]    # small, far away
    chosen = select_rolling_pair(np.stack(bystanders + rolling))
    assert {round(p[:, 0].mean()) for p in chosen} == {round(p[:, 0].mean()) for p in rolling}


def test_prefers_the_pair_in_contact_over_two_separated_large_people():
    grapplers = [person(400, 400, 260), person(450, 430, 250)]
    apart = [person(100, 900, 280), person(900, 900, 280)]       # equally large, far apart
    chosen = select_rolling_pair(np.stack(apart + grapplers))
    assert all(abs(p[:, 0].mean() - 425) < 200 for p in chosen)


def test_fewer_than_three_people_are_returned_unchanged():
    poses = np.stack([person(0, 0, 100), person(50, 50, 100)])
    np.testing.assert_array_equal(select_rolling_pair(poses), poses)


def test_people_without_reliable_keypoints_are_ignored():
    ghost = np.zeros((17, 3))                                    # all confidences 0
    real = [person(300, 300, 200), person(340, 320, 200)]
    chosen = select_rolling_pair(np.stack([ghost] + real))
    assert len(chosen) == 2
    assert all(p[:, 2].sum() > 0 for p in chosen)
