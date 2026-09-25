import numpy as np

from bjj.temporal import estimate_transition_matrix, segments, sharpen_persistence, viterbi


def test_transition_matrix_counts_only_within_a_video():
    labels = np.array([0, 0, 1, 1, 0])
    groups = np.array([0, 0, 0, 1, 1])  # the 1 -> 1 step crosses a cut
    t = estimate_transition_matrix(labels, groups, num_classes=2, smoothing=0.0)
    np.testing.assert_allclose(t[0], [0.5, 0.5])   # 0->0 once, 0->1 once
    np.testing.assert_allclose(t[1], [1.0, 0.0])   # only the 1->0 inside video 1


def test_transition_matrix_rows_sum_to_one_with_smoothing():
    t = estimate_transition_matrix(np.array([0, 0]), np.array([0, 0]), num_classes=3)
    np.testing.assert_allclose(t.sum(axis=1), 1.0)


def test_viterbi_fixes_a_single_flickering_frame():
    # frame 2 is wrongly classified, but positions persist
    probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.4, 0.6], [0.9, 0.1], [0.85, 0.15]])
    transitions = np.array([[0.99, 0.01], [0.01, 0.99]])
    np.testing.assert_array_equal(viterbi(probabilities, transitions), [0, 0, 0, 0, 0])


def test_viterbi_still_follows_a_real_change():
    probabilities = np.array([[0.95, 0.05]] * 4 + [[0.05, 0.95]] * 4)
    transitions = np.array([[0.99, 0.01], [0.01, 0.99]])
    np.testing.assert_array_equal(viterbi(probabilities, transitions), [0, 0, 0, 0, 1, 1, 1, 1])


def test_viterbi_handles_an_empty_sequence():
    assert viterbi(np.zeros((0, 3)), np.eye(3)).shape == (0,)


def test_sharpen_persistence_sets_the_diagonal_and_keeps_rows_normalised():
    t = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.1, 0.1, 0.8]])
    out = sharpen_persistence(t, stay=0.9)
    np.testing.assert_allclose(np.diag(out), 0.9)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)


def test_segments_builds_the_timeline():
    labels = np.array([0, 0, 0, 1, 1, 0])
    times = np.arange(6) * 0.5
    result = segments(labels, times)
    assert [(s["label"], s["frames"]) for s in result] == [(0, 3), (1, 2), (0, 1)]
    assert result[1]["start_time"] == 1.5
