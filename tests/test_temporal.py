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


def test_rescale_transitions_makes_changes_more_likely_at_a_lower_frame_rate():
    from bjj.temporal import rescale_transitions
    sticky = np.array([[0.99, 0.01], [0.02, 0.98]])
    slower = rescale_transitions(sticky, 2.5)      # 25 fps matrix used at 10 fps
    assert slower[0, 1] > sticky[0, 1]             # changing becomes more likely per step
    assert slower[0, 0] < sticky[0, 0]
    np.testing.assert_allclose(slower.sum(axis=1), 1.0)


def test_rescale_transitions_is_a_no_op_at_the_same_frame_rate():
    from bjj.temporal import rescale_transitions
    m = np.array([[0.9, 0.1], [0.3, 0.7]])
    np.testing.assert_allclose(rescale_transitions(m, 1.0), m)


def test_merge_short_runs_absorbs_a_blip():
    from bjj.temporal import merge_short_runs
    labels = np.array([0] * 10 + [1] * 2 + [0] * 10)
    np.testing.assert_array_equal(merge_short_runs(labels, min_frames=5), np.zeros(22, int))


def test_merge_short_runs_keeps_long_enough_positions():
    from bjj.temporal import merge_short_runs
    labels = np.array([0] * 10 + [1] * 8 + [0] * 10)
    np.testing.assert_array_equal(merge_short_runs(labels, min_frames=5), labels)


def test_merge_short_runs_prefers_the_longer_neighbour():
    from bjj.temporal import merge_short_runs
    labels = np.array([0] * 3 + [2] * 2 + [1] * 12)     # blip between a short and a long run
    assert set(merge_short_runs(labels, min_frames=5).tolist()) == {1}


def test_merge_short_runs_is_a_no_op_without_a_threshold():
    from bjj.temporal import merge_short_runs
    labels = np.array([0, 1, 0, 1])
    np.testing.assert_array_equal(merge_short_runs(labels, min_frames=1), labels)


def test_extend_transitions_adds_reachable_classes():
    from bjj.temporal import extend_transitions
    base = np.array([[0.9, 0.1], [0.2, 0.8]])
    out = extend_transitions(base, extra=1)
    assert out.shape == (3, 3)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    assert out[0, 2] > 0                       # the new class is reachable
    assert out[2, 2] > out[2, 0]               # and it persists like the others
    np.testing.assert_allclose(out[0, 1] / out[0, 0], base[0, 1] / base[0, 0], rtol=1e-6)


def test_extend_transitions_without_new_classes_is_a_no_op():
    from bjj.temporal import extend_transitions
    base = np.array([[0.9, 0.1], [0.2, 0.8]])
    np.testing.assert_array_equal(extend_transitions(base, extra=0), base)
