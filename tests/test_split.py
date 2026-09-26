import numpy as np

from bjj.split import held_out_camera_mask, time_block_split


def setup():
    # two sequences with two camera views each, 100 frames per view
    video_ids = np.repeat([0, 1, 2, 3], 100)
    sequences = np.array([0, 0, 1, 1])      # segments 0,1 -> sequence 0; segments 2,3 -> sequence 1
    return video_ids, sequences


def test_test_set_is_the_last_camera_of_each_sequence():
    video_ids, sequences = setup()
    test = held_out_camera_mask(video_ids, sequences)
    np.testing.assert_array_equal(np.unique(video_ids[test]), [1, 3])


def test_the_three_masks_do_not_overlap():
    video_ids, sequences = setup()
    train, val, test = time_block_split(video_ids, sequences)
    assert not (train & val).any()
    assert not (train & test).any()
    assert not (val & test).any()


def test_validation_comes_from_the_training_cameras_only():
    video_ids, sequences = setup()
    _, val, test = time_block_split(video_ids, sequences)
    assert not (val & test).any()
    np.testing.assert_array_equal(np.unique(video_ids[val]), [0, 2])


def test_validation_blocks_are_contiguous():
    video_ids, sequences = setup()
    _, val, _ = time_block_split(video_ids, sequences, val_share=0.2, margin=0)
    for segment in (0, 2):
        block = np.flatnonzero(val & (video_ids == segment))
        assert (np.diff(block) == 1).all()
        assert len(block) == 20


def test_margin_frames_are_dropped_from_training():
    video_ids, sequences = setup()
    train, val, _ = time_block_split(video_ids, sequences, val_share=0.2, margin=5)
    block = np.flatnonzero(val & (video_ids == 0))
    # the 5 frames before and after the block belong to neither train nor validation
    assert not train[block[0] - 5:block[0]].any()
    assert not train[block[-1] + 1:block[-1] + 6].any()
    assert train[block[0] - 6]


def test_a_camera_too_short_for_a_block_is_skipped():
    video_ids = np.repeat([0, 1], 3)
    sequences = np.array([0, 0])
    train, val, test = time_block_split(video_ids, sequences, val_share=0.15)
    assert not val.any()
    assert train.sum() + test.sum() == len(video_ids)
