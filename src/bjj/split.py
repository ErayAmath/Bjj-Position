"""How the data is split into train / validation / test.

This is the decision the whole project stands on. Frames of one roll are nearly identical to
their neighbours, so a random split would put the same moment into training and test and report
an accuracy that does not exist (measured: 81.7 % random vs 10.7 % honest, see the lab notebook).

The split used here:

* **test** — one camera view of every sparring sequence, never touched during training. A new
  camera angle is the closest thing in this dataset to "a gym I have not filmed in yet".
* **validation** — contiguous time blocks cut out of the *training* cameras, with a gap around
  them so the frames next to the block do not leak in. Holding out a second whole camera was
  measured to cost 9 accuracy points, because with only 6 sequences every camera carries a lot
  of the viewpoint diversity.
* **train** — everything else.
"""

from __future__ import annotations

import numpy as np

from bjj.stats import group_segments, segment_class_matrix


def held_out_camera_mask(video_ids: np.ndarray, sequences: np.ndarray) -> np.ndarray:
    """Mask of the frames belonging to the last camera view of each sparring sequence."""
    held_out = [int(np.flatnonzero(sequences == s)[-1]) for s in np.unique(sequences)]
    return np.isin(video_ids, held_out)


def time_block_split(video_ids: np.ndarray, sequences: np.ndarray, val_share: float = 0.15,
                     margin: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(train, val, test) boolean masks.

    val_share: share of each training camera reserved for validation, taken as one contiguous
        block from the middle (a block, not random frames, so the validation set contains whole
        positions rather than neighbours of training frames).
    margin: frames dropped on each side of a validation block; they are too similar to the
        block to be trained on.
    """
    test = held_out_camera_mask(video_ids, sequences)
    val = np.zeros(len(video_ids), dtype=bool)
    buffer = np.zeros(len(video_ids), dtype=bool)

    for segment in np.unique(video_ids[~test]):
        indices = np.flatnonzero(video_ids == segment)
        block = int(len(indices) * val_share)
        if block == 0:
            continue
        start = (len(indices) - block) // 2
        val[indices[start:start + block]] = True
        buffer[indices[max(0, start - margin):start + block + margin]] = True

    train = ~(test | buffer)
    return train, val, test


def split_masks(ann, val_share: float = 0.15, margin: int = 50):
    """Convenience wrapper for an `Annotations` object."""
    sequences = group_segments(segment_class_matrix(ann))
    return time_block_split(ann.video_ids, sequences, val_share, margin)
