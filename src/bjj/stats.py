"""Dataset statistics shared by the EDA plots and the frontend export."""

from __future__ import annotations

import numpy as np

from bjj.data import Annotations, contiguous_runs


def split_class_name(name: str) -> tuple[str, int | None]:
    """'mount1' -> ('mount', 1); symmetric classes like 'standing' -> ('standing', None).

    The trailing 1/2 says which athlete the position is attributed to (e.g. who is on top).
    """
    if name[-1] in "12" and not name[-2].isdigit():
        return name[:-1], int(name[-1])
    return name, None


def class_counts(ann: Annotations) -> dict[str, int]:
    counts = np.bincount(ann.labels, minlength=len(ann.classes))
    return {name: int(c) for name, c in zip(ann.classes, counts)}


def segment_class_matrix(ann: Annotations) -> np.ndarray:
    """(num_segments, num_classes) frame counts."""
    n_seg = int(ann.video_ids.max()) + 1
    matrix = np.zeros((n_seg, len(ann.classes)), dtype=np.int64)
    np.add.at(matrix, (ann.video_ids, ann.labels), 1)
    return matrix


def group_segments(matrix: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Group consecutive segments that are camera views of the same sparring sequence.

    Heuristic: adjacent segments whose class mix has cosine similarity >= threshold belong
    together. On this dataset similarities are either >= 0.9 or <= 0.01, so the threshold
    is not sensitive. Returns one group id per segment.
    """
    mix = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    norms = np.maximum(np.linalg.norm(mix, axis=1), 1e-12)
    similarity = (mix[:-1] * mix[1:]).sum(axis=1) / (norms[:-1] * norms[1:])
    starts_new_group = similarity < threshold
    return np.concatenate([[0], np.cumsum(starts_new_group)]).astype(np.int64)


def pick_sample_run(ann: Annotations, label: int, length: int) -> tuple[int, int] | None:
    """Pick a representative stretch of `length` frames for one class.

    Prefers runs where both athletes are detected. Returns an index range [start, end)
    or None if the class has no run at all. Deterministic: best-scoring run, centred.
    """
    best, best_score = None, -1.0
    for start, end in contiguous_runs(ann.frames, ann.labels):
        if ann.labels[start] != label:
            continue
        both = ann.present[start:end].all(axis=1).mean()
        score = both * min(end - start, length)  # favour complete and long-enough runs
        if score > best_score:
            best, best_score = (start, end), score
    if best is None:
        return None
    start, end = best
    if end - start > length:
        start += (end - start - length) // 2
        end = start + length
    return start, end
