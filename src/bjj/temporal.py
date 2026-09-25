"""Use time: positions last seconds, and not every transition is possible.

A per-frame classifier flickers ("mount, mount, guard, mount"). Two facts fix that:
  * positions persist — the next frame is almost always the same position,
  * transitions are structured — standing never turns into back control in one frame.

Both live in a transition matrix estimated from the dataset. Viterbi then picks the most likely
*sequence* of positions instead of the most likely position per frame.
"""

from __future__ import annotations

import numpy as np


def estimate_transition_matrix(labels: np.ndarray, groups: np.ndarray, num_classes: int,
                               smoothing: float = 1.0) -> np.ndarray:
    """P(next position | current position), counted over consecutive frames.

    groups: video/segment id per frame — transitions across a cut are not real transitions.
    smoothing: add-alpha smoothing, so an unseen transition is unlikely but not impossible
        (a zero would make it impossible forever, and our dataset is small).
    """
    counts = np.full((num_classes, num_classes), smoothing, dtype=np.float64)
    same_video = groups[:-1] == groups[1:]
    np.add.at(counts, (labels[:-1][same_video], labels[1:][same_video]), 1.0)
    return counts / counts.sum(axis=1, keepdims=True)


def sharpen_persistence(transitions: np.ndarray, stay: float | None) -> np.ndarray:
    """Optionally force a minimum probability of staying in the same position.

    The dataset is annotated at 25 fps, so measured self-transitions are already ~0.99. When
    predicting at a lower frame rate the model should be less sticky, and vice versa.
    """
    if stay is None:
        return transitions
    out = transitions * (1 - stay) / np.maximum(transitions.sum(axis=1, keepdims=True) - np.diag(transitions)[:, None], 1e-12)
    np.fill_diagonal(out, stay)
    return out / out.sum(axis=1, keepdims=True)


def viterbi(probabilities: np.ndarray, transitions: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Most likely label sequence given per-frame class probabilities.

    probabilities: (T, C) per-frame class probabilities from the classifier.
    transitions:   (C, C) P(next | current).

    Works in log space: probabilities of long sequences underflow to 0 otherwise.
    Returns (T,) labels.
    """
    T, C = probabilities.shape
    if T == 0:
        return np.zeros(0, dtype=int)
    log_p = np.log(probabilities + epsilon)
    log_t = np.log(transitions + epsilon)

    score = log_p[0].copy()                      # best score of a path ending in each class
    backpointer = np.zeros((T, C), dtype=int)
    for t in range(1, T):
        candidates = score[:, None] + log_t      # (from, to)
        backpointer[t] = candidates.argmax(axis=0)
        score = candidates.max(axis=0) + log_p[t]

    path = np.zeros(T, dtype=int)
    path[-1] = int(score.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = backpointer[t, path[t]]
    return path


def segments(labels: np.ndarray, times: np.ndarray | None = None) -> list[dict]:
    """Turn a label sequence into [{label, start, end, frames}] runs — the round's timeline."""
    if len(labels) == 0:
        return []
    out = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append({
                "label": int(labels[start]),
                "start_frame": int(start),
                "end_frame": int(i - 1),
                "frames": int(i - start),
                "start_time": float(times[start]) if times is not None else None,
                "end_time": float(times[i - 1]) if times is not None else None,
            })
            start = i
    return out
