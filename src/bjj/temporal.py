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


def rescale_transitions(transitions: np.ndarray, frame_ratio: float) -> np.ndarray:
    """Convert a per-frame transition matrix to a different frame rate.

    The matrix is estimated on the dataset at 25 fps, but a video may be analysed at 10 fps.
    One analysis step then covers 2.5 dataset frames, and the chance of having changed position
    in that time is correspondingly higher. Applying the 25 fps matrix unchanged would make the
    model too reluctant to report a change.

    `frame_ratio` = dataset fps / analysis fps. Formally this is the matrix power P^ratio.
    """
    if abs(frame_ratio - 1.0) < 1e-6:
        return transitions
    from scipy.linalg import fractional_matrix_power

    powered = np.real(fractional_matrix_power(transitions, frame_ratio))
    powered = np.clip(powered, 1e-12, None)          # tiny negative values from the numerics
    return powered / powered.sum(axis=1, keepdims=True)


def extend_transitions(transitions: np.ndarray, extra: int, leak: float = 0.01) -> np.ndarray:
    """Grow a transition matrix by `extra` classes that the dataset does not contain.

    Own labels can introduce positions the ViCoS dataset never had (a leg entanglement, say).
    The model's output layer grows, so the transition matrix has to grow with it. The new
    classes get the average persistence of the known ones, and every known class gets a small
    probability `leak` of moving into them — enough to be reachable, not enough to be guessed.
    """
    if extra <= 0:
        return transitions
    size = len(transitions)
    out = np.zeros((size + extra, size + extra))
    out[:size, :size] = transitions * (1 - leak)
    out[:size, size:] = leak / extra
    stay = float(np.mean(np.diag(transitions)))
    for i in range(size, size + extra):
        out[i] = (1 - stay) / (size + extra - 1)
        out[i, i] = stay
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


def merge_short_runs(labels: np.ndarray, min_frames: int) -> np.ndarray:
    """Dissolve runs shorter than `min_frames` into the neighbour they fit best.

    Viterbi already removes most flicker, but it has no notion of how long a position lasts, so
    half-second "positions" survive. In BJJ those are not positions, and a statistic like "how
    often was I passed" counts every one of them as an event.

    The shortest run below the threshold is repeatedly merged into its longer neighbour, which
    keeps genuinely short transitions attached to whichever phase dominates around them.
    """
    if len(labels) == 0 or min_frames <= 1:
        return labels
    runs = [[int(label), start, end] for start, end in _run_bounds(labels)
            for label in [labels[start]]]

    while len(runs) > 1:
        shortest = min(range(len(runs)), key=lambda i: runs[i][2] - runs[i][1])
        length = runs[shortest][2] - runs[shortest][1]
        if length >= min_frames:
            break
        before = runs[shortest - 1] if shortest > 0 else None
        after = runs[shortest + 1] if shortest + 1 < len(runs) else None
        if before and after:
            target = before if (before[2] - before[1]) >= (after[2] - after[1]) else after
        else:
            target = before or after
        target[1] = min(target[1], runs[shortest][1])
        target[2] = max(target[2], runs[shortest][2])
        runs.pop(shortest)
        # neighbours with the same label now touch: fuse them so the loop can terminate
        merged = [runs[0]]
        for run in runs[1:]:
            if run[0] == merged[-1][0]:
                merged[-1][2] = run[2]
            else:
                merged.append(run)
        runs = merged

    out = labels.copy()
    for label, start, end in runs:
        out[start:end] = label
    return out


def _run_bounds(labels: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) index ranges of runs of equal labels."""
    bounds, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            bounds.append((start, i))
            start = i
    return bounds


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
