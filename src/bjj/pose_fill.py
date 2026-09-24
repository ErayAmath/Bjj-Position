"""Fill frames where the pose model lost an athlete, using neighbouring frames.

In a video an athlete rarely disappears for real: usually one frame simply failed while the
frames around it are fine. Between two frames (~0.08 s) nobody moves far, so copying the
missing athlete from the nearest good neighbour is a safe repair.

This is deliberately identity-free: it only restores "how many people are in this frame".
Assigning stable ids over time (tracking) is a separate step.
"""

from __future__ import annotations

import numpy as np

from bjj.pose_eval import mean_distance, person_scale


def _is_duplicate(candidate: np.ndarray, existing: list[np.ndarray], min_ratio: float) -> bool:
    visible = candidate[:, 2] >= 0.3
    scale = person_scale(candidate, visible)
    if scale == 0:
        return True
    return any(mean_distance(candidate, other, visible) < min_ratio * scale for other in existing)


def fill_missing_people(
    frames: list[np.ndarray],
    expected: int = 2,
    max_gap: int = 5,
    min_distance_ratio: float = 0.3,
) -> tuple[list[np.ndarray], int]:
    """Top up frames that have fewer than `expected` people from nearby frames.

    frames: per frame an (P, 17, 3) array of poses, in time order.
    max_gap: how many frames to look back/forward (0.4 s at 12 fps for the default 5).
    A borrowed pose is only added when it is not already present in the frame, so a frame
    that genuinely shows one person does not get a duplicate of that person.

    Returns (repaired frames, number of poses borrowed).
    """
    repaired = [f.copy() for f in frames]
    borrowed = 0
    for i, frame in enumerate(repaired):
        if len(frame) >= expected:
            continue
        people = list(frame)
        # Nearest neighbours first: 1 back, 1 forward, 2 back, ... so the closest frame in time wins.
        for distance in range(1, max_gap + 1):
            for j in (i - distance, i + distance):
                if not (0 <= j < len(frames)) or len(people) >= expected:
                    continue
                for candidate in frames[j]:
                    if len(people) >= expected:
                        break
                    if not _is_duplicate(candidate, people, min_distance_ratio):
                        people.append(candidate)
                        borrowed += 1
            if len(people) >= expected:
                break
        repaired[i] = np.array(people) if people else frame
    return repaired, borrowed
