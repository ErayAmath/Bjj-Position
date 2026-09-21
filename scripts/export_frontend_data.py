"""Condense the raw annotations into small JSON files for the static frontend.

Usage:
    python scripts/export_frontend_data.py [--annotations data/raw/annotations.json]

Writes frontend/data/overview.json and frontend/data/sequences.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bjj.data import SKELETON, contiguous_runs, load_annotations
from bjj.stats import (
    class_counts,
    group_segments,
    pick_sample_run,
    segment_class_matrix,
    split_class_name,
)

ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_LENGTH = 64   # frames per sample sequence
MIN_CONFIDENCE = 0.2   # keypoints below this are drawn as missing


def build_overview(ann) -> dict:
    matrix = segment_class_matrix(ann)
    run_lengths = np.array([end - start for start, end in contiguous_runs(ann.frames, ann.labels)])
    return {
        "num_frames": len(ann),
        "num_classes": len(ann.classes),
        "num_segments": len(matrix),
        "num_sequences": int(group_segments(matrix).max()) + 1,
        "missing_athlete_share": float((~ann.present.all(axis=1)).mean()),
        "classes": [
            {"name": name, "base": split_class_name(name)[0],
             "athlete": split_class_name(name)[1], "count": count}
            for name, count in class_counts(ann).items()
        ],
        "segments": [
            {"id": seg, "sequence": int(group), "frames": int(row.sum()), "counts": row.tolist()}
            for seg, (row, group) in enumerate(zip(matrix, group_segments(matrix)))
        ],
        "run_length": {
            "num_runs": len(run_lengths),
            "median": float(np.median(run_lengths)),
            "p90": float(np.percentile(run_lengths, 90)),
            "max": int(run_lengths.max()),
        },
    }


def build_sequences(ann) -> dict:
    sequences = []
    for label, name in enumerate(ann.classes):
        picked = pick_sample_run(ann, label, SEQUENCE_LENGTH)
        if picked is None:
            continue
        start, end = picked
        frames = []
        for i in range(start, end):
            athletes = []
            for a in range(2):
                if not ann.present[i, a]:
                    athletes.append(None)
                    continue
                # [x, y] rounded to 0.1 px, or null for unreliable keypoints
                athletes.append([
                    [round(float(x), 1), round(float(y), 1)] if c >= MIN_CONFIDENCE else None
                    for x, y, c in ann.poses[i, a]
                ])
            frames.append(athletes)
        sequences.append({
            "name": name,
            "segment": int(ann.video_ids[start]),
            "frame_start": int(ann.frames[start]),
            "frame_end": int(ann.frames[end - 1]),
            "frames": frames,
        })
    return {"skeleton": SKELETON, "sequences": sequences}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=ROOT / "data/raw/annotations.json", type=Path)
    parser.add_argument("--out", default=ROOT / "frontend/data", type=Path)
    args = parser.parse_args()

    ann = load_annotations(args.annotations)
    args.out.mkdir(parents=True, exist_ok=True)
    for filename, payload in [("overview.json", build_overview(ann)),
                              ("sequences.json", build_sequences(ann))]:
        path = args.out / filename
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
