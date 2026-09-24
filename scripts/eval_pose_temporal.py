"""Does using neighbouring frames repair the frames where an athlete was lost?

Runs the pose pipeline over a contiguous stretch of dataset frames (a real video, not a
stratified sample), then applies `fill_missing_people` and re-measures. Only frames whose
annotations contain both athletes are scored, because only there is "both found" meaningful.

Usage:
    python scripts/eval_pose_temporal.py --config "ensemble:ensemble=1,score_thr=0.1"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bjj.data import load_annotations          # noqa: E402
from bjj.pose_eval import match_poses, pck     # noqa: E402
from bjj.pose_fill import fill_missing_people  # noqa: E402
from eval_pose_on_dataset import build_predictor, parse_config  # noqa: E402


def score(frames: list[np.ndarray], manifest: list[dict], gt_poses: np.ndarray) -> dict:
    both = correct = total = scored = 0
    for predictions, item in zip(frames, manifest):
        gt = gt_poses[item["index"]]
        if not (gt[..., 2] >= 0.3).any(axis=1).all():
            continue  # annotations contain only one athlete for this frame
        scored += 1
        matches, _ = match_poses(gt, predictions)
        both += int((matches >= 0).all())
        for g, p in enumerate(matches):
            if p >= 0:
                c, t = pck(gt[g], predictions[p])
                correct, total = correct + c, total + t
    return {"frames_scored": scored,
            "both_found": round(both / scored, 4) if scored else None,
            "pck@0.2": round(correct / total, 4) if total else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="ensemble:ensemble=1,score_thr=0.1")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/raw/clip_manifest.json")
    parser.add_argument("--images", type=Path, default=ROOT / "data/raw/images")
    parser.add_argument("--max-gap", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--out", type=Path, default=ROOT / "results/pose_temporal.csv")
    args = parser.parse_args()

    ann = load_annotations(ROOT / "data/raw/annotations.json")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    name, options = parse_config(args.config)
    predict = build_predictor(options)

    frames = []
    for n, item in enumerate(manifest, 1):
        image = cv2.imread(str(args.images / item["file"]))
        poses = predict(image)
        if len(poses) > 2:  # keep the two most confident people
            poses = poses[np.argsort(poses[..., 2].mean(axis=1))[::-1][:2]]
        frames.append(poses)
        if n % 100 == 0:
            print(f"  {n}/{len(manifest)}")

    rows = [{"config": name, "max_gap": 0, "borrowed": 0, **score(frames, manifest, ann.poses)}]
    for max_gap in args.max_gap:
        repaired, borrowed = fill_missing_people(frames, expected=2, max_gap=max_gap)
        rows.append({"config": name, "max_gap": max_gap, "borrowed": borrowed,
                     **score(repaired, manifest, ann.poses)})
    for row in rows:
        print(f"  max_gap={row['max_gap']:>2}: both_found={row['both_found']:.1%} "
              f"pck@0.2={row['pck@0.2']:.1%} (borrowed {row['borrowed']})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
