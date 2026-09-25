"""Run the pose ensemble over a clip and cache the result for later analyses.

Pose estimation is the slow part of the pipeline (~0.2 s per frame on CPU). Caching it means
the classifier, tracking and smoothing can be re-run in seconds while experimenting.

Usage:
    python scripts/precompute_poses.py --frames data/raw/round_images \
        --manifest data/raw/round_manifest.json --name round_images --fps 25
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from bjj.pipeline.pose import PoseEstimator

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    items = json.loads(args.manifest.read_text(encoding="utf-8"))[:args.limit]
    estimator = PoseEstimator()
    times, detections = [], []
    started = time.perf_counter()
    for n, item in enumerate(items):
        image = cv2.imread(str(args.frames / item["file"]))
        if image is None:
            continue
        times.append(n / args.fps)
        detections.append(estimator(image))
        if (n + 1) % 100 == 0:
            done = n + 1
            rate = (time.perf_counter() - started) / done
            print(f"  {done}/{len(items)} frames ({rate:.2f} s/frame, "
                  f"{(len(items) - done) * rate / 60:.1f} min left)", flush=True)

    cache = ROOT / "data/cache" / f"{args.name}_{args.fps:g}fps.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, times=np.array(times),
                        detections=np.array(detections, dtype=object), allow_pickle=True)
    found = np.mean([len(d) >= 2 for d in detections])
    print(f"wrote {cache} — both athletes detected in {found:.1%} of {len(detections)} frames")


if __name__ == "__main__":
    main()
