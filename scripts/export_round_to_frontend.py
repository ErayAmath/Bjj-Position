"""Convert an analysis JSON into the small file the dashboard loads.

Keeps only what the page draws: the timeline, time per position, and the tracked skeletons
(rounded, optionally sub-sampled).

By default the result is written to frontend/data/private/, which is git-ignored: an analysis
of your own training must never end up in a public repository. Pass --demo to overwrite the
committed demo round instead.

Usage:
    python scripts/export_round_to_frontend.py data/analyses/round_images.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bjj.data import SKELETON

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--every", type=int, default=2, help="keep every n-th frame")
    parser.add_argument("--demo", action="store_true",
                        help="overwrite the public demo round (frontend/data/round.json) instead")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or (ROOT / "frontend/data/round.json" if args.demo
                       else ROOT / "frontend/data/private/round.json")
    report = json.loads(args.analysis.read_text(encoding="utf-8"))
    step = max(1, args.every)
    payload = {
        "source": report["source"],
        "duration_s": report["duration_s"],
        "fps": report["fps"] / step,
        "both_athletes_found": report["both_athletes_found"],
        "both_after_gap_fill": report["both_after_gap_fill"],
        "changes_raw": report["changes_raw"],
        "changes_smoothed": report["changes_smoothed"],
        "classes": report["classes"],
        "time_per_position": report["time_per_position"],
        "time_per_base_position": report["time_per_base_position"],
        "timeline": report["timeline"],
        "skeleton": SKELETON,
        "times": report["times"][::step],
        "labels": report["labels_smoothed"][::step],
        "labels_raw": report["labels_raw"][::step],
        "poses": report.get("poses", [])[::step],
        "accuracy": report.get("accuracy"),
        "private": not args.demo,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KB)")
    if args.demo:
        print("WARNING: this file is committed and public.")
    else:
        print("This folder is git-ignored; the dashboard prefers it over the public demo.")


if __name__ == "__main__":
    main()
