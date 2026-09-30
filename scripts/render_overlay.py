"""Draw an analysis back onto its video: skeletons, track colours and the predicted position.

Numbers tell you how often the model changed its mind; only the video tells you whether it was
right. Output goes next to the analysis in data/ (git-ignored), never into the repository.

Usage:
    python scripts/render_overlay.py data/analyses/roll.json data/videos/roll.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from bjj.data import SKELETON

ROOT = Path(__file__).resolve().parents[1]
COLORS_BGR = [(225, 232, 236), (43, 48, 200)]     # athlete 1 bone white, athlete 2 belt red
PANEL_BGR = (13, 11, 10)


def draw_skeletons(image: np.ndarray, frame_poses, thickness: int) -> None:
    for athlete, keypoints in enumerate(frame_poses):
        if keypoints is None:
            continue
        color = COLORS_BGR[athlete % len(COLORS_BGR)]
        points = [None if p is None else (int(p[0]), int(p[1])) for p in keypoints]
        for i, j in SKELETON:
            if points[i] and points[j]:
                cv2.line(image, points[i], points[j], color, thickness, cv2.LINE_AA)
        for point in points[5:]:
            if point:
                cv2.circle(image, point, thickness + 1, color, -1, cv2.LINE_AA)


def draw_panel(image: np.ndarray, label: str, seconds: float, tracked: int) -> None:
    height, width = image.shape[:2]
    panel = image[:56].copy()
    cv2.rectangle(panel, (0, 0), (width, 56), PANEL_BGR, -1)
    image[:56] = cv2.addWeighted(panel, 0.75, image[:56], 0.25, 0)
    cv2.putText(image, label.upper(), (14, 36), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                (236, 232, 225), 2, cv2.LINE_AA)
    right = f"{seconds:5.1f}s   {tracked}/2 tracked"
    cv2.putText(image, right, (width - 260, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (170, 170, 170), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("--raw", action="store_true", help="show the unsmoothed per-frame prediction")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = json.loads(args.analysis.read_text(encoding="utf-8"))
    poses = report.get("poses")
    if not poses:
        raise SystemExit("this analysis was written with --no-poses; re-run analyze_round.py")
    labels = report["labels_raw" if args.raw else "labels_smoothed"]
    classes, times, fps = report["classes"], report["times"], report["fps"]

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    out = args.out or args.analysis.with_name(f"{args.analysis.stem}_overlay.mp4")
    writer = None
    for n, (time_s, frame_poses, label) in enumerate(zip(times, poses, labels)):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(time_s * source_fps)))
        ok, image = capture.read()
        if not ok:
            break
        if writer is None:
            height, width = image.shape[:2]
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        draw_skeletons(image, frame_poses, thickness=max(2, image.shape[1] // 400))
        draw_panel(image, classes[label], time_s, sum(p is not None for p in frame_poses))
        writer.write(image)
        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(times)} frames")
    capture.release()
    if writer is None:
        raise SystemExit("no frames were written")
    writer.release()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(times)} frames at {fps:g} fps)")


if __name__ == "__main__":
    main()
