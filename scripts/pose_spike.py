"""Phase 0 feasibility spike: does an off-the-shelf pose model work on my sparring footage?

Runs a top-down pipeline (YOLOX person detector -> RTMPose, 17 COCO keypoints, via rtmlib)
on a clip and writes, into a git-ignored folder:
    overlay.mp4    video with detected skeletons drawn on top
    poses.jsonl    one line per sampled frame, keypoints in the dataset's [x, y, conf] layout
    summary.json   how often 0 / 1 / 2 / 3+ people were found, mean keypoint confidence

This is throwaway exploration code; the real pipeline will live in src/bjj/pipeline/.
Person ids are NOT tracked: colours are just per-frame detection order and will flicker.

Frames wider than --max-width are downscaled first. The models work on small inputs anyway
(detector 640x640, pose 192x256 per person), so 4K only costs decoding time. Keypoints in
poses.jsonl are in the coordinates of the downscaled frame (size recorded in summary.json).

Usage:
    python scripts/pose_spike.py data/videos/roll.mp4 --start 30 --seconds 60 --fps 10
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from rtmlib import Body

from bjj.data import SKELETON

ROOT = Path(__file__).resolve().parents[1]
MIN_DRAW_CONFIDENCE = 0.3
COLORS_BGR = [(225, 232, 236), (43, 48, 200), (80, 200, 80), (200, 160, 60)]  # white, red, ...


def downscale(image: np.ndarray, max_width: int) -> np.ndarray:
    """Shrink `image` to at most `max_width` pixels wide, keeping the aspect ratio.

    max_width <= 0 disables resizing. Images already narrow enough are returned unchanged.
    """
    h, w = image.shape[:2]
    if max_width <= 0 or w <= max_width:
        return image
    scale = max_width / w
    # INTER_AREA averages pixels when shrinking -> no aliasing, the usual choice for downscaling.
    return cv2.resize(image, (max_width, round(h * scale)), interpolation=cv2.INTER_AREA)


def video_fps(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


def sampling_step(src_fps: float, fps: float) -> int:
    """Keep every `step`-th source frame. Effective rate is src_fps / step, not exactly `fps`."""
    return max(1, round(src_fps / fps))


def sample_frames(path: Path, start: float, seconds: float, fps: float):
    """Yield (frame_index, time_s, image) at roughly `fps`, from `start` for `seconds`."""
    src_fps = video_fps(path)
    step = sampling_step(src_fps, fps)
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    first = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    last = first + int(seconds * src_fps)
    index = first
    while index < last:
        ok, image = cap.read()
        if not ok:
            break
        if (index - first) % step == 0:
            yield index, index / src_fps, image
        index += 1
    cap.release()


def draw(image: np.ndarray, people: np.ndarray, scores: np.ndarray) -> np.ndarray:
    out = image.copy()
    thickness = max(2, image.shape[1] // 400)
    for p, (kps, conf) in enumerate(zip(people, scores)):
        color = COLORS_BGR[p % len(COLORS_BGR)]
        for i, j in SKELETON:
            if conf[i] >= MIN_DRAW_CONFIDENCE and conf[j] >= MIN_DRAW_CONFIDENCE:
                cv2.line(out, tuple(map(int, kps[i])), tuple(map(int, kps[j])), color, thickness, cv2.LINE_AA)
        for k in range(len(kps)):
            if conf[k] >= MIN_DRAW_CONFIDENCE:
                cv2.circle(out, tuple(map(int, kps[k])), thickness + 1, color, -1, cv2.LINE_AA)
    cv2.putText(out, f"people: {len(people)}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    parser.add_argument("--seconds", type=float, default=60.0, help="clip length in seconds")
    parser.add_argument("--fps", type=float, default=10.0, help="frames per second to analyse")
    parser.add_argument("--mode", default="balanced", choices=["lightweight", "balanced", "performance"])
    parser.add_argument("--max-width", type=int, default=1080,
                        help="downscale wider frames to this width before inference (0 = keep original)")
    parser.add_argument("--out", type=Path, default=None, help="default: data/pose_spike/<video name>")
    args = parser.parse_args()

    out_dir = args.out or ROOT / "data" / "pose_spike" / args.video.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    body = Body(mode=args.mode, backend="onnxruntime", device="cpu")
    # Call detector and pose model separately: with zero boxes, RTMPose would otherwise
    # hallucinate one pose on the full image.
    detector, pose_model = body.det_model, body.pose_model

    # Real sampling rate (e.g. 25 fps source, --fps 10 -> step 2 -> 12.5 fps). The overlay must use
    # this rate, otherwise it plays back slower or faster than reality.
    src_fps = video_fps(args.video)
    effective_fps = src_fps / sampling_step(src_fps, args.fps)

    writer = None
    counts = Counter()
    confidences = []
    started = time.perf_counter()
    with open(out_dir / "poses.jsonl", "w", encoding="utf-8") as f:
        for n, (index, t, image) in enumerate(sample_frames(args.video, args.start, args.seconds, args.fps)):
            image = downscale(image, args.max_width)
            boxes = detector(image)
            if len(boxes):
                keypoints, scores = pose_model(image, bboxes=boxes)
            else:
                keypoints, scores = np.zeros((0, 17, 2)), np.zeros((0, 17))

            counts[min(len(keypoints), 3)] += 1
            confidences.extend(scores.mean(axis=1).tolist())
            f.write(json.dumps({
                "frame": index,
                "time": round(t, 3),
                "boxes": np.round(np.asarray(boxes), 1).tolist(),
                "people": [np.round(np.column_stack([k, s]), 3).tolist() for k, s in zip(keypoints, scores)],
            }) + "\n")

            if writer is None:
                h, w = image.shape[:2]
                writer_size = (w, h)
                writer = cv2.VideoWriter(str(out_dir / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), effective_fps, writer_size)
            writer.write(draw(image, keypoints, scores))
            if n % 50 == 0:
                print(f"frame {index} (t={t:.1f}s): {len(keypoints)} people")

    if writer is None:
        raise SystemExit("No frames read — check --start/--seconds against the video length.")
    writer.release()

    total = sum(counts.values())
    summary = {
        "video": args.video.name,
        "mode": args.mode,
        "frame_size": [writer_size[0], writer_size[1]],  # width, height after --max-width
        "source_fps": round(src_fps, 2),
        "analysed_fps": round(effective_fps, 2),
        "frames_analysed": total,
        "seconds_per_frame": round((time.perf_counter() - started) / total, 3),
        "people_per_frame": {("3+" if k == 3 else str(k)): round(v / total, 3) for k, v in sorted(counts.items())},
        "mean_keypoint_confidence": round(float(np.mean(confidences)), 3) if confidences else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Overlay: {out_dir / 'overlay.mp4'}")


if __name__ == "__main__":
    main()
