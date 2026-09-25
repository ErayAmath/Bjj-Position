"""Analyse one round: video (or a folder of frames) in, position timeline out.

Pipeline: frames -> pose estimation (ensemble) -> tracking -> features -> MLP -> Viterbi
smoothing -> timeline + time spent per position.

Everything runs locally; nothing is uploaded anywhere.

Usage:
    python scripts/analyze_round.py data/videos/roll.mp4 --fps 10
    python scripts/analyze_round.py --frames data/raw/round_images \
        --manifest data/raw/round_manifest.json        # evaluation on dataset frames
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from bjj.features import build_features
from bjj.pipeline.pose import PoseEstimator
from bjj.pipeline.tracking import fill_track_gaps, track_athletes
from bjj.stats import split_class_name
from bjj.temporal import segments, sharpen_persistence, viterbi
from bjj.train import PositionMLP, predict_proba

ROOT = Path(__file__).resolve().parents[1]


def read_video_frames(path: Path, fps: float, max_width: int, start: float, seconds: float | None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / fps))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    first = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    limit = first + int(seconds * src_fps) if seconds else 10**12
    index = first
    while index < limit:
        ok, image = cap.read()
        if not ok:
            break
        if (index - first) % step == 0:
            if max_width and image.shape[1] > max_width:
                scale = max_width / image.shape[1]
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            yield index / src_fps, image
        index += 1
    cap.release()


def read_frame_folder(folder: Path, manifest: Path, fps: float):
    items = json.loads(manifest.read_text(encoding="utf-8"))
    for n, item in enumerate(items):
        image = cv2.imread(str(folder / item["file"]))
        if image is not None:
            yield n / fps, image


def load_or_detect(source, estimator, cache: Path):
    """Pose estimation is the slow part (~0.2 s/frame on CPU) — cache it per clip."""
    if cache.exists():
        stored = np.load(cache, allow_pickle=True)
        print(f"using cached poses from {cache}")
        return stored["times"], list(stored["detections"])

    times, detections = [], []
    for n, (t, image) in enumerate(source, 1):
        times.append(t)
        detections.append(estimator(image))
        if n % 100 == 0:
            print(f"  {n} frames analysed")
    print(f"{len(detections)} frames analysed")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, times=np.array(times),
                        detections=np.array(detections, dtype=object), allow_pickle=True)
    return np.array(times), detections


def load_model(path: Path):
    checkpoint = torch.load(path, weights_only=False)
    model = PositionMLP(checkpoint["num_features"], len(checkpoint["classes"]), checkpoint["hidden"])
    model.load_state_dict(checkpoint["state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), checkpoint, device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, nargs="?")
    parser.add_argument("--frames", type=Path, help="folder of frames instead of a video")
    parser.add_argument("--manifest", type=Path, help="manifest listing the frames")
    parser.add_argument("--model", type=Path, default=ROOT / "results/position_model.pt")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-width", type=int, default=1080)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--max-gap", type=int, default=10, help="frames a lost athlete may be carried forward")
    parser.add_argument("--stay", type=float, default=None,
                        help="probability of staying in the same position per frame (default: dataset estimate)")
    parser.add_argument("--device", default=None, help="cpu or cuda for the pose models")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-poses", action="store_true", help="omit the tracked poses from the report")
    parser.add_argument("--cache", type=Path, default=None,
                        help="npz file with pose detections; reused when it exists (default: data/cache/<name>.npz)")
    args = parser.parse_args()

    model, checkpoint, device = load_model(args.model)
    classes = checkpoint["classes"]
    pose_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    estimator = PoseEstimator(device=pose_device)

    if args.frames:
        source = read_frame_folder(args.frames, args.manifest, args.fps)
        name = args.frames.name
    elif args.video:
        source = read_video_frames(args.video, args.fps, args.max_width, args.start, args.seconds)
        name = args.video.stem
    else:
        raise SystemExit("give a video path or --frames/--manifest")

    cache = args.cache or ROOT / "data/cache" / f"{name}_{args.fps:g}fps.npz"
    times, detections = load_or_detect(source, estimator, cache)

    poses, present = track_athletes(detections)
    poses, usable = fill_track_gaps(poses, present, max_gap=args.max_gap)
    groups = np.zeros(len(poses), dtype=int)   # one continuous clip
    features = build_features(poses, usable, groups, **checkpoint.get("feature_config", {}))
    probabilities = predict_proba(model, features, device)

    raw_labels = probabilities.argmax(1)
    transitions = sharpen_persistence(np.asarray(checkpoint["transitions"]), args.stay)
    smoothed = viterbi(probabilities, transitions)

    timeline = segments(smoothed, times)
    for item in timeline:
        item["position"] = classes[item["label"]]
        item["seconds"] = round(float(item["end_time"] - item["start_time"]), 2)

    total = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    per_position: dict[str, float] = {}
    for item in timeline:
        per_position[item["position"]] = round(per_position.get(item["position"], 0.0) + item["seconds"], 2)

    report = {
        "source": name,
        "frames": len(detections),
        "fps": args.fps,
        "duration_s": round(total, 2),
        "both_athletes_found": round(float(present.all(axis=1).mean()), 4),
        "both_after_gap_fill": round(float(usable.all(axis=1).mean()), 4),
        "changes_raw": int((np.diff(raw_labels) != 0).sum()),
        "changes_smoothed": int((np.diff(smoothed) != 0).sum()),
        "time_per_position": dict(sorted(per_position.items(), key=lambda kv: -kv[1])),
        "time_per_base_position": {},
        "timeline": timeline,
        "classes": classes,
        "labels_raw": raw_labels.tolist(),
        "labels_smoothed": smoothed.tolist(),
        "times": np.round(times, 3).tolist(),
    }
    if not args.no_poses:
        # Tracked poses in image coordinates, for drawing the round in the frontend.
        report["poses"] = [
            [np.round(poses[t, a, :, :2], 1).tolist() if usable[t, a] else None for a in range(2)]
            for t in range(len(times))
        ]
    for position, seconds in per_position.items():
        base = split_class_name(position)[0]
        report["time_per_base_position"][base] = round(
            report["time_per_base_position"].get(base, 0.0) + seconds, 2)

    out = args.out or ROOT / "data/analyses" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"\nRound: {report['duration_s']}s, both athletes found in "
          f"{report['both_athletes_found']:.0%} of frames ({report['both_after_gap_fill']:.0%} after gap fill)")
    print(f"position changes: {report['changes_raw']} raw -> {report['changes_smoothed']} after smoothing")
    print("\ntime per position:")
    for position, seconds in report["time_per_position"].items():
        print(f"  {position:16s} {seconds:6.1f}s  {seconds / max(total, 1e-9):5.1%}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
