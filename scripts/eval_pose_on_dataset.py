"""Measure pose estimation against the dataset's own annotations.

For every sampled image we run a detector + pose model, match the predicted people to the two
annotated athletes and report:

    both_found   share of images where BOTH annotated athletes got a matching prediction
    pck@0.2      share of keypoints within 20 % of the athlete's size (matched athletes only)
    s_per_image  runtime

Configurations are given as "name:key=value,..." so that thresholds can be swept without
editing code, e.g.:

    python scripts/eval_pose_on_dataset.py \
        --config "baseline:score_thr=0.7,nms_thr=0.45" \
        --config "tuned:score_thr=0.3,nms_thr=0.75"

Results go to results/pose_benchmark.csv (overall) and results/pose_benchmark_by_class.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from bjj.data import load_annotations
from bjj.pose_eval import match_poses, pck

ROOT = Path(__file__).resolve().parents[1]


def resize(image: np.ndarray, factor: float) -> np.ndarray:
    """Scale the image (>1 = more pixels per athlete; keypoints are scaled back afterwards)."""
    if factor == 1.0:
        return image
    return cv2.resize(image, None, fx=factor, fy=factor,
                      interpolation=cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA)


def parse_config(text: str) -> tuple[str, dict]:
    """'tuned:score_thr=0.3,rtmo=1' -> ('tuned', {'score_thr': 0.3, 'rtmo': True})."""
    name, _, rest = text.partition(":")
    options: dict = {}
    for item in filter(None, rest.split(",")):
        key, _, value = item.partition("=")
        if key in {"rtmo", "ensemble"}:
            options[key] = value not in {"0", "false", ""}
        elif key in {"mode"}:
            options[key] = value
        else:
            options[key] = float(value)
    return name, options


def merge_poses(primary: np.ndarray, extra: np.ndarray, min_distance_ratio: float = 0.3) -> np.ndarray:
    """Add poses from `extra` that are not already in `primary` (same person, two models)."""
    from bjj.pose_eval import mean_distance, person_scale
    merged = list(primary)
    for candidate in extra:
        visible = candidate[:, 2] >= 0.3
        scale = person_scale(candidate, visible)
        duplicate = any(mean_distance(candidate, kept, visible) < min_distance_ratio * scale
                        for kept in merged)
        if not duplicate:
            merged.append(candidate)
    return np.array(merged) if merged else np.zeros((0, 17, 3))


def build_predictor(options: dict):
    """Return a callable image -> (P, 17, 3) predicted poses."""
    mode = options.get("mode", "balanced")
    upscale = options.get("upscale", 1.0)

    if options.get("ensemble"):
        # Two models fail on different frames: union of top-down and bottom-up detections.
        top_down = build_predictor({k: v for k, v in options.items() if k not in {"ensemble", "score_thr"}})
        bottom_up = build_predictor({**{k: v for k, v in options.items() if k != "ensemble"}, "rtmo": True})
        return lambda image: merge_poses(top_down(image), bottom_up(image))
    if options.get("rtmo"):
        # Bottom-up: one model finds all keypoints at once, no person detector.
        from rtmlib import RTMO, Body
        pose_model = Body(pose="rtmo", mode=mode, backend="onnxruntime", device="cpu").pose_model
        assert isinstance(pose_model, RTMO)
        if "score_thr" in options:
            pose_model.score_thr = options["score_thr"]
        if "nms_thr" in options:
            pose_model.nms_thr = options["nms_thr"]

        def predict(image):
            keypoints, scores = pose_model(resize(image, upscale))
            if not len(keypoints):
                return np.zeros((0, 17, 3))
            poses = np.concatenate([keypoints, scores[..., None]], axis=-1)
            poses[..., :2] /= upscale
            return poses
    else:
        # Top-down: person detector first, then one pose per box.
        from rtmlib import Body
        body = Body(mode=mode, backend="onnxruntime", device="cpu")
        detector, pose_model = body.det_model, body.pose_model
        detector.score_thr = options.get("score_thr", detector.score_thr)
        detector.nms_thr = options.get("nms_thr", detector.nms_thr)

        def predict(image):
            image = resize(image, upscale)
            boxes = detector(image)
            if not len(boxes):
                return np.zeros((0, 17, 3))
            keypoints, scores = pose_model(image, bboxes=boxes)
            poses = np.concatenate([keypoints, scores[..., None]], axis=-1)
            poses[..., :2] /= upscale
            return poses

    return predict


def evaluate(predict, images: Path, manifest: list[dict], gt_poses: np.ndarray, max_people: int):
    per_class = defaultdict(lambda: {"images": 0, "both": 0, "correct": 0, "total": 0, "people": 0})
    started = time.perf_counter()
    for n, item in enumerate(manifest, 1):
        image = cv2.imread(str(images / item["file"]))
        if image is None:
            continue
        predictions = predict(image)
        if max_people and len(predictions) > max_people:
            # Keep the most confident people (spectators score lower than the athletes).
            order = np.argsort(predictions[..., 2].mean(axis=1))[::-1]
            predictions = predictions[order[:max_people]]

        gt = gt_poses[item["index"]]
        matches, _ = match_poses(gt, predictions)
        stats = per_class[item["position"]]
        stats["images"] += 1
        stats["people"] += len(predictions)
        stats["both"] += int((matches >= 0).all())
        for g, p in enumerate(matches):
            if p >= 0:
                correct, total = pck(gt[g], predictions[p])
                stats["correct"] += correct
                stats["total"] += total
        if n % 100 == 0:
            print(f"  {n}/{len(manifest)}")
    return per_class, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/raw/annotations.json")
    parser.add_argument("--images", type=Path, default=ROOT / "data/raw/images")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/raw/sample_manifest.json")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--max-people", type=int, default=2, help="keep the N most confident people (0 = all)")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    ann = load_annotations(args.annotations)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)

    overall_rows, class_rows = [], []
    for text in args.config:
        name, options = parse_config(text)
        print(f"== {name}: {options}")
        per_class, seconds = evaluate(build_predictor(options), args.images, manifest,
                                      ann.poses, args.max_people)

        images = sum(s["images"] for s in per_class.values())
        both = sum(s["both"] for s in per_class.values())
        correct = sum(s["correct"] for s in per_class.values())
        total = sum(s["total"] for s in per_class.values())
        overall_rows.append({
            "config": name, "options": json.dumps(options), "images": images,
            "both_found": round(both / images, 4),
            "pck@0.2": round(correct / total, 4) if total else None,
            "s_per_image": round(seconds / images, 3),
        })
        for position, s in sorted(per_class.items()):
            class_rows.append({
                "config": name, "position": position, "images": s["images"],
                "both_found": round(s["both"] / s["images"], 4),
                "pck@0.2": round(s["correct"] / s["total"], 4) if s["total"] else None,
                "people_per_image": round(s["people"] / s["images"], 2),
            })
        print(f"   both_found={overall_rows[-1]['both_found']:.1%} "
              f"pck@0.2={overall_rows[-1]['pck@0.2']:.1%} {overall_rows[-1]['s_per_image']}s/img")

    for rows, filename in [(overall_rows, "pose_benchmark.csv"), (class_rows, "pose_benchmark_by_class.csv")]:
        path = args.out / filename
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
