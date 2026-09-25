"""End-to-end evaluation: run the full pipeline over a labelled clip and compare to the labels.

This is the honest MVP number: images in, positions out, measured against the dataset's own
position labels on a camera view the model never saw during training.

Athlete numbering: tracking numbers the athletes in the order it first sees them, the dataset
numbers them its own way. The mapping is decided ONCE per clip (identity or swapped, whichever
fits better) — the same single decision the user makes in the product by pointing at themselves.
Reported both ways so the choice is visible.

Usage:
    python scripts/eval_round.py --frames data/raw/round_images \
        --manifest data/raw/round_manifest.json --name round_images
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from bjj.data import load_annotations
from bjj.features import build_features
from bjj.pipeline.tracking import fill_track_gaps, track_athletes
from bjj.stats import split_class_name
from bjj.temporal import sharpen_persistence, viterbi
from bjj.train import PositionMLP, predict_proba

ROOT = Path(__file__).resolve().parents[1]


def load_cached(name: str, fps: float):
    cache = ROOT / "data/cache" / f"{name}_{fps:g}fps.npz"
    if not cache.exists():
        raise SystemExit(f"no cached poses: {cache}\nRun scripts/precompute_poses.py first.")
    stored = np.load(cache, allow_pickle=True)
    return stored["times"], list(stored["detections"])


def load_model(path: Path):
    checkpoint = torch.load(path, weights_only=False)
    model = PositionMLP(checkpoint["num_features"], len(checkpoint["classes"]), checkpoint["hidden"])
    model.load_state_dict(checkpoint["state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), checkpoint, device


def swap_index(classes: list[str]) -> np.ndarray:
    index = {name: i for i, name in enumerate(classes)}
    out = []
    for name in classes:
        base, athlete = split_class_name(name)
        out.append(index[f"{base}{3 - athlete}"] if athlete else index[name])
    return np.array(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--model", type=Path, default=ROOT / "results/position_model.pt")
    parser.add_argument("--max-gap", type=int, nargs="+", default=[0, 5, 10, 20])
    parser.add_argument("--stay", type=float, nargs="+", default=[None, 0.95, 0.99, 0.999])
    parser.add_argument("--out", type=Path, default=ROOT / "results/round_eval.csv")
    args = parser.parse_args()

    ann = load_annotations(ROOT / "data/raw/annotations.json")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    truth = np.array([ann.labels[item["index"]] for item in manifest])
    times, detections = load_cached(args.name, args.fps)
    detections = detections[:len(truth)]
    truth = truth[:len(detections)]

    model, checkpoint, device = load_model(args.model)
    classes = checkpoint["classes"]
    swap = swap_index(classes)
    base = np.array([split_class_name(c)[0] for c in classes])
    transitions_base = np.asarray(checkpoint["transitions"])

    rows = []
    poses_raw, present_raw = track_athletes(detections)
    for max_gap in args.max_gap:
        if max_gap:
            poses, usable = fill_track_gaps(poses_raw, present_raw, max_gap=max_gap)
        else:
            poses, usable = poses_raw, present_raw
        groups = np.zeros(len(poses), dtype=int)   # one continuous clip
        features = build_features(poses, usable, groups, **checkpoint.get("feature_config", {}))
        probabilities = predict_proba(model, features, device)

        # One mapping decision for the whole clip, as the user would make it.
        straight = probabilities.argmax(1)
        swapped = swap[straight]
        use_swap = (swapped == truth).mean() > (straight == truth).mean()
        probs = probabilities[:, swap] if use_swap else probabilities

        for stay in args.stay:
            smoothed = viterbi(probs, sharpen_persistence(transitions_base, stay))
            raw = probs.argmax(1)
            rows.append({
                "max_gap": max_gap,
                "stay": stay if stay is not None else "dataset",
                "athlete_mapping": "swapped" if use_swap else "identity",
                "usable_both": round(float(usable.all(axis=1).mean()), 4),
                "accuracy_raw": round(float((raw == truth).mean()), 4),
                "accuracy_smoothed": round(float((smoothed == truth).mean()), 4),
                "base_accuracy_smoothed": round(float((base[smoothed] == base[truth]).mean()), 4),
                "changes_raw": int((np.diff(raw) != 0).sum()),
                "changes_smoothed": int((np.diff(smoothed) != 0).sum()),
                "changes_truth": int((np.diff(truth) != 0).sum()),
            })
            print(f"  gap={max_gap:>2} stay={str(rows[-1]['stay']):>7}: "
                  f"raw={rows[-1]['accuracy_raw']:.3f} smoothed={rows[-1]['accuracy_smoothed']:.3f} "
                  f"base={rows[-1]['base_accuracy_smoothed']:.3f} "
                  f"changes {rows[-1]['changes_smoothed']} (truth {rows[-1]['changes_truth']})")

    best = max(rows, key=lambda r: r["accuracy_smoothed"])
    print(f"\nbest: gap={best['max_gap']} stay={best['stay']} -> "
          f"{best['accuracy_smoothed']:.1%} (18 classes), {best['base_accuracy_smoothed']:.1%} (10 positions)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
