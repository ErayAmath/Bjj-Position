"""Train the position classifier (MLP) on the annotated keypoints.

Split: one camera view of every sparring sequence is held out for testing, a second one for
validation where a sequence has three. Splitting by sequence would remove whole classes from
training; splitting by frame would leak (neighbouring frames are nearly identical).

Augmentation: every frame is used twice, with the two athletes swapped and the 1/2 suffix of
the label flipped. Without it the model can key on the annotation order instead of the position.

Usage:
    python scripts/train_position_model.py --epochs 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bjj.data import load_annotations
from bjj.features import DEFAULT_CONTEXT, build_features
from bjj.stats import group_segments, segment_class_matrix, split_class_name
from bjj.temporal import estimate_transition_matrix
from bjj.train import TrainConfig, fit, predict_proba

ROOT = Path(__file__).resolve().parents[1]


def swapped_labels(classes: list[str]) -> np.ndarray:
    index = {name: i for i, name in enumerate(classes)}
    out = []
    for name in classes:
        base, athlete = split_class_name(name)
        out.append(index[f"{base}{3 - athlete}"] if athlete else index[name])
    return np.array(out)


def camera_split(ann) -> tuple[np.ndarray, np.ndarray]:
    """(val_mask, test_mask): the last camera of each sequence tests, the second-to-last validates."""
    sequences = group_segments(segment_class_matrix(ann))
    test_segments, val_segments = [], []
    for s in np.unique(sequences):
        cameras = np.flatnonzero(sequences == s)
        test_segments.append(int(cameras[-1]))
        if len(cameras) > 2:
            val_segments.append(int(cameras[-2]))
    return np.isin(ann.video_ids, val_segments), np.isin(ann.video_ids, test_segments)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/raw/annotations.json")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--no-pairs", action="store_true", help="drop the athlete-relation features")
    parser.add_argument("--context", type=int, nargs="*", default=list(DEFAULT_CONTEXT),
                        help="frame offsets appended as context (empty = single frame)")
    parser.add_argument("--out", type=Path, default=ROOT / "results/position_model.pt")
    args = parser.parse_args()

    ann = load_annotations(args.annotations)
    feature_config = {"with_pairs": not args.no_pairs, "context": tuple(args.context)}
    X = build_features(ann.poses, ann.present, ann.video_ids, **feature_config)
    X_swapped = build_features(ann.poses[:, ::-1], ann.present[:, ::-1], ann.video_ids, **feature_config)
    print(f"features per frame: {X.shape[1]}")
    y = ann.labels
    swap = swapped_labels(ann.classes)

    val, test = camera_split(ann)
    train = ~(val | test)
    print(f"frames: train {train.sum()}, val {val.sum()}, test {test.sum()}")

    X_train = np.vstack([X[train], X_swapped[train]])
    y_train = np.concatenate([y[train], swap[y[train]]])

    config = TrainConfig(epochs=args.epochs, hidden=args.hidden, batch_size=args.batch_size)
    model, history = fit(X_train, y_train, len(ann.classes), config, X[val], y[val])

    device = next(model.parameters()).device.type
    results = {}
    for name, mask in [("val", val), ("test", test)]:
        probabilities = predict_proba(model, X[mask], device)
        predictions = probabilities.argmax(1)
        base = np.array([split_class_name(c)[0] for c in ann.classes])
        results[name] = {
            "accuracy": float((predictions == y[mask]).mean()),
            "base_accuracy": float((base[predictions] == base[y[mask]]).mean()),
        }
        print(f"{name}: accuracy={results[name]['accuracy']:.3f} "
              f"base={results[name]['base_accuracy']:.3f}")

    transitions = estimate_transition_matrix(y[train], ann.video_ids[train], len(ann.classes))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "classes": ann.classes,
        "num_features": X.shape[1],
        "feature_config": {"with_pairs": feature_config["with_pairs"],
                           "context": list(feature_config["context"])},
        "hidden": args.hidden,
        "transitions": transitions,
        "results": results,
        "history": history,
    }, args.out)
    (args.out.parent / "position_model_metrics.json").write_text(
        json.dumps({"results": results, "history": history}, indent=2), encoding="utf-8")
    print(f"saved {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
