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

from bjj.augment import degrade, random_viewpoint
from bjj.data import load_annotations
from bjj.features import DEFAULT_CONTEXT_SECONDS, build_features
from bjj.split import split_masks
from bjj.stats import split_class_name
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/raw/annotations.json")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--pairs", action="store_true", help="add the athlete-relation features")
    parser.add_argument("--context", type=float, nargs="*", default=list(DEFAULT_CONTEXT_SECONDS),
                        help="context offsets in seconds (empty = single frame)")
    parser.add_argument("--fps", type=float, default=25.0, help="frame rate of the dataset")
    parser.add_argument("--val-share", type=float, default=0.15)
    parser.add_argument("--val-margin", type=int, default=50,
                        help="frames dropped around each validation block to avoid leakage")
    parser.add_argument("--noise-jitter", type=float, default=0.06,
                        help="keypoint noise as a fraction of torso size (0 disables)")
    parser.add_argument("--noise-drop-keypoint", type=float, default=0.05)
    parser.add_argument("--noise-drop-athlete", type=float, default=0.08)
    parser.add_argument("--noise-copies", type=int, default=1,
                        help="how many degraded copies of the training data to add")
    parser.add_argument("--viewpoint-copies", type=int, default=0,
                        help="copies with a simulated camera position; measured: no help (see README)")
    parser.add_argument("--out", type=Path, default=ROOT / "results/position_model.pt")
    args = parser.parse_args()

    ann = load_annotations(args.annotations)
    feature_config = {"with_pairs": args.pairs, "context_seconds": tuple(args.context)}
    X = build_features(ann.poses, ann.present, ann.video_ids, args.fps, **feature_config)
    X_swapped = build_features(ann.poses[:, ::-1], ann.present[:, ::-1], ann.video_ids,
                               args.fps, **feature_config)
    print(f"features per frame: {X.shape[1]}")
    y = ann.labels
    swap = swapped_labels(ann.classes)

    train, val, test = split_masks(ann, val_share=args.val_share, margin=args.val_margin)
    print(f"frames: train {train.sum()}, val {val.sum()}, test {test.sum()} "
          f"(test = one camera per sequence; validation = time blocks inside training cameras)")

    # The model is trained on verified keypoints but runs on noisy predicted ones, so part of
    # the training data is deliberately degraded (measured: +2 points end to end).
    blocks_X = [X[train], X_swapped[train]]
    blocks_y = [y[train], swap[y[train]]]
    noise = {"jitter": args.noise_jitter, "drop_keypoint": args.noise_drop_keypoint,
             "drop_athlete": args.noise_drop_athlete}
    if any(noise.values()):
        rng = np.random.default_rng(0)
        for copy in range(args.noise_copies):
            for poses, present, labels in [(ann.poses[train], ann.present[train], y[train]),
                                           (ann.poses[train][:, ::-1], ann.present[train][:, ::-1],
                                            swap[y[train]])]:
                noisy_poses, noisy_present = degrade(poses, present, rng=rng, **noise)
                blocks_X.append(build_features(noisy_poses, noisy_present, ann.video_ids[train],
                                               args.fps, **feature_config))
                blocks_y.append(labels)
    # A camera at a different height/angle foreshortens the athletes: measured on own footage,
    # an open guard filmed from the end of the mat has the same geometry as the dataset's turtle.
    if args.viewpoint_copies:
        rng = np.random.default_rng(1)
        for copy in range(args.viewpoint_copies):
            for poses, present, labels in [(ann.poses[train], ann.present[train], y[train]),
                                           (ann.poses[train][:, ::-1], ann.present[train][:, ::-1],
                                            swap[y[train]])]:
                warped = random_viewpoint(poses, present, rng=rng)
                warped, warped_present = degrade(warped, present, rng=rng, **noise)
                blocks_X.append(build_features(warped, warped_present, ann.video_ids[train],
                                               args.fps, **feature_config))
                blocks_y.append(labels)

    X_train = np.vstack(blocks_X)
    y_train = np.concatenate(blocks_y)
    print(f"training rows: {len(X_train)} ({len(blocks_X)} blocks incl. swap and noise)")

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
                           "context_seconds": list(feature_config["context_seconds"])},
        "trained_fps": args.fps,
        "hidden": args.hidden,
        "transitions": transitions,
        "results": results,
        "split": {"val_share": args.val_share, "val_margin": args.val_margin},
        "noise": noise,
        "viewpoint_copies": args.viewpoint_copies,
        "history": history,
    }, args.out)
    (args.out.parent / "position_model_metrics.json").write_text(
        json.dumps({"results": results, "history": history}, indent=2), encoding="utf-8")
    print(f"saved {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
