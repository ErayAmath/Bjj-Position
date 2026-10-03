"""Fine-tune the dataset model on your own labelled footage.

Why this exists (measured, see the lab notebook): the model trained on the ViCoS dataset reaches
84.5 % on that dataset's own held-out camera but only ~32 % on a phone video from a different
gym. The positions there simply look different — an open guard filmed from the end of the mat
has the same geometry as the dataset's turtle. Feature tricks did not close that gap; 60 seconds
of own labels nearly tripled accuracy on the unseen rest of the same video.

Training mixes your frames (repeated, because there are few) with a sample of the dataset, so
the model adapts without forgetting everything it learned.

Evaluation is honest about how little data there is:
  * several videos -> leave one video out, rotating, and report each result
  * one video      -> train on the first part, test on the last, with a gap in between

Usage:
    python scripts/finetune_on_own.py --pair data/cache/roll_10fps.npz data/labels/roll.labels.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from bjj.data import load_annotations
from bjj.features import build_features
from bjj.pipeline.tracking import track_athletes
from bjj.split import split_masks
from bjj.train import PositionMLP, predict_proba, train_step

ROOT = Path(__file__).resolve().parents[1]


def base_names(classes: list[str]) -> np.ndarray:
    return np.array([c[:-1] if c[-1] in "12" and not c[-2].isdigit() else c for c in classes])


def load_pair(cache_path: Path, labels_path: Path, classes: list[str], fps: float, cfg: dict):
    """One labelled video -> (features, labels, times). Frames before the first mark are dropped."""
    cache = np.load(cache_path, allow_pickle=True)
    times = cache["times"]
    poses, present = track_athletes(list(cache["detections"]))
    features = build_features(poses, present, np.zeros(len(poses), int), fps, **cfg).astype(np.float32)

    marks = sorted(json.loads(labels_path.read_text(encoding="utf-8"))["marks"], key=lambda m: m["t"])
    mark_times = np.array([m["t"] for m in marks])
    names = [m["label"] for m in marks]
    index = np.searchsorted(mark_times, times, side="right") - 1
    labelled = index >= 0
    y = np.array([classes.index(names[i]) for i in index[labelled]])
    return features[labelled], y, times[labelled]


def fine_tune(checkpoint: dict, X_own: np.ndarray, y_own: np.ndarray, dataset_sample: tuple,
              epochs: int, repeats: int, learning_rate: float, seed: int = 0):
    """Return a model adapted to the own footage without forgetting the dataset."""
    torch.manual_seed(seed)
    model = PositionMLP(checkpoint["num_features"], len(checkpoint["classes"]), checkpoint["hidden"])
    model.load_state_dict(checkpoint["state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    X_dataset, y_dataset = dataset_sample
    X = np.vstack([np.repeat(X_own, repeats, axis=0), X_dataset])
    y = np.concatenate([np.repeat(y_own, repeats), y_dataset])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X, device=device), torch.tensor(y, device=device)),
        batch_size=512, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            train_step(model, batch_x, batch_y, loss_fn, optimizer)
    return model, device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", nargs=2, action="append", metavar=("CACHE", "LABELS"), required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "results/position_model.pt")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=8, help="how often own frames are repeated")
    parser.add_argument("--dataset-frames", type=int, default=20000, help="dataset frames mixed in")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--margin", type=float, default=2.0, help="seconds dropped around a split")
    parser.add_argument("--out", type=Path, default=ROOT / "results/finetune_own.csv")
    parser.add_argument("--save-model", type=Path, default=ROOT / "results/position_model_own.pt")
    args = parser.parse_args()

    checkpoint = torch.load(args.model, weights_only=False)
    classes, cfg = checkpoint["classes"], checkpoint.get("feature_config", {})
    base = base_names(classes)

    videos = [load_pair(Path(c), Path(l), classes, args.fps, cfg) for c, l in args.pair]
    names = [Path(l).stem for _, l in args.pair]
    print(f"{len(videos)} labelled video(s): " + ", ".join(f"{n} ({len(y)} frames)" for n, (_, y, _) in zip(names, videos)))

    ann = load_annotations(ROOT / "data/raw/annotations.json")
    train_mask, _, _ = split_masks(ann)
    rng = np.random.default_rng(0)
    sample = rng.choice(np.flatnonzero(train_mask), min(args.dataset_frames, int(train_mask.sum())), replace=False)
    dataset_sample = (build_features(ann.poses[sample], ann.present[sample], ann.video_ids[sample],
                                     25.0, **cfg).astype(np.float32), ann.labels[sample])

    rows = []
    folds = []
    if len(videos) > 1:
        for i in range(len(videos)):
            train_X = np.vstack([videos[j][0] for j in range(len(videos)) if j != i])
            train_y = np.concatenate([videos[j][1] for j in range(len(videos)) if j != i])
            folds.append((f"without {names[i]}", train_X, train_y, videos[i][0], videos[i][1]))
    else:
        X, y, times = videos[0]
        cut = times[0] + 0.6 * (times[-1] - times[0])
        train, test = times < cut - args.margin, times > cut + args.margin
        folds.append((f"{names[0]}: first 60 % -> last 40 %", X[train], y[train], X[test], y[test]))

    for name, train_X, train_y, test_X, test_y in folds:
        before_model = PositionMLP(checkpoint["num_features"], len(classes), checkpoint["hidden"])
        before_model.load_state_dict(checkpoint["state_dict"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        before = predict_proba(before_model.to(device), test_X, device).argmax(1)

        model, device = fine_tune(checkpoint, train_X, train_y, dataset_sample,
                                  args.epochs, args.repeats, args.learning_rate)
        after = predict_proba(model, test_X, device).argmax(1)

        row = {
            "fold": name, "train_frames": len(train_y), "test_frames": len(test_y),
            "before_exact": round(float((before == test_y).mean()), 4),
            "before_base": round(float((base[before] == base[test_y]).mean()), 4),
            "after_exact": round(float((after == test_y).mean()), 4),
            "after_base": round(float((base[after] == base[test_y]).mean()), 4),
        }
        rows.append(row)
        print(f"  {name:44s} base {row['before_base']:.1%} -> {row['after_base']:.1%}"
              f"   exact {row['before_exact']:.1%} -> {row['after_exact']:.1%}")

    # Final model: trained on everything that is labelled, for actual use on new videos.
    all_X = np.vstack([v[0] for v in videos])
    all_y = np.concatenate([v[1] for v in videos])
    model, _ = fine_tune(checkpoint, all_X, all_y, dataset_sample,
                         args.epochs, args.repeats, args.learning_rate)
    torch.save({**checkpoint, "state_dict": model.state_dict(),
                "finetuned_on": names, "finetune_frames": int(len(all_y))}, args.save_model)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out.relative_to(ROOT)} and {args.save_model.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
