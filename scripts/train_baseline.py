"""Baseline position classifier on the annotated keypoints.

Deliberately simple (logistic regression on raw keypoints), so that later work — pose
normalisation, an MLP, temporal smoothing — has an honest number to beat.

Split: one camera view of every sparring sequence is held out. Splitting by sequence would
remove whole classes from training (each sequence only drills a few positions); splitting by
frame would leak, because neighbouring frames are nearly identical.

Training uses athlete-swap augmentation: every frame is also used with the two athletes
exchanged and the label's 1/2 suffix flipped. Without it the model could learn "athlete 1 is
the one on the left of the annotation file" instead of the position itself.

Usage:
    python scripts/train_baseline.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bjj.data import Annotations, load_annotations
from bjj.stats import group_segments, segment_class_matrix, split_class_name

ROOT = Path(__file__).resolve().parents[1]


def features(poses: np.ndarray, present: np.ndarray) -> np.ndarray:
    """(N, 2, 17, 3) -> (N, 104): both skeletons flattened plus two "athlete present" flags."""
    flat = poses.reshape(len(poses), -1)
    return np.concatenate([flat, present.astype(np.float32)], axis=1)


def swapped_labels(classes: list[str]) -> np.ndarray:
    """Index of the class with the 1/2 suffix flipped (symmetric classes map to themselves)."""
    index = {name: i for i, name in enumerate(classes)}
    out = []
    for name in classes:
        base, athlete = split_class_name(name)
        out.append(index[f"{base}{3 - athlete}"] if athlete else index[name])
    return np.array(out)


def camera_holdout(ann: Annotations) -> np.ndarray:
    """True for frames of the held-out camera view (the last segment of each sequence)."""
    sequences = group_segments(segment_class_matrix(ann))
    held_out = {int(np.flatnonzero(sequences == s)[-1]) for s in np.unique(sequences)}
    return np.isin(ann.video_ids, list(held_out))


def augment(X: np.ndarray, y: np.ndarray, poses: np.ndarray, present: np.ndarray,
            swap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_swapped = features(poses[:, ::-1], present[:, ::-1])
    return np.vstack([X, X_swapped]), np.concatenate([y, swap[y]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/raw/annotations.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--model-out", type=Path, default=ROOT / "results/baseline_model.json")
    args = parser.parse_args()

    ann = load_annotations(args.annotations)
    X = features(ann.poses, ann.present)
    y = ann.labels
    test = camera_holdout(ann)
    swap = swapped_labels(ann.classes)
    print(f"train frames: {(~test).sum()}, test frames: {test.sum()} (held-out camera per sequence)")

    X_train, y_train = augment(X[~test], y[~test], ann.poses[~test], ann.present[~test], swap)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=400, n_jobs=-1))
    model.fit(X_train, y_train)

    predictions = model.predict(X[test])
    base = np.array([split_class_name(c)[0] for c in ann.classes])
    majority = np.bincount(y_train).argmax()

    rows = [
        {"model": "majority class", "accuracy": round(accuracy_score(y[test], np.full(test.sum(), majority)), 4),
         "base_accuracy": None, "macro_f1": None},
        {"model": "logistic regression (raw keypoints)",
         "accuracy": round(accuracy_score(y[test], predictions), 4),
         "base_accuracy": round(accuracy_score(base[y[test]], base[predictions]), 4),
         "macro_f1": round(f1_score(y[test], predictions, average="macro"), 4)},
    ]
    for row in rows:
        print(row)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "baseline.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Store the fitted coefficients so the end-to-end evaluation can reuse the model.
    scaler, classifier = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    args.model_out.write_text(json.dumps({
        "classes": ann.classes,
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coef": classifier.coef_.tolist(), "intercept": classifier.intercept_.tolist(),
    }), encoding="utf-8")
    print(f"wrote {(args.out / 'baseline.csv').relative_to(ROOT)} and {args.model_out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
