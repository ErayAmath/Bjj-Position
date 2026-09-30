"""Compare an analysis against labels you made yourself in frontend/label.html.

This is the only measurement that answers "does it work on MY footage". Everything else was
measured on the dataset, which was filmed in one gym with fixed cameras.

Reports overall accuracy, accuracy per position, and the confusions that cost the most time,
so that the next improvement can be aimed instead of guessed.

Usage:
    python scripts/eval_own_labels.py data/analyses/roll.json data/labels/roll.labels.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def truth_at(times: np.ndarray, marks: list[dict]) -> np.ndarray:
    """Label of each analysed frame: a mark applies until the next one starts."""
    mark_times = np.array([m["t"] for m in marks])
    labels = [m["label"] for m in marks]
    index = np.searchsorted(mark_times, times, side="right") - 1
    return np.array([labels[i] if i >= 0 else None for i in index], dtype=object)


def base_of(label: str | None) -> str | None:
    if label is None:
        return None
    return label[:-1] if label[-1] in "12" and not label[-2].isdigit() else label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "results/own_footage_eval.csv")
    args = parser.parse_args()

    report = json.loads(args.analysis.read_text(encoding="utf-8"))
    truth_file = json.loads(args.labels.read_text(encoding="utf-8"))
    marks = sorted(truth_file["marks"], key=lambda m: m["t"])
    if not marks:
        raise SystemExit("the label file has no marks")

    times = np.array(report["times"])
    classes = report["classes"]
    predicted = np.array([classes[i] for i in report["labels_smoothed"]], dtype=object)
    predicted_raw = np.array([classes[i] for i in report["labels_raw"]], dtype=object)
    truth = truth_at(times, marks)

    labelled = truth != None                          # noqa: E711 — frames before the first mark
    truth, predicted, predicted_raw = truth[labelled], predicted[labelled], predicted_raw[labelled]
    seconds_per_frame = float(np.median(np.diff(times))) if len(times) > 1 else 0.1

    exact = float((truth == predicted).mean())
    base = float((np.array([base_of(t) for t in truth]) == np.array([base_of(p) for p in predicted])).mean())
    raw_exact = float((truth == predicted_raw).mean())
    print(f"frames compared: {len(truth)} ({len(truth) * seconds_per_frame:.0f} s of labelled video)")
    print(f"accuracy  exact position : {exact:.1%}   (before smoothing: {raw_exact:.1%})")
    print(f"accuracy  base position  : {base:.1%}")

    print("\nper position (by labelled time):")
    rows = []
    for position in sorted(set(truth)):
        mask = truth == position
        accuracy = float((predicted[mask] == position).mean())
        confusion = Counter(predicted[mask][predicted[mask] != position]).most_common(2)
        print(f"  {position:16s} {mask.sum() * seconds_per_frame:5.1f}s  {accuracy:5.1%}"
              f"   most often seen as: {', '.join(f'{name} ({n})' for name, n in confusion) or 'nothing'}")
        rows.append({"position": position, "labelled_seconds": round(mask.sum() * seconds_per_frame, 1),
                     "accuracy": round(accuracy, 4),
                     "confused_with": "; ".join(f"{name}:{n}" for name, n in confusion)})

    print("\nmost expensive confusions (seconds of video):")
    pairs = Counter((t, p) for t, p in zip(truth, predicted) if t != p)
    for (t, p), n in pairs.most_common(8):
        print(f"  {t:16s} -> {p:16s} {n * seconds_per_frame:5.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["position", "labelled_seconds", "accuracy", "confused_with"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {"video": truth_file.get("video"), "frames": len(truth),
               "accuracy_exact": round(exact, 4), "accuracy_exact_unsmoothed": round(raw_exact, 4),
               "accuracy_base": round(base, 4)}
    (args.out.with_suffix(".json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
