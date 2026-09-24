"""Download a small stratified sample of dataset images without fetching the 10 GB archive.

The server supports HTTP range requests, so `remotezip` can read the zip's central directory
and then pull only the members we ask for (~83 KB per image).

Sampling: `--per-class` frames per position class, spread evenly over the sequences that
contain the class, and only frames where the annotations list BOTH athletes — those are the
frames where "did the detector find two people?" is a meaningful question.

Usage:
    python scripts/fetch_dataset_images.py --per-class 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from remotezip import RemoteZip

from bjj.data import load_annotations
from bjj.stats import group_segments, segment_class_matrix

URL = "https://data.vicos.si/datasets/JuiJuitsu/images.zip"
ROOT = Path(__file__).resolve().parents[1]


def choose_frames(ann, per_class: int, seed: int = 0) -> list[int]:
    """Indices of sampled frames: per_class per class, spread over sequences and time."""
    sequences = group_segments(segment_class_matrix(ann))[ann.video_ids]
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for label in range(len(ann.classes)):
        candidates = np.flatnonzero((ann.labels == label) & ann.present.all(axis=1))
        if not len(candidates):
            continue
        # Split the class's frames by sequence, then take evenly spaced frames from each,
        # so the sample covers different camera angles and different moments.
        per_sequence = max(1, per_class // max(1, len(np.unique(sequences[candidates]))))
        for sequence in np.unique(sequences[candidates]):
            pool = candidates[sequences[candidates] == sequence]
            take = min(per_sequence, len(pool))
            chosen.extend(pool[np.linspace(0, len(pool) - 1, take).round().astype(int)].tolist())
    rng.shuffle(chosen)
    return sorted(set(chosen))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/raw/annotations.json")
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--out", type=Path, default=ROOT / "data/raw/images")
    parser.add_argument("--contiguous", type=int, metavar="N",
                        help="instead of the stratified sample: N consecutive frames starting at --from-index")
    parser.add_argument("--from-index", type=int, default=0)
    parser.add_argument("--manifest-name", default="sample_manifest.json")
    args = parser.parse_args()

    ann = load_annotations(args.annotations)
    if args.contiguous:
        indices = list(range(args.from_index, args.from_index + args.contiguous))
    else:
        indices = choose_frames(ann, args.per_class)
    raw = json.loads(args.annotations.read_text(encoding="utf-8"))
    names = [f"{raw[i]['image']}.jpg" for i in indices]

    args.out.mkdir(parents=True, exist_ok=True)
    missing = [n for n in names if not (args.out / n).exists()]
    print(f"{len(indices)} frames selected, {len(missing)} to download")

    downloaded = 0
    if missing:
        with RemoteZip(URL) as z:
            for n, name in enumerate(missing, 1):
                z.extract(name, path=args.out)
                downloaded += (args.out / name).stat().st_size
                if n % 25 == 0:
                    print(f"  {n}/{len(missing)} ({downloaded / 1e6:.1f} MB)")

    manifest = [
        {"index": i, "image": raw[i]["image"], "file": f"{raw[i]['image']}.jpg",
         "position": ann.classes[ann.labels[i]], "segment": int(ann.video_ids[i]),
         "frame": int(ann.frames[i])}
        for i in indices
    ]
    path = args.out.parent / args.manifest_name
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"Downloaded {downloaded / 1e6:.1f} MB; manifest: {path}")


if __name__ == "__main__":
    main()
