"""Convert a video into something browsers can play (H.264 in MP4).

Phone recordings are often HEVC/H.265 (iPhone "High Efficiency"). Chrome and Firefox usually
refuse to decode those, so the labelling tool shows a black player. This converts to H.264.

Uses ffmpeg when it is installed (best quality and speed) and falls back to OpenCV otherwise.

Usage:
    python scripts/convert_video.py data/videos/roll.mov
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2


def with_ffmpeg(source: Path, target: Path, height: int | None) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    scale = ["-vf", f"scale=-2:{height}"] if height else []
    command = [ffmpeg, "-y", "-i", str(source), *scale,
               "-c:v", "libx264", "-preset", "fast", "-crf", "23",
               "-pix_fmt", "yuv420p", "-an", str(target)]
    print("running ffmpeg …")
    return subprocess.run(command, capture_output=True).returncode == 0


def with_opencv(source: Path, target: Path, height: int | None) -> bool:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    frames = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if height and image.shape[0] != height:
            scale = height / image.shape[0]
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if writer is None:
            h, w = image.shape[:2]
            for codec in ("avc1", "H264", "mp4v"):      # avc1 plays in browsers, mp4v often does not
                writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*codec), fps, (w, h))
                if writer.isOpened():
                    print(f"writing with codec {codec}")
                    if codec == "mp4v":
                        print("  warning: mp4v may not play in a browser; install ffmpeg for H.264")
                    break
        writer.write(image)
        frames += 1
    capture.release()
    if writer:
        writer.release()
    return frames > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--height", type=int, default=None,
                        help="scale to this height, e.g. 720 (keeps the aspect ratio)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    target = args.out or args.video.with_name(f"{args.video.stem}_h264.mp4")
    if not (with_ffmpeg(args.video, target, args.height) or with_opencv(args.video, target, args.height)):
        raise SystemExit("conversion failed")
    print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
