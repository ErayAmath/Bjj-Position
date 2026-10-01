"""Serve frontend/ locally, plus the videos in data/videos/ for the labelling tool.

The stdlib `http.server` sends no Cache-Control header, so browsers heuristically reuse stale
JS/CSS after edits; this handler forces a fresh fetch. It also exposes two extra routes so the
labelling tool can open a video without a file dialog:

    GET /api/videos      -> JSON list of the files in data/videos/
    GET /videos/<name>   -> that file (range requests supported by SimpleHTTPRequestHandler)

By default it listens on 127.0.0.1, so only this machine can reach it. Pass --lan to listen on
all interfaces, which makes the dashboard reachable from a phone in the same WiFi — and also
makes those videos reachable for everyone on that network, so only do it on a network you trust.

Usage:
    python scripts/serve_frontend.py [--port 8000] [--lan]
"""

from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
VIDEOS = ROOT / "data" / "videos"
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}


class _RangeReader:
    """Reads at most `remaining` bytes from an open file, for a 206 response."""

    def __init__(self, source, remaining: int):
        self.source, self.remaining = source, remaining

    def read(self, amount: int = -1) -> bytes:
        if self.remaining <= 0:
            return b""
        if amount is None or amount < 0:
            amount = self.remaining
        data = self.source.read(min(amount, self.remaining))
        self.remaining -= len(data)
        return data

    def close(self) -> None:
        self.source.close()


def parse_range(header: str, size: int) -> tuple[int, int] | None:
    """'bytes=1000-' or 'bytes=0-999' -> (start, end) inclusive, or None if unusable."""
    if not header.startswith("bytes="):
        return None
    first, _, last = header[len("bytes="):].split(",")[0].partition("-")
    try:
        if first:
            start = int(first)
            end = int(last) if last else size - 1
        else:                                   # suffix form: last N bytes
            start, end = max(size - int(last), 0), size - 1
    except ValueError:
        return None
    if start >= size or start > end:
        return None
    return start, min(end, size - 1)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        if "Accept-Ranges" not in self._headers_buffer_names():
            self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def _headers_buffer_names(self) -> set[str]:
        return {line.decode("latin-1").split(":", 1)[0]
                for line in getattr(self, "_headers_buffer", []) if b":" in line}

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/api/videos":
            return self._send_video_list()
        super().do_GET()

    def send_head(self):
        """Serve byte ranges. Without this a browser cannot seek inside a video: Python's
        http.server answers a Range request with the whole file, and the player then treats
        the video as not seekable and jumps back to 0."""
        range_header = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not range_header or os.path.isdir(path):
            return super().send_head()
        try:
            source = open(path, "rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None
        size = os.fstat(source.fileno()).st_size
        span = parse_range(range_header, size)
        if span is None:
            source.close()
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        start, end = span
        source.seek(start)
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        return _RangeReader(source, end - start + 1)

    def translate_path(self, path: str) -> str:
        # /videos/<name> comes from data/videos/, everything else from frontend/
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean.startswith("/videos/"):
            name = Path(clean[len("/videos/"):]).name      # no directory traversal
            return str(VIDEOS / name)
        return super().translate_path(path)

    def _send_video_list(self) -> None:
        files = []
        if VIDEOS.is_dir():
            files = sorted(
                ({"name": p.name, "size_mb": round(p.stat().st_size / 1e6, 1)}
                 for p in VIDEOS.iterdir()
                 if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES),
                key=lambda item: item["name"],
            )
        body = json.dumps({"videos": files}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--lan", action="store_true",
                        help="also accept connections from other devices in this network (phone)")
    args = parser.parse_args()

    host = "0.0.0.0" if args.lan else "127.0.0.1"
    with ThreadingHTTPServer((host, args.port), Handler) as server:
        print(f"Serving {FRONTEND} at http://localhost:{args.port}")
        print(f"Videos from {VIDEOS} at http://localhost:{args.port}/videos/<name>")
        if args.lan:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect(("8.8.8.8", 80))     # no packet is sent; reveals the local address
                print(f"Reachable in this network at http://{probe.getsockname()[0]}:{args.port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
