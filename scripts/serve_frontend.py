"""Serve frontend/ locally with caching disabled.

The stdlib `http.server` sends no Cache-Control header, so browsers heuristically reuse
stale JS/CSS after edits. This subclass forces a fresh fetch on every reload.

Usage:
    python scripts/serve_frontend.py [--port 8000]
"""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    handler = partial(NoCacheHandler, directory=str(FRONTEND))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"Serving {FRONTEND} at http://localhost:{args.port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
