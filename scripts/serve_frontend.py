"""Serve frontend/ locally with caching disabled.

The stdlib `http.server` sends no Cache-Control header, so browsers heuristically reuse
stale JS/CSS after edits. This subclass forces a fresh fetch on every reload.

By default it listens on 127.0.0.1, so only this machine can reach it. Pass --lan to listen on
all interfaces, which makes the dashboard reachable from a phone in the same WiFi — and from
anything else on that network, so only do it on a network you trust.

Usage:
    python scripts/serve_frontend.py [--port 8000] [--lan]
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
    parser.add_argument("--lan", action="store_true",
                        help="also accept connections from other devices in this network (phone)")
    args = parser.parse_args()

    host = "0.0.0.0" if args.lan else "127.0.0.1"
    handler = partial(NoCacheHandler, directory=str(FRONTEND))
    with ThreadingHTTPServer((host, args.port), handler) as server:
        print(f"Serving {FRONTEND} at http://localhost:{args.port}")
        if args.lan:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect(("8.8.8.8", 80))     # no packet is sent; reveals the local address
                print(f"Reachable in this network at http://{probe.getsockname()[0]}:{args.port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
