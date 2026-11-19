"""Serve the viewer on this machine: the page (``static/``) and its scene (``/scene.json``, gzipped).

    lumen view sk                          # a registered detector: its GDML under LUMEN_DATA, read with its recipe
    lumen view next100.gdml --open         # any GDML, read without a recipe
    lumen view tank.gdml --recipe my_recipes.py:TANK                   # with a recipe of your own
    lumen view sk --out sk.view.json       # write the scene; ``lumen view --scene sk.view.json`` serves it again

The server binds 127.0.0.1 by default (from a laptop, forward it: ``ssh -L 8765:127.0.0.1:8765 <node>``),
uses only the standard library, and builds the scene once, before it starts.
"""

from __future__ import annotations

import gzip
import json
import sys
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC = Path(__file__).parent / "static"
TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}


def encode(payload) -> bytes:
    """Return the scene as JSON bytes (NaN-free: a non-finite number would not parse in the browser)."""
    return json.dumps(payload, separators=(",", ":"), allow_nan=False, default=_default).encode()


def _default(x):
    import numpy as np

    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"not JSON: {type(x).__name__}")


def handler(scene_gz: bytes):
    """Return a request handler serving the page and the gzipped scene."""

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 -- the http.server method name
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                path = "/index.html"
            if path == "/scene.json":
                return self._send(scene_gz, "application/json", gz=True)
            f = STATIC / path.lstrip("/")
            if f.parent != STATIC or not f.is_file():
                self.send_error(404)
                return None
            return self._send(f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream"))

        def _send(self, body, ctype, gz=False):
            if gz and "gzip" not in self.headers.get("Accept-Encoding", ""):
                body, gz = gzip.decompress(body), False
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if gz:
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):  # quiet: one line per page load, not per file
            if self.path.startswith("/scene.json"):
                sys.stderr.write(f"[lumen view] scene sent to {self.client_address[0]}\n")

    return Handler


def serve(scene_json: bytes, host="127.0.0.1", port=8765, open_browser=False):
    """Serve the page and a scene (JSON bytes) until interrupted."""
    httpd = ThreadingHTTPServer((host, port), handler(gzip.compress(scene_json, 6)))
    url = f"http://{host}:{port}/"
    sys.stderr.write(f"[lumen view] {url}  (Ctrl-C stops)\n")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main(a) -> int:
    """Run ``lumen view`` (see the module docstring) from parsed arguments; return the exit status."""
    if a.scene:
        data = Path(a.scene).read_bytes()
    else:
        if not a.source:
            sys.stderr.write("lumen view: give a detector name or a GDML file (or --scene)\n")
            return 2
        from lumen.viewer.scene import MAX_INSTANCES, build

        sys.stderr.write(f"[lumen view] reading {a.source} as lumen reads it ...\n")
        view = build(a.source, a.detector, a.max_instances or MAX_INSTANCES, recipe=a.recipe, up=getattr(a, "up", None))
        for stage, text in view.errors:
            sys.stderr.write(f"[lumen view] {stage}: {text}\n")
        meta = view.payload.get("meta", {})
        sys.stderr.write(f"[lumen view] {meta.get('counts', {}).get('instances', 0):,} placed volumes, "
                         f"built in {meta.get('seconds', '?')} s\n")  # fmt: skip
        data = encode(view.payload)
        if a.out:
            Path(a.out).write_bytes(data)
            sys.stderr.write(f"[lumen view] wrote {a.out} ({len(data) / 1e6:.1f} MB)\n")
            return 0
    serve(data, a.host, a.port, a.open)
    return 0
