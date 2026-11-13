#!/usr/bin/env python3
"""
GTA IV Face Maps & Skinning Studio Live Server
Binds to 0.0.0.0:8000, permits all preview hosts/origins and iframe embedding,
serves all 20 textures (1024 PNG, 512 PNG, Normal, Specular, DDS, UV Templates),
and streams on-the-fly ZIP bundles with zero extra disk usage.
"""

import io
import json
import mimetypes
import os
import urllib.parse
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
mimetypes.add_type("image/vnd-ms.dds", ".dds")
mimetypes.add_type("application/octet-stream", ".dds")


class StudioHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[GTA4-Studio] {self.address_string()} - {fmt % args}", flush=True)

    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "public, max-age=300")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_HEAD(self):
        self._handle_get(head_only=True)

    def do_GET(self):
        self._handle_get(head_only=False)

    def _handle_get(self, head_only: bool = False):
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)
        qs = urllib.parse.parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            return self._serve_file(os.path.join(ROOT, "web", "index.html"), "text/html; charset=utf-8", head_only)

        if path == "/api/manifest":
            return self._serve_file(os.path.join(ROOT, "metadata.json"), "application/json; charset=utf-8", head_only)

        if path == "/api/download-zip":
            pack = qs.get("pack", ["1024"])[0]
            return self._stream_zip(pack, head_only)

        # Serve static asset directories safely
        rel = path.lstrip("/")
        safe_prefixes = ("textures_1024/", "textures_512/", "normals_512/", "specular_512/", "dds_512/", "templates/", "web/")
        if rel == "GTA_IV_20_Face_Maps_Catalog.jpg" or rel.startswith(safe_prefixes):
            full = os.path.normpath(os.path.join(ROOT, rel))
            if full.startswith(ROOT) and os.path.isfile(full):
                force_dl = qs.get("dl", ["0"])[0] == "1"
                return self._serve_file(full, None, head_only, attachment=force_dl)

        self.send_response(404)
        self._send_cors()
        self.end_headers()
        if not head_only:
            self.wfile.write(b"Not found")

    def _serve_file(self, filepath: str, content_type: str | None, head_only: bool, attachment: bool = False):
        if not os.path.isfile(filepath):
            self.send_response(404)
            self._send_cors()
            self.end_headers()
            return
        if content_type is None:
            ctype, _ = mimetypes.guess_type(filepath)
            content_type = ctype or "application/octet-stream"
        size = os.path.getsize(filepath)
        self.send_response(200)
        self._send_cors()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        if attachment or filepath.endswith(".dds"):
            fname = os.path.basename(filepath)
            self.send_header("Content-Disposition", f'attachment; filename="{fname}"')
        self.end_headers()
        if not head_only:
            with open(filepath, "rb") as f:
                while chunk := f.read(65536):
                    self.wfile.write(chunk)

    def _stream_zip(self, pack: str, head_only: bool):
        folder_map = {
            "1024": [("textures_1024", "textures_1024"), ("templates", "templates")],
            "512": [("textures_512", "textures_512"), ("templates", "templates")],
            "dds": [("dds_512", "dds_512"), ("templates", "templates")],
            "full": [
                ("textures_1024", "textures_1024"),
                ("textures_512", "textures_512"),
                ("dds_512", "dds_512"),
                ("normals_512", "normals_512"),
                ("specular_512", "specular_512"),
                ("templates", "templates"),
            ],
        }
        folders = folder_map.get(pack, folder_map["1024"])
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
            for src_dir, arc_dir in folders:
                abs_dir = os.path.join(ROOT, src_dir)
                if not os.path.isdir(abs_dir):
                    continue
                for fn in sorted(os.listdir(abs_dir)):
                    if fn.startswith("."):
                        continue
                    fp = os.path.join(abs_dir, fn)
                    if os.path.isfile(fp):
                        zf.write(fp, arcname=f"{arc_dir}/{fn}")
            for root_file in ("README.md", "metadata.json", "GTA_IV_20_Face_Maps_Catalog.jpg"):
                rfp = os.path.join(ROOT, root_file)
                if os.path.isfile(rfp):
                    zf.write(rfp, arcname=root_file)

        data = buf.getvalue()
        zip_name = f"GTA_IV_20_Faces_Pack_{pack}.zip"
        self.send_response(200)
        self._send_cors()
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'attachment; filename="{zip_name}"')
        self.end_headers()
        if not head_only:
            self.wfile.write(data)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), StudioHandler)
    print(f"GTA IV Skinning Studio listening on http://0.0.0.0:{port}", flush=True)
    server.serve_forever()
