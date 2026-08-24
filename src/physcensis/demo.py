"""Small dependency-free HTTP demo for the reproduction pipeline."""

from __future__ import annotations

import json
import mimetypes
import re
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from physcensis.agent import OpenAIPredicateAgent, TemplatePredicateAgent
from physcensis.assets import AssetCatalog
from physcensis.config import ReproductionConfig
from physcensis.physics import GenesisBackend, QuasiStaticBackend
from physcensis.pipeline import ScenePipeline


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:42]
    return slug or "scene"


def serve_demo(
    *,
    host: str,
    port: int,
    config: ReproductionConfig,
    backend_name: str,
    output_root: Path,
    examples_dir: Path,
    use_openai: bool,
    model: str,
    catalog: AssetCatalog | None = None,
) -> None:
    repository_root = Path.cwd()
    web_root = Path(__file__).resolve().parent / "web"
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if backend_name == "genesis":
        backend = GenesisBackend(headless=True)
    else:
        backend = QuasiStaticBackend(config.physical.displacement_threshold_m)
    agent = (
        OpenAIPredicateAgent(model=model)
        if use_openai
        else TemplatePredicateAgent(examples_dir)
    )
    pipeline = ScenePipeline(config, backend, catalog=catalog)
    generation_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "PhyScensisDemo/0.1"

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/api/health":
                self._json(
                    {
                        "status": "ok",
                        "backend": backend.name,
                        "agent": "openai" if use_openai else "template",
                    }
                )
                return
            if path == "/preview":
                candidates = (
                    repository_root
                    / "output"
                    / "scenes"
                    / "dense_dishwashing_station_objaverse_v3"
                    / "overview.png",
                    repository_root
                    / "output"
                    / "scenes"
                    / "dense_tool_crate_genesis_v1"
                    / "overview.png",
                    repository_root
                    / "output"
                    / "scenes"
                    / "dense_office_tote_genesis_v1"
                    / "overview.png",
                    repository_root
                    / "output"
                    / "scenes"
                    / "dense_kitchen_sink_objaverse_v2"
                    / "overview.png",
                    repository_root
                    / "output"
                    / "scenes"
                    / "dense_grocery_basket_objaverse_v1"
                    / "overview.png",
                    repository_root
                    / "output"
                    / "scenes"
                    / "dense_kitchen_sink_objaverse_v1"
                    / "overview.png",
                    repository_root
                    / "output"
                    / "scenes"
                    / "dense_grocery_basket_genesis_v2"
                    / "overview.png",
                    repository_root
                    / "output"
                    / "scenes"
                    / "dense_kitchen_sink_genesis_v2"
                    / "overview.png",
                    repository_root
                    / "output"
                    / "scenes"
                    / "dense_grocery_basket_genesis"
                    / "overview.png",
                    repository_root
                    / "output"
                    / "scenes"
                    / "dense_kitchen_sink_genesis"
                    / "overview.png",
                    repository_root
                    / "output"
                    / "scenes"
                    / "dining_table_genesis_v7"
                    / "overview.png",
                    repository_root
                    / "output"
                    / "scenes"
                    / "dining_table_genesis_v6"
                    / "overview.png",
                    repository_root
                    / "output"
                    / "scenes"
                    / "dining_table_genesis_v5"
                    / "overview.png",
                    repository_root
                    / "output"
                    / "scenes"
                    / "dining_table_genesis_v4"
                    / "overview.png",
                    repository_root
                    / "output"
                    / "scenes"
                    / "dining_table_genesis_v2"
                    / "overview.png",
                    repository_root / "output" / "scenes" / "dining_table" / "overview.svg",
                )
                preview = next((candidate for candidate in candidates if candidate.is_file()), None)
                if preview is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                else:
                    self._file(preview)
                return
            if path.startswith("/output/"):
                self._safe_file(output_root, path.removeprefix("/output/"))
                return
            relative = "index.html" if path == "/" else path.lstrip("/")
            self._safe_file(web_root, relative)

        def do_POST(self) -> None:
            if self.path != "/api/generate":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                prompt = str(body.get("prompt", "")).strip()
                if not prompt:
                    raise ValueError("Prompt cannot be empty")
                run_name = f"{int(time.time())}-{_safe_slug(prompt)}"
                run_dir = output_root / run_name
                with generation_lock:
                    result = pipeline.generate(prompt, agent, output_dir=run_dir)
                image_name = None
                if result.artifacts is not None:
                    image_name = (
                        result.artifacts.overview_png.name
                        if result.artifacts.overview_png is not None
                        else result.artifacts.overview_svg.name
                    )
                response: dict[str, Any] = {
                    "success": result.success,
                    "rounds": result.rounds,
                    "summary": result.feedback.summary,
                    "category": result.feedback.category,
                    "metrics": dict(result.feedback.measurements),
                    "issues": [issue.message for issue in result.feedback.issues],
                    "object_count": len(result.scene.objects),
                    "packing_fraction": result.feedback.measurements.get(
                        "packing_fraction", 0.0
                    ),
                    "image": f"/output/{run_name}/{image_name}" if image_name else None,
                    "program": result.program.raw_payload if result.program else None,
                }
                self._json(response, HTTPStatus.OK if result.success else HTTPStatus.UNPROCESSABLE_ENTITY)
            except Exception as exc:  # noqa: BLE001 - convert request failures to JSON
                self._json(
                    {"success": False, "error": f"{type(exc).__name__}: {exc}"},
                    HTTPStatus.BAD_REQUEST,
                )

        def _safe_file(self, root: Path, relative: str) -> None:
            candidate = (root / relative).resolve()
            if root != candidate and root not in candidate.parents:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if not candidate.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._file(candidate)

        def _file(self, path: Path) -> None:
            payload = path.read_bytes()
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[demo] {self.address_string()} {format % args}")

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"PhyScensis demo: http://{host}:{port} ({backend.name} backend)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
