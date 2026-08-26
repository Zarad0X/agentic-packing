#!/usr/bin/env python3
"""Download BlenderKit shortlist thumbnails and render labeled contact sheets."""

from __future__ import annotations

import argparse
import io
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

CELL_WIDTH = 340
CELL_HEIGHT = 290
IMAGE_HEIGHT = 218
COLUMNS = 4


def _placeholder(target: Path, reason: str) -> Path:
    image = Image.new("RGB", (640, 480), "#dedede")
    draw = ImageDraw.Draw(image)
    draw.multiline_text(
        (24, 190),
        f"THUMBNAIL UNAVAILABLE\n{reason[:70]}",
        fill="#8b1a1a",
        font=ImageFont.load_default(size=22),
        spacing=8,
    )
    image.save(target, quality=92)
    return target


def _download(candidate: dict[str, Any], output_dir: Path) -> Path:
    category_dir = output_dir / candidate["category"]
    category_dir.mkdir(parents=True, exist_ok=True)
    target = category_dir / f"{candidate['asset_base_id']}.jpg"
    if target.is_file():
        return target
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        candidate["thumbnail_url"],
        headers={"User-Agent": "agentic-packing-asset-curator/0.1"},
    )
    last_error = "unknown download error"
    for attempt in range(3):
        try:
            with opener.open(request, timeout=60) as response:
                data = response.read()
            with Image.open(io.BytesIO(data)) as source:
                ImageOps.exif_transpose(source).convert("RGB").save(target, quality=92)
            return target
        except (OSError, urllib.error.URLError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(1.5 * (attempt + 1))
    return _placeholder(target, last_error)


def _fit_image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    return ImageOps.contain(image, (CELL_WIDTH - 12, IMAGE_HEIGHT - 12))


def _render(category: str, candidates: list[dict[str, Any]], output_dir: Path) -> Path:
    rows = max(1, (len(candidates) + COLUMNS - 1) // COLUMNS)
    sheet = Image.new("RGB", (CELL_WIDTH * COLUMNS, CELL_HEIGHT * rows), "#ececec")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=14)
    small = ImageFont.load_default(size=12)
    for index, candidate in enumerate(candidates):
        x = (index % COLUMNS) * CELL_WIDTH
        y = (index // COLUMNS) * CELL_HEIGHT
        draw.rectangle((x, y, x + CELL_WIDTH - 1, y + CELL_HEIGHT - 1), outline="#777")
        image = _fit_image(
            output_dir / category / f"{candidate['asset_base_id']}.jpg"
        )
        sheet.paste(image, (x + (CELL_WIDTH - image.width) // 2, y + 3))
        draw.text(
            (x + 7, y + IMAGE_HEIGHT + 3),
            f"{index + 1:02d} {candidate['name'][:42]}",
            fill="#111",
            font=font,
        )
        dimensions = candidate.get("dimensions_m") or [None, None, None]
        dim_text = "x".join("?" if value is None else f"{float(value):.2f}" for value in dimensions)
        draw.text(
            (x + 7, y + IMAGE_HEIGHT + 25),
            f"{candidate['asset_base_id'][:8]}  {candidate['license']}  {dim_text} m",
            fill="#333",
            font=small,
        )
        draw.text(
            (x + 7, y + IMAGE_HEIGHT + 43),
            f"faces={candidate['face_count']:,} tex={candidate['texture_count']} "
            f"objects={candidate['object_count']} rank={candidate['rank']:.1f}",
            fill="#333",
            font=small,
        )
    target = output_dir / f"{category}_contact_sheet.jpg"
    sheet.save(target, quality=94)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortlist", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    data = json.loads(args.shortlist.read_text(encoding="utf-8"))
    candidates = [
        candidate
        for pool in data["categories"].values()
        for candidate in pool["candidates"]
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_download, candidate, args.output_dir) for candidate in candidates]
        for future in as_completed(futures):
            future.result()
    for category, pool in data["categories"].items():
        print(_render(category, pool["candidates"], args.output_dir))


if __name__ == "__main__":
    main()
