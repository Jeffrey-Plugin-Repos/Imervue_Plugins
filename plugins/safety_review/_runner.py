"""Subprocess runner for safety review plugin.

Usage (frozen env):
    python _runner.py <site_packages> single <input> <output> <block_size> <padding> [<mode> <confidence>]
    python _runner.py <site_packages> batch  <json_paths> <output_dir> <block_size> <padding> <overwrite> [<mode> <confidence>]

Protocol — stdout lines:
    PROGRESS:<message>
    OK:<output_path>
    BATCH_PROGRESS:<current>:<total>:<filename>
    BATCH_OK:<success>:<failed>
    ERROR:<message>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# -----------------------------------------------------------------------
# NudeNet labels
# -----------------------------------------------------------------------
# Real-photo mode — only exposed genitalia / anus
MOSAIC_LABELS = frozenset({
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
})

# Anime mode — also include "covered" variants
MOSAIC_LABELS_ANIME = frozenset({
    "FEMALE_GENITALIA_EXPOSED",
    "FEMALE_GENITALIA_COVERED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "ANUS_COVERED",
})

MIN_CONFIDENCE = 0.25


def _labels_for_mode(mode: str) -> frozenset:
    return MOSAIC_LABELS_ANIME if mode == "anime" else MOSAIC_LABELS


def _bootstrap_site_packages(site_packages: str) -> None:
    if site_packages and site_packages not in sys.path:
        sys.path.insert(0, site_packages)


def _mosaic_region(img, x1, y1, x2, y2, block_size):
    from PIL import Image

    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return
    region = img.crop((x1, y1, x2, y2))
    bs = max(2, block_size)
    small = region.resize(
        (max(1, w // bs), max(1, h // bs)),
        resample=Image.Resampling.BILINEAR,
    )
    mosaic = small.resize((w, h), resample=Image.Resampling.NEAREST)
    img.paste(mosaic, (x1, y1))


def _process_one(detector, src, dst, block_size, padding,
                  confidence=MIN_CONFIDENCE, labels=MOSAIC_LABELS):
    """Detect + mosaic one image.  Returns number of regions mosaiced."""
    from PIL import Image

    detections = detector.detect(src)
    regions = [
        d for d in detections
        if d["class"] in labels and d["score"] >= confidence
    ]

    if not regions:
        if os.path.normpath(src) != os.path.normpath(dst):
            import shutil
            shutil.copy2(src, dst)
        return 0

    img = Image.open(src)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")

    for d in regions:
        x1, y1, x2, y2 = d["box"]
        if padding > 0:
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(img.width, x2 + padding)
            y2 = min(img.height, y2 + padding)
        _mosaic_region(img, x1, y1, x2, y2, block_size)

    ext = Path(dst).suffix.lower()
    fmt_map = {
        ".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG",
        ".bmp": "BMP", ".tif": "TIFF", ".tiff": "TIFF",
        ".webp": "WEBP",
    }
    fmt = fmt_map.get(ext, "PNG")
    save_img = img
    if fmt == "JPEG" and save_img.mode == "RGBA":
        save_img = save_img.convert("RGB")
    save_img.save(dst, format=fmt)
    return len(regions)


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 2:
        print("ERROR:Not enough arguments", flush=True)
        sys.exit(1)

    site_packages = args[0]
    mode = args[1]
    _bootstrap_site_packages(site_packages)

    if mode == "single":
        if len(args) < 6:
            print("ERROR:single mode requires: input output block_size padding",
                  flush=True)
            sys.exit(1)
        input_path, output_path = args[2], args[3]
        block_size, padding = int(args[4]), int(args[5])
        # Optional: mode confidence (args[6], args[7])
        det_mode = args[6] if len(args) > 6 else "real"
        confidence = float(args[7]) if len(args) > 7 else MIN_CONFIDENCE
        labels = _labels_for_mode(det_mode)
        try:
            from nudenet import NudeDetector
            print("PROGRESS:Loading NudeNet detector...", flush=True)
            detector = NudeDetector()
            print("PROGRESS:Detecting...", flush=True)
            count = _process_one(detector, input_path, output_path,
                                 block_size, padding,
                                 confidence=confidence, labels=labels)
            if count == 0:
                print("PROGRESS:No genitalia detected", flush=True)
            else:
                print(f"PROGRESS:Mosaiced {count} region(s)", flush=True)
            print(f"OK:{output_path}", flush=True)
        except Exception as exc:
            print(f"ERROR:{exc}", flush=True)
            sys.exit(1)

    elif mode == "batch":
        if len(args) < 7:
            print("ERROR:batch mode requires: json_paths output_dir "
                  "block_size padding overwrite", flush=True)
            sys.exit(1)
        json_paths = args[2]
        output_dir = args[3]
        block_size = int(args[4])
        padding = int(args[5])
        overwrite = args[6].lower() == "true"
        # Optional: mode confidence (args[7], args[8])
        det_mode = args[7] if len(args) > 7 else "real"
        confidence = float(args[8]) if len(args) > 8 else MIN_CONFIDENCE
        labels = _labels_for_mode(det_mode)

        with open(json_paths, encoding="utf-8") as f:
            paths = json.load(f)

        from nudenet import NudeDetector
        print("PROGRESS:Loading NudeNet detector...", flush=True)
        detector = NudeDetector()

        success = 0
        failed = 0
        total = len(paths)

        for i, src in enumerate(paths):
            name = Path(src).name
            print(f"BATCH_PROGRESS:{i}:{total}:{name}", flush=True)
            try:
                if overwrite:
                    dst = src
                else:
                    stem = Path(src).stem
                    suffix = Path(src).suffix or ".png"
                    dst = str(Path(output_dir) / f"{stem}_censored{suffix}")
                    counter = 1
                    while os.path.exists(dst):
                        dst = str(
                            Path(output_dir)
                            / f"{stem}_censored_{counter}{suffix}"
                        )
                        counter += 1
                _process_one(detector, src, dst, block_size, padding,
                             confidence=confidence, labels=labels)
                success += 1
            except Exception as exc:
                print(f"PROGRESS:Error on {name}: {exc}", flush=True)
                failed += 1

        print(f"BATCH_OK:{success}:{failed}", flush=True)
    else:
        print(f"ERROR:Unknown mode: {mode}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
