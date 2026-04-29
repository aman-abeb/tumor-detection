#!/usr/bin/env python3
"""
Legacy offline generator: segmentation + classification tiles under
webapp/static/batch_gallery/ (manifest.json + tile_*.png).

The /batch-test page uses live YOLO batch uploads instead; this script is optional for static demos only.

Run from the repo root:
    python scripts/generate_static_batch.py

Requires BUSI_Jpeg/... and the same checkpoints imported by webapp/app.py.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
WEBAPP_STATIC = ROOT / "webapp" / "static" / "batch_gallery"


def dataset_batch_source_paths() -> list[Path]:
    imgs: list[Path] = []
    base = ROOT / "BUSI_Jpeg"
    for label in ("benign", "malignant"):
        d = base / label
        if not d.is_dir():
            continue
        for pat in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
            for p in d.glob(pat):
                if p.is_file():
                    imgs.append(p.resolve())
    return sorted(set(imgs), key=lambda x: x.as_posix().lower())


def main() -> None:
    sys.path.insert(0, str(ROOT / "webapp"))

    from app import (  # noqa: E402
        BASE_DIR,
        BATCH_MAX_IMAGES,
        CLASS_NAMES,
        _tile_bgr_from_result,
        _tile_error_bgr,
        predict_classification,
        predict_segmentation,
    )

    WEBAPP_STATIC.mkdir(parents=True, exist_ok=True)
    # Clear prior tiles so the folder matches this run exactly
    for old in WEBAPP_STATIC.glob("tile_*.png"):
        old.unlink()

    pool = dataset_batch_source_paths()
    if not pool:
        print("ERROR: No images under BUSI_Jpeg/benign or BUSI_Jpeg/malignant")
        manifest = {"items": [], "error": "no_dataset_images"}
        (WEBAPP_STATIC / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        sys.exit(1)

    k = min(len(pool), BATCH_MAX_IMAGES)
    chosen = random.sample(pool, k=k)

    items = []
    for idx, img_path in enumerate(chosen):
        fname = img_path.name
        tile_name = f"tile_{idx:03d}.png"
        out_path = WEBAPP_STATIC / tile_name
        try:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                raise ValueError("Could not decode image.")
            img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if img_rgb.size == 0:
                raise ValueError("Empty image.")
            _, seg_mask = predict_segmentation(img_rgb)
            cls_idx, probs = predict_classification(img_rgb)

            tile_bgr = _tile_bgr_from_result(
                img_rgb,
                seg_mask,
                fname,
                CLASS_NAMES[cls_idx],
                float(probs[0]),
                float(probs[1]),
            )
            cv2.imwrite(str(out_path), tile_bgr)
            items.append(
                {
                    "filename": fname,
                    "dataset_path": str(img_path.relative_to(BASE_DIR)).replace("\\", "/"),
                    "pred": CLASS_NAMES[cls_idx],
                    "p_benign": float(probs[0]),
                    "p_malignant": float(probs[1]),
                    "tile": tile_name,
                    "ok": True,
                }
            )
        except Exception as exc:
            err_bgr = _tile_error_bgr(fname, str(exc))
            cv2.imwrite(str(out_path), err_bgr)
            items.append(
                {
                    "filename": fname,
                    "dataset_path": str(img_path.relative_to(BASE_DIR)).replace("\\", "/"),
                    "pred": None,
                    "error": str(exc),
                    "tile": tile_name,
                    "ok": False,
                }
            )

    manifest = {"count": len(items), "items": items}
    (WEBAPP_STATIC / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(items)} tiles → {WEBAPP_STATIC}")


if __name__ == "__main__":
    main()
