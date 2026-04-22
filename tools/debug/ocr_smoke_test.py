#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke-тест OCR-детектора на реальном скриншоте, без запуска всего бота.

Запуск:
    python tools/debug/ocr_smoke_test.py path/to/screenshot.png
    python tools/debug/ocr_smoke_test.py screenshot.png --roi 0 400 500 2100
    python tools/debug/ocr_smoke_test.py screenshot.png --fuzzy 0.65

Что делает:
  1. Пробует подхватить доступный OCR-движок (rapidocr / pytesseract).
  2. Запускает распознавание на ROI (или на всём кадре).
  3. Печатает raw-распознанные строки со score.
  4. Если в tools/cfg/config.json есть ALLOWED_ITEM_NAMES — пытается
     замэтчить OCR-строки в этот whitelist и печатает finalлист
     'имя шаблона ← OCR-строка (ratio)'.

Цель — понять, годится ли rapidocr для твоих скриншотов, до того как
включать OCR в bot-конфиге (CFG.OCR.ENABLED=true).
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Позволяем запуск из корня репо без pip-install.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from script.detection import ocr as OCR


def _load_allowed_names(cfg_path: Path):
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        names = cfg.get("ALLOWED_ITEM_NAMES") or []
        return [str(n) for n in names]
    except Exception as e:
        print(f"[WARN] config.json не прочитан: {e}")
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="путь к PNG/JPG со скриншотом")
    ap.add_argument("--roi", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"),
                    help="ROI в координатах кадра (по умолчанию весь кадр)")
    ap.add_argument("--engine", default="auto", choices=["auto", "rapidocr", "tesseract"])
    ap.add_argument("--fuzzy", type=float, default=0.7,
                    help="минимальный SequenceMatcher.ratio для матча в whitelist")
    ap.add_argument("--min-conf", type=float, default=0.4)
    ap.add_argument("--no-whitelist", action="store_true",
                    help="не матчить, печатать только raw OCR")
    args = ap.parse_args()

    engine = OCR.available_engine(args.engine)
    if engine is None:
        print("[FAIL] Нет доступного OCR-движка. Установи одно из:")
        print("  pip install rapidocr-onnxruntime   # рекомендуется")
        print("  pip install pytesseract            # + системный tesseract-ocr (+ -rus)")
        sys.exit(2)
    print(f"[OK] OCR engine: {engine}")

    img = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[FAIL] cv2 не смог прочитать {args.image}")
        sys.exit(2)
    H, W = img.shape[:2]
    print(f"[INFO] image {W}x{H}")

    roi = tuple(args.roi) if args.roi else None

    raw = OCR.run_ocr(img if roi is None else img[roi[1]:roi[3], roi[0]:roi[2]],
                     engine=args.engine, min_conf=args.min_conf)
    print(f"\n=== RAW OCR (score >= {args.min_conf}): {len(raw)} строк ===")
    for i, r in enumerate(raw, 1):
        x, y, w, h = r["box"]
        print(f"  #{i:>3}  score={r['score']:.2f}  box=({x},{y},{w}x{h})  text={r['text']!r}")

    if args.no_whitelist:
        return

    cfg_path = REPO_ROOT / "tools" / "cfg" / "config.json"
    whitelist = _load_allowed_names(cfg_path)
    if not whitelist:
        print("[WARN] ALLOWED_ITEM_NAMES пуст или не найден — пропускаю whitelist-матч.")
        return
    print(f"\n=== WHITELIST MATCH (size={len(whitelist)}, fuzzy_min={args.fuzzy}) ===")
    hits = OCR.detect_item_names_ocr(
        img, roi=roi, whitelist=whitelist,
        engine=args.engine,
        fuzzy_min=args.fuzzy,
        min_conf=args.min_conf,
    )
    if not hits:
        print("  (ничего не замэтчилось)")
        return
    for h in hits:
        name = h["name"]
        ratio = h["score"]
        ocr_text = h.get("ocr_text", "")
        cx, cy = h["center"]
        print(f"  {name!r}  ← {ocr_text!r}   ratio={ratio:.2f}  center=({cx},{cy})")


if __name__ == "__main__":
    main()
