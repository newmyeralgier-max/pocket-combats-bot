import json
import os
import sys

import cv2
import numpy as np

SCREENS_DIR = "C:/bot/screens/"
TPL_DIR = "C:/bot/tpl/chevrons/"
OUT_JSON = "C:/bot/roi_calibrated.json"
CHEVRONS = {
    "items": {"up": os.path.join(TPL_DIR, "items_arrow_up.png"), "down": os.path.join(TPL_DIR, "items_arrow_down.png")},
    "monsters": {
        "up": os.path.join(TPL_DIR, "monsters_arrow_up.png"),
        "down": os.path.join(TPL_DIR, "monsters_arrow_down.png"),
    },
}
THR = 0.82
MAX_FILES = 20


def match_chevron(img, tpl_path, thr=THR):
    tpl = cv2.imread(tpl_path, cv2.IMREAD_COLOR)
    if tpl is None or img is None:
        return None
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tpl_g = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_g, tpl_g, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < thr:
        return None
    h, w = tpl.shape[:2]
    x0, y0 = max_loc
    x1, y1 = x0 + w, y0 + h
    return x0, y0, x1, y1


def normalize_roi(rect, img_shape):
    h, w = img_shape[:2]
    x0, y0, x1, y1 = rect
    return [round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)]


def scan_folder():
    result = {"items": [], "monsters": []}
    files = [f for f in os.listdir(SCREENS_DIR) if f.lower().endswith(".png")]
    files = files[:MAX_FILES]
    for fname in files:
        path = os.path.join(SCREENS_DIR, fname)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        for key in ["items", "monsters"]:
            arrow_up = match_chevron(img, CHEVRONS[key]["up"])
            arrow_down = match_chevron(img, CHEVRONS[key]["down"])
            rect = arrow_up or arrow_down
            if rect:
                roi_pct = normalize_roi(rect, img.shape)
                result[key].append(roi_pct)
    avg = {}
    for key in ["items", "monsters"]:
        arr = result[key]
        if not arr:
            continue
        avg_roi = [(sum(coords) / len(coords)) for coords in zip(*arr)]
        avg[key] = [round(val, 4) for val in avg_roi]
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(avg, f, indent=2)
    print(f"[OK] ROI сохранены в: {OUT_JSON}")
    print(json.dumps(avg, indent=2))


if __name__ == "__main__":
    scan_folder()
