import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np


def preprocess_gray(img, clahe=False, blur_ksize=0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = c.apply(gray)
    if blur_ksize and blur_ksize > 0:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    return gray


def parse_scales(scales_str: str) -> List[float]:
    return [float(s) for s in scales_str.split(",") if s.strip()]


def match_best(img_gray, tmpl_bgr, scales: List[float], method=cv2.TM_CCOEFF_NORMED):
    best = {"val": -1.0, "loc": (0, 0), "scale": 1.0, "w": None, "h": None}
    tmpl_gray = preprocess_gray(tmpl_bgr, clahe=False, blur_ksize=0)
    th, tw = tmpl_gray.shape[:2]
    for s in scales:
        if abs(s - 1.0) > 1e-06:
            t_resized = cv2.resize(
                tmpl_gray, (int(tw * s), int(th * s)), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
            )
        else:
            t_resized = tmpl_gray
        th2, tw2 = t_resized.shape[:2]
        if th2 < 5 or tw2 < 5:
            continue
        if img_gray.shape[0] < th2 or img_gray.shape[1] < tw2:
            continue
        res = cv2.matchTemplate(img_gray, t_resized, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
            score = 1.0 - min_val
            loc = min_loc
        else:
            score = max_val
            loc = max_loc
        if score > best["val"]:
            best.update({"val": float(score), "loc": loc, "scale": float(s), "w": int(tw2), "h": int(th2)})
    return best


def detect_headers(
    img_bgr, templates: Dict[str, np.ndarray], threshold=0.85, scales: List[float] = [1.0], debug_dir=None
):
    img_gray = preprocess_gray(img_bgr, clahe=False, blur_ksize=0)
    states = {}
    for name, tmpl in templates.items():
        best = match_best(img_gray, tmpl, scales, method=cv2.TM_CCOEFF_NORMED)
        found = bool(best["val"] >= threshold)
        x, y = best["loc"]
        w, h = best["w"], best["h"]
        cx, cy = x + w // 2, y + h // 2 if w and h else (None, None)
        states[name] = {
            "found": found,
            "confidence": round(best["val"], 4),
            "top_left": [int(x), int(y)],
            "bottom_right": [int(x + (w or 0)), int(y + (h or 0))],
            "center": [int(cx) if cx is not None else None, int(cy) if cy is not None else None],
            "scale": best["scale"],
            "size": [w, h],
        }
    return states


def classify_by_y_band(states: Dict[str, dict], y_band_tolerance=20):
    ys = [data["center"][1] for data in states.values() if data["found"] and data["center"][1] is not None]
    if not ys:
        return [], list(states.keys()), None
    y_med = int(np.median(ys))
    active, inactive = [], []
    for name, data in states.items():
        if not data["found"] or data["center"][1] is None:
            inactive.append(name)
            continue
        if abs(data["center"][1] - y_med) <= y_band_tolerance:
            active.append(name)
        else:
            inactive.append(name)
    return active, inactive, y_med


def annotate(img_bgr, states: Dict[str, dict], active_names: List[str], out_path: str, y_med=None, y_tol=20):
    vis = img_bgr.copy()
    for name, data in states.items():
        if not data["found"]:
            continue
        pt1 = tuple(data["top_left"])
        pt2 = tuple(data["bottom_right"])
        color = (0, 200, 0) if name in active_names else (0, 0, 220)
        cv2.rectangle(vis, pt1, pt2, color, 2)
        label = f"{name} {data['confidence']:.2f}"
        cv2.putText(vis, label, (pt1[0], max(20, pt1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    if y_med is not None:
        y1, y2 = max(0, y_med - y_tol), min(vis.shape[0] - 1, y_med + y_tol)
        cv2.line(vis, (0, y_med), (vis.shape[1] - 1, y_med), (200, 200, 0), 1)
        cv2.rectangle(vis, (0, y1), (vis.shape[1] - 1, y2), (200, 200, 0), 1)
    cv2.imwrite(out_path, vis)


def save_json(out_path, payload: dict):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
