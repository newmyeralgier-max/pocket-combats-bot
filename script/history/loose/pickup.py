import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

SCREENS_DIR = "C:/bot/screens"
TPL_DIR = "C:/bot/tpl/my"
OUT_DIR = "C:/bot/pickup_out"
ANNOT_DIR = os.path.join(OUT_DIR, "annotated")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ANNOT_DIR, exist_ok=True)
TPL_PICKUP_ACTIVE_1 = os.path.join(TPL_DIR, "pickup.png")
TPL_PICKUP_ACTIVE_2 = os.path.join(TPL_DIR, "pickup_own.png")
TPL_PICKUP_INACTIVE = os.path.join(TPL_DIR, "pickup_other.png")
ACTIVE_THRESH_COLOR = 0.9
INACTIVE_THRESH_GRAY = 0.9
SCALES_COLOR = [0.95, 1.0, 1.05]
SCALES_GRAY = [0.95, 1.0, 1.05]
BORDER = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_bgr(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.isfile(path):
        return None
    return cv2.imread(path, cv2.IMREAD_COLOR)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        g = img
    else:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 5, 35, 35)
    g = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(g)
    return g


def match_color(
    image_bgr: np.ndarray, tpl_bgr: np.ndarray, scales: List[float]
) -> Tuple[float, Tuple[int, int, int, int]]:
    """
    Цветовой template matching: берём 3 канала, корреляцию считаем на каждом и усредняем.
    Возвращаем (score, (x,y,w,h)) лучшего масштаба/позиции.
    """
    H, W = image_bgr.shape[:2]
    best_score = -1.0
    best_box = 0, 0, 0, 0
    img_b, img_g, img_r = cv2.split(image_bgr)
    for s in scales:
        tpl_rs = cv2.resize(
            tpl_bgr,
            (max(5, int(tpl_bgr.shape[1] * s)), max(5, int(tpl_bgr.shape[0] * s))),
            interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC,
        )
        h, w = tpl_rs.shape[:2]
        if H < h or W < w:
            continue
        tpl_b, tpl_g, tpl_r = cv2.split(tpl_rs)
        res_b = cv2.matchTemplate(img_b, tpl_b, cv2.TM_CCOEFF_NORMED)
        res_g = cv2.matchTemplate(img_g, tpl_g, cv2.TM_CCOEFF_NORMED)
        res_r = cv2.matchTemplate(img_r, tpl_r, cv2.TM_CCOEFF_NORMED)
        res = (res_b + res_g + res_r) / 3.0
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = float(max_val)
            best_box = int(max_loc[0]), int(max_loc[1]), int(w), int(h)
    return best_score, best_box


def match_gray(
    image_bgr: np.ndarray, tpl_bgr: np.ndarray, scales: List[float]
) -> Tuple[float, Tuple[int, int, int, int]]:
    img_gray = to_gray(image_bgr)
    H, W = img_gray.shape[:2]
    best_score = -1.0
    best_box = 0, 0, 0, 0
    tpl_gray_full = to_gray(tpl_bgr)
    th, tw = tpl_gray_full.shape[:2]
    for s in scales:
        tws, ths = max(5, int(tw * s)), max(5, int(th * s))
        tpl_gray = cv2.resize(tpl_gray_full, (tws, ths), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC)
        if H < ths or W < tws:
            continue
        res = cv2.matchTemplate(img_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = float(max_val)
            best_box = int(max_loc[0]), int(max_loc[1]), int(tws), int(ths)
    return best_score, best_box


def annotate(image_bgr: np.ndarray, result: Dict[str, any]) -> np.ndarray:
    vis = image_bgr.copy()
    if result["active"]["found"]:
        x, y, w, h = result["active"]["box"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 220, 0), 2)
        cv2.putText(
            vis, f"ACTIVE {result['active']['score']:.3f}", (x, max(20, y - 8)), FONT, 0.6, (0, 220, 0), 2, cv2.LINE_AA
        )
    if result["inactive"]["found"]:
        x, y, w, h = result["inactive"]["box"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 165, 255), 2)
        cv2.putText(
            vis,
            f"INACTIVE {result['inactive']['score']:.3f}",
            (x, y + h + 18),
            FONT,
            0.6,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(vis, f"pickup_state: {result['state']}", (10, 30), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def detect_pickup_on_image(image_path: str) -> Dict[str, any]:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    t_active1 = load_bgr(TPL_PICKUP_ACTIVE_1)
    t_active2 = load_bgr(TPL_PICKUP_ACTIVE_2)
    t_inactive = load_bgr(TPL_PICKUP_INACTIVE)
    best_act_score = -1.0
    best_act_box = 0, 0, 0, 0
    if t_active1 is not None:
        s, b = match_color(img, t_active1, SCALES_COLOR)
        if s > best_act_score:
            best_act_score, best_act_box = s, b
    if t_active2 is not None:
        s, b = match_color(img, t_active2, SCALES_COLOR)
        if s > best_act_score:
            best_act_score, best_act_box = s, b
    best_inact_score = -1.0
    best_inact_box = 0, 0, 0, 0
    if t_inactive is not None:
        best_inact_score, best_inact_box = match_gray(img, t_inactive, SCALES_GRAY)
    active_found = best_act_score >= ACTIVE_THRESH_COLOR
    inactive_found = best_inact_score >= INACTIVE_THRESH_GRAY
    if active_found and not inactive_found:
        state = "active"
    elif inactive_found and not active_found:
        state = "inactive"
    elif active_found and inactive_found:
        state = "active" if best_act_score >= best_inact_score else "inactive"
    else:
        state = "not visible"
    return {
        "file": os.path.basename(image_path),
        "active": {"found": bool(active_found), "score": round(float(best_act_score), 3), "box": best_act_box},
        "inactive": {"found": bool(inactive_found), "score": round(float(best_inact_score), 3), "box": best_inact_box},
        "state": state,
    }


def main():
    files = [
        f
        for f in os.listdir(SCREENS_DIR)
        if f.lower().endswith(".png")
        and (f.startswith("lobby_") or f.startswith("pickup_active_") or f.startswith("pickup_inactive_"))
    ]
    files.sort()
    results: List[Dict[str, any]] = []
    for i, fname in enumerate(files, 1):
        path = os.path.join(SCREENS_DIR, fname)
        try:
            res = detect_pickup_on_image(path)
            results.append(res)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            vis = annotate(img, res)
            cv2.imwrite(os.path.join(ANNOT_DIR, fname), vis)
            print(
                f"[{i}/{len(files)}] {fname} -> pickup_state={res['state']} (act={res['active']['score']:.3f}, inact={res['inactive']['score']:.3f})"
            )
        except Exception as e:
            print(f"[{i}/{len(files)}] {fname} -> ERROR: {e}")
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_DIR, "results.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(
            [
                "file",
                "pickup_state",
                "active_found",
                "active_score",
                "inactive_found",
                "inactive_score",
                "active_box",
                "inactive_box",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r["file"],
                    r["state"],
                    int(r["active"]["found"]),
                    f"{r['active']['score']:.3f}",
                    int(r["inactive"]["found"]),
                    f"{r['inactive']['score']:.3f}",
                    str(r["active"]["box"]),
                    str(r["inactive"]["box"]),
                ]
            )
    print(f"\nГотово. JSON/CSV и аннотации в {OUT_DIR}")


if __name__ == "__main__":
    main()
