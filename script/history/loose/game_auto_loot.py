import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class CfgItems:
    ASSETS = "C:\\bot\\tpl\\my"
    THRESH_NAME = 0.87
    THRESH_HDR = 0.88
    DEBUG_DIR = "C:\\bot\\screens\\_items_debug"
    ITEMS_HDR_KEYS = ["items_label", "items_header", "items_hdr"]
    MONSTERS_HDR_KEYS = ["monster_hdr"]
    ITEM_NAME_KEYS = [
        "yantar",
        "goreloe_derevo",
        "sgorevshaya_spichka",
        "zheleznaya_ruda",
        "bolshoy_luk",
        "latnye_perchatki",
        "despotichnaya_runa",
        "pautina",
    ]


TPL_CACHE: Dict[str, np.ndarray] = {}


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _asset_path(name: str) -> str:
    return os.path.join(CfgItems.ASSETS, f"{name}.png")


def load_tpl(name: str) -> Optional[np.ndarray]:
    if name in TPL_CACHE:
        return TPL_CACHE[name]
    p = _asset_path(name)
    tpl = cv2.imread(p, cv2.IMREAD_COLOR)
    if tpl is None or tpl.size == 0:
        print(f"[items] TPL not found: {p}")
        return None
    TPL_CACHE[name] = tpl
    return tpl


def match_one(img: np.ndarray, tpl: np.ndarray, thr: float) -> List[Dict[str, Any]]:
    res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(res >= thr)
    h, w = tpl.shape[:2]
    hits = []
    for y, x in zip(ys, xs):
        score = float(res[y, x])
        hits.append({"rect": (x, y, w, h), "center": (x + w // 2, y + h // 2), "score": score})
    return hits


def nms(rects: List[Dict[str, Any]], iou_thr: float = 0.4) -> List[Dict[str, Any]]:
    if not rects:
        return []
    boxes = np.array([r["rect"] for r in rects], dtype=np.float32)
    scores = np.array([r["score"] for r in rects], dtype=np.float32)
    x1 = boxes[:, (0)]
    y1 = boxes[:, (1)]
    w = boxes[:, (2)]
    h = boxes[:, (3)]
    x2 = x1 + w
    y2 = y1 + h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        ww = np.maximum(0.0, xx2 - xx1)
        hh = np.maximum(0.0, yy2 - yy1)
        inter = ww * hh
        union = w[i] * h[i] + w[order[1:]] * h[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return [rects[i] for i in keep]


def find_any(img: np.ndarray, names: List[str], thr: float) -> Optional[Tuple[str, Dict[str, Any]]]:
    best = None
    best_key = None
    for n in names:
        tpl = load_tpl(n)
        if tpl is None:
            continue
        hits = match_one(img, tpl, thr)
        if not hits:
            continue
        hit = max(hits, key=lambda d: d["score"])
        if best is None or hit["score"] > best["score"]:
            best, best_key = hit, n
    if not best:
        return None
    return best_key, best


def find_all(img: np.ndarray, names: List[str], thr: float) -> List[Tuple[str, Dict[str, Any]]]:
    out = []
    for n in names:
        tpl = load_tpl(n)
        if tpl is None:
            continue
        for hit in match_one(img, tpl, thr):
            out.append((n, hit))
    return out


def crop_roi_items(img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Строим ROI: от низа 'items_header' до верха 'monster_hdr' (если он есть).
    Возвращаем (roi, offset_xy)
    """
    items_hdr = find_any(img, CfgItems.ITEMS_HDR_KEYS, CfgItems.THRESH_HDR)
    y_top = 0
    if items_hdr:
        _, d = items_hdr
        x, y, w, h = d["rect"]
        y_top = y + h + 8
    monsters_hdr = find_any(img, CfgItems.MONSTERS_HDR_KEYS, CfgItems.THRESH_HDR)
    y_bottom = img.shape[0]
    if monsters_hdr:
        _, d2 = monsters_hdr
        y_bottom = d2["rect"][1] - 6
    y_top = max(0, y_top)
    y_bottom = min(img.shape[0], y_bottom)
    if y_bottom <= y_top + 20:
        y_top = img.shape[0] // 5
        y_bottom = img.shape[0] - img.shape[0] // 4
    roi = img[y_top:y_bottom, :, :].copy()
    return roi, (0, y_top)


def detect_item_entries(img: np.ndarray) -> List[Dict[str, Any]]:
    """
    Возвращает список элементов: [{'name': str, 'center': (x,y), 'rect': (x,y,w,h), 'score': float}]
    Сортировано по y (сверху вниз).
    """
    _ensure_dir(CfgItems.DEBUG_DIR)
    roi, (ox, oy) = crop_roi_items(img)
    debug_vis = img.copy()
    cv2.rectangle(debug_vis, (0, oy), (img.shape[1] - 1, oy + roi.shape[0] - 1), (0, 255, 255), 2)
    raw_hits: List[Tuple[str, Dict[str, Any]]] = find_all(roi, CfgItems.ITEM_NAME_KEYS, CfgItems.THRESH_NAME)
    dets = []
    for name, d in raw_hits:
        x, y, w, h = d["rect"]
        grect = x + ox, y + oy, w, h
        gcenter = d["center"][0] + ox, d["center"][1] + oy
        dets.append({"name": name, "rect": grect, "center": gcenter, "score": d["score"]})
    dets_sorted = sorted(dets, key=lambda r: r["score"], reverse=True)
    dets_nms = nms(dets_sorted, iou_thr=0.35)
    dets_final = sorted(dets_nms, key=lambda r: r["center"][1])
    for d in dets_final:
        x, y, w, h = d["rect"]
        cv2.rectangle(debug_vis, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(
            debug_vis,
            f"{d['name']} {d['score']:.2f}",
            (x, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(os.path.join(CfgItems.DEBUG_DIR, "items_detect_debug.png"), debug_vis)
    print(
        f"[items] detected {len(dets_final)} items: " + ", ".join([f"{d['name']}@{d['score']:.2f}" for d in dets_final])
    )
    return dets_final


def quick_test_on_png(png_path: str):
    img = cv2.imread(png_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"bad image: {png_path}")
        return
    _ = detect_item_entries(img)
    print("[items] done")


if __name__ == "__main__":
    test_path = "C:\\bot\\screens\\_loot_debug\\sample_items_screen.png"
    quick_test_on_png(test_path)
