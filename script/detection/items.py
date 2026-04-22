# -*- coding: utf-8 -*-
import os
import glob
import time
from typing import List, Tuple, Dict, Any, Optional, Set

import numpy as np
import cv2

from script.core.config import (
    CFG, ITEMS_ROI, ALLOWED_ITEM_NAMES,
    EXCLUDE_ZONES, SKIP_SUBSTR, DETECT_ONLY_WHITELIST,
    SAFE_X1, SAFE_Y1, SAFE_X2, SAFE_Y2
)
from script.core.logging import structured_log, snap, log, PERF
from script.loot.utils import clamp, get_thr, try_imread, to_gray, scharr_mag
from script.loot.matcher import find_all_matches, TPL_CACHE, preprocess_gray
from script.loot.tab_detector import detect_tab_states
from script.loot.tpl_loader import load_item_name_templates_unified, get_item_score
from script.detection import ocr as _ocr

IGNORE_SKIP_SUBSTR = CFG.get("FIND", {}).get("IGNORE_SKIP_SUBSTR", False)
IGNORE_WHITELIST = CFG.get("FIND", {}).get("IGNORE_WHITELIST", False)
FORCE_FS_SCAN = CFG.get("FIND", {}).get("FORCE_FS_SCAN", False)
ALLOWED_LOOT_DIRS = CFG.get("ALLOWED_LOOT_DIRS", [])
PREPROC_MODE = CFG.get("FIND", {}).get("PREPROC_MODE", "scharr_v1")

FIND_THRESHOLD = float(CFG.get("FIND", {}).get("THRESHOLD", CFG.get("item_name_threshold", 0.86)))
FIND_SCALES = list(CFG.get("FIND", {}).get("ITEM_SCALES", [0.95, 1.0, 1.05]))
SUPPRESS_Y = int(CFG.get("FIND", {}).get("SUPPRESS_Y_OVERLAP", 12))

def _is_service_file(base: str) -> bool:
    low = base.lower()
    if low.endswith("_mask.png"):
        return True
    if not IGNORE_SKIP_SUBSTR:
        for sub in SKIP_SUBSTR:
            if sub and sub in low:
                return True
    return False

_ITEM_TPL_CACHE_DATA: List[Tuple[str, np.ndarray, str]] = []

def get_item_name_templates(detect_only_names: Optional[Set[str]] = None) -> List[Tuple[str, np.ndarray, str]]:
    global _ITEM_TPL_CACHE_DATA
    accepted, skipped = [], []

    if not _ITEM_TPL_CACHE_DATA:
        try:
            if not FORCE_FS_SCAN:
                tpls = load_item_name_templates_unified(
                    should_skip=lambda base: _is_service_file(base),
                    detect_whitelist=None,
                    allowed_dirs=ALLOWED_LOOT_DIRS,
                )
            else:
                tpls = []
        except Exception as e:
            structured_log("tpls_loader_error", error=str(e))
            tpls = []

        if not tpls:
            files = []
            for d in ALLOWED_LOOT_DIRS:
                for ext in ("*.png", "*.PNG"):
                    files.extend(glob.glob(os.path.join(d, ext)))
            files = sorted(set(p.replace("\\", "/") for p in files))
            for p in files:
                base = os.path.basename(p)
                if _is_service_file(base):
                    skipped.append({"file": base, "reason": "service_or_mask"})
                    continue
                img = try_imread(p)
                if img is None:
                    skipped.append({"file": base, "reason": "load_fail"})
                    continue
                accepted.append((base, img, p))
        else:
            for base, img in tpls:
                if _is_service_file(base):
                    skipped.append({"file": base, "reason": "service_or_mask"})
                    continue
                accepted.append((base, img, None))

        _ITEM_TPL_CACHE_DATA = accepted
        structured_log("items_tpls_cache_loaded", total=len(accepted))

    out = []
    base_detect_wl = None if IGNORE_WHITELIST else (ALLOWED_ITEM_NAMES if DETECT_ONLY_WHITELIST else None)
    effective_detect = detect_only_names if detect_only_names else base_detect_wl

    for base, img, path in _ITEM_TPL_CACHE_DATA:
        if effective_detect and base not in effective_detect:
            continue
        out.append((base, img, path))
    return out

def compute_items_visible_roi(frame_bgr, tabs) -> Tuple[int, int, int, int]:
    H, W = frame_bgr.shape[:2]
    x1, x2 = 0, 500
    top_y = max(SAFE_Y1, 400)
    bottom_y = min(SAFE_Y2, 2100)
    return x1, top_y, x2, bottom_y

def match_multi_scaled(
    gray_img: np.ndarray,
    tpl_gray_full: np.ndarray,
    threshold: float,
    scales: List[float],
    suppress_y: int,
    max_hits_per_scale: int = 7,
    suppress_x: Optional[int] = None,
) -> List[Tuple[int, int, int, int, float]]:
    H, W = gray_img.shape[:2]
    out: List[Tuple[int, int, int, int, float]] = []
    for s in scales:
        th, tw = tpl_gray_full.shape[:2]
        tws, ths = max(5, int(tw * s)), max(5, int(th * s))
        if H < ths or W < tws:
            continue
        tpl_gray = cv2.resize(tpl_gray_full, (tws, ths), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC)
        res = cv2.matchTemplate(gray_img, tpl_gray, cv2.TM_CCOEFF_NORMED)
        RH, RW = res.shape[:2]
        hits = 0
        sy = max(8, int(suppress_y))
        sx = max(0, int(suppress_x if suppress_x is not None else tws // 4))
        while hits < max_hits_per_scale:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if float(max_val) < float(threshold):
                break
            x, y = int(max_loc[0]), int(max_loc[1])
            out.append((x, y, tws, ths, float(max_val)))
            y1 = max(0, y - sy)
            y2 = min(RH - 1, y + sy)
            x1 = max(0, x - sx)
            x2 = min(RW - 1, x + sx)
            res[y1 : y2 + 1, :] = 0
            if sx > 0:
                res[:, x1 : x2 + 1] = 0
            hits += 1
    out.sort(key=lambda r: (r[1], -r[4]))
    merged: List[Tuple[int, int, int, int, float]] = []
    for bx, by, bw, bh, sc in out:
        if not merged:
            merged.append((bx, by, bw, bh, sc))
            continue
        can_add = True
        for mx, my, mw, mh, msc in merged:
            dy = abs(by - my)
            if dy < suppress_y:
                can_add = False
                break
        if can_add:
            merged.append((bx, by, bw, bh, sc))
    return merged

def _collect_hits(matches, img_bgr, base, frame_bgr, x1, y1, boxes_scores, debug_img):
    from script.detection.pickup import frame_hash
    h0, w0 = img_bgr.shape[:2]
    for mx, my, scale, score in matches:
        w = int(round(w0 * scale))
        h = int(round(h0 * scale))
        ay1, ay2 = y1 + my, y1 + my + h
        ax1, ax2 = x1 + mx, x1 + mx + w
        row_crop = frame_bgr[ay1:ay2, ax1:ax2]
        if row_crop.size == 0:
            continue
        row_hash = frame_hash(to_gray(row_crop))
        cy_abs = int((ay1 + ay2) // 2)
        slot_hash = int((row_hash << 8 ^ cy_abs) & 0xFFFFFFFF)
        snap(f"raw_slot_{slot_hash}", row_crop, rects=[(0, 0, w, h)])
        structured_log("item_raw_hit", name=base, score=float(score), y=cy_abs, slot_hash=int(slot_hash))
        boxes_scores.append(((mx, my, w, h), float(score), base, slot_hash))
        cv2.rectangle(debug_img, (ax1, ay1), (ax2, ay2), (0, 255, 0), 2)
        cv2.circle(debug_img, (ax1 + w // 2, ay1 + h // 2), 3, (0, 0, 255), -1)

def merge_same_lines(boxes_scores, line_thr=35):
    if not boxes_scores:
        return []
    keep, seen = [], set()
    for idx, (bbox, score, name, slot_hash) in enumerate(boxes_scores):
        _, y, _, h = bbox
        y_group = int(round(y / float(line_thr)))
        key = y_group, slot_hash
        if key in seen:
            continue
        seen.add(key)
        keep.append(idx)
    return keep

def _vec_direction(vec: List[int]) -> str:
    dy = vec[1] if len(vec) > 1 else 0
    return "up" if dy < 0 else "down" if dy > 0 else "none"

def _split_by_dir(vectors: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
    ups = [v for v in vectors if len(v) > 1 and v[1] < 0]
    downs = [v for v in vectors if len(v) > 1 and v[1] > 0]
    return ups, downs

def _primary_direction_from_base_y(base_y: int, H: int) -> str:
    if base_y > H * 0.66:
        return "up"
    if base_y < H * 0.33:
        return "down"
    return "up"

@PERF.measure("item_search")
def _raw_find_item_names(frame_bgr: np.ndarray, allowed_names: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    if frame_bgr is None:
        return []
    H, W = frame_bgr.shape[:2]
    try:
        tabs = detect_tab_states(frame_bgr)
        rx1, ry1, rx2, ry2 = compute_items_visible_roi(frame_bgr, tabs)
    except Exception:
        rx1, ry1, rx2, ry2 = ITEMS_ROI
    x1 = clamp(int(rx1), 0, W - 1)
    y1 = clamp(int(ry1), 0, H - 1)
    x2 = clamp(int(rx2), 1, W)
    y2 = clamp(int(ry2), 1, H)

    structured_log(
        "items_roi",
        roi=[x1, y1, x2, y2],
        safe=[int(SAFE_X1), int(SAFE_Y1), int(SAFE_X2), int(SAFE_Y2)],
        dynamic_filter_active=bool(allowed_names),
        dynamic_filter_size=(len(allowed_names) if allowed_names else 0),
    )

    if x2 <= x1 or y2 <= y1:
        log(f"[ITEMS] Пустой ROI для списка: {rx1, ry1, rx2, ry2}")
        return []

    crop = frame_bgr[y1:y2, x1:x2]
    if EXCLUDE_ZONES:
        masked = crop.copy()
        Hc, Wc = crop.shape[:2]
        for zx1, zy1, zx2, zy2 in EXCLUDE_ZONES:
            lx1 = clamp(zx1 - x1, 0, Wc)
            ly1 = clamp(zy1 - y1, 0, Hc)
            lx2 = clamp(zx2 - x1, 0, Wc)
            ly2 = clamp(zy2 - y1, 0, Hc)
            if lx2 > lx1 and ly2 > ly1:
                masked[ly1:ly2, lx1:lx2] = 0
        crop = masked

    proc = preprocess_gray(crop, mode=PREPROC_MODE)
    tpls_triplets = get_item_name_templates(detect_only_names=allowed_names)
    if not tpls_triplets:
        return []

    boxes_scores: List[Tuple[Tuple[int, int, int, int], float, str, int]] = []
    grp_thr = int(SUPPRESS_Y)
    max_lines = int(CFG.get("max_lines_to_collect", 12))
    debug_img = frame_bgr.copy()

    PREFILTER_ENABLED = CFG.get("PREFILTER_ENABLED", True)
    PREFILTER_SCALE = get_thr("PREFILTER_SCALE", 0.25)
    PREFILTER_MARGIN = get_thr("PREFILTER_MARGIN", 0.1)

    if PREFILTER_ENABLED:
        proc_small = cv2.resize(proc, (0, 0), fx=PREFILTER_SCALE, fy=PREFILTER_SCALE, interpolation=cv2.INTER_AREA)

    for base, img_bgr, path in tpls_triplets:
        if img_bgr is None:
            continue
        if allowed_names and base not in allowed_names:
            continue
        if DETECT_ONLY_WHITELIST and not IGNORE_WHITELIST and base not in ALLOWED_ITEM_NAMES:
            continue
            
        if PREFILTER_ENABLED:
            tpl_small = TPL_CACHE.get_all_scaled(path or base, img_bgr, PREPROC_MODE, [PREFILTER_SCALE]).get(PREFILTER_SCALE)
            if tpl_small is not None:
                ths, tws = tpl_small.shape[:2]
                pH, pW = proc_small.shape[:2]
                if ths > 0 and tws > 0 and ths <= pH and tws <= pW:
                    res_small = cv2.matchTemplate(proc_small, tpl_small, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res_small)
                    if max_val < float(FIND_THRESHOLD) - PREFILTER_MARGIN:
                        continue

        tpl_scaled_dict = TPL_CACHE.get_all_scaled(path or base, img_bgr, PREPROC_MODE, FIND_SCALES)
        matches = find_all_matches(proc, tpl_scaled_dict, scales=FIND_SCALES, threshold=FIND_THRESHOLD, min_dy=SUPPRESS_Y)
        if matches:
            structured_log("items_multi_hits", template=base, hits=len(matches))
        _collect_hits(matches, img_bgr, base, frame_bgr, x1, y1, boxes_scores, debug_img)

    if not boxes_scores and CFG.get("SCHARR_FALLBACK_ENABLED", True):
        mean_brightness = np.mean(proc)
        if mean_brightness >= 10:
            gg = scharr_mag(proc)
            edge_density = np.count_nonzero(gg > 30) / proc.size
            if edge_density >= 0.01:
                for base, img_bgr, path in tpls_triplets:
                    if img_bgr is None:
                        continue
                    if allowed_names and base not in allowed_names:
                        continue
                    if DETECT_ONLY_WHITELIST and not IGNORE_WHITELIST and base not in ALLOWED_ITEM_NAMES:
                        continue
                    tg_dict = TPL_CACHE.get_all_scharr_scaled(path or base, img_bgr, PREPROC_MODE, FIND_SCALES, scharr_mag)
                    thr2 = max(get_thr("SCHARR_FALLBACK_MIN", 0.83), float(FIND_THRESHOLD) - get_thr("SCHARR_FALLBACK_OFFSET", 0.03))
                    matches = find_all_matches(gg, tg_dict, scales=FIND_SCALES, threshold=thr2, min_dy=SUPPRESS_Y)
                    if matches:
                        structured_log("items_multi_hits_scharr", template=base, hits=len(matches))
                        _collect_hits(matches, img_bgr, base, frame_bgr, x1, y1, boxes_scores, debug_img)

    if CFG.get("use_slot_grouping", True):
        slot_map: Dict[int, Tuple[Tuple[int, int, int, int], float, str, int]] = {}
        for bbox, score, name, slot_hash in boxes_scores:
            _, y_rel, _, h_rel = bbox
            slot_id = int((slot_hash << 8 ^ (y_rel + h_rel // 2)) & 0xFFFFFFFF)
            if slot_id not in slot_map or score > slot_map[slot_id][1]:
                slot_map[slot_id] = (bbox, score, name, slot_hash)
        boxes_unique = list(slot_map.values())
        keep = list(range(len(boxes_unique)))
    else:
        boxes_unique = boxes_scores
        keep = merge_same_lines(boxes_unique, line_thr=grp_thr)

    if len(keep) > max_lines:
        keep = sorted(keep, key=lambda i: boxes_unique[i][0][1])[:max_lines]

    DEFAULT_W = CFG.get("item_score_width", 24)
    DEFAULT_H = CFG.get("item_score_height", 12)
    DEFAULT_SCALE = CFG.get("item_score_scale", 1.0)
    DEFAULT_NAME = ""

    items_scores: List[Tuple[str, float]] = []
    for _, _, name, _ in [boxes_unique[i] for i in keep]:
        try:
            score_val = get_item_score(name, DEFAULT_W, DEFAULT_H, DEFAULT_SCALE, DEFAULT_NAME)
        except Exception as e:
            structured_log("item_score_fail", name=name, error=str(e))
            score_val = 0.0
        items_scores.append((name, float(score_val)))

    items_scores.sort(key=lambda x: x[1], reverse=True)
    top_items = [tpl for tpl, _ in items_scores[:max_lines]]

    found: List[Dict[str, Any]] = []
    for i in keep:
        (x, y, w, h), score, name, slot_hash = boxes_unique[i]
        ax, ay = x1 + x, y1 + y
        found.append({
            "name": name,
            "score": float(score),
            "box": (ax, ay, w, h),
            "center": (ax + w // 2, ay + h // 2),
            "slot_hash": slot_hash,
        })

    # ─── Опциональный OCR-проход (CFG.OCR.ENABLED) ────────────────
    # Дополняет template-матчинг: OCR-хиты, которые не попадают по Y
    # в уже найденные боксы, добавляются как дополнительные кандидаты.
    # Выключено по умолчанию; не ломает существующее поведение.
    ocr_cfg = CFG.get("OCR") or {}
    if ocr_cfg.get("ENABLED"):
        try:
            ocr_hits = _ocr.detect_item_names_ocr(
                frame_bgr,
                roi=(x1, y1, x2, y2),
                whitelist=(allowed_names or ALLOWED_ITEM_NAMES),
                engine=str(ocr_cfg.get("ENGINE", "auto")),
                fuzzy_min=float(ocr_cfg.get("FUZZY_MIN", 0.75)),
                min_conf=float(ocr_cfg.get("MIN_CONF", 0.5)),
            )
        except Exception as e:
            structured_log("ocr_pass_error", error=str(e))
            ocr_hits = []
        if ocr_hits:
            sup_y = max(1, int(SUPPRESS_Y))
            existing_y = [h["center"][1] for h in found]
            added = 0
            for h in ocr_hits:
                cy = h["center"][1]
                if any(abs(cy - ey) < sup_y for ey in existing_y):
                    continue
                found.append(h)
                existing_y.append(cy)
                added += 1
            if added or ocr_cfg.get("DEBUG"):
                structured_log(
                    "ocr_pass",
                    engine=_ocr.available_engine(str(ocr_cfg.get("ENGINE", "auto"))),
                    raw_hits=len(ocr_hits),
                    added=added,
                    template_hits=len(found) - added,
                )

    return found

def find_item_names(get_frame_fn, allowed_names: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    st_frames = int(CFG.get("FIND", {}).get("STABILIZE_FRAMES", 1) or 1)
    best_res, best_count = [], 0
    for i in range(st_frames):
        fr = get_frame_fn()
        if fr is None:
            continue
        res = _raw_find_item_names(fr, allowed_names)
        if len(res) > best_count:
            best_res, best_count = res, len(res)
        if st_frames > 1 and i < st_frames - 1:
            time.sleep(0.15)
    return best_res
