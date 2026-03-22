# C:/bot/script/loot/victory_drop.py
# -*- coding: utf-8 -*-
"""Сканирование экрана победы: детект иконок дропа и фильтрация по whitelist."""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from script.loot import auto_bot
from script.loot import tpl_loader as TL
from script.loot.matcher import find_all_matches, preprocess_gray
from script.loot.utils import clamp, match_scaled, rect_from_rel, get_thr

_IMG_CACHE: Dict[str, np.ndarray] = {}
roi_icons = (0, 1000, 1080, 1500)  # fallback x1, y1, x2, y2

Match = Tuple[int, int, int, int, float, str]  # (x,y,w,h,score,tpl_path)


def _structured_log(step: str, **payload):
    try:
        auto_bot.structured_log(step, **payload)
    except Exception:
        pass


def _to_gray(img: np.ndarray, mode: str = "gray") -> np.ndarray:
    return preprocess_gray(img, mode=mode)


def _ensure_rel(rel: Optional[List[float]], fallback: List[float]) -> List[float]:
    if isinstance(rel, list) and len(rel) == 4:
        return rel
    return fallback


def _cfg(cfg: Dict, *path, default=None):
    cur = cfg
    try:
        for p in path:
            cur = cur[p]
        return cur
    except Exception:
        return default


def _build_registry_index(reg: Dict[str, dict]) -> Dict[str, Tuple[str, str]]:
    """Индекс: tpl_path -> (group, name)."""
    out: Dict[str, Tuple[str, str]] = {}
    for k, v in reg.items():
        if not isinstance(v, dict):
            continue
        p, g = v.get("tpl"), v.get("group")
        if not p or not g:
            continue
        try:
            stem_key = TL._nfkc_lower(k)
        except Exception:
            stem_key = k
        out[p.replace("\\", "/")] = (g, stem_key.split(".", 2)[-1])
    return out


def _collect_icon_templates(reg: Dict[str, dict]) -> List[Tuple[str, np.ndarray]]:
    """Иконки дропа: item.icon.* и rune.icon.*."""
    icons: List[Tuple[str, np.ndarray]] = []
    for k, v in reg.items():
        if not isinstance(v, dict):
            continue
        g = v.get("group") or ""
        if not (g.startswith("item.icon") or g.startswith("rune.icon")):
            continue
        p = v.get("tpl")
        if not p:
            continue
        key = p.replace("\\", "/")
        img = _IMG_CACHE.get(key)
        if img is None:
            try:
                from script.loot.utils import imread_u8
                img = imread_u8(p, cv2.IMREAD_COLOR)
            except Exception:
                img = None
            if img is None:
                img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            _IMG_CACHE[key] = img
        icons.append((key, img))
    _structured_log("victory_icons_loaded", count=len(icons))
    return icons


def _detect_continue_box(frame_bgr: np.ndarray, cfg: Dict) -> Optional[Tuple[int, int, int, int]]:
    """Ищем кнопку CONTINUE для построения ROI дропа."""
    path = _cfg(cfg, "TEMPLATES_PATHS", "CONTINUE") or _cfg(cfg, "TEMPLATES", "CONTINUE")
    if not path or not os.path.exists(path):
        fx, fy, fw, fh = _cfg(cfg, "FALLBACK_ROI", default=(0, 0, frame_bgr.shape[1], frame_bgr.shape[0]))
        return fx, fy, fw, fh

    key = path.replace("\\", "/")
    tpl = _IMG_CACHE.get(key)
    if tpl is None:
        from script.loot.utils import imread_u8
        tpl = imread_u8(path, cv2.IMREAD_COLOR)
        if tpl is None:
            return None
        _IMG_CACHE[key] = tpl

    thr = get_thr("FIGHT_CONTINUE", 0.82)
    scales = list(_cfg(cfg, "FIND", "ITEM_SCALES", default=[0.9, 0.95, 1.0, 1.05]))
    mode = _cfg(cfg, "FIND", "PREPROC_MODE", default="gray") or "gray"
    roi_gray = _to_gray(frame_bgr, mode=mode)
    tpl_gray = _to_gray(tpl, mode=mode)

    H, W = roi_gray.shape[:2]
    best = (None, -1.0, None)
    for s in scales:
        tw = max(5, int(round(tpl_gray.shape[1] * s)))
        th = max(5, int(round(tpl_gray.shape[0] * s)))
        if th > H or tw > W:
            continue
        tgs = cv2.resize(tpl_gray, (tw, th), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC)
        res = cv2.matchTemplate(roi_gray, tgs, cv2.TM_CCOEFF_NORMED)
        _, maxV, _, maxL = cv2.minMaxLoc(res)
        if maxV >= thr and maxV > best[1]:
            best = ((int(maxL[0]), int(maxL[1]), tw, th), float(maxV), s)

    if best[0] is None:
        _structured_log("victory_continue_not_found")
        fx, fy, fw, fh = _cfg(cfg, "FALLBACK_ROI", default=(0, 0, frame_bgr.shape[1], frame_bgr.shape[0]))
        return fx, fy, fw, fh

    x, y, w, h = best[0]
    _structured_log("victory_continue_found", box=[x, y, w, h], score=best[1])
    return x, y, w, h


def _compute_drop_roi(frame_bgr, continue_box, cfg) -> Tuple[int, int, int, int]:
    """ROI иконок дропа на экране победы."""
    H, W = frame_bgr.shape[:2]

    # 1) Абсолютный ROI из конфига
    abs_roi = _cfg(cfg, "ROI", "VICTORY_DROP_ABS")
    if isinstance(abs_roi, list) and len(abs_roi) == 4:
        x1, y1, x2, y2 = [clamp(int(v), 0, max(W, H)) for v in abs_roi]
        if x2 > x1 and y2 > y1:
            _structured_log("victory_drop_roi_abs_cfg", roi=[x1, y1, x2 - x1, y2 - y1])
            return x1, y1, x2 - x1, y2 - y1

    # 2) Полоса над CONTINUE
    if continue_box:
        cx, cy, cw, ch = continue_box
        y2 = clamp(cy - 8, 0, H)
        y1 = clamp(y2 - int(ch * 1.6), 0, y2)
        x1, x2 = max(0, int(W * 0.08)), min(W, int(W * 0.92))
        if y2 - y1 >= 20:
            _structured_log("victory_drop_roi_from_continue", roi=[x1, y1, x2 - x1, y2 - y1])
            return x1, y1, x2 - x1, y2 - y1

    # 3) Жёсткий fallback
    hx1, hy1 = clamp(0, 0, W), clamp(1000, 0, H)
    hx2, hy2 = clamp(1080, 0, W), clamp(1500, 0, H)
    if hx2 > hx1 and hy2 > hy1:
        return hx1, hy1, hx2 - hx1, hy2 - hy1

    # 4) Относительный fallback
    rel = _ensure_rel(_cfg(cfg, "ROI", "VICTORY_DROP_REL"), [0.08, 0.52, 0.92, 0.86])
    x1, y1, x2, y2 = rect_from_rel(W, H, rel)
    return x1, y1, x2 - x1, y2 - y1


def _match_icons_in_roi(frame_bgr, roi_xywh, icons, cfg) -> List[Match]:
    """Находит иконки дропа в ROI."""
    x, y, w, h = roi_xywh
    crop = frame_bgr[y:y + h, x:x + w]
    if crop.size == 0:
        return []
    mode = _cfg(cfg, "FIND", "PREPROC_MODE", default="gray") or "gray"
    roi_proc = _to_gray(crop, mode=mode)
    scales = list(_cfg(cfg, "FIND", "ITEM_SCALES", default=[0.9, 0.95, 1.0, 1.05]))
    thr = get_thr("FIND_DEFAULT", 0.85)
    min_dy = int(_cfg(cfg, "FIND", "MIN_DY", default=10))

    matches: List[Match] = []
    for tpl_path, img in icons:
        tpl_proc = _to_gray(img, mode=mode)
        hits = find_all_matches(roi_proc, tpl_proc, scales=scales, threshold=thr, min_dy=min_dy)
        for mx, my, s, score in hits:
            tw = max(5, int(round(tpl_proc.shape[1] * s)))
            th = max(5, int(round(tpl_proc.shape[0] * s)))
            matches.append((x + mx, y + my, tw, th, float(score), tpl_path))
    matches.sort(key=lambda r: (r[1], -r[4]))
    _structured_log("victory_drop_matches", count=len(matches))
    return matches


def _icon_path_to_item_name_filename(icon_tpl_path: str, reg_index) -> Optional[str]:
    """Иконка -> имя файла name-шаблона (X_name.png)."""
    icon_tpl_path = icon_tpl_path.replace("\\", "/")
    meta = reg_index.get(icon_tpl_path)
    if not meta:
        return None

    group, stem = meta
    if group.startswith("item.icon"):
        name_key = f"item.name.{stem}"
    elif group.startswith("rune.icon"):
        name_key = f"rune.name.{stem}"
    else:
        return None

    try:
        norm_key = TL._nfkc_lower(name_key)
    except Exception:
        norm_key = name_key

    val = TL.REG.get(norm_key)
    if not isinstance(val, dict):
        for k in TL.REG:
            if k.lower() == norm_key.lower():
                val = TL.REG[k]
                break

    if isinstance(val, dict):
        name_path = val.get("tpl")
        if name_path:
            return os.path.basename(name_path)
    return None


def _uniq_keep_order(seq: List[str]) -> List[str]:
    out, seen = [], set()
    for s in seq:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def scan_victory_drop_targets(frame_bgr: np.ndarray, cfg: Dict) -> Dict:
    """
    Сканирует экран победы, ищет иконки дропа, фильтрует по whitelist.
    Возвращает {loot_found, allowed_loot_ids, detected_all}.
    """
    empty = {"loot_found": False, "allowed_loot_ids": [], "detected_all": []}

    # Шаг 0 — проверка CONTINUE
    try:
        cont_path = auto_bot.ui_tpl_path("continue")
        from script.loot.utils import imread_u8
        cont_img = imread_u8(cont_path, cv2.IMREAD_COLOR)
        if cont_img is None:
            return empty
        mode = _cfg(cfg, "FIND", "PREPROC_MODE", default="gray") or "gray"
        fr_p = preprocess_gray(frame_bgr, mode=mode)
        tp_p = preprocess_gray(cont_img, mode=mode)
        scales = list(_cfg(cfg, "FIND", "ITEM_SCALES", default=[0.9, 0.95, 1.0, 1.05]))
        thr = get_thr("FIGHT_CONTINUE", 0.82)
        min_dy = int(_cfg(cfg, "FIND", "MIN_DY", default=10))
        if not find_all_matches(fr_p, tp_p, scales=scales, threshold=thr, min_dy=min_dy):
            return empty
    except Exception:
        return empty

    # Шаг 1 — registry + иконки
    reg = getattr(TL, "REG", {}) or {}
    if not reg:
        return empty
    reg_index = _build_registry_index(reg)
    icons = _collect_icon_templates(reg)

    # Шаг 2 — ROI
    cont_box = _detect_continue_box(frame_bgr, cfg)
    roi = _compute_drop_roi(frame_bgr, cont_box, cfg)

    # Шаг 3 — матчинг иконок
    matches = _match_icons_in_roi(frame_bgr, roi, icons, cfg)
    if not matches:
        return empty

    # Шаг 4 — иконки -> name-шаблоны
    names: List[str] = []
    for _, _, _, _, _, tpl_path in matches:
        fn = _icon_path_to_item_name_filename(tpl_path, reg_index)
        if fn:
            names.append(fn)

    names_unique = _uniq_keep_order(names)

    # Шаг 5 — фильтрация по whitelist
    wl = set(auto_bot.CFG.get("ALLOWED_ITEM_NAMES", []))
    icon_to_name: Dict[str, Optional[str]] = {}
    for _, _, _, _, _, tpl_path in matches:
        icon_to_name[os.path.basename(tpl_path)] = _icon_path_to_item_name_filename(tpl_path, reg_index)

    allowed = [
        n for n in names_unique
        if n in wl or any(ib in wl for ib, nb in icon_to_name.items() if nb == n)
    ]

    _structured_log("victory_drop_targets", detected=names_unique, allowed=allowed, whitelist_size=len(wl))
    return {"loot_found": bool(allowed), "allowed_loot_ids": allowed, "detected_all": names_unique}


def set_victory_targets_from_frame(frame_bgr: np.ndarray) -> Dict:
    cfg = getattr(auto_bot, "CFG", {})
    drop_info = scan_victory_drop_targets(frame_bgr, cfg)
    auto_bot.set_victory_targets(drop_info["allowed_loot_ids"])
    return drop_info
