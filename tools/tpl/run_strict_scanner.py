# -*- coding: utf-8 -*-
"""
run_strict_scanner.py

Утилита-раннер для быстрого прогонки строгого шаблонного поиска
из `find_icon.strict_template_search_on_roi`.

Как пользоваться (пример):
    from run_strict_scanner import load_icon_templates, run_on_image
    tpl = load_icon_templates(r"C:/bot/tpl", "иконки_предметов", max_templates=10)
    run_on_image("C:/path/to/image.png", tpl, out_dir="C:/bot/debug/out_icons_debug")

Это автономный помощник: не меняет основной код, только вызывает его.
"""
import os
import json
from typing import List, Tuple

import cv2
import numpy as np
import importlib.util
from PIL import Image, ImageDraw, ImageFont

# ensure project root is on sys.path so package imports work when this file is
# executed as a script (sys.path[0] would otherwise be this file's dir)
import sys
HERE = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.tpl import find_icon
from tools.tpl import debug_crop_logger
from pathlib import Path
import datetime
# optional overlay verifier (import if available)
try:
    from tools.tpl.overlay_matcher import overlay_local_refine
except Exception:
    overlay_local_refine = None

# load expected coordinates from config file if available (persistent storage)
EXPECTED_COORDS = {}
try:
    cfg_exp = os.path.join(os.path.dirname(__file__), '..', 'cfg', 'expected_coords.json')
    if os.path.exists(cfg_exp):
        with open(cfg_exp, 'r', encoding='utf-8') as _f:
            EX_EXPECTED = json.load(_f)
            # normalize tuples
            for k, v in EX_EXPECTED.items():
                EX_EXPECTED[k] = {tk: (int(coords[0]), int(coords[1])) for tk, coords in v.items()}
            EXPECTED_COORDS = EX_EXPECTED
except Exception:
    EXPECTED_COORDS = {}


def _local_refine_search(image_bgr, tpl_bgr, tpl_alpha, center_x, center_y, search_radius=140, scales=(0.9, 1.0, 1.1), image_hsv=None):
    """Search locally around (center_x, center_y) for the best masked match of tpl inside image.

    Returns (x,y,w,h,masked_agree,masked_corr,combined_score) or None
    """
    H, W = image_bgr.shape[:2]
    # build search window
    x0 = max(0, int(center_x - search_radius))
    y0 = max(0, int(center_y - search_radius))
    x1 = min(W, int(center_x + search_radius))
    y1 = min(H, int(center_y + search_radius))
    window = image_bgr[y0:y1, x0:x1]
    if window.size == 0:
        return None
    best = None
    # prepare template mask and gray
    if tpl_alpha is not None:
        tpl_mask = (tpl_alpha > 0).astype('uint8') * 255
    else:
        tpl_mask = None
    for s in scales:
        try:
            th = max(2, int(tpl_bgr.shape[1] * s)), max(2, int(tpl_bgr.shape[0] * s))
            tpl_rs = cv2.resize(tpl_bgr, th, interpolation=cv2.INTER_AREA)
            if tpl_mask is not None:
                mask_rs = cv2.resize(tpl_mask, th, interpolation=cv2.INTER_NEAREST)
            else:
                mask_rs = None
        except Exception:
            tpl_rs = tpl_bgr
            mask_rs = tpl_mask

        # matchTemplate on V channel with mask if available
            try:
                tpl_hsv = cv2.cvtColor(tpl_rs, cv2.COLOR_BGR2HSV)
                tpl_v = tpl_hsv[:, :, 2]
                # prepare window V channel, prefer provided HSV slice if available
                if image_hsv is not None:
                    # compute slice coords relative to full image
                    win_h = window.shape[0]
                    win_w = window.shape[1]
                    # derive absolute coords around center_x/center_y and clamp
                    x0 = max(0, int(center_x - search_radius))
                    y0 = max(0, int(center_y - search_radius))
                    win_hsv = image_hsv[y0:y0 + win_h, x0:x0 + win_w]
                    win_v = win_hsv[:, :, 2]
                else:
                    win_hsv = cv2.cvtColor(window, cv2.COLOR_BGR2HSV)
                    win_v = win_hsv[:, :, 2]
                # choose safe method when mask is present
                if mask_rs is not None:
                    method = cv2.TM_CCORR_NORMED
                    res = cv2.matchTemplate(win_v, tpl_v, method, mask=mask_rs)
                else:
                    method = cv2.TM_CCOEFF_NORMED
                    res = cv2.matchTemplate(win_v, tpl_v, method)
            except Exception:
                continue
        minv, maxv, minloc, maxloc = cv2.minMaxLoc(res)
        # compute candidate bbox in full image coords
        px, py = maxloc
        w, h = tpl_rs.shape[1], tpl_rs.shape[0]
        img_x = x0 + px
        img_y = y0 + py

        # compute masked agreement and masked_corr for this crop
        crop = image_bgr[img_y:img_y + h, img_x:img_x + w]
        if crop.size == 0:
            continue
        try:
            tpl_hsv = cv2.cvtColor(tpl_rs, cv2.COLOR_BGR2HSV)
            crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            v_tpl = tpl_hsv[:, :, 2].astype(int)
            v_crop = crop_hsv[:, :, 2].astype(int)
        except Exception:
            continue
        if mask_rs is not None:
            mask = (mask_rs > 0).astype('uint8')
        else:
            mask = np.ones_like(v_tpl, dtype='uint8')
        # убрать белые пиксели (HSV: V>250, S<10) из маски до расчёта метрик
        try:
            crop_hsv2 = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            s2 = crop_hsv2[:, :, 1]; v2 = crop_hsv2[:, :, 2]
            white2 = ((v2 > 250) & (s2 < 10)).astype('uint8')
            mask = (mask.astype('uint8') * (1 - white2)).astype('uint8')
        except Exception:
            pass
        if mask.sum() == 0:
            continue
        diff = (abs(v_tpl - v_crop) <= 30).astype('uint8')
        agree = float((diff * mask).sum()) / float(mask.sum())
        # masked_corr
        try:
            tpl_vf = v_tpl.astype('float32') * (mask.astype('float32'))
            crop_vf = v_crop.astype('float32') * (mask.astype('float32'))
            msum = mask.sum()
            mean_tpl = tpl_vf.sum() / msum
            mean_crop = crop_vf.sum() / msum
            a = (tpl_vf - mean_tpl) * (mask.astype('float32'))
            b = (crop_vf - mean_crop) * (mask.astype('float32'))
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            masked_v_corr = float(np.dot(a.ravel(), b.ravel()) / denom) if denom > 0 else 0.0
        except Exception:
            masked_v_corr = 0.0

        # combined score proxy: maxv from matchTemplate weighted by agree
        combined_proxy = float(maxv) * 0.7 + float(agree) * 0.3
        cand = (img_x, img_y, w, h, round(agree, 3), round(masked_v_corr, 3), round(combined_proxy, 3))
        if best is None or cand[6] > best[6]:
            best = cand
    return best


def _imread(path: str, unchanged=False):
    # use find_icon.imread_u if available (handles unicode paths)
    try:
        if hasattr(find_icon, 'imread_u'):
            # if unchanged requested, read with cv2.IMREAD_UNCHANGED
            flags = cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR
            return find_icon.imread_u(path, flags=flags)
    except Exception:
        pass
    # fallback
    flags = cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR
    return cv2.imread(path, flags)


def make_color_mask(bgr: np.ndarray):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (10, 70, 60), (30, 255, 255))
    if cv2.countNonZero(mask) == 0:
        # relax thresholds a bit
        mask = cv2.inRange(hsv, (8, 50, 40), (32, 255, 255))
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def load_icon_templates(tpl_root: str, tpl_group: str, max_templates: int = None, trim: bool = True) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Загружает шаблоны из папки tpl_root/tpl_group.
    Возвращает список кортежей: (tpl_key, tpl_bgr, tpl_gray, tpl_mask_color)
    """
    group_dir = os.path.join(tpl_root, tpl_group)
    if not os.path.isdir(group_dir):
        raise FileNotFoundError(f"Templates folder not found: {group_dir}")

    files = [f for f in os.listdir(group_dir) if f.lower().endswith('.png')]
    files = sorted(files)
    if max_templates is not None:
        files = files[:max_templates]

    templates = []
    
    def _keep_largest_component(bin_mask: np.ndarray, min_area=20):
        try:
            # ensure binary 0/255
            bm = (bin_mask > 0).astype('uint8') * 255
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bm, connectivity=8)
            if num_labels <= 1:
                return bm
            # skip background label 0, find largest label by area
            areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
            areas = sorted(areas, key=lambda x: -x[1])
            for lbl, a in areas:
                if a >= min_area:
                    out = (labels == lbl).astype('uint8') * 255
                    # closing + dilation to fill small holes and widen thin regions
                    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
                    out = cv2.dilate(out, np.ones((3, 3), np.uint8), iterations=1)
                    return out
            return np.zeros_like(bm)
        except Exception:
            return bin_mask
    for fname in files:
        path = os.path.join(group_dir, fname)
        img = _imread(path, unchanged=True)
        if img is None:
            print(f"WARN: can't read template: {path}")
            continue
        # img may have alpha channel
        if img.ndim == 3 and img.shape[2] == 4:
            bgr = img[:, :, :3]
            alpha = img[:, :, 3]
        else:
            bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            alpha = None
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask_color = make_color_mask(bgr)
        # build mask from alpha channel if present
        mask_alpha = (alpha > 0).astype('uint8') * 255 if alpha is not None else None
        # option: trim template to tight bbox around alpha or color mask to avoid extra padding
        # when trim==False we'll keep original bgr and masks so crops include white background
        # initialize defaults so variables exist in all branches
        bgr_trim = bgr
        mask_color_trim = mask_color
        mask_alpha_trim = mask_alpha
        try:
            if trim:
                if mask_alpha is not None and mask_alpha.sum() > 0:
                    ys, xs = np.where(mask_alpha > 0)
                else:
                    ys, xs = np.where(mask_color > 0)
                if len(xs) > 0 and len(ys) > 0:
                    x0, x1 = int(xs.min()), int(xs.max())
                    y0, y1 = int(ys.min()), int(ys.max())
                    bgr_trim = bgr[y0:y1 + 1, x0:x1 + 1]
                    mask_color_trim = mask_color[y0:y1 + 1, x0:x1 + 1]
                    mask_alpha_trim = mask_alpha[y0:y1 + 1, x0:x1 + 1] if mask_alpha is not None else None
                    # save trimmed preview
                    try:
                        outp = os.path.join(os.path.dirname(__file__), '..', 'debug', 'out_icons_debug', 'templates_trimmed')
                        os.makedirs(outp, exist_ok=True)
                        tfn = os.path.join(outp, f"{os.path.splitext(fname)[0]}_trim.png")
                        cv2.imwrite(tfn, bgr_trim)
                    except Exception:
                        pass
            # perform mask cleaning to remove thin artifacts: keep largest CC
            try:
                if mask_alpha_trim is not None:
                    cleaned = _keep_largest_component(mask_alpha_trim, min_area=20)
                    # if cleaned is empty, fall back to original alpha
                    if cleaned.sum() > 0:
                        mask_alpha_trim = cleaned
                else:
                    cleaned = _keep_largest_component(mask_color_trim, min_area=20)
                    if cleaned.sum() > 0:
                        mask_color_trim = cleaned
                # ensure masks are 0/255 uint8
                if mask_color_trim is not None:
                    mask_color_trim = (mask_color_trim > 0).astype('uint8') * 255
                if mask_alpha_trim is not None:
                    mask_alpha_trim = (mask_alpha_trim > 0).astype('uint8') * 255
            except Exception:
                pass
        except Exception:
            bgr_trim = bgr
            mask_color_trim = mask_color
            mask_alpha_trim = mask_alpha
        key = os.path.splitext(fname)[0]
        # keep original bgr as last element for possible use (index 6)
        templates.append((key, bgr_trim, gray, mask_color_trim, mask_alpha_trim, path, bgr))
    print(f"Loaded {len(templates)} templates from {group_dir}")
    return templates


def select_templates(templates, include_keys=None, limit: int = 19):
    """Return a filtered list of templates ensuring include_keys are present and
    filling the rest from the available templates up to `limit`.

    - templates: list as returned by load_icon_templates
    - include_keys: iterable of tpl keys (e.g. 'Янтарь_icon') to always include
    - limit: total number of templates to return
    """
    if include_keys is None:
        include_keys = []
    include_keys = [k for k in include_keys if k]
    tpl_by_key = {t[0]: t for t in templates}
    selected = []
    seen = set()
    # add mandatory ones first (if found)
    for k in include_keys:
        if k in tpl_by_key and k not in seen:
            selected.append(tpl_by_key[k])
            seen.add(k)
    # fill with remaining templates (preserve order) until limit
    for t in templates:
        if len(selected) >= limit:
            break
        if t[0] in seen:
            continue
        selected.append(t)
        seen.add(t[0])
    return selected


def run_on_image(
    image_path: str,
    templates,
    out_dir: str,
    screen_name: str = None,
    min_combined: float = 0.80,
    color_weight: float = 0.7,
    find_all: bool = False,
    use_candidates: bool = False,
    candidate_boxes=None,
    expected_only: bool = False,
    restore_strict: bool = False,
    restore_report_path: str = None,
    restore_threshold: float = 0.5,
    pad_frac: float = 0.35,
    debug_all_matches: bool = False,
    use_overlay_verify: bool = False,
    overlay_threshold: float = 0.80,
    overlay_radius: int = 60,
    overlay_tol_x: int = 50,
    overlay_tol_y: int = 100,
):
    """
    Запускает строгий поиск по одной картинке и возвращает простой JSON-отчёт
    с найденными шаблонами (ключ, combined score, bbox)
    """
    if screen_name is None:
        screen_name = os.path.splitext(os.path.basename(image_path))[0]

    img = _imread(image_path, unchanged=True)
    if img is None:
        raise FileNotFoundError(image_path)

    # precompute HSV and gray versions to avoid repeated conversions in postfilter
    try:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    except Exception:
        img_hsv = None
    try:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        img_gray = None

    # ensure out_dir exists and clean old/unrelated artifacts
    os.makedirs(out_dir, exist_ok=True)
    try:
        def clean_out_dir(pth):
            p = Path(pth)
            if not p.exists():
                return
            now = datetime.datetime.now()
            keep_prefixes = ('strict_', 'self_check', 'crops', 'overlays')
            for child in p.iterdir():
                name = child.name
                keep = False
                for pref in keep_prefixes:
                    if name.startswith(pref) or name == pref:
                        keep = True
                        break
                if keep:
                    continue
                try:
                    mtime = datetime.datetime.fromtimestamp(child.stat().st_mtime)
                    age = (now - mtime).days
                except Exception:
                    age = 9999
                # delete if older than 14 days
                if age > 14:
                    try:
                        if child.is_dir():
                            import shutil
                            shutil.rmtree(child)
                        else:
                            child.unlink()
                    except Exception:
                        pass
        clean_out_dir(out_dir)
    except Exception:
        pass
    # find_icon.strict_template_search_on_roi expects templates as
    # (key, bgr, gray, mask_color) tuples. Our loader keeps extra info
    # (mask_alpha, path). Build a lightweight list for the call and keep
    # full templates for post-filtering.
    templates_for_find = [(t[0], t[1], t[2], t[3]) for t in templates]
    # strict_template_search_on_roi пишет CSV и overlay, и возвращает их пути
    _ = find_icon.strict_template_search_on_roi(img, screen_name, templates_for_find, out_dir)

    # strict_search writes CSV at out_dir/strict_matches_{screen_name}.csv
    csv_path = os.path.join(out_dir, f"strict_matches_{screen_name}.csv")
    matches = []
    if os.path.exists(csv_path):
        try:
            import csv
            with open(csv_path, encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    matches.append(r)
        except Exception as e:
            print(f"Failed to parse CSV {csv_path}: {e}")

    # if debug requested, dump raw match crops and tpl previews for every CSV row
    def _save_raw_match_debug(row):
        try:
            tpl_key = row.get('tpl_key') or row.get('tpl') or row.get('template')
            if not tpl_key:
                return
            # find template entry
            tpl_entry = None
            for t in templates:
                if t[0] == tpl_key:
                    tpl_entry = t
                    break
            if tpl_entry is None:
                return
            x = int(row.get('x', 0) or 0)
            y = int(row.get('y', 0) or 0)
            w = int(row.get('w', 0) or 0)
            h = int(row.get('h', 0) or 0)
            tpl_bgr = tpl_entry[1]
            tpl_alpha = tpl_entry[4]
            # crop from image
            H, W = img.shape[:2]
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(W, x0 + w)
            y1 = min(H, y0 + h)
            crop = img[y0:y1, x0:x1]

            # ✅ Очистить crop от белого фона ПЕРЕД сохранением
            try:
                crop_clean = crop.copy()
                crop_hsv = cv2.cvtColor(crop_clean, cv2.COLOR_BGR2HSV)
                white_crop = ((crop_hsv[:, :, 2] > 245) & (crop_hsv[:, :, 1] < 15))
                non_white_crop = (~white_crop).astype('uint8') * 255
                if non_white_crop.sum() > 0:
                    mean_crop = cv2.mean(crop_clean, mask=non_white_crop)[:3]
                    crop_clean[white_crop] = mean_crop
                crop = crop_clean
            except Exception:
                pass

            # Обрезать template до маски и ресайз
            try:
                if tpl_alpha is not None:
                    mask_bbox = tpl_alpha
                else:
                    mask_bbox = tpl_entry[3]
                ys, xs = np.where(mask_bbox > 0)
                if len(xs) > 0:
                    x0_t, x1_t = int(xs.min()), int(xs.max()) + 1
                    y0_t, y1_t = int(ys.min()), int(ys.max()) + 1
                    tpl_bgr_trim = tpl_bgr[y0_t:y1_t, x0_t:x1_t]
                else:
                    tpl_bgr_trim = tpl_bgr
                tpl_resized = cv2.resize(tpl_bgr_trim, (max(1, crop.shape[1]), max(1, crop.shape[0])), interpolation=cv2.INTER_AREA)
            except Exception:
                tpl_resized = tpl_bgr

            # Очистить tpl_resized от белого
            try:
                tpl_hsv = cv2.cvtColor(tpl_resized, cv2.COLOR_BGR2HSV)
                white_tpl = ((tpl_hsv[:, :, 2] > 245) & (tpl_hsv[:, :, 1] < 15)).astype('uint8')
                non_white_tpl = (white_tpl == 0).astype('uint8') * 255
                if non_white_tpl.sum() > 0:
                    mean_tpl = cv2.mean(tpl_resized, mask=non_white_tpl)[:3]
                    tpl_resized[white_tpl > 0] = mean_tpl
            except Exception:
                pass
            # build mask visualization
            if tpl_alpha is not None:
                mask = tpl_alpha
                mask_rs = cv2.resize(mask, (tpl_resized.shape[1], tpl_resized.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_vis = (mask_rs > 0).astype('uint8') * 255
            else:
                mc = tpl_entry[3]
                mask_rs = cv2.resize(mc, (tpl_resized.shape[1], tpl_resized.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_vis = (mask_rs > 0).astype('uint8') * 255
            # heatmap via matchTemplate on V channel
            try:
                tpl_hsv = cv2.cvtColor(tpl_resized, cv2.COLOR_BGR2HSV)
                crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                tpl_v = tpl_hsv[:, :, 2]
                crop_v = crop_hsv[:, :, 2]
                if crop_v.shape[0] >= tpl_v.shape[0] and crop_v.shape[1] >= tpl_v.shape[1]:
                    res = cv2.matchTemplate(crop_v, tpl_v, cv2.TM_CCOEFF_NORMED)
                    # normalize to 0..255 heatmap sized to res
                    if res is not None and res.size>0:
                        rmin, rmax, _, _ = cv2.minMaxLoc(res)
                        heat = ((res - rmin) / (max(1e-6, (rmax - rmin))) * 255.0).astype('uint8')
                        # upscale heat to crop size for visualization
                        heat_up = cv2.resize(heat, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
                        heat_vis = cv2.applyColorMap(heat_up, cv2.COLORMAP_JET)
                    else:
                        heat_vis = None
                else:
                    heat_vis = None
            except Exception:
                heat_vis = None
            # save steps
            screen_env = os.environ.get('STRICT_SCREEN_NAME', os.path.splitext(os.path.basename(image_path))[0])
            debug_crop_logger.save_step(screen_env, tpl_key, 'raw_tpl_orig', tpl_bgr)
            debug_crop_logger.save_step(screen_env, tpl_key, 'raw_tpl_resized', tpl_resized)
            debug_crop_logger.save_step(screen_env, tpl_key, 'raw_crop', crop)
            if mask_vis is not None:
                debug_crop_logger.save_step(screen_env, tpl_key, 'raw_mask', mask_vis)
            if heat_vis is not None:
                debug_crop_logger.save_step(screen_env, tpl_key, 'raw_heatmap', heat_vis)
            meta = {'csv_row': row, 'bbox': [x0, y0, x1 - x0, y1 - y0]}
            debug_crop_logger.save_metadata(screen_env, tpl_key, meta)
        except Exception:
            pass

    if debug_all_matches:
        for r in matches:
            _save_raw_match_debug(r)

    # if debug_all_matches requested, return all CSV matches (skip postfilter)
    if debug_all_matches:
        find_all = True

    # If user requested to collect all matches, skip postfilter and return CSV rows
    if find_all:
        # convert CSV rows to simple dict shape
        out = []
        for r in matches:
            try:
                out.append({
                    'tpl_key': r.get('tpl_key') or r.get('tpl') or r.get('template'),
                    'combined': float(r.get('combined', 0) or 0),
                    'gray': float(r.get('gray', 0) or 0),
                    'x': int(r.get('x', 0) or 0),
                    'y': int(r.get('y', 0) or 0),
                    'w': int(r.get('w', 0) or 0),
                    'h': int(r.get('h', 0) or 0),
                })
            except Exception:
                continue
        report = {
            'image': image_path,
            'screen_name': screen_name,
            'csv': csv_path if os.path.exists(csv_path) else None,
            'matches': matches,
            'filtered_matches': out,
            'overlay': os.path.join(out_dir, f"strict_overlay_{screen_name}.png") if os.path.exists(os.path.join(out_dir, f"strict_overlay_{screen_name}.png")) else None,
        }
        json_path = os.path.join(out_dir, f"strict_report_{screen_name}.json")
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(report, jf, ensure_ascii=False, indent=2)
        print(f"Report written: {json_path}")
        return report

    # post-filter matches to keep only precise, unique detections
    # expose screen name to postfilter for targeted local refine
    os.environ['STRICT_SCREEN_NAME'] = str(screen_name)
    final_matches = _postfilter_matches(
        matches,
        templates,
        img,
        min_combined=min_combined,
        candidate_boxes=candidate_boxes,
        pad_frac=pad_frac,
        use_overlay=use_overlay_verify,
        overlay_threshold=overlay_threshold,
        overlay_radius=overlay_radius,
        overlay_tol_x=overlay_tol_x,
        overlay_tol_y=overlay_tol_y,
        image_hsv=img_hsv,
        image_gray=img_gray,
        screen_name=screen_name,
        out_dir=out_dir,
    )

    # if expected-only mode requested: prefer only expected templates for this screen
    if expected_only and screen_name in EXPECTED_COORDS:
        # build map of final matches by tpl_key
        final_map = {f['tpl_key']: f for f in final_matches}
        expected_out = []
        for tpl_key, coord in EXPECTED_COORDS[screen_name].items():
            if tpl_key in final_map:
                expected_out.append(final_map[tpl_key])
            else:
                # try local refine to recover missing expected tpl
                tpl_dict = {t[0]: t for t in templates}
                if tpl_key in tpl_dict:
                    tpl_entry = tpl_dict[tpl_key]
                    tpl_bgr = tpl_entry[1]
                    tpl_alpha = tpl_entry[4]
                    res = _local_refine_search(img, tpl_bgr, tpl_alpha, coord[0], coord[1], search_radius=220, scales=(0.85, 0.95, 1.0, 1.05, 1.15, 1.25), image_hsv=img_hsv)
                    if res:
                        x, y, w, h, agree, masked_corr, combined_proxy = res
                        # tighten expected-only acceptance: require stronger proxy and non-negative masked_corr
                        if combined_proxy >= 0.65 and masked_corr >= 0.0:
                            expected_out.append({
                                'tpl_key': tpl_key,
                                'combined': combined_proxy,
                                'masked_agreement': agree,
                                'gray': 0.0,
                                'x': int(x),
                                'y': int(y),
                                'w': int(w),
                                'h': int(h),
                                'detection_method': 'local_refine'
                            })
                    # if still missing and overlay verify requested, try overlay_local_refine in tolerance window
                    if use_overlay_verify and overlay_local_refine is not None and tpl_key in tpl_dict:
                        try:
                            ov = overlay_local_refine(img, tpl_bgr, tpl_alpha, coord[0], coord[1], search_radius=overlay_radius, scales=(0.9, 1.0, 1.1), step=4)
                            if ov:
                                ox, oy, ow, oh, oagree, oinv, ocombined = ov
                                if abs(int(ox) - int(coord[0])) <= overlay_tol_x and abs(int(oy) - int(coord[1])) <= overlay_tol_y and float(oinv) >= float(overlay_threshold):
                                    expected_out.append({
                                        'tpl_key': tpl_key,
                                        'combined': float(oinv),
                                        'masked_agreement': float(oagree),
                                        'gray': 0.0,
                                        'x': int(ox),
                                        'y': int(oy),
                                        'w': int(ow),
                                        'h': int(oh),
                                        'detection_method': 'overlay'
                                    })
                        except Exception:
                            pass
        final_matches = expected_out
    # if requested, attempt to restore matches from a prior strict_verifier report
    if restore_strict and restore_report_path:
        try:
            final_matches = _restore_strict_matches_from_report(restore_report_path, screen_name, final_matches, threshold=restore_threshold)
        except Exception:
            pass
    # cleanup
    try:
        del os.environ['STRICT_SCREEN_NAME']
    except Exception:
        pass

    report = {
        'image': image_path,
        'screen_name': screen_name,
        'csv': csv_path if os.path.exists(csv_path) else None,
        'matches': matches,
        'filtered_matches': final_matches,
        'overlay': os.path.join(out_dir, f"strict_overlay_{screen_name}.png") if os.path.exists(os.path.join(out_dir, f"strict_overlay_{screen_name}.png")) else None,
    }
    # dump quick json
    json_path = os.path.join(out_dir, f"strict_report_{screen_name}.json")
    
    def _safe_numeric(v):
        try:
            vv = float(v)
            if not (vv == vv) or vv == float('inf') or vv == float('-inf'):
                return 0.0
            return vv
        except Exception:
            return v

    # sanitize numeric fields in final matches
    for fm in report.get('filtered_matches', []):
        for k in ('combined', 'gray', 'masked_agreement'):
            if k in fm:
                fm[k] = _safe_numeric(fm[k])

    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(report, jf, ensure_ascii=False, indent=2)
    # also draw UTF-8 overlay with Russian labels (Pillow)
    try:
        # work in RGBA to allow translucent highlights
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('RGBA')
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            font = ImageFont.load_default()

        # separate overlay layer for translucent highlights and labels
        overlay_layer = Image.new('RGBA', pil.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay_layer)

        for fm in final_matches:
            x, y, w, h = int(fm['x']), int(fm['y']), int(fm['w']), int(fm['h'])
            tpl = fm.get('tpl_key') or fm.get('tpl') or fm.get('template') or ''
            method = fm.get('detection_method', fm.get('method', '')) or ''
            # choose color by detection method for clarity
            if method == 'overlay':
                col = (0, 200, 0)
            elif method == 'local_refine':
                col = (0, 120, 200)
            else:
                col = (200, 0, 0)

            # paste the template visual first (so it's visible under highlight outline)
            try:
                tpl_entry = None
                for t in templates:
                    if t[0] == tpl:
                        tpl_entry = t
                        break
                if tpl_entry is not None:
                    tpl_bgr = tpl_entry[1]
                    tpl_alpha = tpl_entry[4]
                    # resize template to detected bbox
                    try:
                        tpl_resized = cv2.resize(tpl_bgr, (max(1, w), max(1, h)), interpolation=cv2.INTER_AREA)
                    except Exception:
                        tpl_resized = tpl_bgr
                    # build RGBA array for paste
                    try:
                        rgba = cv2.cvtColor(tpl_resized, cv2.COLOR_BGR2RGBA)
                    except Exception:
                        # fallback: stack channels
                        h_r, w_r = tpl_resized.shape[:2]
                        rgba = np.zeros((h_r, w_r, 4), dtype='uint8')
                        rgba[:, :, :3] = tpl_resized
                        rgba[:, :, 3] = 255
                    if tpl_alpha is not None:
                        try:
                            alpha_rs = cv2.resize(tpl_alpha, (rgba.shape[1], rgba.shape[0]), interpolation=cv2.INTER_NEAREST)
                            rgba[:, :, 3] = (alpha_rs > 0).astype('uint8') * 255
                        except Exception:
                            pass
                    else:
                        # make near-white pixels transparent to avoid hiding labels
                        mask_white = ((tpl_resized[:, :, 0] >= 245) & (tpl_resized[:, :, 1] >= 245) & (tpl_resized[:, :, 2] >= 245))
                        rgba[:, :, 3] = (~mask_white).astype('uint8') * 255
                    tpl_pil = Image.fromarray(rgba)
                    Wp, Hp = pil.size
                    paste_x = max(0, min(Wp - tpl_pil.width, x))
                    paste_y = max(0, min(Hp - tpl_pil.height, y))
                    # paste onto base image so the template graphic appears
                    pil.paste(tpl_pil, (paste_x, paste_y), tpl_pil)
            except Exception:
                pass

            # draw translucent filled rectangle and solid outline on overlay layer
            try:
                fill = (col[0], col[1], col[2], 64)
                outline = (col[0], col[1], col[2], 255)
                overlay_draw.rectangle([x, y, x + w, y + h], fill=fill, outline=outline, width=3)
                # label with background for readability
                label = f"{tpl} {('(' + method + ')') if method else ''}"
                tw, th = overlay_draw.textsize(label, font=font)
                # background rect
                bx0 = x
                by0 = max(0, y - th - 6)
                bx1 = x + tw + 8
                by1 = y
                overlay_draw.rectangle([bx0, by0, bx1, by1], fill=(0, 0, 0, 180))
                overlay_draw.text((bx0 + 4, by0 + 2), label, font=font, fill=(255, 255, 255, 255))
            except Exception:
                pass

        # composite overlay layer on top and save as RGB
        try:
            composed = Image.alpha_composite(pil, overlay_layer)
            composed_rgb = composed.convert('RGB')
            overlay_path = os.path.join(out_dir, f"strict_overlay_{screen_name}.png")
            composed_rgb.save(overlay_path)
            report['overlay'] = overlay_path
        except Exception:
            # fallback: save base pil converted to RGB
            try:
                pil.convert('RGB').save(os.path.join(out_dir, f"strict_overlay_{screen_name}.png"))
                report['overlay'] = os.path.join(out_dir, f"strict_overlay_{screen_name}.png")
            except Exception:
                pass
    except Exception:
        pass
    print(f"Report written: {json_path}")
    return report


def _bbox_iou(a, b):
    ax, ay, aw, ah = int(a[0]), int(a[1]), int(a[2]), int(a[3])
    bx, by, bw, bh = int(b[0]), int(b[1]), int(b[2]), int(b[3])
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

def _masked_hist_corr(tpl_bgr: np.ndarray, crop_bgr: np.ndarray, mask_u8: np.ndarray) -> float:
    """Сравнение цветового профиля внутри маски по гистограммам HSV."""
    try:
        m = (mask_u8 > 0).astype('uint8')
        tpl_hsv = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2HSV)
        crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        hist_tpl = cv2.calcHist([tpl_hsv], [0, 1], m, [16, 16], [0, 180, 0, 256])
        hist_crp = cv2.calcHist([crop_hsv], [0, 1], m, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist_tpl, hist_tpl)
        cv2.normalize(hist_crp, hist_crp)
        return float(cv2.compareHist(hist_tpl, hist_crp, cv2.HISTCMP_CORREL))
    except Exception:
        return 0.0


def extract_icon_candidates(image_bgr, min_area=800, max_area=20000):
    """Return list of candidate bbox tuples (x,y,w,h) likely containing icons.

    Strategy (tightened):
    - convert to HSV, mark non-white pixels with stricter saturation/value thresholds
    - morphological clean and find contours
    - filter by area, aspect ratio (prefer near-square icons), and color-density
    """
    H, W = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # non-white where saturation is significant OR value not very high (tighter)
    nonwhite = ((s > 50) & (v < 240)).astype('uint8') * 255
    # also include low-value dark icons (tighter)
    dark_mask = (v < 180).astype('uint8') * 255
    mask = cv2.bitwise_or(nonwhite, dark_mask)
    # morph
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area or area > max_area:
            continue
        ar = float(w) / float(h) if h > 0 else 0
        # prefer roughly square icons; reject extreme wide/tall regions
        if ar < 0.6 or ar > 1.8:
            continue
        # pad slightly (small relative padding)
        pad_x = int(min(8, w * 0.12))
        pad_y = int(min(8, h * 0.12))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(W, x + w + pad_x)
        y1 = min(H, y + h + pad_y)
        # проверка доли «цветных» пикселей внутри бокса (по насыщенности/значению)
        try:
            roi_hsv = hsv[y0:y1, x0:x1]
            s_roi = roi_hsv[:, :, 1]
            v_roi = roi_hsv[:, :, 2]
            colorish = ((s_roi > 45) | (v_roi < 235)).astype('uint8')
            ratio_color = float(colorish.sum()) / float(max(1, (y1 - y0) * (x1 - x0)))
            # ensure the region has enough 'colorful' pixels to be a valid icon
            if ratio_color < 0.12:
                continue
            # density: fraction of bbox area covered by contour (helps remove thin UI stripes)
            bbox_area = float((x1 - x0) * (y1 - y0))
            density = float(area) / float(max(1.0, bbox_area))
            if density < 0.12:
                continue
        except Exception:
            ratio_color = 1.0
        boxes.append((x0, y0, x1 - x0, y1 - y0))
    # if we found nothing, fallback to full image
    # also try adding small green blobs (e.g. berries) to candidates — these
    # can be smaller than the generic min_area we used above. This helps
    # catching small green squares that otherwise get missed.
    try:
        color_mask = make_color_mask(image_bgr)
        # find small green contours
        cnts, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < 100 or area > 4000:
                continue
            # small green blobs likely near-square; allow moderate aspect ratios
            ar = float(w) / float(h) if h > 0 else 1.0
            if ar < 0.5 or ar > 2.0:
                continue
            pad_x = int(min(8, w * 0.3))
            pad_y = int(min(8, h * 0.3))
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(W, x + w + pad_x)
            y1 = min(H, y + h + pad_y)
            try:
                roi_hsv = hsv[y0:y1, x0:x1]
                s_roi = roi_hsv[:, :, 1]; v_roi = roi_hsv[:, :, 2]
                colorish = ((s_roi > 45) | (v_roi < 235)).astype('uint8')
                ratio_color = float(colorish.sum()) / float(max(1, (y1 - y0) * (x1 - x0)))
            except Exception:
                ratio_color = 1.0
            if ratio_color < 0.12:
                continue
            boxes.append((x0, y0, x1 - x0, y1 - y0))
    except Exception:
        pass

    if boxes:
        # deduplicate overlapping boxes (simple): sort and unique by rect
        seen = set()
        out = []
        for b in boxes:
            if b in seen:
                continue
            seen.add(b)
            out.append(b)
        return out

    # Try user-provided extractor script (вырезка картинок.py) if available
    try:
        user_path = os.path.join(os.path.dirname(__file__), 'вырезка картинок.py')
        if os.path.exists(user_path):
            spec = importlib.util.spec_from_file_location('user_extractor', user_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, '_adaptive_icon_from_block'):
                box = mod._adaptive_icon_from_block(image_bgr)
                if box:
                    x, y, w, h = box
                    return [(int(x), int(y), int(w), int(h))]
    except Exception:
        pass

    return [(0, 0, W, H)]


def _postfilter_matches(matches, templates, image_bgr, min_combined=0.75, min_masked=0.6, iou_thr=0.25, candidate_boxes=None, pad_frac=0.35,
                        use_overlay: bool = False, overlay_threshold: float = 0.8, overlay_radius: int = 60, overlay_tol_x: int = 50, overlay_tol_y: int = 100,
                        image_hsv=None, image_gray=None, screen_name: str = None, out_dir: str = None):
    """Filter matches by combined score and masked agreement, then cluster and keep one per location.

    Use conservative thresholds taken from tools/cfg/config.json STRICT section when available.
    This function intentionally applies strict acceptance rules to avoid false positives.
    """
    # load strict thresholds from project config if possible
    try:
        cfg_path = os.path.join(os.path.dirname(__file__), '..', 'cfg', 'config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r', encoding='utf-8') as _f:
                cfg_local = json.load(_f)
                strict_cfg = cfg_local.get('STRICT', {})
                cfg_min_combined_final = float(strict_cfg.get('MIN_COMBINED_FINAL', strict_cfg.get('MIN_COMBINED', min_combined)))
                cfg_min_masked = float(strict_cfg.get('MIN_MASKED_AGREE', min_masked))
                cfg_min_gray = float(strict_cfg.get('MIN_GRAY_ACCEPT', 0.40))
                cfg_min_masked_soft = float(strict_cfg.get('MIN_MASKED_SOFT', 0.15))
                cfg_min_masked_required = float(strict_cfg.get('MIN_MASKED_REQUIRED', 0.5))
                per_tpl_masked = strict_cfg.get('PER_TEMPLATE_MIN_MASKED', {}) or {}
                cfg_min_template_coverage = float(strict_cfg.get('MIN_TEMPLATE_COVERAGE', 0.5))
                per_tpl_coverage = strict_cfg.get('PER_TEMPLATE_MIN_COVERAGE', {}) or {}
                cfg_min_masked_corr = float(strict_cfg.get('MIN_MASKED_CORR', 0.4))
                cfg_min_hist_corr = float(strict_cfg.get('MIN_HIST_CORR', 0.65))
                cfg_min_tpl_h = int(strict_cfg.get('MIN_TEMPLATE_HEIGHT', 40))
                cfg_max_tpl_h = int(strict_cfg.get('MAX_TEMPLATE_HEIGHT', 70))
                per_tpl_masked_corr = strict_cfg.get('PER_TEMPLATE_MIN_MASKED_CORR', {}) or {}
                per_tpl_hist_corr = strict_cfg.get('PER_TEMPLATE_MIN_HIST_CORR', {}) or {}
                per_tpl_edge_accept = strict_cfg.get('PER_TEMPLATE_MIN_EDGE_ACCEPT', {}) or {}
                per_tpl_no_orb = strict_cfg.get('PER_TEMPLATE_NO_ORB_VERIFY', {}) or {}
                cfg_min_edge = float(strict_cfg.get('MIN_EDGE_ACCEPT', 0.18))
                per_tpl = strict_cfg.get('PER_TEMPLATE_MIN_COMBINED', {}) or {}
        else:
            cfg_min_combined_final = float(min_combined)
            cfg_min_masked = float(min_masked)
            cfg_min_gray = 0.40
            cfg_min_edge = 0.18
            per_tpl = {}
    except Exception:
        cfg_min_combined_final = float(min_combined)
        cfg_min_masked = float(min_masked)
        cfg_min_gray = 0.40
        cfg_min_edge = 0.18
        per_tpl = {}
    if not matches:
        return []
    # build template dict by key
    tpl_dict = {t[0]: t for t in templates}

    accepted = []
    reject_reasons = []
    for m in matches:
        try:
            reasons = []
            tpl_key = m.get('tpl_key') or m.get('tpl') or m.get('template')
            if tpl_key is None:
                # record missing template key
                try:
                    reject_reasons.append({'row': m, 'reasons': ['missing_tpl_key']})
                except Exception:
                    pass
                continue
            # ignore non-finite values
            # parse numeric scores and skip non-finite values (inf/nan)
            # parse combined/gray and sanitize non-finite
            try:
                combined = float(m.get('combined', 0) or 0)
                if not (combined == combined and combined != float('inf')):
                    # skip non-finite scores
                    reasons.append('non_finite_combined')
                    reject_reasons.append({'row': m, 'reasons': reasons})
                    continue
            except Exception:
                reasons.append('bad_combined_parse')
                reject_reasons.append({'row': m, 'reasons': reasons})
                continue
            try:
                gray_score = float(m.get('gray', 0) or 0)
                if not (gray_score == gray_score and gray_score != float('inf')):
                    gray_score = 0.0
            except Exception:
                gray_score = 0.0
            # early fast reject for very low combined scores (catch obvious noise)
            if combined < 0.60:
                reasons.append(f'combined_too_low:{combined:.3f}<0.60')
                reject_reasons.append({'row': m, 'reasons': reasons})
                continue
            x = int(m.get('x', 0))
            y = int(m.get('y', 0))
            w = int(m.get('w', 0))
            h = int(m.get('h', 0))
            tpl_entry = tpl_dict.get(tpl_key)
            if tpl_entry is None:
                # try basename variants
                k2 = tpl_key + '.png' if not tpl_key.lower().endswith('.png') else tpl_key
                for t in templates:
                    if t[5].endswith(k2) or os.path.splitext(os.path.basename(t[5]))[0] == tpl_key:
                        tpl_entry = t
                        break
            if tpl_entry is None:
                continue
            # tpl_entry: (key, bgr_trimmed, gray, mask_color_trimmed, mask_alpha_trimmed, path, bgr_original)
            # ✅ ИСПРАВЛЕНО: используем trimmed версию (индекс 1), оригинал в индексе 6
            tpl_bgr = tpl_entry[1]
            tpl_alpha = tpl_entry[4]

            # crop candidate region from image
            H, W = image_bgr.shape[:2]
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(W, x0 + w)
            y1 = min(H, y0 + h)
            crop = image_bgr[y0:y1, x0:x1]
            if crop.size == 0:
                reasons.append('empty_crop')
                reject_reasons.append({'row': m, 'reasons': reasons})
                continue
            # ✅ НОВОЕ: Сначала обрезать template до маски, потом resize
            try:
                # Определить bbox маски
                if tpl_alpha is not None:
                    mask_for_bbox = tpl_alpha
                else:
                    mask_for_bbox = tpl_entry[3]

                ys, xs = np.where(mask_for_bbox > 0)
                if len(xs) > 0 and len(ys) > 0:
                    x0_t, x1_t = int(xs.min()), int(xs.max()) + 1
                    y0_t, y1_t = int(ys.min()), int(ys.max()) + 1
                    tpl_bgr_trimmed = tpl_bgr[y0_t:y1_t, x0_t:x1_t]
                    tpl_alpha_trimmed = tpl_alpha[y0_t:y1_t, x0_t:x1_t] if tpl_alpha is not None else None
                    mask_color_trimmed = tpl_entry[3][y0_t:y1_t, x0_t:x1_t]
                else:
                    tpl_bgr_trimmed = tpl_bgr
                    tpl_alpha_trimmed = tpl_alpha
                    mask_color_trimmed = tpl_entry[3]

                # Теперь resize trimmed template
                tpl_resized = cv2.resize(tpl_bgr_trimmed, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_AREA)

                # Также resize соответствующую маску
                if tpl_alpha_trimmed is not None:
                    tpl_alpha_for_resize = tpl_alpha_trimmed
                else:
                    tpl_alpha_for_resize = mask_color_trimmed
            except Exception:
                tpl_resized = tpl_bgr
                tpl_alpha_for_resize = tpl_alpha if tpl_alpha is not None else tpl_entry[3]

            # ✅ НОВОЕ: Удалить белый фон из tpl_resized
            try:
                tpl_hsv_clean = cv2.cvtColor(tpl_resized, cv2.COLOR_BGR2HSV)
                s_tpl = tpl_hsv_clean[:, :, 1]
                v_tpl = tpl_hsv_clean[:, :, 2]
                white_tpl = ((v_tpl > 245) & (s_tpl < 15)).astype('uint8')
                non_white = (white_tpl == 0).astype('uint8') * 255
                if non_white.sum() > 0:
                    mean_tpl_color = cv2.mean(tpl_resized, mask=non_white)[:3]
                    tpl_resized[white_tpl > 0] = mean_tpl_color
            except Exception:
                pass

            # compute masked agreement using alpha if available else color mask
            if tpl_alpha_for_resize is not None:
                tpl_alpha_res = cv2.resize(tpl_alpha_for_resize, (tpl_resized.shape[1], tpl_resized.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask = (tpl_alpha_res > 0).astype('uint8')
            else:
                # fallback: use color mask computed previously (entry[3])
                mask_color = tpl_entry[3]
                mask = cv2.resize(mask_color, (tpl_resized.shape[1], tpl_resized.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask = (mask > 0).astype('uint8')

            # ✅ УЛУЧШЕНО: Более строгое удаление белого из маски
            try:
                if image_hsv is not None:
                    crop_hsv_local = image_hsv[y0:y1, x0:x1]
                else:
                    crop_hsv_local = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                s_ch = crop_hsv_local[:, :, 1]
                v_ch = crop_hsv_local[:, :, 2]
                # Более строгие пороги для белого: V>240 и S<20
                white_px = ((v_ch > 240) & (s_ch < 20)).astype('uint8')
                mask = (mask.astype('uint8') * (1 - white_px)).astype('uint8')

                # Также исключить очень темные пиксели (шум)
                very_dark = (v_ch < 15).astype('uint8')
                mask = (mask.astype('uint8') * (1 - very_dark)).astype('uint8')
            except Exception:
                pass

            if mask.sum() == 0:
                reasons.append('mask_empty_after_white_exclusion')
                reject_reasons.append({'row': m, 'reasons': reasons})
                continue
            # if mask is very sparse (thin stripes / artifacts), try an
            # aggressive cleanup: erosion to remove thin artifacts followed by
            # keeping the largest connected component. This helps avoid
            # matching on thin stripes inside templates.
            try:
                # mask currently 0/1 uint8
                h_m, w_m = mask.shape[:2]
                mask_area = float(mask.sum())
                density = mask_area / float(max(1, h_m * w_m))
                # if density low, erode to remove thin structures
                if density < 0.20:
                    try:
                        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
                    except Exception:
                        pass
                # keep largest connected component to drop small stripes
                try:
                    bm = (mask > 0).astype('uint8') * 255
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bm, connectivity=8)
                    if num_labels > 1:
                        areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
                        areas = sorted(areas, key=lambda x: -x[1])
                        # select largest component with reasonable area
                        for lbl, a in areas:
                            if a >= 8:
                                mask = (labels == lbl).astype('uint8')
                                break
                except Exception:
                    pass
                # small morphological dilation to tolerate tiny misalignments
                try:
                    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
                except Exception:
                    pass
            except Exception:
                pass

            # compare V channel difference (relaxed threshold)
            tpl_hsv = cv2.cvtColor(tpl_resized, cv2.COLOR_BGR2HSV)
            if image_hsv is not None:
                # use precomputed HSV for the full image if available (slice ROI)
                crop_hsv = image_hsv[y0:y1, x0:x1]
            else:
                crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            v_tpl = tpl_hsv[:, :, 2].astype(int)
            v_crop = crop_hsv[:, :, 2].astype(int)
            diff = (abs(v_tpl - v_crop) <= 30).astype('uint8')
            agree = float((diff * mask).sum()) / float(mask.sum())
            # compute coverage within masked pixels (fraction of mask pixels that match)
            try:
                total_tpl_px = float(tpl_resized.shape[0] * tpl_resized.shape[1])
                coverage_full = float(diff.sum()) / total_tpl_px if total_tpl_px > 0 else 0.0
                coverage_masked = float((diff * mask).sum()) / float(mask.sum()) if mask.sum() > 0 else 0.0
            except Exception:
                coverage_full = 0.0
                coverage_masked = 0.0
            # compute simple edge agreement as an additional signal (do it regardless)
            try:
                tpl_edge = cv2.Canny(cv2.cvtColor(tpl_resized, cv2.COLOR_BGR2GRAY), 50, 150)
                if image_gray is not None:
                    crop_gray = image_gray[y0:y1, x0:x1]
                else:
                    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                crop_edge = cv2.Canny(crop_gray, 50, 150)
                # masked normalized correlation between edges
                tpl_e = (tpl_edge.astype('float32') * (mask.astype('float32')))
                crop_e = (crop_edge.astype('float32') * (mask.astype('float32')))
                denom = (np.linalg.norm(tpl_e) * np.linalg.norm(crop_e))
                edge_score = float(np.dot(tpl_e.ravel(), crop_e.ravel()) / denom) if denom > 0 else 0.0
            except Exception:
                edge_score = 0.0

            # compute masked normalized cross-correlation on V-channel (structural agreement)
            try:
                tpl_vf = v_tpl.astype('float32') * (mask.astype('float32'))
                crop_vf = v_crop.astype('float32') * (mask.astype('float32'))
                # subtract mean on masked pixels
                msum = mask.sum()
                if msum > 0:
                    mean_tpl = tpl_vf.sum() / msum
                    mean_crop = crop_vf.sum() / msum
                    a = (tpl_vf - mean_tpl) * (mask.astype('float32'))
                    b = (crop_vf - mean_crop) * (mask.astype('float32'))
                    denom2 = (np.linalg.norm(a) * np.linalg.norm(b))
                    masked_v_corr = float(np.dot(a.ravel(), b.ravel()) / denom2) if denom2 > 0 else 0.0
                else:
                    masked_v_corr = 0.0
            except Exception:
                masked_v_corr = 0.0

            # доп. цветовая корреляция внутри маски
            try:
                hist_corr = _masked_hist_corr(tpl_resized, crop, (mask > 0).astype('uint8'))
            except Exception:
                hist_corr = 0.0

            # Acceptance rule (strict): require combined >= global final threshold AND
            # at least one strong auxiliary signal (masked_agreement or masked_corr or edge)
            thr_edge = float(cfg_min_edge)
            # вспомогательные сигналы: усиливаем структуру — masked_agreement ИЛИ masked_corr ИЛИ edge_score.
            aux_ok = (agree >= cfg_min_masked) or (masked_v_corr >= max(0.35, cfg_min_masked_corr)) or (edge_score >= thr_edge)
            tpl_override = per_tpl.get(tpl_key)
            tpl_min = float(tpl_override) if tpl_override is not None else float(cfg_min_combined_final)
            # if candidate boxes provided, require detection bbox to intersect a padded candidate box
            # (relaxed from previous center-in-candidate rule which was too strict).
            # However, allow strong combined matches to bypass candidate filtering in case
            # the extractor missed the region. Only enforce intersection for borderline matches.
            if candidate_boxes:
                # if match is strong enough, skip candidate intersection requirement
                try:
                    combined_val = float(combined)
                except Exception:
                    combined_val = 0.0
                # require intersection only for matches weaker than the configured final threshold
                bypass_thresh = float(cfg_min_combined_final)
                if combined_val < bypass_thresh:
                    inside = False
                    # pad fraction controls how much we inflate candidate boxes when
                    # checking intersection with detection bbox. This can be tuned to
                    # recover icons near edges or when extractor is conservative.
                    # Default kept at 0.35 but can be passed through from caller.
                    pad_frac_local = float(pad_frac)
                    for cb in candidate_boxes:
                        cbx, cby, cbw, cbh = cb
                        pad_px = int(max(cbw, cbh) * pad_frac_local)
                        # padded candidate rect (clamped to image bounds W,H)
                        pcbx = max(0, cbx - pad_px)
                        pcby = max(0, cby - pad_px)
                        pcbx2 = min(W, cbx + cbw + pad_px)
                        pcby2 = min(H, cby + cbh + pad_px)
                        # intersection between detection bbox and padded candidate
                        ix0 = max(x, pcbx)
                        iy0 = max(y, pcby)
                        ix1 = min(x + w, pcbx2)
                        iy1 = min(y + h, pcby2)
                        if ix1 > ix0 and iy1 > iy0:
                            inside = True
                            break
                    if not inside:
                        reasons.append('not_in_candidate_boxes')
                        reject_reasons.append({'row': m, 'reasons': reasons})
                        continue
            if (combined >= float(tpl_min)) and aux_ok and (float(gray_score) >= float(cfg_min_gray)):
                accepted.append(
                    {
                        **m,
                        'masked_agreement': round(agree, 3),
                        'coverage': round(coverage_full, 3),
                        'coverage_masked': round(coverage_masked, 3),
                        'masked_corr': round(masked_v_corr, 3),
                        'hist_corr': round(hist_corr, 3),
                        'edge_score': round(edge_score, 3),
                        'bbox': (x, y, w, h),
                        'detection_method': 'strict',
                    }
                )
            else:
                # failed primary acceptance (combined/aux) — if gray specifically caused rejection, record it
                if (combined >= float(tpl_min)) and aux_ok and (float(gray_score) < float(cfg_min_gray)):
                    try:
                        reasons.append(f'gray_too_low(gray={gray_score:.3f}<{cfg_min_gray})')
                    except Exception:
                        reasons.append('gray_too_low')
                else:
                    reasons.append('failed_combined_or_aux')
                # include some metrics to help debugging
                reasons.append(f'combined:{combined:.3f}')
                reasons.append(f'agree:{agree:.3f}')
                reasons.append(f'masked_v_corr:{masked_v_corr:.3f}')
                reasons.append(f'edge:{edge_score:.3f}')
                reasons.append(f'hist_corr:{hist_corr:.3f}')
                reject_reasons.append({'row': m, 'reasons': reasons})
        except Exception:
            continue

    # cluster accepted by IoU and keep best combined per cluster
    accepted_sorted = sorted(accepted, key=lambda r: -float(r.get('combined', 0)))
    clusters = []
    for a in accepted_sorted:
        placed = False
        for cl in clusters:
            if _bbox_iou(a['bbox'], cl[0]['bbox']) >= iou_thr:
                cl.append(a)
                placed = True
                break
        if not placed:
            clusters.append([a])

    final = [max(cl, key=lambda t: float(t.get('combined', 0))) for cl in clusters]
    # prepare tidy output
    out = []
    for f in final:
        try:
            combined_v = float(f.get('combined', 0))
            gray_v = float(f.get('gray', 0) or 0)
            masked_v = float(f.get('masked_agreement', 0) or 0)
            cov_v = float(f.get('coverage', 0) or 0)
        except Exception:
            continue
        # Enforce gray threshold at final stage as well (reject and log reason)
        try:
            if float(gray_v) < float(cfg_min_gray):
                try:
                    reasons = [f'gray_too_low(gray={gray_v:.3f}<{cfg_min_gray})']
                    reject_reasons.append({'row': f, 'reasons': reasons})
                except Exception:
                    pass
                continue
        except Exception:
            try:
                reject_reasons.append({'row': f, 'reasons': ['bad_gray_value']})
            except Exception:
                pass
            continue
        try:
            tpl_k = f.get('tpl_key')
            tpl_override = per_tpl.get(tpl_k)
            tpl_min_final = float(tpl_override) if tpl_override is not None else float(cfg_min_combined_final)
        except Exception:
            tpl_min_final = float(cfg_min_combined_final)

        # per-template required masked agreement (fallback to global required)
        try:
            tpl_mask_req = float(per_tpl_masked.get(tpl_k, cfg_min_masked_required))
        except Exception:
            tpl_mask_req = float(cfg_min_masked_required)

        # per-template coverage requirement (fallback to global)
        try:
            tpl_cov_req = float(per_tpl_coverage.get(tpl_k, cfg_min_template_coverage))
        except Exception:
            tpl_cov_req = float(cfg_min_template_coverage)

        # require combined >= threshold (global or per-template)
        if not (((combined_v >= float(cfg_min_combined_final)) or (combined_v >= tpl_min_final))):
            continue

        # per-template masked_corr requirement (fallback to global)
        try:
            tpl_mask_corr_req = float(per_tpl_masked_corr.get(tpl_k, cfg_min_masked_corr))
        except Exception:
            tpl_mask_corr_req = float(cfg_min_masked_corr)

        # per-template hist_corr and edge thresholds (fallback to global)
        try:
            tpl_hist_corr_req = float(per_tpl_hist_corr.get(tpl_k, cfg_min_hist_corr))
        except Exception:
            tpl_hist_corr_req = float(cfg_min_hist_corr)
        try:
            tpl_edge_req = float(per_tpl_edge_accept.get(tpl_k, cfg_min_edge))
        except Exception:
            tpl_edge_req = float(cfg_min_edge)
        # per-template ORB disable flag
        try:
            no_orb = bool(per_tpl_no_orb.get(tpl_k, False))
        except Exception:
            no_orb = False

        # структурные и дополнительные метрики: masked_corr, hist_corr, edge
        masked_corr_v = float(f.get('masked_corr', 0.0))
        hist_corr_v = float(f.get('hist_corr', 0.0))
        edge_v = float(f.get('edge_score', 0.0))
        # optional size check (use config bounds if present)
        try:
            tpl_h_val = int(f.get('h', 0))
        except Exception:
            tpl_h_val = 0

        # require at least one strong structural metric (agree or masked_corr)
        if not ((masked_v >= tpl_mask_req) or (masked_corr_v >= tpl_mask_corr_req)):
            continue
        # require color agreement and edge strength
        try:
            if hist_corr_v < float(tpl_hist_corr_req):
                continue
        except Exception:
            if hist_corr_v < 0.65:
                continue
        if edge_v < float(tpl_edge_req):
            continue
        # enforce height bounds if configured
        try:
            if tpl_h_val > 0 and (int(cfg_min_tpl_h) > 0 and int(cfg_max_tpl_h) > 0):
                if not (int(cfg_min_tpl_h) <= tpl_h_val <= int(cfg_max_tpl_h)):
                    continue
        except Exception:
            pass

        # ORB-верификация для спорных совпадений: если малый запас по combined и структурным метрикам
        accept = True
        try:
            tpl_k = f.get('tpl_key')
            tpl_entry = tpl_dict.get(tpl_k)
            if tpl_entry is not None:
                tpl_bgr_chk = tpl_entry[1]
                x0 = int(f['bbox'][0]); y0 = int(f['bbox'][1]); w0 = int(f['bbox'][2]); h0 = int(f['bbox'][3])
                crop_chk = image_bgr[y0:y0 + h0, x0:x0 + w0]
                # спорный, если combined близко к порогу или masked_agreement/masked_corr без сильного запаса
                borderline = (combined_v < (tpl_min_final + 0.03)) or ((masked_corr_v < (tpl_mask_corr_req + 0.05)) and (masked_v < (tpl_mask_req + 0.05)))
                if borderline and (not no_orb):
                    try:
                        tpl_resized_chk = cv2.resize(tpl_bgr_chk, (max(1, w0), max(1, h0)), interpolation=cv2.INTER_AREA)
                        orb_ok = _orb_verify(tpl_resized_chk, crop_chk, min_matches=10)
                        if not orb_ok:
                            accept = False
                    except Exception:
                        pass
        except Exception:
            accept = True
        if not accept:
            # skip this final detection due to ORB mismatch
            continue
        out.append({
            'tpl_key': f.get('tpl_key'),
            'combined': combined_v,
            'masked_agreement': masked_v,
            'gray': gray_v,
            'x': int(f['bbox'][0]),
            'y': int(f['bbox'][1]),
            'w': int(f['bbox'][2]),
            'h': int(f['bbox'][3]),
        })
        # save debug artifacts for this final detection
        try:
            tpl_k = f.get('tpl_key')
            tpl_entry = tpl_dict.get(tpl_k)
            if tpl_entry is not None:
                tpl_bgr = tpl_entry[1]
                tpl_alpha = tpl_entry[4]
                x0 = int(f['bbox'][0])
                y0 = int(f['bbox'][1])
                w0 = int(f['bbox'][2])
                h0 = int(f['bbox'][3])
                crop = image_bgr[y0:y0 + h0, x0:x0 + w0]
                # ✅ Очистить crop от белого фона перед сохранением
                try:
                    crop_clean = crop.copy()
                    crop_hsv = cv2.cvtColor(crop_clean, cv2.COLOR_BGR2HSV)
                    white_crop = ((crop_hsv[:, :, 2] > 245) & (crop_hsv[:, :, 1] < 15))
                    non_white_crop = (~white_crop).astype('uint8') * 255
                    if non_white_crop.sum() > 0:
                        mean_crop = cv2.mean(crop_clean, mask=non_white_crop)[:3]
                        crop_clean[white_crop] = mean_crop
                    crop = crop_clean
                except Exception:
                    pass
                screen_env = os.environ.get('STRICT_SCREEN_NAME', 'unknown')
                debug_crop_logger.save_step(screen_env, tpl_k, 'tpl_orig', tpl_bgr)
                try:
                    # Обрезать template до маски перед ресайзом для сохранения
                    try:
                        if tpl_alpha is not None:
                            mask_bbox = tpl_alpha
                        else:
                            mask_bbox = tpl_entry[3]
                        ys, xs = np.where(mask_bbox > 0)
                        if len(xs) > 0:
                            x0_t, x1_t = int(xs.min()), int(xs.max()) + 1
                            y0_t, y1_t = int(ys.min()), int(ys.max()) + 1
                            tpl_bgr_trim = tpl_bgr[y0_t:y1_t, x0_t:x1_t]
                        else:
                            tpl_bgr_trim = tpl_bgr
                        tpl_res_save = cv2.resize(tpl_bgr_trim, (max(1, w0), max(1, h0)), interpolation=cv2.INTER_AREA)
                        # clean white in template for saved preview
                        try:
                            tpl_hsv = cv2.cvtColor(tpl_res_save, cv2.COLOR_BGR2HSV)
                            white_tpl = ((tpl_hsv[:, :, 2] > 245) & (tpl_hsv[:, :, 1] < 15)).astype('uint8')
                            non_white_tpl = (white_tpl == 0).astype('uint8') * 255
                            if non_white_tpl.sum() > 0:
                                mean_tpl = cv2.mean(tpl_res_save, mask=non_white_tpl)[:3]
                                tpl_res_save[white_tpl > 0] = mean_tpl
                        except Exception:
                            pass
                    except Exception:
                        tpl_res_save = tpl_bgr
                    debug_crop_logger.save_step(screen_env, tpl_k, 'tpl_resized', tpl_res_save)
                except Exception:
                    debug_crop_logger.save_step(screen_env, tpl_k, 'tpl_resized', tpl_bgr)
                debug_crop_logger.save_step(screen_env, tpl_k, 'crop', crop)
                ov = debug_crop_logger.make_overlay(tpl_bgr if tpl_bgr is not None else np.zeros((1, 1, 3), dtype=np.uint8), crop, tpl_alpha)
                if ov is not None:
                    debug_crop_logger.save_step(screen_env, tpl_k, 'overlay', ov)
                # include additional metrics and thresholds for debugging
                try:
                    hist_corr_v = float(f.get('hist_corr', 0.0))
                except Exception:
                    hist_corr_v = 0.0
                try:
                    masked_corr_v = float(f.get('masked_corr', 0.0))
                except Exception:
                    masked_corr_v = 0.0
                try:
                    edge_score_v = float(f.get('edge_score', 0.0))
                except Exception:
                    edge_score_v = 0.0
                meta = {
                    'combined': combined_v,
                    'masked_agreement': masked_v,
                    'gray': gray_v,
                    'hist_corr': hist_corr_v,
                    'edge_score': edge_score_v,
                    'masked_corr': masked_corr_v,
                    'bbox': [x0, y0, w0, h0],
                    'thresholds': {
                        'min_gray': float(cfg_min_gray),
                        'min_combined': float(cfg_min_combined_final)
                    },
                    'passed_gray': bool(float(gray_v) >= float(cfg_min_gray)),
                    'passed_combined': bool(float(combined_v) >= float(cfg_min_combined_final))
                }
                debug_crop_logger.save_metadata(screen_env, tpl_k, meta)
        except Exception:
            pass
    # targeted local refinement for problematic templates
    try:
        # determine caller-provided image context: attempt to find screen_name from environment
        # but _postfilter_matches receives image_bgr; we need templates and possibly screen_name
        # The run_on_image passes candidate_boxes but not screen_name; higher-level caller will
        # call _apply_targeted_local_refine separately when available. For safety, try to read
        # SCREEN_NAME from environment variable set by run_on_image (we set it before calling)
        screen_name = os.environ.get('STRICT_SCREEN_NAME')
        out = _apply_targeted_local_refine(
            out,
            image_bgr,
            templates,
            screen_name,
            use_overlay=use_overlay,
            overlay_threshold=overlay_threshold,
            overlay_radius=overlay_radius,
            overlay_tol_x=overlay_tol_x,
            overlay_tol_y=overlay_tol_y,
            image_hsv=image_hsv,
            image_gray=image_gray,
        )
    except Exception:
        pass
    # write reject reasons for debugging if requested
    try:
        if out_dir and screen_name and reject_reasons:
            os.makedirs(out_dir, exist_ok=True)
            rrp = os.path.join(out_dir, f"strict_reject_reasons_{screen_name}.json")
            try:
                with open(rrp, 'w', encoding='utf-8') as rf:
                    json.dump(reject_reasons, rf, ensure_ascii=False, indent=2)
            except Exception:
                pass
    except Exception:
        pass
    return out


def _apply_targeted_local_refine(out, image_bgr, templates, screen_name, use_overlay: bool = False, overlay_threshold: float = 0.8, overlay_radius: int = 60, overlay_tol_x: int = 50, overlay_tol_y: int = 100, image_hsv=None, image_gray=None):
    """Try to recover specific problematic templates (like Зелёные_ягоды_icon)
    by running a localized template search around expected coords and inserting
    the best candidate if it passes a relaxed combined proxy threshold.
    This is intentionally narrow: only affects templates present in EXPECTED_COORDS
    for the given screen_name.
    """
    if screen_name not in EXPECTED_COORDS:
        return out
    tpl_dict = {t[0]: t for t in templates}
    added = []
    for tpl_key, coord in EXPECTED_COORDS[screen_name].items():
        # skip if already present in out
        if any(x.get('tpl_key') == tpl_key for x in out):
            continue
        if tpl_key not in tpl_dict:
            continue
        tpl_entry = tpl_dict[tpl_key]
        tpl_bgr = tpl_entry[1]
        tpl_alpha = tpl_entry[4]
        # run local refine
        res = _local_refine_search(image_bgr, tpl_bgr, tpl_alpha, coord[0], coord[1], search_radius=160, image_hsv=image_hsv)
        accepted_candidate = None
        if res:
            x, y, w, h, agree, masked_corr, combined_proxy = res
            # only accept if combined_proxy reasonably high and masked_corr non-negative
            if combined_proxy >= 0.65 and masked_corr >= 0.0:
                accepted_candidate = {
                    'tpl_key': tpl_key,
                    'combined': combined_proxy,
                    'masked_agreement': agree,
                    'gray': 0.0,
                    'x': int(x),
                    'y': int(y),
                    'w': int(w),
                    'h': int(h),
                    'detection_method': 'local_refine'
                }
        # if not accepted via local_refine and overlay is enabled, try overlay_local_refine
        if accepted_candidate is None and use_overlay and overlay_local_refine is not None:
            try:
                ov = overlay_local_refine(image_bgr, tpl_bgr, tpl_alpha, coord[0], coord[1], search_radius=overlay_radius, scales=(0.9, 1.0, 1.1), step=4)
                if ov:
                    ox, oy, ow, oh, oagree, oinv, ocombined = ov
                    # oinv is inverse-weighted score (1 - weighted); accept when confidence high and within tolerance
                    if abs(int(ox) - int(coord[0])) <= overlay_tol_x and abs(int(oy) - int(coord[1])) <= overlay_tol_y and float(oinv) >= float(overlay_threshold):
                        accepted_candidate = {
                            'tpl_key': tpl_key,
                            'combined': float(oinv),
                            'masked_agreement': float(oagree),
                            'gray': 0.0,
                            'x': int(ox),
                            'y': int(oy),
                            'w': int(ow),
                            'h': int(oh),
                            'detection_method': 'overlay'
                        }
            except Exception:
                pass
        if accepted_candidate is not None:
            # ensure method field exists (local_refine or overlay)
            if 'detection_method' not in accepted_candidate:
                accepted_candidate['detection_method'] = 'local_refine'
            added.append(accepted_candidate)
    if added:
        # merge added entries into out (avoid duplicates)
        keys = {x.get('tpl_key') for x in out}
        for a in added:
            if a['tpl_key'] not in keys:
                out.append(a)
    return out


def _orb_verify(tpl_img, crop_img, min_matches=10):
    """Quick ORB feature check: return True if enough good matches found.

    tpl_img, crop_img are BGR images (numpy arrays). We convert to gray,
    detect ORB features and use BFMatcher with Hamming distance.
    """
    try:
        tpl_gray = cv2.cvtColor(tpl_img, cv2.COLOR_BGR2GRAY)
        crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(500)
        kp1, des1 = orb.detectAndCompute(tpl_gray, None)
        kp2, des2 = orb.detectAndCompute(crop_gray, None)
        if des1 is None or des2 is None:
            return False
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        # count good matches by distance threshold
        good = [m for m in matches if m.distance < 60]
        return len(good) >= int(min_matches)
    except Exception:
        return False


def _restore_strict_matches_from_report(report_path: str, screen_name: str, current_out: list, threshold: float = 0.5):
    """Restore matches from a strict verifier report file for given screen_name.

    This function is defensive: it accepts a few possible report layouts and only
    reinstates entries whose `masked_agreement` (or equivalent) >= threshold and
    which are not already present in current_out.
    """
    if not report_path or not os.path.exists(report_path):
        return current_out
    try:
        with open(report_path, 'r', encoding='utf-8') as rf:
            rep = json.load(rf)
    except Exception:
        return current_out

    # discover candidate list for screen_name in various possible layouts
    candidates = []
    if isinstance(rep, dict):
        if screen_name in rep and isinstance(rep[screen_name], list):
            candidates = rep[screen_name]
        elif 'results' in rep and isinstance(rep['results'], dict) and screen_name in rep['results']:
            candidates = rep['results'][screen_name]
        elif 'screens' in rep and isinstance(rep['screens'], dict) and screen_name in rep['screens']:
            candidates = rep['screens'][screen_name]
        else:
            # perhaps report is a flat list under 'items' or top-level list
            if isinstance(rep.get('items'), list):
                candidates = rep.get('items')
            elif isinstance(rep.get('entries'), list):
                candidates = rep.get('entries')
            elif isinstance(rep, list):
                candidates = rep

    if not candidates:
        # try fallback to per-screen strict_report_{screen_name}.json if available
        try:
            fallback = os.path.join(os.path.dirname(__file__), '..', 'debug', 'out_icons_debug', f'strict_report_{screen_name}.json')
            if os.path.exists(fallback):
                with open(fallback, 'r', encoding='utf-8') as rf:
                    rep2 = json.load(rf)
                    # try to extract filtered_matches or matches
                    if isinstance(rep2, dict) and 'filtered_matches' in rep2 and isinstance(rep2['filtered_matches'], list):
                        candidates = rep2['filtered_matches']
        except Exception:
            pass
        if not candidates:
            return current_out

    existing_keys = {c.get('tpl_key') or c.get('tpl') or c.get('template') for c in current_out}
    added = []
    for e in candidates:
        try:
            # find screen-specific entry by checking if this entry belongs to our screen
            # some reports include 'screen' or 'image' fields
            if isinstance(e, dict) and e.get('screen') and e.get('screen') != screen_name:
                continue
            tpl_key = e.get('tpl_key') or e.get('tpl') or e.get('template') or e.get('tplname')
            if not tpl_key:
                continue
            if tpl_key in existing_keys:
                continue
            ma = float(e.get('masked_agreement', e.get('masked_agree', e.get('masked', 0)) or 0) or 0)
            if ma < float(threshold):
                continue
            # bbox handling: prefer 'bbox' then x,y,w,h
            bbox = None
            if 'bbox' in e and isinstance(e['bbox'], (list, tuple)) and len(e['bbox']) >= 4:
                bbox = e['bbox']
            else:
                try:
                    bx = int(e.get('x', 0))
                    by = int(e.get('y', 0))
                    bw = int(e.get('w', e.get('width', 0) or 0))
                    bh = int(e.get('h', e.get('height', 0) or 0))
                    bbox = [bx, by, bw, bh]
                except Exception:
                    bbox = None
            if bbox is None:
                continue
            combined = float(e.get('combined', e.get('combined_proxy', 0) or 0) or 0)
            added.append({
                'tpl_key': tpl_key,
                'combined': combined,
                'masked_agreement': round(ma, 3),
                'gray': float(e.get('gray', 0) or 0),
                'x': int(bbox[0]),
                'y': int(bbox[1]),
                'w': int(bbox[2]),
                'h': int(bbox[3]),
            })
        except Exception:
            continue

    if added:
        # append but keep deterministic order (existing first, then added)
        return current_out + added
    return current_out


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--tpl-root', default=r"C:\bot\tpl")
    p.add_argument('--tpl-group', default='иконки_предметов')
    p.add_argument('--image')
    p.add_argument('--out', default=r"C:\bot\debug\out_icons_debug")
    p.add_argument('--max-templates', type=int, default=50)
    p.add_argument('--tpl-names', default=None, help='Comma-separated template keys to always include (e.g. "Янтарь_icon,Апельсин_icon")')
    p.add_argument('--limit', type=int, default=19, help='Total number of templates to use (including mandatory ones)')
    p.add_argument('--min-combined', type=float, default=0.80, help='Min combined threshold passed to postfilter')
    p.add_argument('--color-weight', type=float, default=0.7, help='Color weight (not used by postfilter but kept for parity)')
    p.add_argument('--find-all', action='store_true', help='Return all CSV matches without postfiltering')
    p.add_argument('--use-candidates', action='store_true', help='Restrict matches to areas likely containing icons (fast pre-extraction)')
    p.add_argument('--expected-only', action='store_true', help='Only return expected templates (use local refinement to recover missing ones)')
    p.add_argument('--show-templates', action='store_true', help='Print selected templates and exit')
    p.add_argument('--strict-verify', action='store_true', help='Run 1:1 strict verification for EXPECTED_COORDS after scanning')
    p.add_argument('--restore-strict', action='store_true', help='Restore matches from strict_verifier report into final output')
    p.add_argument('--restore-report', default=os.path.join(r"C:\bot\debug\out_icons_debug", 'strict_verification_report.json'), help='Path to strict verification report for restoration')
    p.add_argument('--restore-threshold', type=float, default=0.5, help='Masked agreement threshold to accept restored strict matches')
    p.add_argument('--pad-frac', type=float, default=0.35, help='Pad fraction used when intersecting candidate boxes with detections')
    p.add_argument('--no-trim', action='store_true', help='Do not trim templates to mask bbox; keep original background')
    p.add_argument('--use-overlay-verify', action='store_true', help='Use overlay_local_refine to verify/recover expected templates')
    p.add_argument('--overlay-threshold', type=float, default=0.80, help='Overlay acceptance confidence threshold')
    p.add_argument('--overlay-radius', type=int, default=60, help='Overlay local search radius (pixels)')
    p.add_argument('--overlay-tol-x', type=int, default=50, help='Overlay acceptance tolerance in X (pixels)')
    p.add_argument('--overlay-tol-y', type=int, default=100, help='Overlay acceptance tolerance in Y (pixels)')
    p.add_argument('--debug-all-matches', action='store_true', help='Save per-CSV-row debug crops/masks/heatmaps')
    args = p.parse_args()

    trim = not getattr(args, 'no_trim', False)
    tpls = load_icon_templates(args.tpl_root, args.tpl_group, max_templates=args.max_templates, trim=trim)
    # if tpl-names provided, parse and select templates ensuring mandatory inclusion
    tpl_names = []
    if args.tpl_names:
        tpl_names = [x.strip() for x in args.tpl_names.split(',') if x.strip()]
    # always ensure these mandatory templates are present
    mandatory = ['Янтарь_icon', 'Зелёные_ягоды_icon']
    # merge user-specified and mandatory, preserving order and uniqueness
    merged_keys = []
    for k in (tpl_names + mandatory):
        if k and k not in merged_keys:
            merged_keys.append(k)
    # if any mandatory key is missing from loaded templates (because of max_templates limit),
    # reload the full group so we can include them
    loaded_keys = {t[0] for t in tpls}
    need_full_reload = any(mk for mk in mandatory if mk not in loaded_keys)
    if need_full_reload:
        print("Mandatory template missing in truncated list, reloading full template set to ensure inclusion")
        tpls = load_icon_templates(args.tpl_root, args.tpl_group, max_templates=None)
    tpls_sel = select_templates(tpls, include_keys=merged_keys, limit=args.limit)
    print(f"Using {len(tpls_sel)} templates (limit={args.limit})")
    if getattr(args, 'show_templates', False):
        print("Selected templates:")
        for t in tpls_sel:
            print(t[0])
        sys.exit(0)
    if args.image:
        # optionally extract icon candidate boxes and pass to postfilter
        use_cand = getattr(args, 'use_candidates', False)
        if use_cand:
            img_full = _imread(args.image, unchanged=True)
            cand_boxes = extract_icon_candidates(img_full)
        else:
            cand_boxes = None
        run_on_image(
            args.image,
            tpls_sel,
            args.out,
            debug_all_matches=getattr(args, 'debug_all_matches', False),
            min_combined=args.min_combined,
            color_weight=args.color_weight,
            find_all=getattr(args, 'find_all', False),
            # pass candidate boxes for stricter filtering
            use_candidates=use_cand,
            candidate_boxes=cand_boxes,
            restore_strict=getattr(args, 'restore_strict', False),
            restore_report_path=getattr(args, 'restore_report', None),
            restore_threshold=getattr(args, 'restore_threshold', 0.5),
            pad_frac=getattr(args, 'pad_frac', 0.35),
            use_overlay_verify=getattr(args, 'use_overlay_verify', False),
            overlay_threshold=getattr(args, 'overlay_threshold', 0.80),
            overlay_radius=getattr(args, 'overlay_radius', 60),
            overlay_tol_x=getattr(args, 'overlay_tol_x', 50),
            overlay_tol_y=getattr(args, 'overlay_tol_y', 100),
        )
    else:
        print('No image specified. Use --image <path> to run on one image.')
