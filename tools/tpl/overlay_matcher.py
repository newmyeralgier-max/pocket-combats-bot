# -*- coding: utf-8 -*-
"""
overlay_matcher.py

Прототип: локальный поиск на основе "наложения" (alpha-aware) — для каждого ожидаемого
шаблона пробуем наложить шаблон (с alpha) на участок экрана и измеряем среднюю абсолютную
ошибку внутри маски. Это помогает избежать неправильной обрезки шаблонов и служит
альтернативой matchTemplate+mask.

Запуск: из корня репозитория
    python tools/tpl/overlay_matcher.py

Сохраняет простой JSON-отчёт в debug/out_icons_debug/overlay_report_{screen}.json
и печатает краткий свод.
"""
import os
import json
import math
import numpy as np
import cv2
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.tpl import run_strict_scanner as runner


def _imread(path: str):
    return runner._imread(path, unchanged=True)


def overlay_local_refine(image_bgr, tpl_bgr, tpl_alpha, center_x, center_y, search_radius=140, scales=(0.9, 1.0, 1.1), step=4):
    H, W = image_bgr.shape[:2]
    x0 = max(0, int(center_x - search_radius))
    y0 = max(0, int(center_y - search_radius))
    x1 = min(W, int(center_x + search_radius))
    y1 = min(H, int(center_y + search_radius))
    window = image_bgr[y0:y1, x0:x1]
    if window.size == 0:
        return None

    best = None
    # prepare alpha mask if missing
    has_alpha = tpl_alpha is not None and tpl_alpha.sum() > 0
    for s in scales:
        try:
            th = max(2, int(tpl_bgr.shape[1] * s)), max(2, int(tpl_bgr.shape[0] * s))
            tpl_rs = cv2.resize(tpl_bgr, th, interpolation=cv2.INTER_AREA)
            if has_alpha:
                mask_rs = cv2.resize(tpl_alpha, th, interpolation=cv2.INTER_NEAREST)
            else:
                # fallback: compute color mask
                mc = runner.make_color_mask(tpl_bgr)
                mask_rs = cv2.resize(mc, th, interpolation=cv2.INTER_NEAREST)
        except Exception:
            tpl_rs = tpl_bgr.copy()
            mask_rs = tpl_alpha.copy() if tpl_alpha is not None else runner.make_color_mask(tpl_bgr)

        h, w = tpl_rs.shape[:2]
        if window.shape[0] < h or window.shape[1] < w:
            continue

        # normalize mask to 0/1 float
        mask_f = (mask_rs > 0).astype('float32') / 255.0
        mask_sum = max(1.0, float(mask_f.sum()))

        # slide tpl over window with given step
        for py in range(0, window.shape[0] - h + 1, max(1, step)):
            for px in range(0, window.shape[1] - w + 1, max(1, step)):
                crop = window[py:py + h, px:px + w]
                # normalize crop channels to BGR
                if crop.ndim == 3 and crop.shape[2] == 4:
                    crop = crop[:, :, :3]
                if crop.ndim == 2:
                    crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
                # compute per-pixel absolute difference weighted by mask
                diff = np.abs(tpl_rs.astype('float32') - crop.astype('float32'))
                # weight by mask (only where tpl has content)
                # convert to single-channel average over color channels
                d_chan = diff.mean(axis=2)
                weighted = (d_chan * mask_f).sum() / mask_sum / 255.0
                # also compute masked agreement similar to existing code: count pixels within tolerance
                try:
                    tpl_hsv = cv2.cvtColor(tpl_rs, cv2.COLOR_BGR2HSV)
                    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    v_tpl = tpl_hsv[:, :, 2].astype(int)
                    v_crop = crop_hsv[:, :, 2].astype(int)
                    agree = float(((np.abs(v_tpl - v_crop) <= 30).astype('uint8') * (mask_f > 0)).sum()) / mask_sum
                except Exception:
                    agree = 0.0

                # lower error is better; compute similarity once and combine with agree
                similarity = 1.0 - weighted
                combined = similarity * 0.7 + float(agree) * 0.3
                # map back to full-image coords
                img_x = x0 + px
                img_y = y0 + py
                cand = (img_x, img_y, w, h, round(agree, 3), round(similarity, 3), round(combined, 3))
                if best is None or cand[6] > best[6]:
                    best = cand
    return best


def run_for_expected(tpl_root, tpl_group, out_dir, expected_path=None, screens=None):
    if expected_path is None:
        expected_path = os.path.join(HERE, '..', 'cfg', 'expected_coords.json')
    if not os.path.exists(expected_path):
        print('expected_coords.json not found:', expected_path)
        return
    with open(expected_path, 'r', encoding='utf-8') as f:
        expected = json.load(f)

    tpls = runner.load_icon_templates(tpl_root, tpl_group, max_templates=1000, trim=False)
    tpl_dict = {t[0]: t for t in tpls}

    if screens is None:
        screens = list(expected.keys())

    os.makedirs(out_dir, exist_ok=True)
    reports = {}
    for screen in screens:
        img_path = None
        # guess image path from back/screens
        candidate = os.path.join(PROJECT_ROOT, 'back', 'screens', f"{screen}.png")
        if os.path.exists(candidate):
            img_path = candidate
        else:
            # try without extension or direct path
            candidate2 = os.path.join(PROJECT_ROOT, 'back', 'screens', screen)
            if os.path.exists(candidate2):
                img_path = candidate2
        if not img_path:
            print('Screen image not found for', screen)
            continue
        img = _imread(img_path)
        H, W = img.shape[:2]
        reports[screen] = {'image': img_path, 'results': []}
        for tpl_key, coord in expected.get(screen, {}).items():
            if tpl_key not in tpl_dict:
                reports[screen]['results'].append({'tpl_key': tpl_key, 'error': 'tpl_missing'})
                continue
            tpl_entry = tpl_dict[tpl_key]
            tpl_bgr = tpl_entry[1]
            tpl_alpha = tpl_entry[4]
            res = overlay_local_refine(img, tpl_bgr, tpl_alpha, coord[0], coord[1], search_radius=220, scales=(0.85, 0.95, 1.0, 1.05, 1.15), step=4)
            if res:
                x, y, w, h, agree, inv_score, combined = res
                # ensure JSON-serializable native types
                reports[screen]['results'].append({
                    'tpl_key': tpl_key,
                    'x': int(x),
                    'y': int(y),
                    'w': int(w),
                    'h': int(h),
                    'agree': float(agree),
                    'overlay_confidence': float(inv_score),
                    'combined': float(combined)
                })
            else:
                reports[screen]['results'].append({'tpl_key': tpl_key, 'error': 'no_candidate'})
        # write report per-screen
        outp = os.path.join(out_dir, f'overlay_report_{screen}.json')
        with open(outp, 'w', encoding='utf-8') as of:
            json.dump(reports[screen], of, ensure_ascii=False, indent=2)
        print('Wrote', outp)

    # write aggregate report
    agg = os.path.join(out_dir, 'overlay_report_all.json')
    with open(agg, 'w', encoding='utf-8') as af:
        json.dump(reports, af, ensure_ascii=False, indent=2)
    print('Wrote aggregate report:', agg)
    return reports


if __name__ == '__main__':
    TPL_ROOT = os.path.join(PROJECT_ROOT, 'tpl')
    TPL_GROUP = 'иконки_предметов'
    OUT = os.path.join(PROJECT_ROOT, 'debug', 'out_icons_debug')
    run_for_expected(TPL_ROOT, TPL_GROUP, OUT)
