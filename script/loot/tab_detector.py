# tab_detector.py — определение состояния вкладок (вещи / монстры)
import random

import cv2
import numpy as np

from script.loot.utils import (
    BASE_DIR,
    clamp,
    imread_u8,
    match_scaled,
    scharr_mag,
    to_gray,
    ui_tpl_path,
    get_thr,
)

# ── Пути шаблонов ────────────────────────────────────────────────────
TPL_FIGHTS_LABEL = ui_tpl_path("fights_tab")
TPL_ITEMS_LABEL = ui_tpl_path("items_tab")
TPL_MONSTERS_LABEL = ui_tpl_path("monsters_tab")
TPL_SYSCHAT = ui_tpl_path("syschat")

EXTRA_LABELS = {"fights": TPL_FIGHTS_LABEL}

CHEVRONS = {
    "вещи": {"open": ui_tpl_path("items_open"), "close": ui_tpl_path("items_closed")},
    "монстры": {"open": ui_tpl_path("monsters_open"), "close": ui_tpl_path("monsters_closed")},
}

# ── Константы ────────────────────────────────────────────────────────
TOLERANCE_Y = 72
BAND_MIN_HEIGHT = 40
BAND_Y_RATIO = 0.1
CHEVRON_THRESHOLD = get_thr("TAB_CHEVRON", 0.86)
CHEVRON_VERIFY_THRESHOLD = get_thr("TAB_CHEVRON_VERIFY", 0.82)
LABEL_THRESHOLD = get_thr("TAB_LABEL", 0.86)
LABEL_THRESHOLD_FALLBACK = get_thr("TAB_LABEL_FALLBACK", 0.83)
SCALES = [0.9, 0.95, 1.0, 1.05, 1.1]
CHEVRON_SEARCH_X = 780, 1080
CHEVRON_SEARCH_Y_PAD = 120


# ── Вспомогательные ──────────────────────────────────────────────────

def jitter(val, spread=4):
    return val + random.randint(-spread, spread)


def _load_bgr(path: str):
    return imread_u8(path, cv2.IMREAD_COLOR) if path else None


def _match(gray_img, tpl_bgr, threshold):
    """match_scaled обёртка: принимает BGR шаблон, конвертит в gray."""
    if tpl_bgr is None:
        return False, 0.0, (0, 0, 0, 0)
    return match_scaled(gray_img, to_gray(tpl_bgr), threshold, SCALES)


def _match_any(gray_img, tpl_bgr_list, threshold):
    """Лучший матч среди списка BGR шаблонов."""
    best = (False, 0.0, (0, 0, 0, 0), None)
    for idx, tpl_bgr in enumerate(tpl_bgr_list):
        if tpl_bgr is None:
            continue
        found, score, box = _match(gray_img, tpl_bgr, threshold)
        if score > best[1]:
            best = (found, score, box, idx)
    return best


def _detect_and_mask_syschat(bgr):
    gray = to_gray(bgr)
    tpl = _load_bgr(TPL_SYSCHAT)
    if tpl is None:
        return gray, None, 0.0
    found, score, (x, y, w, h) = _match(gray, tpl, get_thr("SYSCHAT", 0.85))
    if not found:
        return gray, None, score
    masked = bgr.copy()
    y1 = max(0, y - 8)
    y2 = min(bgr.shape[0], y + h + 8)
    roi = masked[y1:y2, :]
    roi = cv2.GaussianBlur(roi, (7, 7), 0)
    roi = cv2.addWeighted(roi, 0.7, np.zeros_like(roi), 0.3, 0)
    masked[y1:y2, :] = roi
    return to_gray(masked), (x, y, w, h), score


def _match_chevron_banded(gray, tpl_bgr, band_y, threshold):
    if tpl_bgr is None:
        return False, 0.0, (0, 0, 0, 0)
    x1, x2 = CHEVRON_SEARCH_X
    y1 = max(0, band_y - CHEVRON_SEARCH_Y_PAD)
    y2 = min(gray.shape[0], band_y + CHEVRON_SEARCH_Y_PAD)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return False, 0.0, (0, 0, 0, 0)
    found, score, (x, y, w, h) = _match(roi, tpl_bgr, threshold)
    if found:
        return True, score, (x1 + x, y1 + y, w, h)
    return _match(gray, tpl_bgr, threshold)


# ── Основная функция ─────────────────────────────────────────────────

def detect_tab_states(bgr: np.ndarray):
    """
    Возвращает для 'вещи' и 'монстры':
      - state: 'открыта'/'закрыта'/'не видно'/'не определено'/'стрелка уехала'
      - label_found, label_box, band_y, scores, arrows
    """
    gray = to_gray(bgr)
    H, W = gray.shape[:2]
    band_y_default = int(H * BAND_Y_RATIO)
    sys_gray, sys_box, _ = _detect_and_mask_syschat(bgr)

    # Detect fights tab occlusion ONCE
    occluded_by_extra = False
    tpl_fights = _load_bgr(TPL_FIGHTS_LABEL)
    if tpl_fights is not None:
        fights_found, fights_score, _ = _match(gray, tpl_fights, LABEL_THRESHOLD_FALLBACK)
        if fights_found:
            occluded_by_extra = True
            print(f"[TABS] Обнаружена вкладка 'Сражения', score={fights_score:.3f}")

    out = {"occluded_by_extra": occluded_by_extra}

    for label in ("вещи", "монстры"):
        label_tpl = _load_bgr(TPL_ITEMS_LABEL if label == "вещи" else TPL_MONSTERS_LABEL)

        lbl_found, lbl_score, (lx, ly, lw, lh) = _match(sys_gray, label_tpl, LABEL_THRESHOLD)
        if not lbl_found:
            lbl_found, lbl_score, (lx, ly, lw, lh) = _match(gray, label_tpl, LABEL_THRESHOLD_FALLBACK)
        if not lbl_found and label_tpl is not None:
            gg = scharr_mag(gray)
            tpl_g = scharr_mag(to_gray(label_tpl))
            found_g, score_g, (gx, gy, gw, gh) = match_scaled(gg, tpl_g, LABEL_THRESHOLD_FALLBACK, SCALES)
            if found_g:
                lbl_found, lbl_score, (lx, ly, lw, lh) = True, score_g, (gx, gy, gw, gh)

        label_box = (lx, ly, lw, lh) if lbl_found else None
        band_y = ly + lh // 2 if lbl_found else band_y_default

        ch_open = _load_bgr(CHEVRONS[label]["open"])
        ch_close = _load_bgr(CHEVRONS[label]["close"])
        open_found, open_score, (ox, oy, ow, oh) = _match_chevron_banded(gray, ch_open, band_y, CHEVRON_THRESHOLD)
        close_found, close_score, (cx, cy, cw, chh) = _match_chevron_banded(gray, ch_close, band_y, CHEVRON_THRESHOLD)

        if not lbl_found:
            if open_found and (not close_found or open_score >= close_score):
                band_y = oy + oh // 2
            elif close_found:
                band_y = cy + chh // 2

        up_present = bool(open_found)
        down_present = bool(close_found)
        up_y = oy + oh // 2 if open_found else None
        down_y = cy + chh // 2 if close_found else None
        up_in = bool(up_present and up_y is not None and abs(up_y - band_y) <= TOLERANCE_Y)
        down_in = bool(down_present and down_y is not None and abs(down_y - band_y) <= TOLERANCE_Y)

        if not lbl_found and not (open_found or close_found):
            state = "не видно"
        elif up_in and not down_in:
            state = "открыта"
        elif down_in and not up_in:
            state = "закрыта"
        elif up_in and down_in:
            # Если оба найдены с похожими счетами в одном месте — это скорее всего ошибка шаблона.
            # Требуем разницу хотя бы в 0.02 или используем лучший.
            diff = abs(open_score - close_score)
            if diff < 0.02 and label == "монстры":
                # Фолбэк для монстров: если разница ничтожна, считаем закрытым (безопаснее) 
                # или пробуем найти более строгий признак.
                state = "закрыта" if close_score > open_score else "открыта"
            else:
                state = "открыта" if open_score >= close_score else "закрыта"
        elif (open_found or close_found) and not (up_in or down_in):

            state = "стрелка уехала"
        else:
            state = "не определено"

        out[label] = {
            "state": state,
            "label_found": bool(lbl_found),
            "label_box": label_box,
            "band_y": int(band_y),
            "scores": {
                "label": round(float(lbl_score), 3),
                "up": round(float(open_score), 3),
                "down": round(float(close_score), 3),
            },
            "arrows": {
                "up": {"present": up_present, "x": None, "y": up_y, "in_band": up_in},
                "down": {"present": down_present, "x": None, "y": down_y, "in_band": down_in},
            },
        }
    return out
