import os

import cv2
import numpy as np

TPL_DIR_MY = "C:/bot/tpl/my"
TPL_DIR_CHEVRONS = "C:/bot/tpl/chevrons"
TPL_ITEMS_LABEL = ui_tpl_path("items_tab.png")
TPL_MONSTERS_LABEL = ui_tpl_path("monsters_tab.png")
CHEVRONS = {
    "вещи": {
        "open": os.path.join(TPL_DIR_CHEVRONS, "items_open.png"),
        "close": os.path.join(TPL_DIR_CHEVRONS, "items_closed.png"),
    },
    "монстры": {
        "open": os.path.join(TPL_DIR_CHEVRONS, "monsters_open.png"),
        "close": os.path.join(TPL_DIR_CHEVRONS, "monsters_closed.png"),
    },
}
TPL_SYSCHAT = ui_tpl_path("syschat.png")
TOLERANCE_Y = 72
BAND_MIN_HEIGHT = 40
BAND_Y_RATIO = 0.1
CHEVRON_THRESHOLD = 0.86
CHEVRON_VERIFY_THRESHOLD = 0.82
LABEL_THRESHOLD = 0.86
LABEL_THRESHOLD_FALLBACK = 0.83
SCALES = [0.9, 0.95, 1.0, 1.05, 1.1]
CHEVRON_SEARCH_X = 780, 1080
CHEVRON_SEARCH_Y_PAD = 120


def imread_u8(path: str, flags: int = cv2.IMREAD_COLOR):
    try:
        with open(path, "rb") as f:
            data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, flags)
        return img
    except Exception as e:
        print(f"[IMG] read fail: {path} // {e}")
        return None


def _load_bgr(path: str):
    return imread_u8(path, cv2.IMREAD_COLOR) if path else None


def _to_gray(img: np.ndarray) -> np.ndarray:
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 5, 35, 35)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    return g


def _scharr_mag(gray: np.ndarray) -> np.ndarray:
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _match_scaled(gray_img: np.ndarray, tpl_bgr, threshold: float):
    if tpl_bgr is None:
        return False, 0.0, (0, 0, 0, 0)
    best = False, 0.0, (0, 0, 0, 0)
    tpl_gray_full = _to_gray(tpl_bgr)
    th, tw = tpl_gray_full.shape[:2]
    H, W = gray_img.shape[:2]
    for s in SCALES:
        tws, ths = max(5, int(tw * s)), max(5, int(th * s))
        if H < ths or W < tws:
            continue
        tpl_gray = cv2.resize(tpl_gray_full, (tws, ths), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC)
        res = cv2.matchTemplate(gray_img, tpl_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best[1]:
            best = max_val >= threshold, float(max_val), (int(max_loc[0]), int(max_loc[1]), int(tws), int(ths))
    return best


def _match_scaled_any(gray_img: np.ndarray, tpl_bgr_list, threshold: float):
    best = False, 0.0, (0, 0, 0, 0), None
    H, W = gray_img.shape[:2]
    for idx, tpl_bgr in enumerate(tpl_bgr_list):
        if tpl_bgr is None:
            continue
        tpl_gray_full = _to_gray(tpl_bgr)
        th, tw = tpl_gray_full.shape[:2]
        for s in SCALES:
            tws, ths = max(5, int(tw * s)), max(5, int(th * s))
            if H < ths or W < tws:
                continue
            tpl_gray = cv2.resize(
                tpl_gray_full, (tws, ths), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
            )
            res = cv2.matchTemplate(gray_img, tpl_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best[1]:
                best = max_val >= threshold, float(max_val), (int(max_loc[0]), int(max_loc[1]), int(tws), int(ths)), idx
    return best


def _detect_and_mask_syschat(bgr: np.ndarray):
    gray = _to_gray(bgr)
    tpl = _load_bgr(TPL_SYSCHAT)
    if tpl is None:
        return gray, None, 0.0
    found, score, (x, y, w, h) = _match_scaled(gray, tpl, 0.85)
    if not found:
        return gray, None, score
    masked = bgr.copy()
    y1, y2 = max(0, y - 8), min(bgr.shape[0], y + h + 8)
    roi = masked[y1:y2, :]
    roi = cv2.GaussianBlur(roi, (7, 7), 0)
    roi = cv2.addWeighted(roi, 0.7, np.zeros_like(roi), 0.3, 0)
    masked[y1:y2, :] = roi
    return _to_gray(masked), (x, y, w, h), score


def _match_chevron_banded(gray: np.ndarray, tpl_bgr, band_y: int, threshold: float):
    if tpl_bgr is None:
        return False, 0.0, (0, 0, 0, 0)
    x1, x2 = CHEVRON_SEARCH_X
    y1 = max(0, band_y - CHEVRON_SEARCH_Y_PAD)
    y2 = min(gray.shape[0], band_y + CHEVRON_SEARCH_Y_PAD)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return False, 0.0, (0, 0, 0, 0)
    found, score, (x, y, w, h) = _match_scaled(roi, tpl_bgr, threshold)
    if found:
        return True, score, (x1 + x, y1 + y, w, h)
    return _match_scaled(gray, tpl_bgr, threshold)


def detect_tab_states(bgr: np.ndarray):
    """
    Возвращает для 'вещи' и 'монстры':
      - state: 'открыта'/'закрыта'/'не видно'/'не определено'/'стрелка уехала'
      - label_found: найден ли лейбл вкладки (для клика)
      - label_box: (x,y,w,h) лейбла или None
      - band_y: Y центра полосы вкладки (по лейблу или по лучшему chevron)
      - scores: {label, up, down} — up=open, down=close
      - arrows: совместимость со старым форматом
    """
    gray = _to_gray(bgr)
    H, W = gray.shape[:2]
    band_y_default = int(H * BAND_Y_RATIO)
    sys_gray, sys_box, _ = _detect_and_mask_syschat(bgr)
    out = {}
    for label in ("вещи", "монстры"):
        label_tpl = _load_bgr(TPL_ITEMS_LABEL if label == "вещи" else TPL_MONSTERS_LABEL)
        lbl_found, lbl_score, (lx, ly, lw, lh) = _match_scaled(sys_gray, label_tpl, LABEL_THRESHOLD)
        if not lbl_found:
            lbl_found, lbl_score, (lx, ly, lw, lh) = _match_scaled(gray, label_tpl, LABEL_THRESHOLD_FALLBACK)
        if not lbl_found and label_tpl is not None:
            gg = _scharr_mag(gray)
            tpl_g = _scharr_mag(_to_gray(label_tpl))
            found_g, score_g, (gx, gy, gw, gh) = _match_scaled(gg, tpl_g, LABEL_THRESHOLD_FALLBACK)
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
