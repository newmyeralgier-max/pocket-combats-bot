import json
import math
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class CFG:
    TH_ARROW = 0.72
    TH_SLOT_SCORE = 0.012
    TM_METHOD = cv2.TM_CCOEFF_NORMED
    SLOTS_PER_PAGE = 6
    LAYOUT = "right"
    DEVICE_ID = None
    DRY_RUN = True
    SLEEP_AFTER_TAP = 0.25
    SLEEP_AFTER_SWIPE = 0.45
    SLEEP_AFTER_SCREENSHOT = 0.08
    TAB_ROI_FRAC = 0.7, 0.05, 0.98, 0.2
    SLOT_W_FRAC = 0.24
    SLOT_H_FRAC = 0.085
    SLOT_STEP_Y_FRAC = 0.098
    OFFSET_DY_FROM_ARROW_FRAC = 0.04
    BASE_Y_FRAC_FALLBACK = 0.26
    FINE_DX = 0
    FINE_DY = 0
    TPL_OPEN_PATH = "templates/open_tab.png"
    TPL_CLOSE_PATH = "templates/close_tab.png"
    DEBUG_DIR = "debug_loot"
    SAVE_DEBUG = False
    SWIPE_X_FRAC = 0.85
    SWIPE_Y1_FRAC = 0.78
    SWIPE_Y2_FRAC = 0.32
    SWIPE_DURATION_MS = 280


def ts() -> str:
    return time.strftime("%H:%M:%S")


def log_info(msg: str):
    print(f"[{ts()}][INFO] {msg}")


def log_warn(msg: str):
    print(f"[{ts()}][WARN] {msg}")


def log_err(msg: str):
    print(f"[{ts()}][ERR ] {msg}")


def imread_unicode(path: str, flags=cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Надежная загрузка изображения с русскими путями."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        return img
    except Exception as e:
        log_err(f"imread_unicode fail: {e} path={path}")
        return None


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ext = os.path.splitext(path)[1].lower()
        result, enc = cv2.imencode(ext if ext else ".png", img)
        if not result:
            log_err(f"imwrite failed for {path}")
            return False
        with open(path, "wb") as f:
            enc.tofile(f)
        return True
    except Exception as e:
        log_err(f"imwrite_unicode fail: {e}")
        return False


def clamp_rect(rect: Tuple[int, int, int, int], W: int, H: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = rect
    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(0, min(x2, W))
    y2 = max(0, min(y2, H))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def rect_center(rect: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = rect
    return (x1 + x2) // 2, (y1 + y2) // 2


def make_rect_from_xywh(x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
    return int(x), int(y), int(x + w), int(y + h)


def crop_roi(img: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = roi
    return img[y1:y2, x1:x2]


def adb_shell(args: List[str], device: Optional[str] = None) -> Tuple[int, bytes, bytes]:
    cmd = ["adb"]
    if device:
        cmd += ["-s", device]
    cmd += ["shell"] + args
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    return proc.returncode, out, err


def adb_exec_out(args: List[str], device: Optional[str] = None) -> Tuple[int, bytes, bytes]:
    cmd = ["adb"]
    if device:
        cmd += ["-s", device]
    cmd += ["exec-out"] + args
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    return proc.returncode, out, err


def send_tap(x: int, y: int, device: Optional[str] = CFG.DEVICE_ID):
    if CFG.DRY_RUN:
        log_info(f"TAP (dry) at ({x},{y})")
        return
    code, out, err = adb_shell(["input", "tap", str(x), str(y)], device=device)
    if code != 0:
        log_err(f"adb tap failed: {err.decode('utf-8', 'ignore')}")
    time.sleep(CFG.SLEEP_AFTER_TAP)


def send_swipe(
    x1: int, y1: int, x2: int, y2: int, duration_ms: int = CFG.SWIPE_DURATION_MS, device: Optional[str] = CFG.DEVICE_ID
):
    if CFG.DRY_RUN:
        log_info(f"SWIPE (dry) from ({x1},{y1}) to ({x2},{y2}) for {duration_ms}ms")
        time.sleep(CFG.SLEEP_AFTER_SWIPE)
        return
    code, out, err = adb_shell(["input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)], device=device)
    if code != 0:
        log_err(f"adb swipe failed: {err.decode('utf-8', 'ignore')}")
    time.sleep(CFG.SLEEP_AFTER_SWIPE)


def capture_screen(device: Optional[str] = CFG.DEVICE_ID) -> Optional[np.ndarray]:
    code, out, err = adb_exec_out(["screencap", "-p"], device=device)
    if code != 0 or not out:
        log_err(f"screencap failed: {err.decode('utf-8', 'ignore')}")
        return None
    arr = np.frombuffer(out, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        log_err("cv2.imdecode returned None for screencap")
        return None
    time.sleep(CFG.SLEEP_AFTER_SCREENSHOT)
    return img


def match_one_safe(
    img: np.ndarray, tpl: Optional[np.ndarray], roi: Optional[Tuple[int, int, int, int]], th: float
) -> Optional[Dict]:
    """
    Универсальный матч внутри ROI. Возвращает dict с rect, center, score.
    """
    if img is None or tpl is None:
        return None
    H, W = img.shape[:2]
    if roi is not None:
        roi = clamp_rect(roi, W, H)
        crop = crop_roi(img, roi)
        offx, offy = roi[0], roi[1]
    else:
        crop = img
        offx, offy = 0, 0
    h, w = tpl.shape[:2]
    if h <= 0 or w <= 0 or crop.shape[0] < h or crop.shape[1] < w:
        return None
    res = cv2.matchTemplate(crop, tpl, CFG.TM_METHOD)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    score = float(max_val)
    if score < th:
        return None
    x1 = offx + max_loc[0]
    y1 = offy + max_loc[1]
    x2 = x1 + w
    y2 = y1 + h
    rect = clamp_rect((x1, y1, x2, y2), W, H)
    cx, cy = rect_center(rect)
    return {"rect": rect, "center": (cx, cy), "score": score, "tpl_wh": (w, h), "roi_used": roi}


def detect_tab_state(
    img: np.ndarray,
    tpl_open: Optional[np.ndarray],
    tpl_close: Optional[np.ndarray],
    tab_roi: Optional[Tuple[int, int, int, int]],
    th: Optional[float] = None,
) -> Tuple[bool, Optional[Dict], str]:
    """
    Определяет состояние таба по стрелкам open/close в заданном ROI.
    Возвращает (no_arrows, best_match, label) где:
      - no_arrows: True если ни одна стрелка не найдена
      - best_match: dict матча или None
      - label: "OPEN_TAB", "CLOSE_TAB", "NO_ARROW"
    """
    if th is None:
        th = CFG.TH_ARROW
    m_open = match_one_safe(img, tpl_open, tab_roi, th) if tpl_open is not None else None
    m_close = match_one_safe(img, tpl_close, tab_roi, th) if tpl_close is not None else None

    def _score(m):
        return -1.0 if m is None else float(m.get("score", -1.0))

    s_open = _score(m_open)
    s_close = _score(m_close)
    best = None
    label = "NO_ARROW"
    if s_open >= 0 or s_close >= 0:
        if s_open >= s_close and s_open >= 0:
            best, label = m_open, "OPEN_TAB"
        elif s_close > s_open and s_close >= 0:
            best, label = m_close, "CLOSE_TAB"
    no_arrows = best is None
    return no_arrows, best, label


def compute_vertical_slots(
    img_shape: Tuple[int, int, int],
    arrow_match: Optional[Dict],
    tab_roi: Optional[Tuple[int, int, int, int]],
    layout: str = CFG.LAYOUT,
    count: int = CFG.SLOTS_PER_PAGE,
) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int]]]:
    """
    Строит список ROI слотов и центры тапов, используя фиксированный X.
    - Если найдена стрелка — якоримся по её центру X, а базовую Y берём от низа стрелки + offset.
    - Если стрелки нет — используем tab_roi (если дан) или фоллбек-фракции CFG.TAB_ROI_FRAC.
    """
    H, W = img_shape[:2]
    if arrow_match is not None and "rect" in arrow_match:
        ax, ay = rect_center(arrow_match["rect"])
        arrow_rect = arrow_match["rect"]
        base_y = int(arrow_rect[3] + CFG.OFFSET_DY_FROM_ARROW_FRAC * H) + int(CFG.FINE_DY)
        anchor_x = ax
    elif tab_roi is not None:
        rx, ry = rect_center(tab_roi)
        anchor_x = rx
        base_y = int(CFG.BASE_Y_FRAC_FALLBACK * H) + int(CFG.FINE_DY)
    else:
        x1f, y1f, x2f, y2f = CFG.TAB_ROI_FRAC
        x1 = int(x1f * W)
        y1 = int(y1f * H)
        x2 = int(x2f * W)
        y2 = int(y2f * H)
        rx, ry = rect_center((x1, y1, x2, y2))
        anchor_x = rx
        base_y = int(CFG.BASE_Y_FRAC_FALLBACK * H) + int(CFG.FINE_DY)
    slot_w = int(CFG.SLOT_W_FRAC * W)
    slot_h = int(CFG.SLOT_H_FRAC * H)
    step_y = int(CFG.SLOT_STEP_Y_FRAC * H)
    if layout == "right":
        x = int(anchor_x - slot_w * 0.55) + int(CFG.FINE_DX)
    else:
        x = int(anchor_x - slot_w * 0.45) + int(CFG.FINE_DX)
    rois = []
    taps = []
    for i in range(count):
        y = base_y + i * step_y
        rect = make_rect_from_xywh(x, y, slot_w, slot_h)
        rect = clamp_rect(rect, W, H)
        rois.append(rect)
        taps.append(rect_center(rect))
    return rois, taps


def slot_score_nonempty(img: np.ndarray, roi: Tuple[int, int, int, int]) -> float:
    """
    Простая метрика: комбинация дисперсии (Laplacian var) и доли границ (Canny).
    Возвращает небольшой положительный score. Чем выше — тем "сложнее" и вероятнее слот непустой.
    """
    crop = crop_roi(img, roi)
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var_lap = float(lap.var())
    v = np.median(gray)
    lo = max(0, int(0.66 * v))
    hi = min(255, int(1.33 * v))
    edges = cv2.Canny(gray, lo, hi)
    edge_ratio = float((edges > 0).sum()) / float(edges.size + 1e-06)
    norm_lap = math.log1p(var_lap) / 10.0
    score = norm_lap * 0.7 + edge_ratio * 0.3
    return score


def is_slot_lootable(
    img: np.ndarray, roi: Tuple[int, int, int, int], th: float = CFG.TH_SLOT_SCORE
) -> Tuple[bool, float]:
    """
    Универсальная проверка: если нет своих шаблонов, считаем слот "непустым", когда визуальная сложность > порога.
    Можно заменить на match нужной иконки/кнопки.
    """
    s = slot_score_nonempty(img, roi)
    return s >= th, s


def ensure_tab_open(
    img: np.ndarray,
    tpl_open: Optional[np.ndarray],
    tpl_close: Optional[np.ndarray],
    tab_roi: Optional[Tuple[int, int, int, int]],
) -> Tuple[np.ndarray, Optional[Dict], str]:
    """
    Если видим OPEN_TAB (иконка «открыть») — тапаем по стрелке, чтобы раскрыть.
    Если видим CLOSE_TAB — уже раскрыто.
    Если стрелки нет — продолжаем по fallback.
    Возвращает (актуальный скрин, best_match, label)
    """
    no_arrows, best, label = detect_tab_state(img, tpl_open, tpl_close, tab_roi, CFG.TH_ARROW)
    log_info(
        f"tab_state={label} score={None if not best else round(best['score'], 3)} rect={None if not best else best['rect']}"
    )
    if label == "OPEN_TAB" and best is not None:
        cx, cy = best["center"]
        log_info(f"tab closed → tap arrow to open at {best['center']}")
        send_tap(cx, cy)
        new_img = capture_screen()
        if new_img is not None:
            img = new_img
        time.sleep(0.05)
        no_arrows, best, label = detect_tab_state(img, tpl_open, tpl_close, tab_roi, CFG.TH_ARROW)
        log_info(f"post-open tab_state={label}")
    return img, best, label


def worker_once(
    img: np.ndarray,
    tpl_open: Optional[np.ndarray],
    tpl_close: Optional[np.ndarray],
    tab_roi: Optional[Tuple[int, int, int, int]],
    layout: str = CFG.LAYOUT,
    slots_per_page: int = CFG.SLOTS_PER_PAGE,
) -> Dict:
    """
    Один рабочий проход:
      1) Обеспечить открытое состояние таба
      2) Рассчитать вертикальные слоты
      3) Проверить слоты и при необходимости тапнуть по первому подходящему
    """
    img, best_arrow, label = ensure_tab_open(img, tpl_open, tpl_close, tab_roi)
    rois, taps = compute_vertical_slots(img.shape, best_arrow, tab_roi, layout=layout, count=slots_per_page)
    log_info(f"slots computed: {len(rois)}")
    taken = None
    slot_infos = []
    for idx, (roi, tap_pt) in enumerate(zip(rois, taps)):
        ok, score = is_slot_lootable(img, roi)
        slot_infos.append({"idx": idx, "roi": roi, "score": round(score, 5), "lootable": ok})
        log_info(f"slot#{idx}: score={round(score, 5)} lootable={ok} roi={roi}")
        if ok and taken is None:
            x, y = tap_pt
            log_info(f"action: tap slot#{idx} at ({x},{y})")
            send_tap(x, y)
            taken = {"idx": idx, "tap": (x, y)}
            break
    return {
        "tab_label": label,
        "arrow": None if best_arrow is None else {"rect": best_arrow["rect"], "score": best_arrow["score"]},
        "slots": slot_infos,
        "taken": taken,
    }


def load_templates() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    open_tpl = imread_unicode(CFG.TPL_OPEN_PATH)
    close_tpl = imread_unicode(CFG.TPL_CLOSE_PATH)
    if open_tpl is None:
        log_warn(f"tpl_open not loaded: {CFG.TPL_OPEN_PATH}")
    if close_tpl is None:
        log_warn(f"tpl_close not loaded: {CFG.TPL_CLOSE_PATH}")
    return open_tpl, close_tpl


def frac_roi_to_abs(
    img_shape: Tuple[int, int, int], frac_rect: Tuple[float, float, float, float]
) -> Tuple[int, int, int, int]:
    H, W = img_shape[:2]
    x1 = int(frac_rect[0] * W)
    y1 = int(frac_rect[1] * H)
    x2 = int(frac_rect[2] * W)
    y2 = int(frac_rect[3] * H)
    return clamp_rect((x1, y1, x2, y2), W, H)


def main_once_from_device():
    img = capture_screen()
    if img is None:
        log_err("No screen captured")
        return
    tpl_open, tpl_close = load_templates()
    tab_roi = frac_roi_to_abs(img.shape, CFG.TAB_ROI_FRAC)
    result = worker_once(img, tpl_open, tpl_close, tab_roi, layout=CFG.LAYOUT, slots_per_page=CFG.SLOTS_PER_PAGE)
    log_info("RESULT: " + json.dumps(result, ensure_ascii=False))
    if CFG.SAVE_DEBUG:
        dbg_path = os.path.join(CFG.DEBUG_DIR, f"frame_{int(time.time())}.png")
        imwrite_unicode(dbg_path, img)


def demo_from_file(image_path: str):
    img = imread_unicode(image_path)
    if img is None:
        log_err(f"Failed to read input image {image_path}")
        return
    tpl_open, tpl_close = load_templates()
    tab_roi = frac_roi_to_abs(img.shape, CFG.TAB_ROI_FRAC)
    result = worker_once(img, tpl_open, tpl_close, tab_roi, layout=CFG.LAYOUT, slots_per_page=CFG.SLOTS_PER_PAGE)
    log_info("RESULT: " + json.dumps(result, ensure_ascii=False))
    if CFG.SAVE_DEBUG:
        dbg_path = os.path.join(CFG.DEBUG_DIR, f"frame_{int(time.time())}.png")
        imwrite_unicode(dbg_path, img)


if __name__ == "__main__":
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        CFG.DRY_RUN = True
        demo_from_file(sys.argv[1])
    else:
        main_once_from_device()
