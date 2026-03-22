"""
items_chevron_diag.py
Диагностика состояния вкладки "Разбросанные вещи" по стрелке справа от названия.
- Опирается только на chevron (стрелку) в узкой полосе справа от items_tab.png.
- Логи: C:ot\\log\\items_diag_YYYY-MM-DD.txt
- Снимки: C:ot\\log\\items_diag\\snapshots
Предпосылки:
- При старте считаем: вещи закрыты, монстры открыты (константа).
- В игре стрелка вертикальная; open_direction см. CONFIG.OPEN_DIRECTION.

Требуемые файлы шаблонов в C:ot	pl\\my\\:
- items_tab.png           (название вкладки "Разбросанные вещи")
- (опционально) items_header.png — не используется в решении, но может помочь в будущем.
"""

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np

BASE_DIR = "C:\\bot"
TPL_DIR = os.path.join(BASE_DIR, "tpl", "my")
LOG_DIR = os.path.join(BASE_DIR, "log")
DIAG_DIR = os.path.join(LOG_DIR, "items_diag")
SNAP_DIR = os.path.join(DIAG_DIR, "snapshots")
os.makedirs(SNAP_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, f"items_diag_{datetime.now().date()}.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,%(msecs)03d | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger("items_chevron_diag")
ADB_SERIAL = os.getenv("ADB_SERIAL", "").strip() or None


def _adb(args: List[str], timeout=8):
    base = ["adb"]
    if ADB_SERIAL:
        base += ["-s", ADB_SERIAL]
    return subprocess.run(base + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)


def adb_tap(x: int, y: int, delay=0.35):
    _adb(["shell", "input", "tap", str(int(x)), str(int(y))])
    time.sleep(delay)


def screencap() -> Optional[np.ndarray]:
    try:
        proc = _adb(["exec-out", "screencap", "-p"], timeout=6)
        data = np.frombuffer(proc.stdout, np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    try:
        remote = "/sdcard/__scr_items_chev.png"
        _adb(["shell", "screencap", "-p", remote], timeout=6)
        local = os.path.join(SNAP_DIR, "__last.png")
        _adb(["pull", remote, local], timeout=6)
        img = cv2.imread(local, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def save_snap(img: np.ndarray, tag: str) -> str:
    path = os.path.join(SNAP_DIR, f"{int(time.time())}_{tag}.png")
    try:
        cv2.imwrite(path, img)
    except Exception:
        pass
    return path


@dataclass
class CONFIG:
    ORIENTATION: str = "vertical"
    OPEN_DIRECTION: str = "down"
    ITEMS_LABEL: str = os.path.join(TPL_DIR, "items_tab.png")
    LABEL_SEARCH_REL: Tuple[float, float, float, float] = (0.0, 0.6, 0.7, 1.0)
    LABEL_SCALES: Tuple[float, ...] = (0.9, 1.0, 1.1, 1.2)
    LABEL_MIN_SCORE: float = 0.6
    ROI_RIGHT_WIDTH_REL: float = 0.1
    ROI_RIGHT_PAD_PX: int = 2
    MIN_BLACK_RATIO: float = 0.02
    MIN_MARGIN: float = 0.12
    VERIFY_DELAY: float = 0.8
    MAX_RETRIES: int = 2


def load_tmpl(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        log.warning(f"Нет шаблона: {path}")
        return None
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def match_label(screen_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int, float]]:
    tmpl = load_tmpl(CONFIG.ITEMS_LABEL)
    if tmpl is None or screen_bgr is None:
        return None
    gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    x1 = int(CONFIG.LABEL_SEARCH_REL[0] * w)
    y1 = int(CONFIG.LABEL_SEARCH_REL[1] * h)
    x2 = int(CONFIG.LABEL_SEARCH_REL[2] * w)
    y2 = int(CONFIG.LABEL_SEARCH_REL[3] * h)
    roi = gray[y1:y2, x1:x2]
    th, tw = tmpl.shape[:2]
    best = None
    for sc in CONFIG.LABEL_SCALES:
        ts = cv2.resize(tmpl, (int(tw * sc), int(th * sc)), interpolation=cv2.INTER_AREA)
        if ts.shape[0] >= roi.shape[0] or ts.shape[1] >= roi.shape[1]:
            continue
        res = cv2.matchTemplate(roi, ts, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if best is None or max_val > best[0]:
            bx, by = max_loc
            bw, bh = ts.shape[1], ts.shape[0]
            best = max_val, x1 + bx, y1 + by, bw, bh, sc
    if best and best[0] >= CONFIG.LABEL_MIN_SCORE:
        score, lx, ly, lw, lh, sc = best
        log.info(f"[label] score={score:.3f} at {lx, ly, lw, lh} scale={sc:.2f}")
        return lx, ly, lw, lh, score
    log.info("[label] не найдено в ожидаемой зоне")
    return None


def chevron_roi_from_label(label_bbox: Tuple[int, int, int, int], screen_shape) -> Optional[Tuple[int, int, int, int]]:
    lx, ly, lw, lh = label_bbox
    H, W = screen_shape[:2]
    x1 = lx + lw + CONFIG.ROI_RIGHT_PAD_PX
    x2 = min(W, x1 + int(CONFIG.ROI_RIGHT_WIDTH_REL * W))
    y1 = ly
    y2 = min(H, ly + lh)
    if x2 <= x1 + 4 or y2 <= y1 + 4:
        return None
    return x1, y1, x2, y2


def binarize_roi(roi_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    norm = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(norm, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def direction_from_bw(bw: np.ndarray) -> Tuple[str, float, float]:
    h, w = bw.shape[:2]
    black = (bw > 0).astype(np.uint8)
    black_ratio = black.sum() / float(h * w + 1e-06)
    if CONFIG.ORIENTATION == "horizontal":
        left_sum = black[:, : w // 2].sum()
        right_sum = black[:, w // 2 :].sum()
        total = left_sum + right_sum + 1e-06
        margin = abs(left_sum - right_sum) / total
        if black_ratio < CONFIG.MIN_BLACK_RATIO or margin < CONFIG.MIN_MARGIN:
            return "unknown", black_ratio, margin
        return "left" if left_sum > right_sum else "right", black_ratio, margin
    else:
        top_sum = black[: h // 2, :].sum()
        bot_sum = black[h // 2 :, :].sum()
        total = top_sum + bot_sum + 1e-06
        margin = abs(top_sum - bot_sum) / total
        if black_ratio < CONFIG.MIN_BLACK_RATIO or margin < CONFIG.MIN_MARGIN:
            return "unknown", black_ratio, margin
        return "up" if top_sum > bot_sum else "down", black_ratio, margin


def is_open_direction(direction: str) -> Optional[bool]:
    if direction == "unknown":
        return None
    return direction == CONFIG.OPEN_DIRECTION


def annotate(
    screen_bgr: np.ndarray,
    label_rect: Optional[Tuple[int, int, int, int]],
    roi_rect: Optional[Tuple[int, int, int, int]],
    direction: str,
    black_ratio: float,
    margin: float,
    decision: str,
) -> np.ndarray:
    vis = screen_bgr.copy()
    if label_rect:
        x, y, w, h = label_rect
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 165, 255), 2)
        cv2.putText(vis, "items_tab", (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
    if roi_rect:
        x1, y1, x2, y2 = roi_rect
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            vis, "chevron_roi", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA
        )
    cv2.putText(
        vis,
        f"dir={direction} black={black_ratio:.3f} margin={margin:.2f} -> {decision}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return vis


def detect_items_state(
    screen_bgr: np.ndarray,
) -> Tuple[str, Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]], str, float, float]:
    label = match_label(screen_bgr)
    if not label:
        return "no_label", None, None, "unknown", 0.0, 0.0
    lx, ly, lw, lh, _ = label
    roi = chevron_roi_from_label((lx, ly, lw, lh), screen_bgr.shape)
    if not roi:
        return "no_roi", (lx, ly, lw, lh), None, "unknown", 0.0, 0.0
    x1, y1, x2, y2 = roi
    roi_img = screen_bgr[y1:y2, x1:x2].copy()
    bw = binarize_roi(roi_img)
    direction, black_ratio, margin = direction_from_bw(bw)
    open_close = is_open_direction(direction)
    decision = "open" if open_close is True else "close" if open_close is False else "unknown"
    cv2.imwrite(os.path.join(SNAP_DIR, f"{int(time.time())}_roi.png"), roi_img)
    cv2.imwrite(os.path.join(SNAP_DIR, f"{int(time.time())}_bw.png"), bw)
    return decision, (lx, ly, lw, lh), roi, direction, black_ratio, margin


def tap_items_by_label(screen_bgr: np.ndarray) -> bool:
    label = match_label(screen_bgr)
    if not label:
        log.info("[tap] items_tab не найден")
        return False
    lx, ly, lw, lh, _ = label
    cx, cy = lx + lw // 2, ly + lh // 2
    log.info(f"[tap] по имени items_tab -> ({cx},{cy})")
    adb_tap(cx, cy, delay=0.7)
    return True


def ensure_items_state(desired_open: bool) -> bool:
    img = screencap()
    if img is None:
        log.info("[ensure] нет скрина")
        return False
    state, lrect, rrect, direction, black, margin = detect_items_state(img)
    vis = annotate(img, lrect, rrect, direction, black, margin, state)
    save_snap(vis, f"state_{state}")
    if desired_open and state == "open" or not desired_open and state == "close":
        log.info(f"[ensure] уже {state}")
        return True
    ok = tap_items_by_label(img)
    if not ok:
        return False
    for retry in range(CONFIG.MAX_RETRIES + 1):
        time.sleep(CONFIG.VERIFY_DELAY)
        img2 = screencap()
        if img2 is None:
            continue
        state2, lrect2, rrect2, direction2, black2, margin2 = detect_items_state(img2)
        vis2 = annotate(img2, lrect2, rrect2, direction2, black2, margin2, state2)
        save_snap(vis2, f"verify_{state2}")
        log.info(f"[verify] dir={direction2} black={black2:.3f} margin={margin2:.2f} -> {state2}")
        if desired_open and state2 == "open":
            return True
        if not desired_open and state2 == "close":
            return True
    return False


def main():
    log.info("=== DIAG: старт. Предполагаем: вещи закрыты, монстры открыты ===")
    img = screencap()
    if img is None:
        log.info("Нет скриншота — проверь ADB")
        return
    st, lrect, rrect, d, b, m = detect_items_state(img)
    vis = annotate(img, lrect, rrect, d, b, m, st)
    p = save_snap(vis, f"initial_{st}")
    log.info(f"[initial] state={st} dir={d} black={b:.3f} margin={m:.2f} snap={os.path.basename(p)}")
    log.info("--- Пытаюсь ОТКРЫТЬ вещи ---")
    ok_open = ensure_items_state(True)
    log.info(f"[result-open] ok={ok_open}")
    log.info("--- Пытаюсь ЗАКРЫТЬ вещи ---")
    ok_close = ensure_items_state(False)
    log.info(f"[result-close] ok={ok_close}")
    log.info("--- Пытаюсь ОТКРЫТЬ вещи (повтор) ---")
    ok_open2 = ensure_items_state(True)
    log.info(f"[result-open-2] ok={ok_open2}")
    log.info("=== DIAG: завершено ===")


if __name__ == "__main__":
    main()
