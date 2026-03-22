"""
script/loot/script/loot/script/loot/script/loot/script/loot/script/loot/script/loot/auto_bot.py — надёжный автолотер под твою структуру C:/bot/
Основные требования:
 - Использует шаблоны в C:/bot/tpl/my и C:/bot/tpl/chevrons
 - Бесконечный цикл автолотинга (можно ограничить в config.json)
 - Логи в C:/bot/debug/debug/debug/debug/debug/debug/debug/debug/log/run_log.txt
 - Отладочные снимки в C:/bot/_debug (если DEBUG=true)
 - Не использует item.png — берет все *.png из tpl/my, кроме служебных
 - Учитывает безопасную зону тапов X=[0..1080], Y=[400..2100]
"""

import datetime
import glob
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

BASE_DIR = "C:/bot"
sys.path.append(BASE_DIR)
try:
    from script.loot.utils import _tpl, now_ts, snap, snap_roi, tap, to_gray

    HAVE_UTILS = True
except ImportError:
    HAVE_UTILS = False
BASE_DIR = "C:/bot"
SCRIPT_DIR = os.path.join(BASE_DIR, "script")
SCREENS_DIR = os.path.join(BASE_DIR, "screens")
OUT_DIR = os.path.join(BASE_DIR, "out")
DEBUG_DIR = os.path.join(BASE_DIR, "_debug")
TPL_DIR_MY = os.path.join(BASE_DIR, "tpl", "my")
TPL_CHEVRONS_DIR = os.path.join(BASE_DIR, "tpl", "chevrons")
LOG_DIR = os.path.join(BASE_DIR, "log")
CFG_DIR = os.path.join(BASE_DIR, "cfg")
CFG_FILE = os.path.join(CFG_DIR, "config.json")
LOG_FILE = os.path.join(LOG_DIR, "run_log.txt")
for d in (SCREENS_DIR, OUT_DIR, DEBUG_DIR, LOG_DIR, CFG_DIR):
    os.makedirs(d, exist_ok=True)
DEFAULT_CFG: Dict[str, Any] = {
    "device_id": None,
    "dry_run": False,
    "DEBUG": True,
    "save_debug_images": True,
    "sleep_after_tap": 0.65,
    "step_delay": 0.35,
    "delay_between_loops": 5.0,
    "max_loot_rounds": 0,
    "max_tab_attempts": 2,
    "item_name_threshold": 0.86,
    "item_scales": [0.95, 1.0, 1.05],
    "max_lines_to_collect": 12,
    "item_group_step": 150,
    "item_group_threshold": 75,
    "pickup_rel": [0.72, 0.83, 0.95, 0.94],
    "color_ratio_threshold": 0.12,
    "safe_x1": 0,
    "safe_x2": 1080,
    "safe_y1": 400,
    "safe_y2": 2100,
    "max_swipe_attempts": 1,
    "max_screenshot_attempts": 2,
    "pickup_timeout_sec": 6.0,
    "chevron_threshold": 0.86,
    "label_threshold": 0.86,
    "label_threshold_fallback": 0.83,
}


def now_ts() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def rect_from_rel(W: int, H: int, rel: List[float]) -> Tuple[int, int, int, int]:
    x1 = int(rel[0] * W)
    y1 = int(rel[1] * H)
    x2 = int(rel[2] * W)
    y2 = int(rel[3] * H)
    return x1, y1, x2, y2


def to_gray_local(img: np.ndarray) -> np.ndarray:
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 5, 35, 35)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    return g


def scharr_mag(img_gray: np.ndarray) -> np.ndarray:
    gx = cv2.Scharr(img_gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img_gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def tm_match_scaled(
    gray_img: np.ndarray, tpl_gray_full: np.ndarray, scales: List[float]
) -> Tuple[float, Tuple[int, int, int, int]]:
    best_val, best_box = -1.0, (0, 0, 0, 0)
    th, tw = tpl_gray_full.shape[:2]
    H, W = gray_img.shape[:2]
    for s in scales:
        tws, ths = max(5, int(tw * s)), max(5, int(th * s))
        if ths > H or tws > W:
            continue
        tpl_gray = cv2.resize(tpl_gray_full, (tws, ths), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC)
        res = cv2.matchTemplate(gray_img, tpl_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val = float(max_val)
            best_box = int(max_loc[0]), int(max_loc[1]), int(tws), int(ths)
    return best_val, best_box


def dbg_save(
    name: str,
    img: np.ndarray,
    rects: Optional[List[Tuple[int, int, int, int]]] = None,
    points: Optional[List[Tuple[int, int]]] = None,
    color=(0, 255, 0),
):
    if not SAVE_DEBUG_IMAGES:
        return
    img2 = img.copy()
    if rects:
        for x, y, w, h in rects:
            cv2.rectangle(img2, (x, y), (x + w, y + h), color, 2)
    if points:
        for px, py in points:
            cv2.circle(img2, (px, py), 8, (0, 0, 255), 2)
    path = os.path.join(DEBUG_DIR, f"{now_ts()}_{name}.png")
    cv2.imwrite(path, img2)
    return path


def structured_log(step: str, **payload):
    rec = {"ts": datetime.datetime.now().isoformat(timespec="milliseconds"), "step": step, **payload}
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except:
        pass


if not HAVE_UTILS:
    to_gray = to_gray_local

    def snap(name, frame, rects=None, rect_colors=None, points=None):
        dbg_save(name, frame, rects, points)

    def snap_roi(name, roi, roi_rect):
        dbg_save(f"{name}_roi", roi, rects=[(0, 0, roi.shape[1], roi.shape[0])])

    def tap(x: int, y: int, reason: str = "tap"):
        x = clamp(x, SAFE_X1, SAFE_X2)
        y = clamp(y, SAFE_Y1, SAFE_Y2)
        structured_log("tap", x=x, y=y, reason=reason)
        if not DRY_RUN:
            adb_cmd(["shell", "input", "tap", str(x), str(y)])
        time.sleep(SLEEP_AFTER_TAP)

    def _tpl(path: str):
        try:
            return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except:
            return None


def log(msg: str):
    line = f"[{now_ts()}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_config() -> Dict[str, Any]:
    cfg = DEFAULT_CFG.copy()
    if os.path.isfile(CFG_FILE):
        try:
            user = json.load(open(CFG_FILE, "r", encoding="utf-8"))
            for k, v in user.items():
                cfg[k] = v
            log("[CFG] Загружен tools/tools/tools/tools/tools/tools/tools/cfg/config.json")
        except Exception as e:
            log(f"[CFG] Не удалось прочитать config.json: {e} — используем дефолт")
    else:
        try:
            with open(CFG_FILE, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CFG, f, ensure_ascii=False, indent=2)
            log("[CFG] Сохранён дефолтный config.json (можно править ROI/тайминги)")
        except Exception as e:
            log(f"[CFG] Не удалось сохранить дефолтный config.json: {e}")
    return cfg


CFG = load_config()
DRY_RUN = CFG.get("DRY_RUN", CFG.get("dry_run", False))
DEBUG = CFG.get("DEBUG", True)
SAVE_DEBUG_IMAGES = CFG.get("save_debug_images", CFG.get("DEBUG", True))
SCREEN_W = CFG.get("SCREEN_W", 1080)
SCREEN_H = CFG.get("SCREEN_H", 2460)
safe = CFG.get("SAFE_TAP_AREA", {"x1": 0, "y1": 400, "x2": 1080, "y2": 2100})
SAFE_X1, SAFE_Y1, SAFE_X2, SAFE_Y2 = safe["x1"], safe["y1"], safe["x2"], safe["y2"]
ITEMS_ROI = CFG.get("ITEMS_ROI", [0, 400, 500, 2100])
STEP_DELAY = CFG.get("STEP_DELAY", 0.3)
SLEEP_AFTER_TAP = CFG.get("SLEEP_AFTER_TAP", 0.35)
MATCH = CFG.get("MATCH", {})
PICKUP_TPL_THRESHOLD = MATCH.get("PICKUP_TPL_THRESHOLD", 0.86)
PICKUP_SCALES = MATCH.get("PICKUP_SCALES", [0.9, 0.95, 1.0, 1.05])
PICKUP_COLOR_SAT_MIN = MATCH.get("PICKUP_COLOR_SAT_MIN", 90)
PICKUP_COLOR_VAL_MIN = MATCH.get("PICKUP_COLOR_VAL_MIN", 160)
PICKUP_COLOR_RATIO_THRESHOLD = MATCH.get("PICKUP_COLOR_RATIO_THRESHOLD", 0.12)
PICKUP_METHOD_WEIGHTS = MATCH.get("PICKUP_METHOD_WEIGHTS", {"templ": 0.6, "edge": 0.25, "color": 0.15})
PICKUP_ACTIVE_THRESHOLD = MATCH.get("PICKUP_ACTIVE_THRESHOLD", 0.92)
PICKUP_INACTIVE_THRESHOLD = MATCH.get("PICKUP_INACTIVE_THRESHOLD", 0.9)
ITEM_NAME_THRESHOLD = MATCH.get("ITEM_NAME_THRESHOLD", 0.86)
ROI_CFG = CFG.get("ROI", {})
PICKUP_REL = ROI_CFG.get("PICKUP_REL", [0.72, 0.83, 0.95, 0.94])
PICKUP_REL_CARD_FALLBACK = ROI_CFG.get("PICKUP_REL_CARD_FALLBACK", [0.6, 0.7, 0.98, 0.98])
SWP = CFG.get("SWIPE", {})
MAX_SWIPE_ATTEMPTS = SWP.get("MAX_SWIPE_ATTEMPTS", 6)
SWIPE_VECTORS = SWP.get("VECTORS", [[0, 650], [0, 550], [0, 450]])
SWIPE_DURATION_MS = SWP.get("DURATION_MS", 260)
SWIPE_PAUSE_MS = SWP.get("PAUSE_MS", 500)
SWIPE_STOP_ON_REPEAT_HASH = SWP.get("STOP_ON_REPEAT_HASH", True)
ORDER = CFG.get("ORDER", {})
RESCAN_AFTER_PICKUP = ORDER.get("RESCAN_AFTER_PICKUP", True)
LOGIC = CFG.get("LOGIC", {})
ALLOW_PICKUP_OTHER = LOGIC.get("ALLOW_PICKUP_OTHER", False)
ABORT_ON_FIRST_INACTIVE = LOGIC.get("ABORT_ON_FIRST_INACTIVE", True)
TPLS = CFG.get("TEMPLATES", {})
TPL_PICKUP_ACTIVE_LIST = TPLS.get("PICKUP_ACTIVE", [])
TPL_PICKUP_INACTIVE = TPLS.get("PICKUP_INACTIVE", None)
DEVICE_ID: Optional[str] = CFG.get("device_id")
DRY_RUN = bool(CFG.get("dry_run"))
DEBUG = bool(CFG.get("DEBUG"))
SAVE_DEBUG_IMAGES = bool(CFG.get("save_debug_images"))
SLEEP_AFTER_TAP = float(CFG.get("sleep_after_tap", 0.65))
STEP_DELAY = float(CFG.get("step_delay", 0.35))
DELAY_BETWEEN_LOOPS = float(CFG.get("delay_between_loops", 5.0))
MAX_LOOT_ROUNDS = int(CFG.get("max_loot_rounds", 0))
MAX_TAB_ATTEMPTS = int(CFG.get("max_tab_attempts", 2))
ITEM_NAME_THRESHOLD = float(CFG.get("item_name_threshold", 0.86))
ITEM_SCALES = list(CFG.get("item_scales", [0.95, 1.0, 1.05]))
MAX_LINES_TO_COLLECT = int(CFG.get("max_lines_to_collect", 12))
ITEM_GROUP_STEP = int(CFG.get("item_group_step", 150))
ITEM_GROUP_THRESHOLD = int(CFG.get("item_group_threshold", 75))
PICKUP_REL = tuple(CFG.get("pickup_rel", [0.72, 0.83, 0.95, 0.94]))
COLOR_RATIO_THRESHOLD = float(CFG.get("color_ratio_threshold", 0.12))
SAFE_X1, SAFE_X2 = int(CFG.get("safe_x1", 0)), int(CFG.get("safe_x2", 1080))
SAFE_Y1, SAFE_Y2 = int(CFG.get("safe_y1", 400)), int(CFG.get("safe_y2", 2100))
MAX_SWIPE_ATTEMPTS = int(CFG.get("max_swipe_attempts", 1))
MAX_SCREENSHOT_ATTEMPTS = int(CFG.get("max_screenshot_attempts", 2))
PICKUP_TIMEOUT_SEC = float(CFG.get("pickup_timeout_sec", 6.0))
CHEVRON_THRESHOLD = float(CFG.get("chevron_threshold", 0.86))
LABEL_THRESHOLD = float(CFG.get("label_threshold", 0.86))
LABEL_THRESHOLD_FALLBACK = float(CFG.get("label_threshold_fallback", 0.83))


def adb_cmd(args: List[str], timeout: float = 5.0) -> subprocess.CompletedProcess:
    base = ["adb"]
    if DEVICE_ID:
        base = ["adb", "-s", DEVICE_ID]
    try:
        return subprocess.run(base + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired:
        log("[WARN] ADB command timeout")
        raise


def screenshot_bgr(attempts: int = MAX_SCREENSHOT_ATTEMPTS) -> Optional[np.ndarray]:
    for i in range(attempts):
        try:
            proc = adb_cmd(["exec-out", "screencap", "-p"], timeout=15.0)
            data = proc.stdout
            if not data:
                raise RuntimeError("Пустой скриншот")
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("Не удалось декодировать PNG скриншот")
            if SAVE_DEBUG_IMAGES:
                cv2.imwrite(os.path.join(DEBUG_DIR, f"raw_{now_ts()}.png"), img)
            return img
        except Exception as e:
            log(f"[WARN] Скриншот не получен ({i + 1}/{attempts}): {e}")
            time.sleep(0.8)
    return None


def tap(x: int, y: int, reason: str = ""):
    x = max(SAFE_X1, min(SAFE_X2, int(x)))
    y = max(SAFE_Y1, min(SAFE_Y2, int(y)))
    if DRY_RUN:
        log(f"DRY_RUN TAP at ({x},{y}){'  // ' + reason if reason else ''}")
        return
    try:
        adb_cmd(["shell", "input", "tap", str(x), str(y)])
        log(f"TAP at ({x},{y}){'  // ' + reason if reason else ''}")
    except Exception as e:
        log(f"[ERROR] tap adb failed: {e}")
    time.sleep(SLEEP_AFTER_TAP)


def swipe_mid_down(duration_ms: int = 260):
    try:
        frame = screenshot_bgr()
        H, W = frame.shape[:2]
    except:
        W, H = 1080, 2460
    x = W // 2
    y1, y2 = min(SAFE_Y2, 1900), max(SAFE_Y1, 1400)
    if DRY_RUN:
        log(f"DRY_RUN SWIPE {x},{y1} -> {x},{y2}")
        return
    try:
        adb_cmd(["shell", "input", "swipe", str(x), str(y1), str(x), str(y2), str(duration_ms)])
        log("[SCROLL] swipe down to reveal pickup button")
    except Exception as e:
        log(f"[WARN] swipe failed: {e}")
    time.sleep(0.5)


def device_swipe(dx: int, dy: int, duration_ms: int):
    x = (SAFE_X1 + SAFE_X2) // 2
    y_from = clamp(SAFE_Y1 + 200, SAFE_Y1, SAFE_Y2)
    y_to = clamp(y_from + dy, SAFE_Y1, SAFE_Y2)
    x_to = clamp(x + dx, SAFE_X1, SAFE_X2)
    structured_log("swipe", start=[x, y_from], end=[x_to, y_to], duration=duration_ms)
    if not DRY_RUN:
        adb_cmd(["shell", "input", "swipe", str(x), str(y_from), str(x_to), str(y_to), str(duration_ms)])
    time.sleep(SWIPE_PAUSE_MS / 1000.0)


def frame_hash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    h = 0
    for i, v in enumerate(diff.flatten()):
        if v:
            h |= 1 << i
    return h


def swipe_until_visible_pickup(get_frame_fn, card_rect: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
    """
    Пытается вывести кнопку в видимую область: свайп -> поиск -> стоп по условиям.
    get_frame_fn: функция без аргументов, возвращающая текущий скрин (np.ndarray BGR)
    """
    prev_hash = None
    for i in range(MAX_SWIPE_ATTEMPTS):
        frame = get_frame_fn()
        st = pickup_state(frame, card_rect=card_rect)
        if st["state"] == "active" or st["state"] == "inactive":
            return st
        gray = to_gray(frame)
        cur_hash = frame_hash(gray)
        if SWIPE_STOP_ON_REPEAT_HASH and prev_hash is not None and cur_hash == prev_hash:
            structured_log("swipe_stop_same_hash", attempt=i, hash=hex(cur_hash))
            break
        prev_hash = cur_hash
        vec = SWIPE_VECTORS[min(i, len(SWIPE_VECTORS) - 1)]
        device_swipe(vec[0], vec[1], SWIPE_DURATION_MS)
    frame = get_frame_fn()
    return pickup_state(frame, card_rect=card_rect)


def imread_u8(path: str, flags: int = cv2.IMREAD_COLOR):
    try:
        with open(path, "rb") as f:
            data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, flags)
    except Exception as e:
        log(f"[IMG] read fail: {path} // {e}")
        return None


def save_debug(img: np.ndarray, name: str):
    if not SAVE_DEBUG_IMAGES:
        return
    try:
        cv2.imwrite(os.path.join(DEBUG_DIR, name), img)
    except Exception as e:
        log(f"[DEBUG] cannot save {name}: {e}")


def snap(
    label: str,
    frame: np.ndarray,
    rects: List[Tuple[int, int, int, int]] = None,
    rect_colors: List[Tuple[int, int, int]] = None,
    points: List[Tuple[int, int]] = None,
    point_colors: List[Tuple[int, int, int]] = None,
):
    if not SAVE_DEBUG_IMAGES:
        return
    try:
        img = frame.copy()
        rects = rects or []
        rect_colors = rect_colors or []
        points = points or []
        point_colors = point_colors or []
        for i, r in enumerate(rects):
            color = rect_colors[i] if i < len(rect_colors) else (0, 255, 0)
            x, y, w, h = r
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        for i, p in enumerate(points):
            color = point_colors[i] if i < len(point_colors) else (0, 0, 255)
            cv2.circle(img, (int(p[0]), int(p[1])), 8, color, 2)
        name = f"annot_{label}_{now_ts()}.png"
        cv2.imwrite(os.path.join(DEBUG_DIR, name), img)
    except Exception as e:
        log(f"[DEBUG] snap failed: {e}")


def snap_roi(label: str, roi: np.ndarray, at_rect: Tuple[int, int, int, int]):
    if not SAVE_DEBUG_IMAGES:
        return
    try:
        x, y, w, h = at_rect
        tag = f"{label}_x{x}_y{y}_w{w}_h{h}_{now_ts()}.png"
        cv2.imwrite(os.path.join(DEBUG_DIR, f"pickup_roi_{tag}"), roi)
    except Exception as e:
        log(f"[DEBUG] snap_roi failed: {e}")


def detect_chat_top_y(frame_bgr: np.ndarray) -> int:
    syschat_path = ui_tpl_path("syschat.png")
    H, W = frame_bgr.shape[:2]
    tpl = imread_u8(syschat_path, cv2.IMREAD_COLOR)
    if tpl is not None:
        ok, score, (x, y, w, h) = match_scaled(to_gray(frame_bgr), to_gray(tpl), 0.85, [0.9, 1.0, 1.1])
        if ok:
            return y
    return int(H * 0.88)


def to_gray(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.bilateralFilter(g, 5, 35, 35)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    return g


def _scharr_mag(gray: np.ndarray) -> np.ndarray:
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def match_scaled(
    gray_img: np.ndarray, tpl_gray_full: np.ndarray, threshold: float, scales: List[float]
) -> Tuple[bool, float, Tuple[int, int, int, int]]:
    best_ok = False
    best_score = 0.0
    best_box = 0, 0, 0, 0
    th, tw = tpl_gray_full.shape[:2]
    H, W = gray_img.shape[:2]
    for s in scales:
        tws, ths = max(5, int(tw * s)), max(5, int(th * s))
        if H < ths or W < tws:
            continue
        tpl_gray = cv2.resize(tpl_gray_full, (tws, ths), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC)
        res = cv2.matchTemplate(gray_img, tpl_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = float(max_val)
            best_ok = best_score >= threshold
            best_box = int(max_loc[0]), int(max_loc[1]), int(tws), int(ths)
    return best_ok, best_score, best_box


sys.path.append(os.path.join(BASE_DIR, "script"))
sys.path.append(BASE_DIR)
try:
    from tab_detector import detect_tab_states
except Exception as e:
    log(f"[ERROR] Cannot import tab_detector.detect_tab_states: {e}")

    def detect_tab_states(bgr):
        H, W = bgr.shape[:2]
        return {
            "вещи": {
                "state": "не видно",
                "label_found": False,
                "label_box": None,
                "band_y": int(H * 0.1),
                "scores": {"label": 0.0, "up": 0.0, "down": 0.0},
            },
            "монстры": {
                "state": "не видно",
                "label_found": False,
                "label_box": None,
                "band_y": int(H * 0.1),
                "scores": {"label": 0.0, "up": 0.0, "down": 0.0},
            },
        }


SKIP_FILES = {
    "monsters_tab.png",
    "items_tab.png",
    "open_tab.png",
    "close_tab.png",
    "pickup.png",
    "pickup_own.png",
    "pickup_own.png",
    "pickup_other.png",
    "syschat.png",
    "start.png",
    "start.PNG",
    "items_hdr.png",
    "items_label.png",
}


def load_item_name_templates():
    tpls = []
    for p in sorted(glob.glob(ui_tpl_path("*.png"))):
        base = os.path.basename(p)
        if base in SKIP_FILES:
            continue
        img = imread_u8(p, cv2.IMREAD_COLOR)
        if img is not None:
            tpls.append((base, img))
        else:
            log(f"[TPL] not readable: {p}")
    return tpls


def merge_same_lines(boxes_scores, line_thr=ITEM_GROUP_THRESHOLD):
    if not boxes_scores:
        return []
    centers = []
    for i, ((x, y, w, h), s, name) in enumerate(boxes_scores):
        cy = y + h // 2
        centers.append((cy, i))
    centers.sort()
    groups, current = [], [centers[0][1]]
    last_cy = centers[0][0]
    for cy, idx in centers[1:]:
        if abs(cy - last_cy) <= line_thr:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
        last_cy = cy
    groups.append(current)
    keep_idxs = [max(g, key=lambda i: boxes_scores[i][1]) for g in groups]
    return keep_idxs


def find_item_names(frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
    H, W = frame_bgr.shape[:2]
    x1, x2 = 0, min(500, W)
    y1, y2 = SAFE_Y1, min(SAFE_Y2, H)
    crop = frame_bgr[y1:y2, x1:x2]
    gray = to_gray(crop)
    boxes_scores = []
    tpls = load_item_name_templates()
    for name, tpl in tpls:
        ok, score, (x, y, w, h) = match_scaled(gray, to_gray(tpl), ITEM_NAME_THRESHOLD, ITEM_SCALES)
        if ok:
            boxes_scores.append(((x, y, w, h), score, name))
    if not boxes_scores:
        gg = _scharr_mag(gray)
        for name, tpl in tpls:
            tg = _scharr_mag(to_gray(tpl))
            ok, score, (x, y, w, h) = match_scaled(gg, tg, max(0.83, ITEM_NAME_THRESHOLD - 0.03), ITEM_SCALES)
            if ok:
                boxes_scores.append(((x, y, w, h), score, name))
    log(f"[ITEMS] найдено совпадений: {len(boxes_scores)} в ROI=[{x1},{y1},{x2},{y2}] до группировки")
    keep = merge_same_lines(boxes_scores, line_thr=ITEM_GROUP_THRESHOLD)
    if len(keep) > MAX_LINES_TO_COLLECT:
        keep = sorted(keep, key=lambda i: boxes_scores[i][0][1])[:MAX_LINES_TO_COLLECT]
    rects_abs = []
    found = []
    for i in keep:
        (x, y, w, h), score, name = boxes_scores[i]
        ax, ay = x1 + x, y1 + y
        rects_abs.append((ax, ay, w, h))
        found.append({"name": name, "score": float(score), "center": (ax + w // 2, ay + h // 2), "box": (ax, ay, w, h)})
    try:
        snap("items", frame_bgr, rects=rects_abs)
    except:
        pass
    log(f"[ITEMS] строк после группировки: {len(found)} (шаг≈{ITEM_GROUP_STEP}, порог={ITEM_GROUP_THRESHOLD})")
    return found


def rect_from_rel(w: int, h: int, rel: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    x1r, y1r, x2r, y2r = rel
    ax1, ay1, ax2, ay2 = int(w * x1r), int(h * y1r), int(w * x2r), int(h * y2r)
    return ax1, ay1, ax2, ay2


def _tpl(path: str) -> Optional[np.ndarray]:
    if not path:
        return None
    return imread_u8(path, cv2.IMREAD_COLOR)


def pickup_state(frame_bgr: np.ndarray, card_rect: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
    """
    Многоканальный детект кнопки "Подобрать".
    Приоритет: шаблоны (active/inactive) + edge + color. ROI — динамическая (под карточкой) либо глобальная из конфигурации.
    Возврат:
      {state: 'active'|'inactive'|'out_of_view', score: float, box: (x,y,w,h)|None, method: str, roi_rect: (x1,y1,w,h)}
    """
    H, W = frame_bgr.shape[:2]
    if card_rect:
        x, y, w, h = card_rect
        rx1 = x
        ry1 = y + int(h * 0.35)
        rx2 = min(W, x + w)
        ry2 = min(H, y + h)
        if ry2 - ry1 < 40 or rx2 - rx1 < 80:
            rx1, ry1, rx2, ry2 = rect_from_rel(W, H, PICKUP_REL)
    else:
        rx1, ry1, rx2, ry2 = rect_from_rel(W, H, PICKUP_REL)
    rx1 = clamp(rx1, 0, W - 1)
    rx2 = clamp(rx2, 1, W)
    ry1 = clamp(ry1, 0, H - 1)
    ry2 = clamp(ry2, 1, H)
    roi_rect = rx1, ry1, max(1, rx2 - rx1), max(1, ry2 - ry1)
    chat_top = detect_chat_top_y(frame_bgr)
    rx, ry, rw, rh = roi_rect
    ry2 = min(ry + rh, chat_top - 8)
    if ry2 <= ry:
        return {
            "state": "out_of_view",
            "score": None,
            "box": None,
            "method": "roi_clipped",
            "roi_rect": (rx, ry, rw, 1),
        }
    roi_rect = rx, ry, rw, ry2 - ry
    roi = frame_bgr[ry:ry2, rx : rx + rw]
    if roi.size == 0:
        return {"state": "out_of_view", "score": None, "box": None, "method": "out_of_view", "roi_rect": roi_rect}
    gray_full = to_gray(frame_bgr)
    gray_roi = to_gray(roi)
    edge_roi = scharr_mag(gray_roi)
    act_tpls = []
    for p in TPL_PICKUP_ACTIVE_LIST:
        tpl = _tpl(p)
        if tpl is not None:
            act_tpls.append(tpl)
    inact_tpl = _tpl(TPL_PICKUP_INACTIVE) if TPL_PICKUP_INACTIVE else None
    best_templ = {"kind": None, "score": -1.0, "box": None}
    for tpl in act_tpls:
        tpl_gray = to_gray(tpl)
        score, (x, y, w, h) = tm_match_scaled(gray_roi, tpl_gray, PICKUP_SCALES)
        if score > best_templ["score"]:
            best_templ = {"kind": "active", "score": score, "box": (rx1 + x, ry1 + y, w, h)}
    if inact_tpl is not None:
        tpl_gray = to_gray(inact_tpl)
        score, (x, y, w, h) = tm_match_scaled(gray_roi, tpl_gray, PICKUP_SCALES)
        if score > best_templ["score"]:
            best_templ = {"kind": "inactive", "score": score, "box": (rx1 + x, ry1 + y, w, h)}
    templ_score = max(0.0, best_templ["score"])
    edge_score = 0.0
    if best_templ["box"]:
        bx, by, bw, bh = best_templ["box"]
        tpl_src = (
            act_tpls[0]
            if best_templ["kind"] == "active" and len(act_tpls) > 0
            else inact_tpl if inact_tpl is not None else None
        )
        if tpl_src is not None:
            tpl_e = scharr_mag(to_gray(tpl_src))
            edge_score, _ = tm_match_scaled(edge_roi, tpl_e, PICKUP_SCALES)
    else:
        edge_score = 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, (1)] >= PICKUP_COLOR_SAT_MIN) & (hsv[:, :, (2)] >= PICKUP_COLOR_VAL_MIN)
    color_ratio = float(mask.sum()) / float(max(1, mask.size))
    w_tpl = PICKUP_METHOD_WEIGHTS.get("templ", 0.6)
    w_edge = PICKUP_METHOD_WEIGHTS.get("edge", 0.25)
    w_col = PICKUP_METHOD_WEIGHTS.get("color", 0.15)
    mixed = w_tpl * templ_score + w_edge * edge_score + w_col * color_ratio
    if templ_score < 0.3 and edge_score < 0.2 and color_ratio < 0.05:
        return {
            "state": "out_of_view",
            "score": float(templ_score),
            "box": best_templ["box"],
            "method": "empty_roi",
            "roi_rect": roi_rect,
        }
    state = None
    method = "mixed"
    if best_templ["kind"] == "inactive" and templ_score >= PICKUP_INACTIVE_THRESHOLD:
        state = "inactive"
        method = "tpl"
    elif best_templ["kind"] == "active" and templ_score >= PICKUP_ACTIVE_THRESHOLD:
        state = "active"
        method = "tpl"
    elif mixed >= max(PICKUP_ACTIVE_THRESHOLD, 0.92):
        state = "active"
    else:
        return {
            "state": "out_of_view",
            "score": mixed,
            "box": best_templ["box"],
            "method": method,
            "roi_rect": roi_rect,
        }
    try:
        dbg_save("pickup_roi", frame_bgr, rects=[roi_rect], color=(0, 200, 255))
        if best_templ["box"]:
            dbg_save("pickup_tpl", frame_bgr, rects=[best_templ["box"]], color=(255, 165, 0))
    except:
        pass
    structured_log(
        "pickup_seek",
        state=state,
        templ_score=round(templ_score, 3),
        edge_score=round(edge_score, 3),
        color_ratio=round(color_ratio, 3),
        mixed=round(mixed, 3),
        method=method,
        roi_rect=roi_rect,
        box=best_templ["box"],
    )
    return {
        "state": state,
        "score": float(max(templ_score, mixed)),
        "box": best_templ["box"],
        "method": method,
        "roi_rect": roi_rect,
    }


def collapse_item_panel(last_click_xy: Tuple[int, int]):
    x, y = last_click_xy
    tap(x, y, reason="collapse_item_panel")
    time.sleep(0.35)


def ensure_tab_state(tab_label: str, target_state: str) -> bool:

    def wait_and_check() -> bool:
        time.sleep(0.8)
        st = detect_tab_states(screenshot_bgr())[tab_label]["state"]
        return st == target_state

    frame = screenshot_bgr()
    if frame is None:
        return False
    cur = detect_tab_states(frame)[tab_label]
    log(
        f"[TAB:{tab_label}] state={cur['state']} label_found={int(cur['label_found'])} lbl={cur['scores']['label']:.3f} up={cur['scores']['up']:.3f} dn={cur['scores']['down']:.3f} band_y={cur['band_y']}"
    )
    if cur["state"] == target_state:
        return True
    if cur["label_found"] and cur["label_box"]:
        lx, ly, lw, lh = cur["label_box"]
        for frac in (0.5, 0.7, 0.3):
            cx = int(lx + lw * frac)
            cy = int(ly + lh * 0.5)
            tap(cx, cy, reason=f"{tab_label}:label_frac={frac:.1f} -> toggle")
            if wait_and_check():
                return True
    band_y = int(cur["band_y"])
    tap(180, band_y, reason=f"{tab_label}:band_line -> toggle")
    if wait_and_check():
        return True
    for cx in (1050, 950):
        tap(cx, band_y, reason=f"{tab_label}:chevron -> toggle")
        if wait_and_check():
            return True
    log(f"[WARN] {tab_label}: state не меняется после label/band/chevron — пропускаем")
    return False


def compute_row_hash(frame_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> int:
    x, y, w, h = box
    x1 = max(0, x - 5)
    y1 = max(0, y - 2)
    x2 = min(frame_bgr.shape[1], x + w + 220)
    y2 = min(frame_bgr.shape[0], y + h + 2)
    crop = frame_bgr[y1:y2, x1:x2]
    g = to_gray(crop)
    return frame_hash(g)


def auto_loot_once():
    log("=== AUTO-LOOT START (steps 8–12) ===")
    ensure_tab_state("монстры", "закрыта")
    ensure_tab_state("вещи", "открыта")
    frame = screenshot_bgr()
    items = find_item_names(frame)
    structured_log("list_scan", count=len(items) if items else 0)
    if not items:
        log("[FLOW] В списке нет предметов")
        log("=== AUTO-LOOT END ===")
        return
    items.sort(key=lambda d: d["box"][1])
    processed_hashes = set()
    idx = 0
    while idx < len(items):
        it = items[idx]
        name = it["name"]
        cx, cy = it["center"]
        x, y, w, h = it["box"]
        base_hash = compute_row_hash(frame, it["box"])
        if base_hash in processed_hashes:
            idx += 1
            continue
        log(f"[ITEM] click '{name}' at ({cx}, {cy}) score={it['score']:.3f}")
        structured_log("focus_item", item=name, box=it["box"], center=(cx, cy), base_hash=hex(base_hash))
        tap(cx, cy, reason="open_item_card")
        time.sleep(STEP_DELAY)
        frame_card = screenshot_bgr()
        H, W = frame_card.shape[:2]
        chat_top = detect_chat_top_y(frame_card)
        card_rect = int(W * 0.56), SAFE_Y1, int(W * 0.42), max(20, min(SAFE_Y2, chat_top - 8) - SAFE_Y1)

        def get_frame():
            return screenshot_bgr()

        st = swipe_until_visible_pickup(get_frame_fn=get_frame, card_rect=card_rect)
        state, box = st["state"], st["box"]
        log(f"[PICKUP] state={state} score={st['score']} method={st['method']}")
        if state == "inactive":
            if ABORT_ON_FIRST_INACTIVE:
                log(f"[FLOW] Неактивная кнопка у '{name}' — сворачиваем и завершаем лут")
                collapse_item_panel((cx, cy))
                ensure_tab_state("вещи", "закрыта")
                log("=== AUTO-LOOT END ===")
                return
            else:
                log(f"[FLOW] '{name}': кнопка неактивна — пропускаем")
                collapse_item_panel((cx, cy))
                processed_hashes.add(base_hash)
                idx += 1
                continue
        if state == "active":
            if box:
                bx, by = box[0] + box[2] // 2, box[1] + box[3] // 2
            else:
                rx, ry, rw, rh = st["roi_rect"]
                bx, by = rx + rw // 2, ry + rh // 2
            dbg_save(
                "pickup_click", frame, rects=[box] if box else [st["roi_rect"]], points=[(bx, by)], color=(0, 255, 0)
            )
            tap(bx, by, reason="pickup")
            time.sleep(1.0)
            frame_after = screenshot_bgr()
            try:
                items_after = find_item_names(frame_after)
            except Exception as e:
                items_after = []
                log(f"[VERIFY] Ошибка повторного поиска имён: {e}")
            disappeared = True
            for si in items_after or []:
                if si["name"] == name:
                    h_after = compute_row_hash(frame_after, si["box"])
                    if h_after == base_hash:
                        disappeared = False
                    break
            structured_log("verify", item=name, disappeared=disappeared)
            if disappeared:
                log(f"[FLOW] '{name}' подобран — карточка исчезла/изменился слот")
                collapse_item_panel((cx, cy))
                processed_hashes.add(base_hash)
                if RESCAN_AFTER_PICKUP:
                    frame = screenshot_bgr()
                    items = find_item_names(frame) or []
                    items.sort(key=lambda d: d["box"][1])
                    structured_log("rescan", count=len(items))
                    next_idx = 0
                    for j, obj in enumerate(items):
                        if obj["box"][1] > y:
                            next_idx = j
                            break
                        next_idx = min(j + 1, len(items))
                    idx = next_idx
                else:
                    idx += 1
                continue
            else:
                log(f"[WARN] '{name}' не исчез — повторная попытка")
                tap(bx, by, reason="pickup_retry")
                time.sleep(1.0)
                frame_after2 = screenshot_bgr()
                disappeared2 = True
                try:
                    items_after2 = find_item_names(frame_after2) or []
                    for si in items_after2:
                        if si["name"] == name:
                            h_after2 = compute_row_hash(frame_after2, si["box"])
                            if h_after2 == base_hash:
                                disappeared2 = False
                            break
                except Exception as e:
                    log(f"[VERIFY] Ошибка повторной верификации: {e}")
                structured_log("verify_retry", item=name, disappeared=disappeared2)
                collapse_item_panel((cx, cy))
                if disappeared2:
                    processed_hashes.add(base_hash)
                frame = screenshot_bgr()
                items = find_item_names(frame) or []
                items.sort(key=lambda d: d["box"][1])
                structured_log("rescan", count=len(items))
                next_idx = 0
                for j, obj in enumerate(items):
                    if obj["box"][1] > y:
                        next_idx = j
                        break
                    next_idx = min(j + 1, len(items))
                idx = next_idx
                continue
        log(f"[FLOW] '{name}': кнопка не найдена — пропускаем")
        collapse_item_panel((cx, cy))
        processed_hashes.add(base_hash)
        idx += 1
    ensure_tab_state("вещи", "закрыта")
    log("=== AUTO-LOOT END ===")


def main():
    try:
        adb_cmd(["get-state"])
    except Exception as e:
        log(f"[ERROR] ADB недоступен: {e}")
        return
    rounds = 0
    log("[MAIN] Entering main loop")
    try:
        while True:
            rounds += 1
            if MAX_LOOT_ROUNDS and rounds > MAX_LOOT_ROUNDS:
                log(f"[MAIN] reached max_loot_rounds={MAX_LOOT_ROUNDS}, exiting")
                break
            try:
                auto_loot_once()
            except Exception as e:
                log(f"[ERROR] Exception in auto_loot_once: {e}")
            log(f"[MAIN] Sleeping {DELAY_BETWEEN_LOOPS}s before next loop")
            time.sleep(DELAY_BETWEEN_LOOPS)
    except KeyboardInterrupt:
        log("[MAIN] Interrupted by user (KeyboardInterrupt)")
    except Exception as e:
        log(f"[MAIN] Unexpected error: {e}")
    finally:
        log("[MAIN] Exiting main")


if __name__ == "__main__":
    main()
