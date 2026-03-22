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

sys.path.append(os.path.join("C:/bot", "tools"))
from script.loot.utils import short_time_tag

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
STEP_COUNTER = 0


def next_step(label: str):
    global STEP_COUNTER
    STEP_COUNTER += 1
    return f"step_{STEP_COUNTER:04d}_{label}"


def mark_step(label: str) -> int:
    """
    Явно поднимает шаг и возвращает step_id (int).
    Используй перед большим действием чтобы связать логи и снапы.
    """
    global STEP_COUNTER
    STEP_COUNTER += 1
    structured_log("step_mark", step_id=STEP_COUNTER, label=label)
    return STEP_COUNTER


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
    "max_screenshot_attempts": 2,
    "pickup_timeout_sec": 6.0,
    "chevron_threshold": 0.86,
    "label_threshold": 0.86,
    "label_threshold_fallback": 0.83,
}


def now_ts():
    return short_time_tag(include_seconds=True)


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
            with open(CFG_FILE, "r", encoding="utf-8") as f:
                user = json.load(f)
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
FIND = CFG.get("FIND", {})
DETECT_ONLY_WHITELIST = bool(FIND.get("DETECT_ONLY_WHITELIST", False))
SKIP_SUBSTR = list(FIND.get("SKIP_SUBSTR", ["tab", "chevron", "hdr", "label"]))
EXCLUDE_ZONES = list(FIND.get("EXCLUDE_ZONES", [[0, 1900, 500, 2100]]))
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
DELAY_BETWEEN_LOOPS = float(CFG.get("DELAY_BETWEEN_LOOPS", CFG.get("delay_between_loops", 5.0)))
MAX_LOOT_ROUNDS = int(CFG.get("max_loot_rounds", 0))
MATCH = CFG.get("MATCH", {})
PICKUP_TPL_THRESHOLD = MATCH.get("PICKUP_TPL_THRESHOLD", 0.86)
PICKUP_SCALES = MATCH.get("PICKUP_SCALES", [0.9, 0.95, 1.0, 1.05])
PICKUP_COLOR_SAT_MIN = MATCH.get("PICKUP_COLOR_SAT_MIN", 90)
PICKUP_COLOR_VAL_MIN = MATCH.get("PICKUP_COLOR_VAL_MIN", 160)
PICKUP_COLOR_RATIO_THRESHOLD = MATCH.get("PICKUP_COLOR_RATIO_THRESHOLD", 0.12)
PICKUP_METHOD_WEIGHTS = MATCH.get("PICKUP_METHOD_WEIGHTS", {"templ": 0.6, "edge": 0.25, "color": 0.15})
PICKUP_ACTIVE_THRESHOLD = MATCH.get("PICKUP_ACTIVE_THRESHOLD", 0.92)
PICKUP_INACTIVE_THRESHOLD = MATCH.get("PICKUP_INACTIVE_THRESHOLD", 0.9)
ITEM_NAME_THRESHOLD = MATCH.get("ITEM_NAME_THRESHOLD", CFG.get("item_name_threshold", 0.86))
ROI_CFG = CFG.get("ROI", {})
PICKUP_REL = ROI_CFG.get("PICKUP_REL", CFG.get("pickup_rel", [0.72, 0.83, 0.95, 0.94]))
PICKUP_REL_CARD_FALLBACK = ROI_CFG.get("PICKUP_REL_CARD_FALLBACK", [0.56, 0.35, 0.98, 0.96])
SWP = CFG.get("SWIPE", {})
MAX_SWIPE_ATTEMPTS = int(CFG.get("max_swipe_attempts", SWP.get("MAX_SWIPE_ATTEMPTS", 4)))
SWIPE_VECTORS = SWP.get("VECTORS", [[0, 700], [0, 600], [0, 500], [0, 400]])
SWIPE_DURATION_MS = SWP.get("DURATION_MS", 300)
SWIPE_PAUSE_MS = SWP.get("PAUSE_MS", 500)
SWIPE_STOP_ON_REPEAT_HASH = SWP.get("STOP_ON_REPEAT_HASH", True)
SWIPE_SAME_HASH_STOP_N = int(SWP.get("SAME_HASH_STOP_N", 2))
TIMINGS = CFG.get("TIMINGS", {})
CARD_OPEN_DELAY_MS = int(TIMINGS.get("CARD_OPEN_DELAY_MS", 250))
POST_SWIPE_DELAY_MS = int(TIMINGS.get("POST_SWIPE_DELAY_MS", 200))
VERIFY1_MS = int(TIMINGS.get("VERIFY_ITEM_REMOVED_1_MS", 1200))
VERIFY2_MS = int(TIMINGS.get("VERIFY_ITEM_REMOVED_2_MS", 800))
ORDER = CFG.get("ORDER", {})
RESCAN_AFTER_PICKUP = ORDER.get("RESCAN_AFTER_PICKUP", True)
LOGIC = CFG.get("LOGIC", {})
ALLOW_PICKUP_OTHER = LOGIC.get("ALLOW_PICKUP_OTHER", False)
ABORT_ON_FIRST_INACTIVE = LOGIC.get("ABORT_ON_FIRST_INACTIVE", True)
TPLS = CFG.get("TEMPLATES", {})
TPL_PICKUP_ACTIVE_LIST = TPLS.get("PICKUP_ACTIVE", [])
TPL_PICKUP_INACTIVE = TPLS.get("PICKUP_INACTIVE", None)
ALLOWED_ITEM_NAMES = set(CFG.get("ALLOWED_ITEM_NAMES", []))
FIND_STABILIZE_FRAMES = int(CFG.get("FIND", {}).get("STABILIZE_FRAMES", 2))
DEVICE_ID: Optional[str] = CFG.get("device_id")
MAX_SCREENSHOT_ATTEMPTS = int(CFG.get("MAX_SCREENSHOT_ATTEMPTS", CFG.get("max_screenshot_attempts", 2)))
ITEM_SCALES = list(CFG.get("item_scales", [0.95, 1.0, 1.05]))
from typing import Dict, Tuple

slot_lifecycle: Dict[Tuple[int, int], dict] = {}


def update_slot_lifecycle(found_items: list, stage: str) -> None:
    """
    Обновление паспорта жизни слота.
    stage: 'detected', 'picked_by_bot', 'taken_by_other'
    """
    now_ts_str = datetime.datetime.now().isoformat(timespec="seconds")
    seen_keys = set()
    for it in found_items:
        key = int(it["slot_hash"]), int(it["box"][1])
        seen_keys.add(key)
        if key not in slot_lifecycle:
            slot_lifecycle[key] = {
                "name": it["name"],
                "first_seen_ts": now_ts_str,
                "last_seen_ts": now_ts_str,
                "status": stage,
            }
        else:
            slot_lifecycle[key]["last_seen_ts"] = now_ts_str
            if slot_lifecycle[key]["status"] not in ("picked_by_bot", "taken_by_other"):
                slot_lifecycle[key]["status"] = stage
    if stage == "detected":
        for key, data in slot_lifecycle.items():
            if key not in seen_keys and data["status"] not in ("picked_by_bot", "taken_by_other"):
                data["status"] = "taken_by_other"
                data["last_seen_ts"] = now_ts_str


def finalize_slot_lifecycle() -> None:
    """
    Пишем в лог полный отчёт по всем слотам за бой.
    """
    structured_log(
        "slot_lifecycle_report", slots=[{"hash": h, "y": y, **data} for (h, y), data in slot_lifecycle.items()]
    )


def dump_queue(tag: str, items_list: list) -> None:
    structured_log(
        "items_queue",
        tag=tag,
        items=[{"name": it["name"], "y": int(it["box"][1]), "hash": int(it["slot_hash"])} for it in items_list],
    )


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def compute_items_visible_roi(frame_bgr, tabs) -> Tuple[int, int, int, int]:
    H, W = frame_bgr.shape[:2]
    fights = bool(tabs.get("occluded_by_extra"))
    top_y = 750 if fights else 600
    bottom_y = min(2100 if fights else 1900, SAFE_Y2)
    x1, x2 = 0, 500
    return x1, top_y, x2, bottom_y


def rect_from_rel(W: int, H: int, rel: List[float]) -> Tuple[int, int, int, int]:
    x1 = int(rel[0] * W)
    y1 = int(rel[1] * H)
    x2 = int(rel[2] * W)
    y2 = int(rel[3] * H)
    return x1, y1, x2, y2


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


def imread_u8(path: str, flags: int = cv2.IMREAD_COLOR):
    try:
        with open(path, "rb") as f:
            data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, flags)
    except Exception as e:
        log(f"[IMG] read fail: {path} // {e}")
        return None


def dbg_name(name, step_id=None):
    if step_id is not None:
        return f"{step_id:04d}_{name}"
    return next_step(name) if CFG.get("DEBUG_PACK_BY_STEP", False) else name


def dbg_save(
    name: str,
    img: np.ndarray,
    rects: Optional[List[Tuple[int, int, int, int]]] = None,
    points: Optional[List[Tuple[int, int]]] = None,
    color=(0, 255, 0),
):
    if not SAVE_DEBUG_IMAGES or img is None:
        return
    img2 = img.copy()
    if rects:
        for x, y, w, h in rects:
            cv2.rectangle(img2, (x, y), (x + w, y + h), color, 2)
    if points:
        for px, py in points:
            cv2.circle(img2, (px, py), 8, (0, 0, 255), 2)
    path = os.path.join(DEBUG_DIR, f"{now_ts()}_{dbg_name('' + name)}.png")
    cv2.imwrite(path, img2)
    return path


def snap(label: str, frame: np.ndarray, step_id=None, rects=None, rect_colors=None, points=None, point_colors=None):
    if not SAVE_DEBUG_IMAGES or frame is None:
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
        name = f"annot_{dbg_name(label, step_id)}_{now_ts()}.png"
        cv2.imwrite(os.path.join(DEBUG_DIR, name), img)
    except Exception as e:
        log(f"[DEBUG] snap failed: {e}")


def snap_roi(label: str, roi: np.ndarray, at_rect: Tuple[int, int, int, int]):
    if not SAVE_DEBUG_IMAGES or roi is None:
        return
    try:
        x, y, w, h = at_rect
        tag = f"{dbg_name(label)}_x{x}_y{y}_w{w}_h{h}_{now_ts()}.png"
        cv2.imwrite(os.path.join(DEBUG_DIR, f"pickup_roi_{tag}"), roi)
    except Exception as e:
        log(f"[DEBUG] snap_roi failed: {e}")


def structured_log(step: str, **payload):
    rec = {"ts": datetime.datetime.now().isoformat(timespec="milliseconds"), "step": step, **payload}
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except:
        pass


def adb_cmd(args: List[str], timeout: float = 5.0) -> subprocess.CompletedProcess:
    base = ["adb"]
    if DEVICE_ID:
        base = ["adb", "-s", DEVICE_ID]
    try:
        return subprocess.run(base + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired:
        log("[WARN] ADB command timeout")
        raise


def screenshot_bgr(attempts: Optional[int] = None) -> Optional[np.ndarray]:
    attempts = attempts or MAX_SCREENSHOT_ATTEMPTS
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
                cv2.imwrite(os.path.join(DEBUG_DIR, f"{now_ts()}_{dbg_name('raw')}.png"), img)
            return img
        except Exception as e:
            log(f"[WARN] Скриншот не получен ({i + 1}/{attempts}): {e}")
            time.sleep(0.8)
    return None


def tap(x: int, y: int, reason: str = ""):
    x = max(SAFE_X1, min(SAFE_X2, int(x)))
    y = max(SAFE_Y1, min(SAFE_Y2, int(y)))
    structured_log("tap", x=x, y=y, reason=reason)
    if DRY_RUN:
        log(f"DRY_RUN TAP at ({x},{y}){'  // ' + reason if reason else ''}")
        time.sleep(SLEEP_AFTER_TAP)
        return
    try:
        adb_cmd(["shell", "input", "tap", str(x), str(y)])
        log(f"TAP at ({x},{y}){'  // ' + reason if reason else ''}")
    except Exception as e:
        log(f"[ERROR] tap adb failed: {e}")
    time.sleep(SLEEP_AFTER_TAP)


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


def page_hash_from_bgr(frame_bgr: np.ndarray) -> int:
    return frame_hash(to_gray(frame_bgr))


def detect_chat_top_y(frame_bgr: np.ndarray) -> int:
    syschat_path = ui_tpl_path("syschat.png")
    H, W = frame_bgr.shape[:2]
    tpl = imread_u8(syschat_path, cv2.IMREAD_COLOR)
    if tpl is not None:
        ok, score, (x, y, w, h) = match_scaled(to_gray(frame_bgr), to_gray(tpl), 0.85, [0.9, 1.0, 1.1])
        if ok:
            return y
    return int(H * 0.88)


def _tpl(path: str) -> Optional[np.ndarray]:
    if not path:
        return None
    return imread_u8(path, cv2.IMREAD_COLOR)


def pickup_state(frame_bgr: np.ndarray, card_rect: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
    H, W = frame_bgr.shape[:2]

    def _eval_roi(x1: int, y1: int, x2: int, y2: int, tag: str):
        chat_top = detect_chat_top_y(frame_bgr)
        x1 = clamp(x1, 0, W - 1)
        x2 = clamp(x2, 1, W)
        y1 = clamp(y1, 0, H - 1)
        y2 = clamp(y2, 1, H)
        y2 = min(y2, chat_top - 8)
        if x2 <= x1 or y2 <= y1:
            return {
                "state": "out_of_view",
                "score": None,
                "box": None,
                "method": f"{tag}_roi_clipped",
                "roi_rect": (x1, y1, max(1, x2 - x1), 1),
            }
        roi_rect = x1, y1, x2 - x1, y2 - y1
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return {
                "state": "out_of_view",
                "score": None,
                "box": None,
                "method": f"{tag}_empty_roi",
                "roi_rect": roi_rect,
            }
        snap_roi(f"{tag}", roi, roi_rect)
        gray_roi = to_gray(roi)
        edge_roi = _scharr_mag(gray_roi)
        act_tpls = []
        for p in TPL_PICKUP_ACTIVE_LIST:
            img = _tpl(p)
            if img is not None:
                act_tpls.append(img)
        inact_tpl = _tpl(TPL_PICKUP_INACTIVE) if TPL_PICKUP_INACTIVE else None
        best_templ = {"kind": None, "score": -1.0, "box": None}
        for tpl in act_tpls:
            ok, score, (tx, ty, tw, th) = match_scaled(gray_roi, to_gray(tpl), PICKUP_TPL_THRESHOLD, PICKUP_SCALES)
            if score > best_templ["score"]:
                best_templ = {"kind": "active", "score": score, "box": (x1 + tx, y1 + ty, tw, th)}
        if inact_tpl is not None:
            ok, score, (tx, ty, tw, th) = match_scaled(
                gray_roi, to_gray(inact_tpl), PICKUP_TPL_THRESHOLD, PICKUP_SCALES
            )
            if score > best_templ["score"]:
                best_templ = {"kind": "inactive", "score": score, "box": (x1 + tx, y1 + ty, tw, th)}
        templ_score = max(0.0, best_templ["score"])
        edge_score = 0.0
        if best_templ["box"]:
            tpl_src = (
                act_tpls[0]
                if best_templ["kind"] == "active" and act_tpls
                else inact_tpl if inact_tpl is not None else None
            )
            if tpl_src is not None:
                tpl_e = _scharr_mag(to_gray(tpl_src))
                ok, e_score, _ = match_scaled(edge_roi, tpl_e, 0.0, PICKUP_SCALES)
                edge_score = float(e_score)
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
                "score": templ_score,
                "box": best_templ["box"],
                "method": f"{tag}_empty_features",
                "roi_rect": roi_rect,
            }
        if best_templ["kind"] == "inactive" and templ_score >= PICKUP_INACTIVE_THRESHOLD:
            return {
                "state": "inactive",
                "score": templ_score,
                "box": best_templ["box"],
                "method": f"{tag}_tpl",
                "roi_rect": roi_rect,
            }
        elif best_templ["kind"] == "active" and templ_score >= PICKUP_ACTIVE_THRESHOLD:
            return {
                "state": "active",
                "score": templ_score,
                "box": best_templ["box"],
                "method": f"{tag}_tpl",
                "roi_rect": roi_rect,
            }
        elif mixed >= PICKUP_ACTIVE_THRESHOLD:
            return {
                "state": "active",
                "score": mixed,
                "box": best_templ["box"],
                "method": f"{tag}_mixed",
                "roi_rect": roi_rect,
            }
        else:
            return {
                "state": "out_of_view",
                "score": mixed,
                "box": best_templ["box"],
                "method": f"{tag}_mixed",
                "roi_rect": roi_rect,
            }

    if card_rect:
        x, y, w, h = card_rect
        rx1 = x
        ry1 = y + int(h * 0.3)
        rx2 = min(W, x + w)
        ry2 = min(H, y + h)
        st_card = _eval_roi(rx1, ry1, rx2, ry2, tag="card")
        if st_card["state"] in ("active", "inactive"):
            return st_card
        structured_log("pickup_card_roi_failed", method=st_card.get("method"), score=st_card.get("score"))
    px1, py1, px2, py2 = rect_from_rel(W, H, PICKUP_REL)
    st_rel = _eval_roi(px1, py1, px2, py2, tag="rel")
    if st_rel["state"] in ("active", "inactive"):
        return st_rel
    fx1, fy1, fx2, fy2 = rect_from_rel(W, H, PICKUP_REL_CARD_FALLBACK)
    st_fallback = _eval_roi(fx1, fy1, fx2, fy2, tag="fallback")
    return st_fallback


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


def swipe_until_visible_pickup(
    get_frame_fn, base_y: int, card_rect: Optional[Tuple[int, int, int, int]] = None
) -> Dict[str, Any]:
    """
    Алгоритм:
    1. Если active/inactive в первом кадре — выходим.
    2. Если box пустой — всегда свайпаем.
    3. Если box есть и центр ниже середины — свайпаем.
    4. Первый свайп — половина SAFE-зоны вверх.
    5. Если всё ещё out_of_view — micro-свайп и повторная проверка.
    """
    frame0 = get_frame_fn()
    st0 = pickup_state(frame0, card_rect=card_rect)
    structured_log("pickup_check_init", step_id=STEP_COUNTER, state=st0["state"])
    snap("pickup_init", frame0, step_id=STEP_COUNTER)
    if st0["state"] in ("active", "inactive"):
        return st0
    safe_top = SAFE_Y1
    safe_bottom = SAFE_Y2
    span = safe_bottom - safe_top
    midline = safe_top + span // 2
    has_box = bool(st0.get("box"))
    if has_box:
        _, y, _, h = st0["box"]
        cy = y + h // 2
    else:
        cy = None
    need_swipe = not has_box or cy > midline
    swipe_sid = mark_step("pickup_swipe_sequence")
    structured_log(
        "plan_swipe_to_pickup",
        step_id=swipe_sid,
        midline=int(midline),
        cy=None if cy is None else int(cy),
        need_swipe=bool(need_swipe),
    )
    if not need_swipe:
        structured_log("early_exit_pickup_visible_or_upper_half", cy=cy, midline=midline)
        return st0
    structured_log(
        "act_swipe", step_id=swipe_sid, kind="strong_up", dy=int(-max(300, span // 2)), duration_ms=SWIPE_DURATION_MS
    )
    dy_strong = -max(300, span // 2)
    device_swipe(0, dy_strong, SWIPE_DURATION_MS)
    time.sleep(POST_SWIPE_DELAY_MS / 1000.0)
    frame1 = get_frame_fn()
    st1 = pickup_state(frame1, card_rect=card_rect)
    structured_log("outcome_swipe_check", step_id=swipe_sid, after="strong_up", state=st1["state"])
    if st1["state"] in ("active", "inactive"):
        return st1
    structured_log("act_swipe", step_id=swipe_sid, kind="micro_up", dy=-150, duration_ms=SWIPE_DURATION_MS)
    device_swipe(0, -150, SWIPE_DURATION_MS)
    time.sleep(POST_SWIPE_DELAY_MS / 1000.0)
    frame2 = get_frame_fn()
    st2 = pickup_state(frame2, card_rect=card_rect)
    structured_log("outcome_swipe_check", step_id=swipe_sid, after="micro_up", state=st2["state"])
    return st2


def collapse_if_open(panel_open: bool, click_xy: Tuple[int, int], reason: str = "collapse_item_panel") -> bool:
    if not panel_open:
        structured_log("collapse_skip", because="panel_already_closed")
        return False
    x, y = click_xy
    tap(x, y, reason=reason)
    time.sleep(0.3)
    structured_log("collapsed")
    return False


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


EXTRA_LABELS = {"fights": ui_tpl_path("fights_tab.png")}


def detect_tab_states(frame_bgr: np.ndarray) -> Dict[str, Any]:
    tabs = detect_tab_states(frame_bgr)
    occluded_by_extra = False
    fights_path = EXTRA_LABELS.get("fights")
    tpl = imread_u8(fights_path, cv2.IMREAD_COLOR) if fights_path else None
    if tpl is not None:
        ok, score, (x, y, w, h) = match_scaled(to_gray(frame_bgr), to_gray(tpl), 0.85, [0.9, 1.0, 1.1])
        if ok:
            occluded_by_extra = True
            snap("extra_tab_fights", frame_bgr, rects=[(x, y, w, h)])
            structured_log("tab_occluded_by_extra_detected", score=float(score), box=[x, y, w, h])
    tabs["occluded_by_extra"] = occluded_by_extra
    return tabs


SKIP_FILES = {
    "monsters_tab.png",
    "items_tab.png",
    "open_tab.png",
    "close_tab.png",
    "pickup.png",
    "pickup_own.png",
    "pickup_other.png",
    "syschat.png",
    "start.png",
    "monsters_hdr.png",
    "items_hdr.png",
    "items_label.png",
}


def _stem(name: str) -> str:
    base = os.path.basename(name)
    s, _ = os.path.splitext(base)
    return s


def should_skip_tpl(base: str) -> bool:
    if base in {
        "monsters_tab.png",
        "items_tab.png",
        "open_tab.png",
        "close_tab.png",
        "pickup.png",
        "pickup_own.png",
        "pickup_other.png",
        "syschat.png",
        "start.png",
        "start.PNG",
        "items_hdr.png",
        "items_label.png",
    }:
        return True
    low = base.lower()
    for sub in SKIP_SUBSTR:
        if sub and sub in low:
            return True
    return False


def load_item_name_templates():
    """
    Загружаем шаблоны имён:
    - исключаем служебные по should_skip_tpl
    - опционально фильтруем по ALLOWED_ITEM_NAMES (DETECT_ONLY_WHITELIST)
    """
    tpls = []
    names_loaded = []
    for p in sorted(glob.glob(ui_tpl_path("*.png"))):
        base = os.path.basename(p)
        if should_skip_tpl(base):
            continue
        if DETECT_ONLY_WHITELIST:
            stem = _stem(base)
            allow_stems = {_stem(n) for n in ALLOWED_ITEM_NAMES}
            if stem not in allow_stems:
                continue
        img = imread_u8(p, cv2.IMREAD_COLOR)
        if img is None:
            log(f"[TPL] not readable: {p}")
            continue
        tpls.append((base, img))
        names_loaded.append(base)
    black = set(CFG.get("FIND", {}).get("BLACKLIST", []))
    tpls = [(name, img) for name, img in tpls if name not in black]
    structured_log("items_tpls_loaded", count=len(tpls), names=names_loaded[:50])
    return tpls


def merge_same_lines(boxes_scores, line_thr=75):
    """
    Группировка совпадений по вертикали с учётом уникального slot_hash.
    boxes_scores: [ ((x, y, w, h), score, name, slot_hash), ... ]
    line_thr: допуск по центру Y для объединения в одну группу.
    """
    if not boxes_scores:
        return []
    keep_indices = []
    seen_groups = set()
    for idx, (bbox, score, name, slot_hash) in enumerate(boxes_scores):
        _, y, _, h = bbox
        y_group = int(round(y / float(line_thr)))
        key = y_group, slot_hash
        if key in seen_groups:
            continue
        seen_groups.add(key)
        keep_indices.append(idx)
    return keep_indices


def find_item_names(frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
    H, W = frame_bgr.shape[:2]
    rx1, ry1, rx2, ry2 = ITEMS_ROI
    x1 = clamp(int(rx1), 0, W - 1)
    y1 = clamp(int(ry1), 0, H - 1)
    x2 = clamp(int(rx2), 1, W)
    y2 = clamp(int(ry2), 1, H)
    if x2 <= x1 or y2 <= y1:
        log(f"[ITEMS] Пустой ROI для списка: {ITEMS_ROI}")
        return []
    crop = frame_bgr[y1:y2, x1:x2]
    gray = to_gray(crop)
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
        gray = to_gray(crop)
        structured_log("items_exclude_zones_applied", zones=EXCLUDE_ZONES)
    boxes_scores = []
    tpls = load_item_name_templates()
    for name, tpl in tpls:
        ok, score, (x, y, w, h) = match_scaled(gray, to_gray(tpl), ITEM_NAME_THRESHOLD, ITEM_SCALES)
        if ok:
            row_crop = frame_bgr[y1 + y : y1 + y + h, x1 + x : x1 + x + w]
            row_hash = frame_hash(to_gray(row_crop))
            boxes_scores.append(((x, y, w, h), score, name, int(row_hash)))
    if not boxes_scores:
        gg = _scharr_mag(gray)
        for name, tpl in tpls:
            tg = _scharr_mag(to_gray(tpl))
            ok, score, (x, y, w, h) = match_scaled(gg, tg, max(0.83, ITEM_NAME_THRESHOLD - 0.03), ITEM_SCALES)
            if ok:
                row_crop = frame_bgr[y1 + y : y1 + y + h, x1 + x : x1 + x + w]
                row_hash = frame_hash(to_gray(row_crop))
                boxes_scores.append(((x, y, w, h), score, name, int(row_hash)))
    log(f"[ITEMS] найдено совпадений: {len(boxes_scores)} в ROI=[{x1},{y1},{x2},{y2}] до группировки")
    keep = merge_same_lines(boxes_scores, line_thr=int(CFG.get("item_group_threshold", 75)))
    if len(keep) > int(CFG.get("max_lines_to_collect", 12)):
        keep = sorted(keep, key=lambda i: boxes_scores[i][0][1])[: int(CFG.get("max_lines_to_collect", 12))]
    rects_abs, found = [], []
    for i in keep:
        (x, y, w, h), score, name, slot_hash = boxes_scores[i]
        ax, ay = x1 + x, y1 + y
        rects_abs.append((ax, ay, w, h))
        found.append(
            {
                "name": name,
                "score": float(score),
                "center": (ax + w // 2, ay + h // 2),
                "box": (ax, ay, w, h),
                "slot_hash": slot_hash,
            }
        )
    centers = [it["center"] for it in found]
    snap("items", frame_bgr, rects=rects_abs, points=centers)
    structured_log("items_found_detail", count=len(found), names=[it["name"] for it in found])
    log(f"[ITEMS] строк после группировки: {len(found)} (порог={int(CFG.get('item_group_threshold', 75))})")
    update_slot_lifecycle(found, stage="detected")
    return found


def ensure_tab_state(tab_label: str, target_state: str) -> bool:

    def wait_and_check() -> bool:
        time.sleep(0.8)
        st = detect_tab_states(screenshot_bgr())[tab_label]["state"]
        return st == target_state

    frame = screenshot_bgr()
    if frame is None:
        return False
    tabs = detect_tab_states(frame)
    cur = tabs[tab_label]
    log(
        f"[TAB:{tab_label}] state={cur['state']} label_found={int(cur['label_found'])} lbl={cur['scores']['label']:.3f} up={cur['scores']['up']:.3f} dn={cur['scores']['down']:.3f} band_y={cur['band_y']}"
    )
    if tab_label == "вещи" and tabs.get("occluded_by_extra"):
        log("[TABS] вкладка 'вещи' скрыта вкладкой 'Сражения'")
        occ_sid = mark_step("tab_reveal_fights")
        structured_log("plan_tab_reveal", step_id=occ_sid, reason="fights_occlusion", swipe=[0, 200], duration_ms=200)
        structured_log(
            "act_swipe",
            step_id=occ_sid,
            start=[(SAFE_X1 + SAFE_X2) // 2, SAFE_Y1 + 200],
            end=[(SAFE_X1 + SAFE_X2) // 2, SAFE_Y1 + 200 + 200],
            duration_ms=200,
        )
        device_swipe(0, 200, 200)
        time.sleep(0.3)
        tabs = detect_tab_states(screenshot_bgr())
        structured_log("outcome_tab_reveal", step_id=occ_sid, occluded_by_extra=bool(tabs.get("occluded_by_extra")))
        cur = tabs[tab_label]
    plan_sid = mark_step(f"tab_toggle_{tab_label}")
    structured_log(
        "plan_tab_toggle",
        step_id=plan_sid,
        tab=tab_label,
        target=target_state,
        order=["label(0.5/0.7/0.3)", "band_line", "chevrons(1050,950)"],
        band_y=int(cur["band_y"]),
        label_box=cur["label_box"],
        scores=cur["scores"],
    )
    if cur["state"] == target_state:
        return True
    if cur["label_found"] and cur["label_box"]:
        lx, ly, lw, lh = cur["label_box"]
        for frac in (0.5, 0.7, 0.3):
            cx = int(lx + lw * frac)
            cy = int(ly + lh * 0.5)
            structured_log("act_tab_toggle", step_id=plan_sid, method="label", variant=f"{frac:.1f}", x=cx, y=cy)
            tap(cx, cy, reason=f"{tab_label}:label_frac={frac:.1f} -> toggle")
            if wait_and_check():
                structured_log(
                    "outcome_tab_toggle", step_id=plan_sid, result="success", via="label", variant=f"{frac:.1f}"
                )
                return True
    band_y = int(cur["band_y"])
    if band_y >= SAFE_Y1:
        structured_log("act_tab_toggle", step_id=plan_sid, method="band_line", x=180, y=band_y)
        tap(180, band_y, reason=f"{tab_label}:band_line -> toggle")
        if wait_and_check():
            structured_log("outcome_tab_toggle", step_id=plan_sid, result="success", via="band_line")
            return True
    else:
        structured_log(
            "skip_band_line_tap", step_id=plan_sid, reason="band_y_below_safe", band_y=band_y, safe_y1=SAFE_Y1
        )
    for cx in (1050, 950):
        if band_y >= SAFE_Y1:
            structured_log("act_tab_toggle", step_id=plan_sid, method="chevron", x=cx, y=band_y)
            tap(cx, band_y, reason=f"{tab_label}:chevron -> toggle")
            if wait_and_check():
                structured_log("outcome_tab_toggle", step_id=plan_sid, result="success", via="chevron", x=cx, y=band_y)
                return True
    structured_log("outcome_tab_toggle", step_id=plan_sid, result="failed", reason="no_state_change")
    log(f"[WARN] {tab_label}: state не меняется после label/band/chevron — пропускаем")
    return False


def verify_item_removed(name: str, base_hash: int, base_y: int) -> bool:
    time.sleep(VERIFY1_MS / 1000.0)
    slot_thr = int(CFG.get("item_group_threshold", 75))
    for _ in range(2):
        frame = screenshot_bgr()
        items_after = find_item_names(frame) or []
        same_line = [it for it in items_after if abs(int(it["box"][1]) - int(base_y)) <= slot_thr]
        if not same_line:
            return True
        replaced = any(int(it["slot_hash"]) != int(base_hash) for it in same_line)
        if replaced:
            return True
        time.sleep(VERIFY2_MS / 1000.0)
    return False


def auto_loot_once():
    log("=== AUTO-LOOT START (strict top→bottom) ===")
    min_y_cursor = SAFE_Y1
    ensure_tab_state("монстры", "закрыта")
    ensure_tab_state("вещи", "открыта")
    log("[PRE-LOOT] Вкладки в нужном состоянии — делаем стартовый лист")
    structured_log("pre_loot_initial_swipe", reason="monsters_closed_and_items_open")
    dy_init = -(SAFE_Y2 - SAFE_Y1) // 2
    device_swipe(0, dy_init, SWIPE_DURATION_MS)
    time.sleep(POST_SWIPE_DELAY_MS / 1000.0)
    processed_slots_success: set[tuple[int, int]] = set()
    processed_slots_skipped: set[tuple[int, int]] = set()
    occlusion_empty_checks = 0
    last_page_hash = None
    frame = screenshot_bgr()
    if frame is None:
        return
    page_hash = page_hash_from_bgr(frame)

    def in_set(h, y, S):
        for ph, py in S:
            if h == ph and abs(y - py) <= int(CFG.get("item_group_threshold", 75)):
                return True
        return False

    def is_processed_success(item) -> bool:
        h = int(item["slot_hash"])
        y = int(item["box"][1])
        return in_set(h, y, processed_slots_success)

    def is_marked_skipped(item) -> bool:
        h = int(item["slot_hash"])
        y = int(item["box"][1])
        return in_set(h, y, processed_slots_skipped)

    frame = screenshot_bgr()
    items = find_item_names(frame) or []
    if FIND_STABILIZE_FRAMES > 1:
        base = {(it["name"], int(it["slot_hash"]), int(it["box"][1])) for it in items}
        for _ in range(FIND_STABILIZE_FRAMES - 1):
            time.sleep(0.1)
            frame2 = screenshot_bgr()
            items2 = find_item_names(frame2) or []
            curr = {(it["name"], int(it["slot_hash"]), int(it["box"][1])) for it in items2}
            base &= curr
        items = [it for it in items if (it["name"], int(it["slot_hash"]), int(it["box"][1])) in base]
    items.sort(key=lambda d: d["box"][1])
    dump_queue("initial", items)
    structured_log("list_scan", count=len(items))
    if page_hash == last_page_hash:
        items = [
            it
            for it in items
            if not in_set(int(it["slot_hash"]), int(it["box"][1]), processed_slots_success)
            and not in_set(int(it["slot_hash"]), int(it["box"][1]), processed_slots_skipped)
        ]
        structured_log("filter_same_page", same=True, kept=len(items))
    else:
        processed_slots_skipped.clear()
        processed_slots_success.clear()
        structured_log("filter_same_page", same=False, reset_sets=True)
    last_page_hash = page_hash
    while items:
        current = items[0]
        name = current["name"]
        cx, cy = current["center"]
        base_hash = int(current["slot_hash"])
        base_y = int(current["box"][1])
        if is_processed_success(current):
            items.pop(0)
            continue
        log(f"[ITEM] open '{name}' at {current['center']} y={base_y} hash={hex(base_hash)}")
        structured_log("focus_item", item=name, box=current["box"], center=current["center"], base_hash=hex(base_hash))
        if not ALLOW_PICKUP_OTHER and name not in ALLOWED_ITEM_NAMES:
            log(f"[FLOW] '{name}' не в whitelist — пропуск без открытия")
            structured_log("skip_non_whitelist", item=name, box=current["box"])
            processed_slots_success.add((base_hash, base_y))
            items.pop(0)
            continue
        tap(cx, cy, reason="open_item_card")
        panel_open = True
        time.sleep(max(0.001, CARD_OPEN_DELAY_MS / 1000.0))
        frame_card = screenshot_bgr()
        if frame_card is None:
            panel_open = collapse_if_open(panel_open, (cx, cy))
            processed_slots_skipped.add((base_hash, base_y))
            items.pop(0)
            continue
        H, W = frame_card.shape[:2]
        chat_top = detect_chat_top_y(frame_card)
        top = clamp(cy - 300, SAFE_Y1, SAFE_Y2 - 60)
        bottom = clamp(min(chat_top - 8, SAFE_Y2), top + 60, SAFE_Y2)
        card_rect = int(W * 0.56), top, int(W * 0.42), bottom - top
        st = swipe_until_visible_pickup(screenshot_bgr, base_y=base_y, card_rect=card_rect)
        state, box = st["state"], st.get("box")
        log(f"[PICKUP] state={state} score={st.get('score')} method={st.get('method')}")
        if state == "inactive":
            panel_open = collapse_if_open(panel_open, (cx, cy))
            if ABORT_ON_FIRST_INACTIVE:
                structured_log("inactive_abort_context", current={"name": name, "y": base_y})
                structured_log("exit_reason", reason="pickup_button_inactive")
                finalize_slot_lifecycle()
                ensure_tab_state("вещи", "закрыта")
                log("=== AUTO-LOOT END ===")
                return
            else:
                processed_slots_skipped.add((base_hash, base_y))
                items.pop(0)
                continue
        if state == "active" and box:
            bx, by = box[0] + box[2] // 2, box[1] + box[3] // 2
            tap(bx, by, reason="pickup")
            if verify_item_removed(name, base_hash, base_y):
                log(f"[FLOW] '{name}' подобран — подтверждено")
            else:
                log(f"[WARN] '{name}' не исчез — но помечаем обработанным")
            processed_slots_success.add((base_hash, base_y))
            update_slot_lifecycle([current], stage="picked_by_bot")
        if state == "out_of_view":
            processed_slots_skipped.add((base_hash, base_y))
            update_slot_lifecycle([current], stage="taken_by_other")
        rescan_sid = mark_step("rescan_items")
        structured_log("plan_rescan", step_id=rescan_sid, reason="post_item_action_or_scroll")
        frame = screenshot_bgr()
        page_hash = page_hash_from_bgr(frame)
        structured_log("act_rescan", step_id=rescan_sid, page_hash=int(page_hash))
        all_items = find_item_names(frame) or []
        filtered = [
            it for it in all_items if not in_set(int(it["slot_hash"]), int(it["box"][1]), processed_slots_success)
        ]
        items = sorted(filtered, key=lambda d: d["box"][1])
        dump_queue("after_rescan", items)
        structured_log("outcome_rescan", step_id=rescan_sid, count=len(items), page_hash=int(page_hash))
        last_page_hash = page_hash
        tabs = detect_tab_states(frame)
        if not items and tabs.get("occluded_by_extra"):
            structured_log("exit_reason", reason="items_tab_hidden_until_fight_end")
            log("[FLOW] прекращаем лут — вкладка вещей скрыта сражениями")
            finalize_slot_lifecycle()
            return
        if not items:
            pag_sid = mark_step("pagination")
            structured_log("plan_pagination", step_id=pag_sid, vector=[0, 800], duration_ms=600)
            take_frame = screenshot_bgr()
            before_path = dbg_save("before_pagination_swipe", take_frame)
            structured_log(
                "act_pagination",
                step_id=pag_sid,
                page_hash_before=int(page_hash_from_bgr(take_frame)),
                image_before=before_path,
            )
            swipe_dx, swipe_dy = 0, 800
            device_swipe(swipe_dx, swipe_dy, 600)
            time.sleep(0.5)
            take_frame = screenshot_bgr()
            after_path = dbg_save("after_pagination_swipe", take_frame)
            page_hash = page_hash_from_bgr(take_frame)
            last_page_hash = page_hash
            new_items = find_item_names(take_frame) or []
            filtered_new = []
            for it in new_items:
                ih, iy = int(it["slot_hash"]), int(it["box"][1])
                if in_set(ih, iy, processed_slots_success):
                    continue
                filtered_new.append(it)
            items = sorted(filtered_new, key=lambda d: d["box"][1])
            structured_log(
                "outcome_pagination",
                step_id=pag_sid,
                count=len(items),
                page_hash_after=int(page_hash),
                image_after=after_path,
            )
            tabs = detect_tab_states(take_frame)
            if tabs.get("occluded_by_extra"):
                structured_log("exit_reason", reason="items_tab_hidden_until_fight_end")
                log("[FLOW] прекращаем лут — вкладка вещей скрыта сражениями")
                finalize_slot_lifecycle()
                return
    finalize_slot_lifecycle()
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
