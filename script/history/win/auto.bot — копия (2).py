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
SCREENS_DIR = os.path.join(BASE_DIR, "screens")
OUT_DIR = os.path.join(BASE_DIR, "out")
DEBUG_DIR = os.path.join(BASE_DIR, "_debug")
TPL_DIR_MY = os.path.join(BASE_DIR, "tpl", "my")
TPL_CHEVRONS_DIR = os.path.join(BASE_DIR, "tpl", "chevrons")
LOG_DIR = os.path.join(BASE_DIR, "log")
LOG_FILE = os.path.join(LOG_DIR, "run_log.txt")
os.makedirs(SCREENS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
try:
    sys.path.append(os.path.join(BASE_DIR, "lib"))
    from tab_detector import detect_tab_states
except Exception as e:
    print(
        "Не могу импортировать detect_tab_states из C:/bot/lib/script/loot/script/loot/script/loot/script/loot/script/loot/script/loot/script/loot/tab_detector.py"
    )
    raise
DEVICE_ID: Optional[str] = None
DRY_RUN = False
SLEEP_AFTER_TAP = 0.65
MAX_TAB_ATTEMPTS = 2
ITEM_NAME_THRESHOLD = 0.86
ITEM_SCALES = [0.95, 1.0, 1.05]
MAX_LINES_TO_COLLECT = 6
PICKUP_REL = 0.72, 0.83, 0.95, 0.94
COLOR_RATIO_THRESHOLD = 0.12
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
    "start.PNG",
    "items_hdr.png",
    "items_label.png",
}


def now_ts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def log(msg: str):
    line = f"[{now_ts()}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except:
        pass


def adb_cmd(args: List[str], timeout: float = 5.0) -> subprocess.CompletedProcess:
    base = ["adb"]
    if DEVICE_ID:
        base = ["adb", "-s", DEVICE_ID]
    return subprocess.run(base + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)


def imread_u8(path: str, flags: int = cv2.IMREAD_COLOR):
    try:
        with open(path, "rb") as f:
            data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, flags)
        return img
    except Exception as e:
        log(f"[IMG] read fail: {path} // {e}")
        return None


SAFE_X1, SAFE_X2 = 0, 1080
SAFE_Y1, SAFE_Y2 = 400, 2100


def tap(x: int, y: int, reason: str = ""):
    x = max(SAFE_X1, min(SAFE_X2, int(x)))
    y = max(SAFE_Y1, min(SAFE_Y2, int(y)))
    if DRY_RUN:
        log(f"DRY_RUN TAP at ({x},{y}){'  // ' + reason if reason else ''}")
        return
    adb_cmd(["shell", "input", "tap", str(x), str(y)])
    log(f"TAP at ({x},{y}){'  // ' + reason if reason else ''}")


def screenshot_bgr() -> np.ndarray:
    try:
        proc = adb_cmd(["exec-out", "screencap", "-p"], timeout=15.0)
    except subprocess.TimeoutExpired:
        log("[WARN] Скриншот не получен за 15с — повторная попытка")
        proc = adb_cmd(["exec-out", "screencap", "-p"], timeout=15.0)
    data = proc.stdout
    if not data:
        raise RuntimeError("Пустой скриншот из adb exec-out screencap -p")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Не удалось декодировать PNG скриншот")
    return img


def save_debug(img: np.ndarray, name: str):
    cv2.imwrite(os.path.join(DEBUG_DIR, name), img)


def to_gray(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.bilateralFilter(g, 5, 35, 35)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    return g


def match_scaled(
    gray_img: np.ndarray, tpl_bgr: np.ndarray, threshold: float, scales: List[float]
) -> Tuple[bool, float, Tuple[int, int, int, int]]:
    best = False, 0.0, (0, 0, 0, 0)
    tpl_gray_full = to_gray(tpl_bgr)
    th, tw = tpl_gray_full.shape[:2]
    H, W = gray_img.shape[:2]
    for s in scales:
        tws, ths = max(5, int(tw * s)), max(5, int(th * s))
        if H < ths or W < tws:
            continue
        tpl_gray = cv2.resize(tpl_gray_full, (tws, ths), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC)
        res = cv2.matchTemplate(gray_img, tpl_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best[1]:
            best = max_val >= threshold, float(max_val), (int(max_loc[0]), int(max_loc[1]), int(tws), int(ths))
    return best


def _scharr_mag(gray: np.ndarray) -> np.ndarray:
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def ensure_tab_state(tab_label: str, target_state: str) -> bool:
    """
    Приводит вкладку tab_label к состоянию target_state.
    Возвращает True, если удалось, False — если нет (сценарий продолжится).
    """
    frame = screenshot_bgr()
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
            time.sleep(SLEEP_AFTER_TAP)
            frame2 = screenshot_bgr()
            if detect_tab_states(frame2)[tab_label]["state"] == target_state:
                return True
    band_y = int(cur["band_y"])
    tap(180, band_y, reason=f"{tab_label}:band_line -> toggle")
    time.sleep(SLEEP_AFTER_TAP)
    frame3 = screenshot_bgr()
    if detect_tab_states(frame3)[tab_label]["state"] == target_state:
        return True
    log(f"[WARN] {tab_label}: state не меняется, пропускаем дальнейшие попытки")
    return False


def load_item_name_templates() -> List[Tuple[str, np.ndarray]]:
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


def find_item_names(frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
    H, W = frame_bgr.shape[:2]
    x1, x2 = 0, 500
    y1, y2 = 400, 2100
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
            ok, score, (x, y, w, h) = match_scaled(gg, tg, 0.83, ITEM_SCALES)
            if ok:
                boxes_scores.append(((x, y, w, h), score, name))
    log(f"[ITEMS] найдено совпадений: {len(boxes_scores)} в ROI=[{x1},{y1},{x2},{y2}] до группировки")
    keep = merge_same_lines(boxes_scores, line_thr=75)
    if len(keep) > MAX_LINES_TO_COLLECT:
        keep = sorted(keep, key=lambda i: boxes_scores[i][0][1])[:MAX_LINES_TO_COLLECT]
    found = []
    for i in keep:
        (x, y, w, h), score, name = boxes_scores[i]
        found.append(
            {
                "name": name,
                "score": float(score),
                "center": (x1 + x + w // 2, y1 + y + h // 2),
                "box": (x1 + x, y1 + y, w, h),
            }
        )
    log(f"[ITEMS] строк после группировки: {len(found)} (шаг≈150, порог=75)")
    return found


def merge_same_lines(boxes_scores: List[Tuple[Tuple[int, int, int, int], float, str]], line_thr: int = 75):
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


def rect_from_rel(w: int, h: int, rel: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    x1r, x2r, y1r, y2r = rel
    return int(w * x1r), int(h * y1r), int(w * x2r), int(h * y2r)


def pickup_state(frame_bgr: np.ndarray) -> Tuple[str, Optional[float]]:
    H, W = frame_bgr.shape[:2]
    gray = to_gray(frame_bgr)

    def try_tpl(tpl_bgr, label, best_state_score):
        best_state, best_score = best_state_score
        if tpl_bgr is None:
            return best_state, best_score
        ok, sc, _ = match_scaled(gray, tpl_bgr, 0.86, [0.9, 0.95, 1.0, 1.05])
        if ok and (best_score is None or sc > best_score):
            return label, float(sc)
        return best_state, best_score

    active_tpl_main = imread_u8(ui_tpl_path("pickup.png"))
    active_tpl_own = None
    inactive_tpl = imread_u8(ui_tpl_path("pickup_other.png"))
    best_state, best_score = None, None
    best_state, best_score = try_tpl(active_tpl_main, "active", (best_state, best_score))
    best_state, best_score = try_tpl(active_tpl_own, "active", (best_state, best_score))
    best_state, best_score = try_tpl(inactive_tpl, "inactive", (best_state, best_score))
    if best_state is not None:
        return best_state, best_score
    rx1, ry1, rx2, ry2 = rect_from_rel(W, H, PICKUP_REL)
    roi = frame_bgr[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        log("[WARN] Кнопка подбора вне экрана — предмет внизу, требуется скролл")
        return "out_of_view", None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, (1)] > 90) & (hsv[:, :, (2)] > 160)
    ratio = float(mask.sum()) / float(max(1, mask.size))
    log(f"[PICKUP] fallback color ratio={ratio:.3f} -> {'active' if ratio >= COLOR_RATIO_THRESHOLD else 'inactive'}")
    return "active" if ratio >= COLOR_RATIO_THRESHOLD else "inactive", ratio


def collapse_item_panel(frame_bgr: np.ndarray, last_click: Tuple[int, int]):
    x, y = last_click
    tap(x, y, reason="collapse_item_panel")
    time.sleep(0.35)


def auto_loot_once():
    log("=== AUTO-LOOT START (steps 8–12) ===")
    frame0 = screenshot_bgr()
    st = detect_tab_states(frame0)
    st_mon = st["монстры"]["state"]
    st_itm = st["вещи"]["state"]
    log(f"[FLOW] initial tabs: монстры={st_mon}, вещи={st_itm}")
    if st_mon == "открыта":
        if not ensure_tab_state("монстры", "закрыта"):
            log("[FLOW] Не удалось закрыть 'монстры'")
    frame1 = screenshot_bgr()
    st2 = detect_tab_states(frame1)
    if st2["вещи"]["state"] != "открыта":
        if not ensure_tab_state("вещи", "открыта"):
            log("[FLOW] Не удалось открыть 'вещи'")
    frame = screenshot_bgr()
    items = find_item_names(frame)
    if not items:
        log("[FLOW] В списке нет предметов")
        log("=== AUTO-LOOT END ===")
        return
    items.sort(key=lambda d: d["box"][1])
    for it in items:
        name = it["name"]
        cx, cy = it["center"]
        log(f"[ITEM] click '{name}' at ({cx}, {cy}) score={it['score']:.3f}")
        tap(cx, cy)
        time.sleep(0.45)
        frame2 = screenshot_bgr()
        state, score = pickup_state(frame2)
        log(f"[PICKUP] state={state} score={'%.3f' % score if score is not None else 'None'}")
        if state == "active":
            H, W = frame2.shape[:2]
            px1, py1, px2, py2 = rect_from_rel(W, H, PICKUP_REL)
            bx, by = (px1 + px2) // 2, (py1 + py2) // 2
            tap(bx, by, reason="pickup")
            time.sleep(0.45)
            collapse_item_panel(frame2, (cx, cy))
            time.sleep(0.25)
            continue
        state, score = pickup_state(frame2)
        if state == "out_of_view":
            log("[FLOW] Кнопка за пределами экрана — выходим из лута (нужен скролл)")
            break
        log(f"[FLOW] Первая неактивная кнопка у '{name}' — сворачиваем и завершаем лут")
        collapse_item_panel(frame2, (cx, cy))
        time.sleep(0.25)
        break
    log("=== AUTO-LOOT END ===")


def main():
    try:
        adb_cmd(["get-state"])
    except Exception as e:
        log(f"ADB недоступен: {e}")
        return
    auto_loot_once()


if __name__ == "__main__":
    main()
