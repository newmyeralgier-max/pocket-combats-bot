import glob
import logging
import os
import re
import subprocess
import time
from pathlib import Path

import cv2

DEVICE = "192.168.0.100:5555"
TPL_DIR = "C:\\bot\\tpl\\my"
DEBUG_DIR = "C:\\bot\\debug"
ADB = "adb"
THR_UI = 0.85
THR_ITEM = 0.85
EXCLUDE_PREFIXES_ITEMS = (
    "monster_",
    "victory",
    "attack",
    "start",
    "items_",
    "popup_",
    "moves",
    "preffered_skill_text",
    "syschat",
    "default",
)
EXCLUDE_EXACT_ITEMS = {"monster_params.png", "monster_hp.png", "close_tab.png", "open_tab.png"}
PRIORITY_PREFIXES = "pickup_own", "pickup_other", "pickup_"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(message)s]", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("loot")


def adb_shell(cmd: str):
    return subprocess.run([ADB, "-s", DEVICE, "shell", cmd], capture_output=True, text=True)


def adb_pull(src: str, dst: str):
    return subprocess.run([ADB, "-s", DEVICE, "pull", src, dst], capture_output=True, text=True)


def screencap(save_path: str):
    tmp = "/sdcard/screen.png"
    adb_shell("screencap -p /sdcard/screen.png")
    res = adb_pull(tmp, save_path)
    size_info = f"{len(open(save_path, 'rb').read()):,}".replace(",", " ")
    log.info(f"[adb] pulled → {Path(save_path).name} ({size_info} bytes)")


def click(xy):
    x, y = xy
    adb_shell(f"input tap {int(x)} {int(y)}")


CYR_RE = re.compile("[А-Яа-яЁё]")


def find_template(tpl_path, image, threshold=0.85):
    tpl = cv2.imread(tpl_path)
    if tpl is None:
        log.warning(f"[vision] не удалось прочитать шаблон: {tpl_path}")
        return None, 0.0, None
    res = cv2.matchTemplate(image, tpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    if score >= threshold:
        h, w = tpl.shape[:2]
        x0, y0 = loc
        center = x0 + w // 2, y0 + h // 2
        rect = x0, y0, w, h
        return center, score, rect
    return None, float(score), None


def find_any(paths, image, threshold):
    best = None
    for p in paths:
        center, score, rect = find_template(p, image, threshold)
        if center:
            return {"tpl": p, "center": center, "score": score, "rect": rect}
        if best is None or score > best["score"]:
            best = {"tpl": p, "center": None, "score": score, "rect": None}
    return best


def list_png(dirpath):
    return sorted(glob.glob(os.path.join(dirpath, "*.png")))


def is_excluded_in_items(name: str) -> bool:
    base = name.lower()
    if base in EXCLUDE_EXACT_ITEMS:
        return True
    for pref in EXCLUDE_PREFIXES_ITEMS:
        if base.startswith(pref):
            return True
    if base in ("items_hdr.png", "items_header.png", "items_label.png", "items_tab.png"):
        return True
    return False


def load_templates_for_items():
    files = [f for f in list_png(TPL_DIR) if not CYR_RE.search(Path(f).stem)]
    pri, rest = [], []
    for f in files:
        name = Path(f).name
        if is_excluded_in_items(name):
            continue
        if name.startswith(PRIORITY_PREFIXES):
            pri.append(f)
        else:
            rest.append(f)
    ordered = sorted(pri) + sorted(rest)
    log.info(f"[tpl] загружено={len(ordered)} (items-only) dir={TPL_DIR}")
    return ordered


def ensure_items_context(image):
    check_keys = ["items_hdr.png", "items_header.png", "items_label.png"]
    for key in check_keys:
        p = os.path.join(TPL_DIR, key)
        center, sc, rect = find_template(p, image, THR_UI)
        if center:
            log.info(f"[state] items UI виден через {key} → score={sc:.3f}")
            roi = compute_items_roi(image.shape, rect)
            return True, roi
    tab = os.path.join(TPL_DIR, "items_tab.png")
    center, sc, _ = find_template(tab, image, THR_UI)
    if center:
        log.info(f"[action] открываем items_tab (score={sc:.3f})")
        click(center)
        time.sleep(0.5)
        tmp = Path(DEBUG_DIR, "_items_check.png")
        screencap(str(tmp))
        img2 = cv2.imread(str(tmp))
        for key in check_keys:
            p = os.path.join(TPL_DIR, key)
            center2, sc2, rect2 = find_template(p, img2, THR_UI)
            if center2:
                log.info(f"[state] items UI появился через {key} → score={sc2:.3f}")
                roi = compute_items_roi(img2.shape, rect2)
                return True, roi
    log.warning("[state] items UI не обнаружен")
    return False, None


def compute_items_roi(img_shape, header_rect):
    H, W = img_shape[0], img_shape[1]
    if header_rect:
        x0, y0, w, h = header_rect
        top = max(0, y0 + h + 8)
    else:
        top = 620
    bottom = max(top + 50, H - 140)
    left, right = 0, W
    roi = left, top, right, bottom
    log.info(f"[roi] items_roi={roi} (WxH={W}x{H})")
    return roi


def crop(image, roi):
    x0, y0, x1, y1 = roi
    return image[y0:y1, x0:x1]


def main():
    log.info("[start] loot_worker")
    log.info(f"[adb] устройство OK: {DEVICE}")
    Path(DEBUG_DIR).mkdir(parents=True, exist_ok=True)
    diag_path = os.path.join(DEBUG_DIR, f"diag_{time.strftime('%Y%m%d_%H%M%S')}.png")
    screencap(diag_path)
    image = cv2.imread(diag_path)
    if image is None:
        log.error("[error] не удалось загрузить скрин")
        return
    log.info(f"[scan] {DEVICE}")
    in_items, items_roi = ensure_items_context(image)
    if not in_items:
        log.info("[summary] skipped: items UI not visible")
        return
    templates = load_templates_for_items()
    if not templates:
        log.warning("[tpl] нет подходящих шаблонов для items-контекста")
        log.info("[summary] idle")
        return
    tried = 0
    clicked = False
    roi_img = crop(image, items_roi)
    pri = [p for p in templates if Path(p).name.startswith(PRIORITY_PREFIXES)]
    rest = [p for p in templates if p not in pri]
    for group_name, group in (("priority", pri), ("items", rest)):
        best = None
        best_score = -1.0
        for tpl in group:
            tried += 1
            center, score, rect = find_template(tpl, roi_img, THR_ITEM)
            if center:
                gx = items_roi[0] + center[0]
                gy = items_roi[1] + center[1]
                log.info(
                    f'[done] {DEVICE} -> click "{Path(tpl).name}" score={score:.3f} at ({gx}, {gy}) [{group_name}]'
                )
                click((gx, gy))
                log.info("[summary] accepted")
                clicked = True
                break
            if score > best_score:
                best_score = score
                best = Path(tpl).name
        if clicked:
            break
        log.info(f"[diag] no match in {group_name} (best={best} score={best_score:.3f})")
    if not clicked:
        log.info(f"[done] {DEVICE} -> nothing found (tried {tried} tpl within ROI)")
        log.info("[summary] idle")


if __name__ == "__main__":
    main()
