import os
import subprocess
import time
from datetime import datetime

import cv2
import numpy as np

DRY_RUN = True
BOT_ROOT = "C:\\bot"
TEMPLATE_PATH = os.path.join(BOT_ROOT, "tpl", "служебные")
MY_TPL = os.path.join(TEMPLATE_PATH, "my")
CHEVRONS = os.path.join(TEMPLATE_PATH, "chevrons")
LOG_PATH = os.path.join(BOT_ROOT, "loot_run.txt")
ADB_PATH = "adb"
SCREEN_PATH = os.path.join(BOT_ROOT, "screens", "screen.png")
TAB_NAMES = {"monsters": os.path.join(MY_TPL, "monsters_tab.png"), "items": os.path.join(MY_TPL, "items_tab.png")}
CHEVRON_TPLS = {
    "monsters_open": os.path.join(CHEVRONS, "monsters_open.png"),
    "monsters_closed": os.path.join(CHEVRONS, "monsters_closed.png"),
    "items_open": os.path.join(CHEVRONS, "items_open.png"),
    "items_closed": os.path.join(CHEVRONS, "items_closed.png"),
}
TAB_COORDS = {"monsters": (150, 850), "items": (150, 1050)}
SWIPE_UP = "620", "700", "620", "300", "100"
SWIPE_DOWN = "620", "300", "620", "700", "100"
MATCH_THRESH = 0.87
CHEVRON_X_ZONE = 960, 1040


def log(text):
    timestamp = datetime.now().strftime("%H:%M:%S")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {text}\n")


def log_header():
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 50 + "\n" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n" + "=" * 50 + "\n")


log_header()


def load_template(path):
    if not os.path.exists(path):
        log(f"❌ Шаблон не найден: {path}")
        return None
    tpl = cv2.imread(path)
    if tpl is None or tpl.shape[0] < 2 or tpl.shape[1] < 2:
        log(f"❌ Пустой/битый шаблон: {path}")
        return None
    return tpl


def run_adb(args):
    result = subprocess.run([ADB_PATH] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout.strip()


def tap(x, y):
    if DRY_RUN:
        log(f"[DRY] Тап по: {x}, {y}")
    else:
        run_adb(["shell", "input", "tap", str(x), str(y)])


def swipe(cmd):
    if DRY_RUN:
        log(f"[DRY] Свайп: {' '.join(cmd)}")
    else:
        run_adb(["shell", "input", "swipe"] + list(cmd))


def get_screen():
    run_adb(["shell", "screencap", "-p", "/sdcard/screen.png"])
    run_adb(["pull", "/sdcard/screen.png", SCREEN_PATH])
    img = cv2.imread(SCREEN_PATH)
    return img


def match_template(img, tpl, roi=None, threshold=MATCH_THRESH):
    if roi:
        x, y, w, h = roi
        if w < tpl.shape[1] or h < tpl.shape[0]:
            new_w = max(w, tpl.shape[1])
            new_h = max(h, tpl.shape[0])
            log(f"⚠️ ROI расширен: {x, y, new_w, new_h}")
            w, h = new_w, new_h
        roi_img = img[y : y + h, x : x + w]
    else:
        roi_img = img
    res = cv2.matchTemplate(roi_img, tpl, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val >= threshold:
        loc = max_loc
        log(f"✅ Найдено совпадение: {max_val:.3f} в {loc}")
        return loc[0] + x if roi else loc[0], loc[1] + y if roi else loc[1]
    else:
        log(f"🔍 Совпадение недостаточное: {max_val:.3f}")
        return None


def ensure_tab_state(screen, tab_key, desired_state):
    chevron_tpl = load_template(CHEVRON_TPLS[f"{tab_key}_{desired_state}"])
    x1, x2 = CHEVRON_X_ZONE
    roi = x1, 0, x2 - x1, screen.shape[0]
    pos = match_template(screen, chevron_tpl, roi=roi)
    if pos:
        log(f"✔ Вкладка '{tab_key}' уже в состоянии: {desired_state}")
        return True
    else:
        tab_tpl = load_template(TAB_NAMES[tab_key])
        retry = 0
        while retry < 3:
            screen = get_screen()
            tab_pos = match_template(screen, tab_tpl)
            if tab_pos:
                tap(tab_pos[0], tab_pos[1])
                log(f"⚙ Переключено '{tab_key}' → {desired_state}")
                time.sleep(1)
                return True
            swipe(SWIPE_DOWN)
            time.sleep(0.5)
            retry += 1
        log(f"❌ Не удалось найти/переключить вкладку '{tab_key}'")
        return False


def main():
    screen = get_screen()
    success = ensure_tab_state(screen, "monsters", "open")
    if not success:
        return
    screen = get_screen()
    success = ensure_tab_state(screen, "items", "closed")
    if not success:
        return
    log("🏁 Стартовое состояние установлено: Монстры=open, Вещи=closed")


if __name__ == "__main__":
    main()
