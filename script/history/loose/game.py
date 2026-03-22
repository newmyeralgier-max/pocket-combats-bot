"""
game_auto_adb.py
Poco X4 GT | ADB over Wi‑Fi | .com.pocketcombats | Resolution: 1080x2460

Paths:
- Templates: C:/bot/tpl/my
- Logs:      C:/bot/logs
- Debug:     C:/bot/screens

Scope:
- 1) Close overlays (syschat, monster_params) via close_1/close_2
- 2) Tabs baseline: Items closed, Monsters open
- 3) Scroll down
- 4) Tap attack icon
- 5) Combat: open moves, try preferred skill, else normal attack
- 6) Victory → Continue
- 7) Scroll down
- 8) Close Monsters
- 9) Open Items
- 10) Loot first item popup: pickup (active=pickup.png/pickup_own.png; inactive=pickup_other.png)
- 11) Restore windows
- All tabs (except overlays) toggle by name taps
"""

import os
import random
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

USER_DEVICE_IP = "192.168.0.100:5555"
PACKAGE = "com.pocketcombats"
LAUNCH_VIA_MONKEY = True
GAME_ACTIVITY = "com.pocketcombats/.MainActivity"
DEVICE_W, DEVICE_H = 1080, 2460
ADB_BIN = "adb"
ADB_SERIAL = None
DRY_RUN = False
TPL_DIR = "C:/bot/tpl/my"
LOGS_DIR = "C:/bot/logs"
DEBUG_DIR = "C:/bot/screens"
THRESH_DEFAULT = 0.84
THRESH_STRICT = 0.88
SCALES = [0.92, 0.96, 1.0, 1.04, 1.08]
JITTER_MIN, JITTER_MAX = 0.1, 0.35
POST_TAP_PAUSE = 0.12
POST_SWIPE_PAUSE = 0.25
SCROLL_DOWN = int(DEVICE_W * 0.5), int(DEVICE_H * 0.8), int(DEVICE_W * 0.5), int(DEVICE_H * 0.35), 280
TPL = {
    "syschat": "syschat.png",
    "mparams": "monster_params.png",
    "close1": "close_1.png",
    "close2": "close_2.png",
    "items_tab": "items_tab.png",
    "items_label": "items_label.png",
    "monster_tab": "monster_tab.png",
    "monster_hdr": "monster_hdr.png",
    "attack": "attack.png",
    "moves": "moves.png",
    "preferred_skill": "preffered_skill.png",
    "preferred_skill_txt": "preffered_skill_text.png",
    "victory": "victory.png",
    "continue": "continue.png",
    "popup_any": "popup_auto.png",
    "pickup_active": "pickup.png",
    "pickup_active_alt": "pickup_own.png",
    "pickup_inactive": "pickup_other.png",
}
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOGS_DIR, "autobattle.log")


def log(s: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} | {s}"
    print(line)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def jitter(min_s=JITTER_MIN, max_s=JITTER_MAX):
    time.sleep(random.uniform(min_s, max_s))


def adb_cmd() -> List[str]:
    base = [ADB_BIN]
    if ADB_SERIAL:
        base += ["-s", ADB_SERIAL]
    return base


def adb_run(args: List[str], capture=False, binary=False, timeout=None):
    cmd = adb_cmd() + args
    log(f"[ADB] {' '.join(cmd)}")
    if capture:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    else:
        return subprocess.run(cmd, timeout=timeout)


def adb_connect(ip_port: str, retries=3, wait_s=1.5) -> bool:
    for i in range(1, retries + 1):
        res = adb_run(["connect", ip_port], capture=True)
        out = (res.stdout or b"").decode("utf-8", errors="ignore").lower()
        if "connected to" in out or "already connected to" in out:
            log(f"[OK] Connected to {ip_port}")
            return True
        log(f"[WARN] connect attempt {i} failed: {out.strip()}")
        time.sleep(wait_s)
    return False


def adb_devices() -> List[str]:
    res = adb_run(["devices"], capture=True)
    out = (res.stdout or b"").decode("utf-8", errors="ignore").splitlines()
    devices = []
    for line in out[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            devices.append(parts[0])
    return devices


def adb_tap(x: int, y: int, tag=""):
    if DRY_RUN:
        log(f"[DRY] tap {tag} ({x},{y})")
        return
    adb_run(["shell", "input", "tap", str(x), str(y)])
    time.sleep(POST_TAP_PAUSE)
    jitter()


def adb_swipe(x1: int, y1: int, x2: int, y2: int, duration_ms=250, tag=""):
    if DRY_RUN:
        log(f"[DRY] swipe {tag} ({x1},{y1})->({x2},{y2}) {duration_ms}ms")
        return
    adb_run(["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)])
    time.sleep(POST_SWIPE_PAUSE)
    jitter()


def adb_key(keycode: int, tag=""):
    if DRY_RUN:
        log(f"[DRY] key {tag} code={keycode}")
        return
    adb_run(["shell", "input", "keyevent", str(keycode)])
    jitter()


def adb_start_game():
    if LAUNCH_VIA_MONKEY:
        adb_run(["shell", "monkey", "-p", PACKAGE, "-c", "android.intent.category.LAUNCHER", "1"])
    else:
        adb_run(["shell", "am", "start", "-n", GAME_ACTIVITY])


def adb_screencap_cv() -> Optional[np.ndarray]:
    try:
        res = adb_run(["exec-out", "screencap", "-p"], capture=True, timeout=5)
        data = res.stdout
        if not data:
            log("[ERR] screencap returned empty")
            return None
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        log(f"[ERR] screencap: {e}")
        return None


def save_debug_img(img_bgr: np.ndarray, prefix: str):
    try:
        fname = f"{prefix}_{int(time.time() * 1000)}.png"
        cv2.imwrite(os.path.join(DEBUG_DIR, fname), img_bgr)
    except Exception:
        pass


TPL_CACHE: Dict[str, np.ndarray] = {}


def load_tpl(key: str, as_gray=True) -> np.ndarray:
    global TPL_CACHE
    if key in TPL_CACHE:
        return TPL_CACHE[key]
    fname = TPL.get(key)
    if not fname:
        raise ValueError(f"No template key: {key}")
    path = os.path.join(TPL_DIR, fname)
    flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
    tpl = cv2.imread(path, flag)
    if tpl is None:
        raise FileNotFoundError(f"Template not found: {path}")
    TPL_CACHE[key] = tpl
    return tpl


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def match_one(img_gray: np.ndarray, tpl_gray: np.ndarray, threshold=THRESH_DEFAULT, scales=SCALES):
    best = None
    for s in scales:
        h = max(8, int(tpl_gray.shape[0] * s))
        w = max(8, int(tpl_gray.shape[1] * s))
        tpl_r = cv2.resize(tpl_gray, (w, h), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC)
        res = cv2.matchTemplate(img_gray, tpl_r, cv2.TM_CCOEFF_NORMED)
        _, maxVal, _, maxLoc = cv2.minMaxLoc(res)
        if maxVal >= threshold and (best is None or maxVal > best["score"]):
            best = {"pt": maxLoc, "wh": (w, h), "score": float(maxVal)}
    return best


def find_key(img_bgr: np.ndarray, key: str, threshold=THRESH_DEFAULT):
    tpl = load_tpl(key)
    m = match_one(to_gray(img_bgr), tpl, threshold=threshold)
    if m:
        x, y = m["pt"]
        w, h = m["wh"]
        return x, y, w, h, m["score"]
    return None


def find_any(img_bgr: np.ndarray, keys: List[str], threshold=THRESH_DEFAULT):
    best = None
    for k in keys:
        r = find_key(img_bgr, k, threshold=threshold)
        if r and (best is None or r[4] > best[4]):
            best = k, *r
    return best


def wait_for(keys: List[str], timeout=6.0, threshold=THRESH_DEFAULT):
    t0 = time.time()
    while time.time() - t0 < timeout:
        img = adb_screencap_cv()
        if img is None:
            continue
        hit = find_any(img, keys, threshold=threshold)
        if hit:
            return img, hit
        time.sleep(0.15)
    return None, None


def center(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x, y, w, h = box
    return x + w // 2, y + h // 2


def tap_on_key(key: str, threshold=THRESH_DEFAULT, tag=""):
    img = adb_screencap_cv()
    if img is None:
        log(f"[{tag}] screencap failed")
        return False
    hit = find_key(img, key, threshold=threshold)
    if not hit:
        log(f"[{tag}] '{key}' not found")
        return False
    x, y, w, h, sc = hit
    cx, cy = center((x, y, w, h))
    log(f"[{tag}] tap '{key}' score={sc:.3f} at ({cx},{cy})")
    adb_tap(cx, cy, tag=key)
    return True


def swipe_down(tag="scroll"):
    x1, y1, x2, y2, d = SCROLL_DOWN
    log(f"[{tag}] swipe down")
    adb_swipe(x1, y1, x2, y2, d, tag=tag)


def close_overlays(max_clicks=3):
    for i in range(max_clicks):
        img = adb_screencap_cv()
        if img is None:
            continue
        close_hit = find_any(img, ["close1", "close2"], threshold=THRESH_STRICT)
        if not close_hit:
            if not find_any(img, ["syschat", "mparams"], threshold=THRESH_DEFAULT):
                log("[overlay] no overlays")
                return True
            else:
                log("[overlay] overlays remain but no close button; retry")
                time.sleep(0.2)
                continue
        _, x, y, w, h, sc = close_hit
        cx, cy = center((x, y, w, h))
        log(f"[overlay] close via score={sc:.2f} at ({cx},{cy})")
        adb_tap(cx, cy, tag="close_overlay")
        time.sleep(0.25)
    return False


def ensure_tabs_baseline():
    """
    Требуемый вид:
    - Items closed
    - Monsters open
    Используем два признака:
      items_label виден => items открыт (нужно закрыть)
      monster_hdr или наличие monster_tab => monsters открыть
    """
    ok = True
    img = adb_screencap_cv()
    if img is None:
        return False
    items_open = find_key(img, "items_label", threshold=THRESH_STRICT) is not None
    if items_open:
        log("[tabs] items are open → closing by tapping items_tab")
        if not tap_on_key("items_tab", threshold=THRESH_DEFAULT, tag="tabs"):
            ok = False
        time.sleep(0.25)
    img2 = adb_screencap_cv() or img
    monsters_visible = find_key(img2, "monster_hdr", threshold=THRESH_DEFAULT) is not None
    if not monsters_visible:
        log("[tabs] monsters likely closed → tapping monster_tab")
        if not tap_on_key("monster_tab", threshold=THRESH_DEFAULT, tag="tabs"):
            ok = False
        time.sleep(0.25)
    return ok


def do_attack_flow():
    swipe_down(tag="pre-attack-scroll")
    if not tap_on_key("attack", threshold=THRESH_DEFAULT, tag="attack"):
        log("[attack] attack icon not found")
        return False
    img, hit = wait_for(["moves"], timeout=6.0, threshold=THRESH_DEFAULT)
    if not hit:
        log("[combat] 'moves' not appeared; try anyway")
    else:
        tap_on_key("moves", threshold=THRESH_DEFAULT, tag="moves")
    if tap_on_key("preferred_skill", threshold=THRESH_DEFAULT, tag="skill") or tap_on_key(
        "preferred_skill_txt", threshold=THRESH_DEFAULT, tag="skill_txt"
    ):
        log("[combat] used preferred skill")
    else:
        log("[combat] preferred not found → fallback to attack")
        tap_on_key("attack", threshold=THRESH_DEFAULT, tag="attack_fallback")
    img, hit = wait_for(["victory"], timeout=10.0, threshold=THRESH_DEFAULT)
    if hit:
        log("[combat] Victory detected")
        time.sleep(0.5)
        tap_on_key("continue", threshold=THRESH_DEFAULT, tag="continue")
        time.sleep(0.4)
    else:
        log("[combat] Victory not detected in time")
    return True


def close_monsters_open_items():
    swipe_down(tag="post-combat-scroll")
    img = adb_screencap_cv()
    if img is not None and find_key(img, "monster_hdr", threshold=THRESH_DEFAULT):
        tap_on_key("monster_tab", threshold=THRESH_DEFAULT, tag="close_monsters")
    tap_on_key("items_tab", threshold=THRESH_DEFAULT, tag="open_items")
    time.sleep(0.25)


def open_first_item_popup():
    """
    Открываем попап первого предмета.
    Стратегия:
      1) Открыта вкладка items (см. выше).
      2) Ищем popup_any — если уже открыт, ок.
      3) Иначе пробуем тапать в область под items_label:
         два вертикальных оффсета (при необходимости подстроить).
    """
    img = adb_screencap_cv()
    if img is None:
        return False
    if find_key(img, "popup_any", threshold=THRESH_DEFAULT):
        log("[loot] popup already open")
        return True
    label = find_key(img, "items_label", threshold=THRESH_STRICT)
    if not label:
        log("[loot] items_label not found")
        return False
    x, y, w, h, sc = label
    candidates = [(x + 40, y + h + 60), (x + 40, y + h + 110)]
    for i, (tx, ty) in enumerate(candidates, 1):
        log(f"[loot] try open item #{i} at ({tx},{ty})")
        adb_tap(tx, ty, tag=f"item_{i}")
        time.sleep(0.35)
        img2 = adb_screencap_cv()
        if img2 is not None and find_key(img2, "popup_any", threshold=THRESH_DEFAULT):
            log("[loot] popup opened")
            return True
    log("[loot] failed to open popup")
    return False


def detect_pickup_state() -> Tuple[str, Optional[Tuple[int, int, int, int]]]:
    img = adb_screencap_cv()
    if img is None:
        return "unknown", None
    hit = find_any(img, ["pickup_active", "pickup_active_alt"], threshold=THRESH_DEFAULT)
    if hit:
        _, x, y, w, h, sc = hit
        return "active", (x, y, w, h)
    ina = find_key(img, "pickup_inactive", threshold=THRESH_DEFAULT)
    if ina:
        x, y, w, h, sc = ina
        return "inactive", (x, y, w, h)
    if not find_key(img, "popup_any", threshold=THRESH_DEFAULT):
        return "no_popup", None
    return "unknown", None


def loot_flow_until_inactive(max_clicks=15):
    """
    - Открыть попап первого предмета
    - Пока 'подобрать' активна — нажимать
    - Как только неактивна — закрыть попап и завершить лут
    """
    if not open_first_item_popup():
        log("[loot] cannot open first item popup")
        return False
    total = 0
    while total < max_clicks:
        state, box = detect_pickup_state()
        if state == "active" and box:
            cx, cy = center(box)
            log(f"[loot] pickup active → tap at ({cx},{cy})")
            adb_tap(cx, cy, tag="pickup")
            total += 1
            time.sleep(0.35)
            continue
        elif state in ("inactive", "unknown"):
            log(f"[loot] pickup {state} → close popup")
            img = adb_screencap_cv()
            if img is not None:
                pop = find_key(img, "popup_any", threshold=THRESH_DEFAULT)
                if pop:
                    px, py, pw, ph, _ = pop
                    adb_tap(px + pw // 2, max(0, py - 20), tag="close_popup_head")
                    time.sleep(0.25)
            return True
        elif state == "no_popup":
            log("[loot] popup disappeared → try reopen")
            if not open_first_item_popup():
                return True
        else:
            log("[loot] state unknown → small wait")
            time.sleep(0.25)
    log("[loot] max_clicks reached")
    return True


def restore_windows():
    """
    Привести окна к исходному виду:
    - Items closed
    - Monsters open
    """
    return ensure_tabs_baseline()


def run_cycle_once():
    close_overlays()
    ensure_tabs_baseline()
    do_attack_flow()
    close_monsters_open_items()
    loot_flow_until_inactive(max_clicks=20)
    restore_windows()


def ensure_connected_and_started():
    if not adb_connect(USER_DEVICE_IP, retries=4, wait_s=1.5):
        log("[FATAL] cannot connect to device")
        sys.exit(1)
    devs = adb_devices()
    if not devs:
        log("[FATAL] no adb devices")
        sys.exit(1)
    if not ADB_SERIAL:
        log(f"[INFO] active device(s): {devs}")
    adb_start_game()
    time.sleep(2.0)


def graceful_exit(signum, frame):
    log(f"[EXIT] signal {signum}")
    sys.exit(0)


def main():
    random.seed(int(time.time()))
    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)
    log("=== game_auto_adb started ===")
    ensure_connected_and_started()
    try:
        while True:
            run_cycle_once()
            time.sleep(1.0)
    except KeyboardInterrupt:
        graceful_exit("KB", None)


if __name__ == "__main__":
    main()
