import json
import logging
import os
import shutil
import subprocess
import sys
import time
from typing import Optional, Tuple

from config_bot import (
    BASE_DIR,
    DEBUG_DIR,
    DEVICE_IP,
    DIAG_LOG,
    GAME_MAIN_ACTIVITY,
    GAME_PACKAGE,
    LOG_DIR,
    SCRIPT_DIR,
    TPL_ALPHA,
    TPL_BW,
    TPL_BW_INV,
    TPL_DIR,
    TPL_EDGES,
    TPL_MY,
    USE_WIFI_FIRST,
)

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(DIAG_LOG, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("diag")


def ensure_dirs():
    for p in [BASE_DIR, SCRIPT_DIR, TPL_DIR, TPL_MY, TPL_ALPHA, TPL_EDGES, TPL_BW, TPL_BW_INV, DEBUG_DIR, LOG_DIR]:
        os.makedirs(p, exist_ok=True)


def run(cmd, timeout=15, capture=True):
    try:
        if capture:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
            return (
                res.returncode,
                res.stdout.decode("utf-8", errors="ignore"),
                res.stderr.decode("utf-8", errors="ignore"),
            )
        else:
            res = subprocess.run(cmd, timeout=timeout)
            return res.returncode, "", ""
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except Exception as e:
        return 1, "", str(e)


def ensure_adb() -> bool:
    code, out, err = run(["adb", "version"])
    if code != 0:
        log.warning("ADB не найден в PATH. Попытка запустить сервер всё равно.")
    run(["adb", "start-server"])
    code, out, err = run(["adb", "version"])
    if code == 0:
        log.info(out.strip().splitlines()[0])
        return True
    log.error("ADB недоступен. Установи Android Platform-Tools и добавь adb в PATH.")
    return False


def list_devices():
    code, out, err = run(["adb", "devices"])
    devs = []
    for line in out.splitlines():
        if "\tdevice" in line and not line.lower().startswith("list of"):
            devs.append(line.split("\t")[0])
    return devs


def connect_wifi(ip_port: str) -> bool:
    if not ip_port:
        return False
    log.info(f"[wifi] подключение к {ip_port}...")
    code, out, err = run(["adb", "connect", ip_port])
    if "connected to" in out or "already connected" in out.lower():
        log.info(out.strip())
        return True
    log.warning(f"Wi‑Fi не подключён: {out or err}".strip())
    return False


def ensure_device(ip_pref: str, use_wifi_first=True) -> Optional[str]:
    if use_wifi_first and ip_pref:
        connect_wifi(ip_pref)
        time.sleep(0.5)
        devs = list_devices()
        if ip_pref in devs:
            return ip_pref
    devs = list_devices()
    if devs:
        if ip_pref and ip_pref in devs:
            return ip_pref
        return devs[0]
    return None


def ensure_python_deps():
    missing = []
    for mod in ["cv2", "numpy", "PIL", "pytesseract"]:
        try:
            __import__(mod if mod != "PIL" else "PIL.Image")
        except Exception:
            missing.append(mod)
    if not missing:
        log.info("Python зависимости уже есть (opencv, numpy, pillow, pytesseract).")
        return

    def pip_install(pkg):
        log.info(f"Устанавливаю {pkg} ...")
        code, out, err = run([sys.executable, "-m", "pip", "install", pkg], timeout=300)
        if code != 0:
            log.warning(f"Не удалось установить {pkg}: {out or err}")

    mapping = {"cv2": "opencv-python-headless", "numpy": "numpy", "PIL": "Pillow", "pytesseract": "pytesseract"}
    for m in missing:
        pip_install(mapping[m])


def ensure_tesseract_windows() -> bool:
    try:
        import pytesseract

        try:
            v = pytesseract.get_tesseract_version()
            log.info(f"Tesseract OK: {v}")
            return True
        except Exception:
            pass
        candidates = [
            "C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
            "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
        ]
        for c in candidates:
            if os.path.exists(c):
                pytesseract.pytesseract.tesseract_cmd = c
                try:
                    v = pytesseract.get_tesseract_version()
                    log.info(f"Tesseract найден: {v} ({c})")
                    return True
                except Exception:
                    continue
        log.warning("Tesseract не найден. OCR будет отключён, это не критично.")
        return False
    except Exception:
        log.warning("pytesseract не установлен — OCR будет отключён.")
        return False


def get_screen_resolution(serial: str) -> Optional[Tuple[int, int]]:
    code, out, err = run(["adb", "-s", serial, "shell", "wm", "size"])
    for line in out.splitlines():
        if "Physical size:" in line:
            try:
                s = line.split(":")[1].strip()
                w, h = s.split("x")
                return int(w), int(h)
            except Exception:
                pass
    return None


def is_process_running(serial: str, package: str) -> bool:
    code, out, err = run(["adb", "-s", serial, "shell", "pidof", package])
    return out.strip() != ""


def launch_game(serial: str, package: str, activity_full: str):
    comp = (
        f"{package}/{activity_full}"
        if not activity_full.startswith(package)
        else f"{package}/{activity_full.split('.')[-1]}"
    )
    if "/" not in comp:
        comp = f"{package}/{activity_full}"
    log.info(f"Запускаю игру: {comp}")
    run(["adb", "-s", serial, "shell", "am", "start", "-n", comp])


def run_all() -> dict:
    ensure_dirs()
    ok_adb = ensure_adb()
    ensure_python_deps()
    ensure_tesseract_windows()
    serial = ensure_device(DEVICE_IP, USE_WIFI_FIRST) if ok_adb else None
    if not serial:
        log.error("Устройство не найдено. Подключи USB или проверь IP:PORT.")
        return {"serial": None}
    res = get_screen_resolution(serial)
    if res:
        log.info(f"Разрешение: {res[0]}x{res[1]}")
    else:
        log.warning("Не удалось определить разрешение экрана.")
    if GAME_PACKAGE:
        if not is_process_running(serial, GAME_PACKAGE):
            launch_game(serial, GAME_PACKAGE, GAME_MAIN_ACTIVITY)
            time.sleep(2.0)
    return {"serial": serial, "resolution": res}
