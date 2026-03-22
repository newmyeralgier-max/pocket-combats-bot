import os
import re
import subprocess
from typing import List, Optional


def _adb_cmd(serial: str, *args: str):
    return ["adb", "-s", serial, *args]


def _adb_out(serial: str, args: List[str], decode: bool = True):
    proc = subprocess.run(_adb_cmd(serial, *args), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if decode:
        return proc.stdout.decode("utf-8", errors="ignore")
    return proc.stdout


def tap(serial: str, x: int, y: int):
    _adb_out(serial, ["shell", "input", "tap", str(x), str(y)])


def swipe(serial: str, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 200):
    _adb_out(serial, ["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)])


def screenshot(serial: str, save_to: Optional[str] = None):
    png = _adb_out(serial, ["exec-out", "screencap", "-p"], decode=False)
    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        with open(save_to, "wb") as f:
            f.write(png)
        return save_to
    return png


def _extract_pkg(text: str) -> str:
    patterns = [
        "mCurrentFocus=.*?\\s([A-Za-z0-9\\._]+)/",
        "mFocusedApp.*?ActivityRecord\\{.*?\\s([A-Za-z0-9\\._]+)/",
        "topResumedActivity:.*?\\s([A-Za-z0-9\\._]+)/",
        "ResumedActivity:.*?\\s([A-Za-z0-9\\._]+)/",
        "mResumedActivity:.*?\\s([A-Za-z0-9\\._]+)/",
        "\\bACTIVITY\\s+([A-Za-z0-9\\._]+)/",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return ""


def get_current_package(serial: str) -> str:
    for args in [
        ["shell", "dumpsys", "window", "windows"],
        ["shell", "dumpsys", "activity", "activities"],
        ["shell", "cmd", "activity", "top"],
    ]:
        try:
            out = _adb_out(serial, args)
            pkg = _extract_pkg(out)
            if pkg:
                return pkg
        except Exception:
            pass
    return ""


def is_process_running(serial: str, package: str) -> bool:
    try:
        out = _adb_out(serial, ["shell", "pidof", package])
        if out.strip():
            return True
    except Exception:
        pass
    try:
        out = _adb_out(serial, ["shell", "ps", "-A"])
        for line in out.splitlines():
            if package in line and "grep" not in line:
                return True
    except Exception:
        pass
    return False


def launch_game(serial: str, package: str, activity: Optional[str] = None):
    try:
        if activity:
            comp = f"{package}/{activity}" if activity.startswith(".") else f"{package}/{activity}"
            _adb_out(serial, ["shell", "am", "start", "-n", comp])
        else:
            _adb_out(serial, ["shell", "monkey", "-p", package, "-c", "android.intent.category.LAUNCHER", "1"])
    except Exception as e:
        print(f"[launch_game] error: {e}")
