import subprocess
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np


class ADB:

    def __init__(self, serial: Optional[str] = None, adb_path: str = "adb"):
        self.adb_path = adb_path
        self.serial = serial or self._auto_pick_serial()

    def _auto_pick_serial(self) -> Optional[str]:
        try:
            out = subprocess.check_output([self.adb_path, "devices"], stderr=subprocess.STDOUT, timeout=5)
            lines = out.decode("utf-8", "ignore").strip().splitlines()
            devices = []
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 2 and parts[1] == "device":
                    devices.append(parts[0])
            if len(devices) == 1:
                return devices[0]
            elif len(devices) == 0:
                print("[ADB] Нет подключённых устройств. Выполни: adb connect HOST:PORT")
                return None
            else:
                print("[ADB] Несколько устройств. Укажи serial в ADB(...).")
                return None
        except Exception as e:
            print(f"[ADB] Не удалось получить список устройств: {e}")
            return None

    def _base_cmd(self) -> List[str]:
        cmd = [self.adb_path]
        if self.serial:
            cmd += ["-s", self.serial]
        return cmd

    def run(self, args, timeout: int = 5) -> subprocess.CompletedProcess:
        cmd = self._base_cmd() + args
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)

    def sh(self, shell_cmd: str, timeout: int = 5) -> subprocess.CompletedProcess:
        return self.run(["shell", shell_cmd], timeout=timeout)

    def connect(self, ip_port: str, retries: int = 3, wait_s: float = 1.5) -> bool:
        for i in range(1, retries + 1):
            cp = self.run(["connect", ip_port], timeout=5)
            out = (cp.stdout or b"").decode("utf-8", "ignore").lower()
            if "connected to" in out or "already connected" in out:
                print(f"[ADB] Connected to {ip_port}")
                return True
            print(f"[ADB] connect attempt {i} failed: {out.strip()}")
            time.sleep(wait_s)
        return False

    def tap(self, x: int, y: int):
        self.sh(f"input tap {int(x)} {int(y)}")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 220):
        self.sh(f"input swipe {int(x1)} {int(y1)} {int(x2)} {int(y2)} {int(duration_ms)}")

    def key_back(self):
        self.sh("input keyevent 4")

    def get_size(self) -> Tuple[int, int]:
        try:
            cp = self.sh("wm size", timeout=3)
            txt = (cp.stdout or b"").decode("utf-8", "ignore")
            for line in txt.splitlines():
                if "Physical size" in line:
                    part = line.split(":")[1].strip()
                    w, h = part.split("x")
                    return int(w), int(h)
        except Exception:
            pass
        img = self.screencap_cv()
        if img is None:
            return 0, 0
        h, w = img.shape[:2]
        return w, h

    def screencap_cv(self, retries: int = 3, pause_s: float = 0.08) -> Optional[np.ndarray]:
        last_err = None
        for _ in range(retries):
            try:
                cp = self.run(["exec-out", "screencap", "-p"], timeout=5)
                if cp.returncode == 0 and cp.stdout:
                    arr = np.frombuffer(cp.stdout, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None and img.size > 0:
                        return img
            except Exception as e:
                last_err = e
            time.sleep(pause_s)
        if last_err:
            print(f"[ADB] screencap_cv: {last_err}")
        return None
