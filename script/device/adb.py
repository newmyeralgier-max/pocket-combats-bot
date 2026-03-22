# -*- coding: utf-8 -*-
import os
import time
import subprocess
import random
from typing import List, Optional

import cv2
import numpy as np

from script.core.config import (
    CFG, DEVICE_ID, MAX_SCREENSHOT_ATTEMPTS, DRY_RUN, SAVE_DEBUG_IMAGES, DEBUG_DIR,
    SAFE_X1, SAFE_Y1, SAFE_X2, SAFE_Y2, SLEEP_AFTER_TAP,
    SWIPE_PAUSE_MS
)
from script.core.logging import log, structured_log, dbg_name, snap, now_ts, PERF
from script.screen.capture import create_capture, ScreenCapture
from script.loot.utils import clamp

_CAPTURE_OBJ: Optional[ScreenCapture] = None

class FastADB:
    def __init__(self, device_id: Optional[str] = None):
        self.device_id = device_id
        self.process: Optional[subprocess.Popen] = None
        self._start_shell()

    def _start_shell(self):
        if self.process:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except:
                try:
                    self.process.kill()
                except:
                    pass
        cmd = ["adb"]
        if self.device_id:
            cmd.extend(["-s", str(self.device_id)])
        cmd.append("shell")
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            text=True
        )
        log(f"[INFO] Persistent ADB shell started for {self.device_id or 'default'}")

    def run_cmd(self, cmd_str: str):
        if not self.process or self.process.poll() is not None:
            self._start_shell()
        
        proc = self.process
        if not proc or not proc.stdin:
            log("[ERROR] FastADB process or stdin is missing")
            return

        try:
            proc.stdin.write(cmd_str + "\n")
            proc.stdin.flush()
        except Exception as e:
            log(f"[ERROR] FastADB write failed: {e}")
            self._start_shell()
            proc = self.process
            if proc and proc.stdin:
                try:
                    proc.stdin.write(cmd_str + "\n")
                    proc.stdin.flush()
                except:
                    log("[ERROR] FastADB recovery failed")


_FAST_ADB: Optional[FastADB] = None

def get_fast_adb() -> FastADB:
    global _FAST_ADB
    if _FAST_ADB is None:
        _FAST_ADB = FastADB(DEVICE_ID)
    return _FAST_ADB

def get_capture() -> ScreenCapture:
    global _CAPTURE_OBJ
    if _CAPTURE_OBJ is None:
        method = CFG.get("CAPTURE_METHOD", "adb")
        window_title = CFG.get("CAPTURE_WINDOW_TITLE", "BlueStacks")
        _CAPTURE_OBJ = create_capture(
            method=method,
            device_id=DEVICE_ID,
            max_retries=MAX_SCREENSHOT_ATTEMPTS,
            backoff_base=0.5,
            window_title=window_title
        )
    return _CAPTURE_OBJ

def screenshot_bgr(attempts: Optional[int] = None) -> Optional[np.ndarray]:
    with PERF.measure("screenshot"):
        cap = get_capture()
        if attempts is not None and hasattr(cap, "max_retries"):
            cap.max_retries = attempts
        img = cap.grab()
        if img is None:
            return None
        if SAVE_DEBUG_IMAGES:
            cv2.imwrite(os.path.join(DEBUG_DIR, f"{now_ts()}_{dbg_name('raw')}.png"), img)
        return img

def adb_cmd(args: List[str], timeout: float = 5.0, max_retries: int = 3, backoff_base: float = 0.5) -> subprocess.CompletedProcess:
    base = ["adb"]
    if DEVICE_ID:
        base = ["adb", "-s", DEVICE_ID]
    for attempt in range(max_retries):
        try:
            return subprocess.run(base + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        except subprocess.TimeoutExpired:
            log(f"[WARN] ADB command timeout on attempt {attempt + 1}/{max_retries}")
            if attempt == max_retries - 1:
                log("[ERROR] ADB recovery failed after max retries")
                raise
            if attempt == max_retries - 2:
                log("[WARN] Restarting ADB server...")
                os.system("adb kill-server && adb start-server")
            time.sleep(backoff_base * (2 ** attempt))
    raise RuntimeError("Unexpected end of adb_cmd")

def tap(x: int, y: int, reason: str = ""):
    with PERF.measure("tap"):
        x = max(SAFE_X1, min(SAFE_X2, int(x)))
        y = max(SAFE_Y1, min(SAFE_Y2, int(y)))
        structured_log("tap", x=x, y=y, reason=reason)
        if DRY_RUN:
            log(f"DRY_RUN TAP at ({x},{y}){'  // ' + reason if reason else ''}")
            time.sleep(SLEEP_AFTER_TAP)
            return
        try:
            get_fast_adb().run_cmd(f"input touchscreen tap {x} {y}")
            log(f"TAP at ({x},{y}){'  // ' + reason if reason else ''}")
        except Exception as e:
            log(f"[ERROR] tap adb failed: {e}")
        time.sleep(SLEEP_AFTER_TAP)

def tap_raw(x: int, y: int, reason: str = ""):
    with PERF.measure("tap"):
        structured_log("tap_raw", x=int(x), y=int(y), reason=reason)
        if DRY_RUN:
            log(f"DRY_RUN TAP_RAW at ({x},{y}){'  // ' + reason if reason else ''}")
            time.sleep(SLEEP_AFTER_TAP)
            return
        try:
            get_fast_adb().run_cmd(f"input touchscreen tap {int(x)} {int(y)}")
            log(f"TAP_RAW at ({x},{y}){'  // ' + reason if reason else ''}")
        except Exception as e:
            log(f"[ERROR] tap_raw adb failed: {e}")
        time.sleep(SLEEP_AFTER_TAP)

def tap_fast(x: int, y: int, reason: str = ""):
    with PERF.measure("tap"):
        x = max(SAFE_X1, min(SAFE_X2, int(x)))
        y = max(SAFE_Y1, min(SAFE_Y2, int(y)))
        structured_log("tap_fast", x=x, y=y, reason=reason)
        if DRY_RUN:
            log(f"DRY_RUN TAP_FAST at ({x},{y}){'  // ' + reason if reason else ''}")
            return
        try:
            get_fast_adb().run_cmd(f"input touchscreen tap {x} {y}")
            log(f"TAP_FAST at ({x},{y}){'  // ' + reason if reason else ''}")
        except Exception as e:
            log(f"[ERROR] tap_fast adb failed: {e}")

def device_swipe(dx: int, dy: int, duration_ms: int):
    grid = 50
    jitter_x = 6
    jitter_y = 8
    margin_x = 12
    margin_y = 20
    span_y = max(1, SAFE_Y2 - SAFE_Y1)

    x_min_wish, x_max_wish = 200, 800
    x_min = max(SAFE_X1 + margin_x, x_min_wish)
    x_max = min(SAFE_X2 - margin_x, x_max_wish)
    if x_min > x_max:
        x_min, x_max = SAFE_X1 + margin_x, SAFE_X2 - margin_x

    x_anchor_raw = random.randint(x_min, x_max)
    x_anchor = int(round(x_anchor_raw / grid) * grid)
    x_anchor = clamp(x_anchor + random.randint(-jitter_x, jitter_x), x_min, x_max)

    if dy < 0:
        y_from_base = SAFE_Y2 - random.randint(int(span_y * 0.12), int(span_y * 0.20))
    elif dy > 0:
        y_from_base = SAFE_Y1 + random.randint(int(span_y * 0.12), int(span_y * 0.20))
    else:
        y_from_base = SAFE_Y1 + random.randint(int(span_y * 0.35), int(span_y * 0.60))

    y_from = int(round(y_from_base / grid) * grid)
    y_from = clamp(y_from + random.randint(-jitter_y, jitter_y), SAFE_Y1 + margin_y, SAFE_Y2 - margin_y)

    y_to = clamp(y_from + dy, SAFE_Y1 + margin_y, SAFE_Y2 - margin_y)
    x_to = clamp(x_anchor + dx, x_min, x_max)

    dy_real = y_to - y_from
    min_travel_px = max(90, int(span_y * 0.06))
    if abs(dy_real) < min_travel_px and dy != 0:
        if dy < 0:
            y_from = clamp(SAFE_Y2 - random.randint(int(span_y * 0.18), int(span_y * 0.24)), SAFE_Y1 + margin_y, SAFE_Y2 - margin_y)
        else:
            y_from = clamp(SAFE_Y1 + random.randint(int(span_y * 0.18), int(span_y * 0.24)), SAFE_Y1 + margin_y, SAFE_Y2 - margin_y)
        y_to = clamp(y_from + dy, SAFE_Y1 + margin_y, SAFE_Y2 - margin_y)
        dy_real = y_to - y_from

    dur = int(max(80.0, duration_ms * random.uniform(0.82, 1.15)))

    structured_log(
        "swipe_plan_human",
        start=[int(x_anchor), int(y_from)],
        end=[int(x_to), int(y_to)],
        dy_req=int(dy),
        dy_real=int(dy_real),
        duration_ms=int(dur),
    )

    if SAVE_DEBUG_IMAGES:
        pre = screenshot_bgr()
        if pre is not None:
            snap("swipe_before", pre, points=[(int(x_anchor), int(y_from)), (int(x_to), int(y_to))])

    if not DRY_RUN:
        try:
            get_fast_adb().run_cmd(
                f"input touchscreen swipe {int(x_anchor)} {int(y_from)} {int(x_to)} {int(y_to)} {int(dur)}"
            )
            log(f"SWIPE from ({int(x_anchor)},{int(y_from)}) to ({int(x_to)},{int(y_to)}) speed={dur}ms")
        except Exception as e:
            log(f"[ERROR] swipe adb failed: {e}")

    time.sleep(max(0.0, SWIPE_PAUSE_MS / 1000.0 + random.uniform(-0.05, 0.10)))

    if SAVE_DEBUG_IMAGES:
        post = screenshot_bgr()
        if post is not None:
            snap("swipe_after", post, points=[(int(x_anchor), int(y_from)), (int(x_to), int(y_to))])

def keyevent(code: int):
    """Sends a key event via FastADB (e.g., 4 for Back)."""
    try:
        get_fast_adb().run_cmd(f"input keyevent {code}")
        log(f"KEYEVENT {code}")
    except Exception as e:
        log(f"[ERROR] keyevent adb failed: {e}")


