# -*- coding: utf-8 -*-
import os
import json
import time
import datetime
import atexit
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from script.loot.utils import short_time_tag
from script.core.config import LOG_FILE, DEBUG_DIR, SAVE_DEBUG_IMAGES, CFG

# Ensure dirs exist
_log_dir = os.path.dirname(LOG_FILE)
if _log_dir:
    os.makedirs(_log_dir, exist_ok=True)
if DEBUG_DIR:
    os.makedirs(DEBUG_DIR, exist_ok=True)

STEP_COUNTER: int = 0

def next_step(label: str) -> str:
    global STEP_COUNTER
    STEP_COUNTER += 1
    return f"step_{STEP_COUNTER:04d}_{label}"

def mark_step(label: str) -> int:
    global STEP_COUNTER
    STEP_COUNTER += 1
    structured_log("step_mark", step_id=STEP_COUNTER, label=label)
    return STEP_COUNTER

def now_ts() -> str:
    return short_time_tag(include_seconds=True)

def log(msg: str):
    line = f"[{now_ts()}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def dbg_name(name: str, step_id: Optional[int] = None) -> str:
    if step_id is not None:
        return f"{step_id:04d}_{name}"
    return next_step(name) if CFG.get("DEBUG_PACK_BY_STEP", False) else name

def dbg_save(
    name: str,
    img: np.ndarray,
    rects: Optional[List[Tuple[int, int, int, int]]] = None,
    points: Optional[List[Tuple[int, int]]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
):
    if not SAVE_DEBUG_IMAGES or img is None:
        return
    img2 = img.copy()
    if rects:
        for x, y, w, h in rects:
            cv2.rectangle(img2, (x, y), (x + w, y + h), color, 2)
    if points:
        for px, py in points:
            cv2.circle(img2, (int(px), int(py)), 8, (0, 0, 255), 2)
    path = os.path.join(DEBUG_DIR, f"{now_ts()}_{dbg_name(name)}.png")
    cv2.imwrite(path, img2)
    return path

def snap(label: str, frame: np.ndarray, step_id: Optional[int] = None, rects=None, rect_colors=None, points=None, point_colors=None):
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

class BufferedStructuredLogger:
    def __init__(self, log_file: str, max_buffer: int = 50, flush_interval: float = 2.0):
        self.log_file = log_file
        self.max_buffer = max_buffer
        self.flush_interval = flush_interval
        self._buffer: List[dict] = []
        self._last_flush = time.perf_counter()
        atexit.register(self.flush)

    def log(self, step: str, **payload):
        rec = {"ts": datetime.datetime.now().isoformat(timespec="milliseconds"), "step": step, **payload}
        self._buffer.append(rec)
        if len(self._buffer) >= self.max_buffer or (time.perf_counter() - self._last_flush) >= self.flush_interval:
            self.flush()

    def flush(self):
        if not self._buffer:
            return
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                for rec in self._buffer:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._buffer.clear()
            self._last_flush = time.perf_counter()
        except Exception as e:
            log(f"[DEBUG] structured_log write failed: {e}")

_logger_v2 = BufferedStructuredLogger(LOG_FILE)
def structured_log(step: str, **payload):
    _logger_v2.log(step, **payload)

class PerfTimer:
    def __init__(self):
        self._timings: Dict[str, List[float]] = defaultdict(list)
        self._counts: Dict[str, int] = defaultdict(int)

    @contextmanager
    def measure(self, label: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self._timings[label].append(dt)
            self._counts[label] += 1
            if len(self._timings[label]) > 1000:
                self._timings[label] = self._timings[label][-500:]

    def report(self) -> Dict[str, dict]:
        res = {}
        for k, vals in self._timings.items():
            if not vals:
                continue
            res[k] = {
                "avg": sum(vals) / len(vals),
                "max": max(vals),
                "min": min(vals),
                "count": self._counts[k]
            }
        return res

PERF = PerfTimer()
