# quick_tap.py — быстрый тап по зоне лейбла
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import cv2

from script.loot.utils import clamp, ui_tpl_path


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int

    def right(self) -> int:
        return self.x + self.w

    def bottom(self) -> int:
        return self.y + self.h

    def center(self) -> Tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2


def clamp_rect_to(rect: Rect, bounds: Rect) -> Rect:
    x1 = clamp(rect.x, bounds.x, bounds.right())
    y1 = clamp(rect.y, bounds.y, bounds.bottom())
    x2 = clamp(rect.right(), bounds.x, bounds.right())
    y2 = clamp(rect.bottom(), bounds.y, bounds.bottom())
    return Rect(x1, y1, max(0, x2 - x1), max(0, y2 - y1))


def load_qt_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    qt = dict(cfg.get("QUICK_TAP", {}))
    defaults = {
        "X_TARGET_MODE": "box_center",
        "X_ABSOLUTE": 950,
        "X_JITTER_PX": 3,
        "Y_OFFSET_PCT": 0.2,
        "ZONE_HEIGHT_PCT": 0.1,
        "ZONE_HEIGHT_MIN_PX": 6,
        "ZONE_HEIGHT_MAX_PX": 12,
        "Y_JITTER_PX": 0,
        "TAPS": 2,
        "DELAY_BETWEEN_TAPS_MS": 40,
        "POST_SERIES_DELAY_MS": 120,
        "DEBUG_OVERLAY": False,
        "DEBUG_DIR": "debug/quick_tap",
    }
    for k, v in defaults.items():
        qt.setdefault(k, v)
    return qt


def compute_tap_zone(label_box: Rect, safe_rect: Rect, qt: Dict[str, Any]) -> Tuple[int, int, Rect]:
    cx, cy = label_box.center()
    zone_h_pct = float(qt["ZONE_HEIGHT_PCT"])
    zone_h = int(round(label_box.h * zone_h_pct))
    zone_h = clamp(zone_h, int(qt["ZONE_HEIGHT_MIN_PX"]), int(qt["ZONE_HEIGHT_MAX_PX"]))
    if zone_h <= 0:
        zone_h = int(qt["ZONE_HEIGHT_MIN_PX"])
    offset_up = int(round(label_box.h * float(qt["Y_OFFSET_PCT"])))
    y_top = cy - offset_up
    y_bot = y_top + zone_h
    if qt["X_TARGET_MODE"] == "absolute":
        x = int(qt["X_ABSOLUTE"])
    else:
        x = cx
    x = clamp(x, safe_rect.x, safe_rect.right())
    y_top = clamp(y_top, safe_rect.y, safe_rect.bottom())
    y_bot = clamp(y_bot, safe_rect.y, safe_rect.bottom())
    y = y_top + (y_bot - y_top) // 2
    zone_rect = Rect(
        x=max(safe_rect.x, label_box.x), y=y_top, w=min(label_box.w, safe_rect.right() - label_box.x), h=y_bot - y_top
    )
    return x, y, zone_rect


def apply_jitter(x: int, y: int, qt: Dict[str, Any]) -> Tuple[int, int]:
    jx = int(qt["X_JITTER_PX"])
    jy = int(qt["Y_JITTER_PX"])
    if jx > 0:
        x += random.randint(-jx, jx)
    if jy > 0:
        y += random.randint(-jy, jy)
    return x, y


def draw_debug_overlay(frame_bgr, label_box: Rect, zone_rect: Rect, tap_pt: Tuple[int, int]):
    if frame_bgr is None:
        return None
    img = frame_bgr.copy()
    cv2.rectangle(img, (label_box.x, label_box.y), (label_box.right(), label_box.bottom()), (0, 255, 255), 2)
    cv2.rectangle(img, (zone_rect.x, zone_rect.y), (zone_rect.right(), zone_rect.bottom()), (0, 128, 255), 2)
    cv2.circle(img, tap_pt, 4, (0, 0, 255), -1)
    cv2.line(img, (0, tap_pt[1]), (max(zone_rect.right(), label_box.right()) + 50, tap_pt[1]), (0, 0, 255), 1)
    return img


def quick_tap_for_label(
    tap_fn: Callable[[int, int], None],
    label_box_xywh: Tuple[int, int, int, int],
    safe_xywh: Tuple[int, int, int, int],
    cfg: Dict[str, Any],
    frame_bgr=None,
    slot_id: Optional[int] = None,
    label_text: Optional[str] = None,
) -> Tuple[int, int]:
    """Тапает TAPS раз в зону над центром текста. Возвращает (x, y) последнего тапа."""
    qt = load_qt_cfg(cfg)
    label_box = Rect(*map(int, label_box_xywh))
    safe_rect = Rect(*map(int, safe_xywh))
    x, y, zone = compute_tap_zone(label_box, safe_rect, qt)
    x, y = apply_jitter(x, y, qt)
    slot = f"{slot_id}" if slot_id is not None else "-"
    name = label_text or "-"
    print(
        f"[QT] slot={slot} label='{name}' box=({label_box.x},{label_box.y},{label_box.w},{label_box.h}) "
        f"cy={label_box.y + label_box.h // 2} offset_up={int(label_box.h * float(qt['Y_OFFSET_PCT']))} "
        f"zone_h={zone.h} x={x} y={y} mode={qt['X_TARGET_MODE']}"
    )
    taps = int(qt["TAPS"])
    delay_btwn = int(qt["DELAY_BETWEEN_TAPS_MS"]) / 1000.0
    for i in range(taps):
        tap_fn(int(x), int(y))
        if i < taps - 1:
            time.sleep(delay_btwn)
    time.sleep(int(qt["POST_SERIES_DELAY_MS"]) / 1000.0)
    if bool(qt["DEBUG_OVERLAY"]):
        os.makedirs(qt["DEBUG_DIR"], exist_ok=True)
        overlay = draw_debug_overlay(frame_bgr, label_box, zone, (int(x), int(y)))
        if overlay is not None:
            ts = int(time.time() * 1000)
            fname = f"qt_slot{slot}_{name}_{ts}.png".replace(" ", "_")
            cv2.imwrite(os.path.join(qt["DEBUG_DIR"], fname), overlay)
    return int(x), int(y)
