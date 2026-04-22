# -*- coding: utf-8 -*-
"""
Резолюшен-агностичный слой координат.

Постановка: по всему коду (fight.py, fsm_main.py, victory_drop.py и т.д.)
натыкан референсный фрейм 1080×2460: хардкод `tap_raw(..., 2250)`,
`VICTORY_DROP_ABS=[0,1000,1080,1500]`, `ITEMS_ROI=[0,400,500,2100]` и т.п.
На эмуляторе с другим разрешением (или на телефоне с другой DPI) все
эти тапы и ROI едут мимо.

Этот модуль предоставляет конверсию:
  • `rel_x(r)`, `rel_y(r)` — относительные 0..1 → пиксели (округлённо).
  • `rel_point(rx, ry)` → (x, y).
  • `rel_roi(rx1, ry1, rx2, ry2)` → (x1, y1, x2, y2).
  • `scale_roi_from_ref(roi)` — старый абсолютный ROI из эталонного
    разрешения (`CFG.REF_SCREEN_W` × `CFG.REF_SCREEN_H`, дефолт 1080×2460)
    → под текущий размер экрана.
  • `detect_and_apply_screen_size(frame_bgr)` — опциональный авто-детект.

Поведение без флагов не меняется: если `CFG.SCREEN_AUTO_DETECT != true`
и никто не вызывает эти хелперы — всё работает как раньше, через
`SCREEN_W`/`SCREEN_H` из конфига (дефолты 1080/2460).

Конфиг-ключи:
  SCREEN_W, SCREEN_H              — текущее разрешение (CFG уже хранит)
  REF_SCREEN_W, REF_SCREEN_H      — эталон для scale_roi_from_ref
                                     (default 1080/2460 — на чём изначально
                                     нарезались все ROI/таплы)
  SCREEN_AUTO_DETECT              — если true, первый успешный screenshot
                                     обновит SCREEN_W/SCREEN_H
"""

from __future__ import annotations

from typing import Any, Optional, Tuple


def _cfg_int(cfg: Any, key: str, default: int) -> int:
    try:
        return int((cfg or {}).get(key, default) or default)
    except Exception:
        return default


def _ref_size(cfg: Any) -> Tuple[int, int]:
    """Референсное разрешение, на котором сделаны хардкод-координаты."""
    return (
        _cfg_int(cfg, "REF_SCREEN_W", 1080),
        _cfg_int(cfg, "REF_SCREEN_H", 2460),
    )


def current_size(cfg: Any) -> Tuple[int, int]:
    """Текущее разрешение экрана из конфига."""
    return (
        _cfg_int(cfg, "SCREEN_W", 1080),
        _cfg_int(cfg, "SCREEN_H", 2460),
    )


def rel_x(cfg: Any, r: float) -> int:
    """Относительная X (0..1) → пиксель."""
    w, _ = current_size(cfg)
    return int(round(float(r) * w))


def rel_y(cfg: Any, r: float) -> int:
    """Относительная Y (0..1) → пиксель."""
    _, h = current_size(cfg)
    return int(round(float(r) * h))


def rel_point(cfg: Any, rx: float, ry: float) -> Tuple[int, int]:
    w, h = current_size(cfg)
    return (int(round(float(rx) * w)), int(round(float(ry) * h)))


def rel_roi(
    cfg: Any,
    rx1: float, ry1: float, rx2: float, ry2: float,
) -> Tuple[int, int, int, int]:
    w, h = current_size(cfg)
    return (
        int(round(float(rx1) * w)),
        int(round(float(ry1) * h)),
        int(round(float(rx2) * w)),
        int(round(float(ry2) * h)),
    )


def scale_xy_from_ref(cfg: Any, x_ref: float, y_ref: float) -> Tuple[int, int]:
    """(x, y), заданные в эталонном 1080×2460, → под текущий экран."""
    rw, rh = _ref_size(cfg)
    w, h = current_size(cfg)
    if rw <= 0 or rh <= 0:
        return (int(round(x_ref)), int(round(y_ref)))
    return (
        int(round(float(x_ref) * w / rw)),
        int(round(float(y_ref) * h / rh)),
    )


def scale_roi_from_ref(cfg: Any, roi) -> Tuple[int, int, int, int]:
    """ROI [x1,y1,x2,y2] из эталонного разрешения → под текущий экран."""
    if roi is None or len(roi) != 4:
        return (0, 0, 0, 0)
    x1, y1, x2, y2 = [float(v) for v in roi]
    rw, rh = _ref_size(cfg)
    w, h = current_size(cfg)
    if rw <= 0 or rh <= 0 or (w == rw and h == rh):
        return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
    sx, sy = w / rw, h / rh
    return (
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    )


def is_relative_roi(roi) -> bool:
    """True, если ROI задан в 0..1 (все значения <= 1.0)."""
    if roi is None or len(roi) != 4:
        return False
    try:
        return all(0.0 <= float(v) <= 1.0 for v in roi)
    except Exception:
        return False


def auto_roi(cfg: Any, roi) -> Tuple[int, int, int, int]:
    """Универсальный парсинг ROI:
        • [0.0..1.0] × 4          → rel_roi (ratio);
        • иначе                   → scale_roi_from_ref (эталонные пиксели).
    """
    if is_relative_roi(roi):
        return rel_roi(cfg, *[float(v) for v in roi])
    return scale_roi_from_ref(cfg, roi)


# ────────────────────────────────────────────────────────────────
# Опциональный авто-детект.
# ────────────────────────────────────────────────────────────────


def detect_screen_size(frame_bgr) -> Optional[Tuple[int, int]]:
    """Возвращает (W, H) реального кадра, либо None."""
    if frame_bgr is None:
        return None
    try:
        h, w = frame_bgr.shape[:2]
        if w > 0 and h > 0:
            return (int(w), int(h))
    except Exception:
        pass
    return None


def apply_detected_size(cfg: Any, frame_bgr) -> Optional[Tuple[int, int]]:
    """Если в кадре другой размер, чем в cfg.SCREEN_W/H, — ОБНОВЛЯЕТ cfg
    (in-place) и возвращает новый (W, H). Иначе None.

    Важно: изменяет только словарь cfg. Модули, которые уже захватили
    SCREEN_W/SCREEN_H в свои `int` константы при импорте (fight.py и т.д.),
    должны перечитать их из cfg — или использовать coords.* хелперы
    (они читают cfg каждый раз).
    """
    detected = detect_screen_size(frame_bgr)
    if detected is None or not isinstance(cfg, dict):
        return None
    cur_w, cur_h = current_size(cfg)
    if detected == (cur_w, cur_h):
        return None
    cfg["SCREEN_W"], cfg["SCREEN_H"] = detected
    return detected
