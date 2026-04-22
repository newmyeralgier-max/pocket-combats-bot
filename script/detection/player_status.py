# -*- coding: utf-8 -*-
"""
Детекторы состояния персонажа: смерть (экран воскрешения) и уровень HP.

Используется в script/main/fsm_main.py::STATE_RECOVER, когда
`CFG.RECOVER.ENABLED = true`. До включения флага в fsm ничего не
вызывается — по умолчанию бот ведёт себя как раньше.

Все детекторы опциональны и «failsafe»:
  • если шаблон не задан в конфиге или не найден на диске — детектор
    молча возвращает False/None и не бросает исключений;
  • если ROI задан криво или HP-бар не виден — возвращаем None
    (как «не знаю»), а не 0.0.

Вызывающий код (FSM) должен трактовать False/None как «предположим,
что всё ок, продолжаем». Это консервативная политика: лучше
несколько пустых боёв, чем зависнуть в recover навсегда.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:  # cv2 есть всегда в продакшене, но на CI без opencv юнит-тесты не падают
    import cv2  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


# ────────────────────────────────────────────────────────────────
# Шаблон-детектор (обёртка над matchTemplate с защитой от ошибок).
# ────────────────────────────────────────────────────────────────


def _match_template_score(
    frame_bgr: np.ndarray,
    tpl_path: str,
) -> Optional[float]:
    """Возвращает max score от cv2.matchTemplate, либо None."""
    if cv2 is None or not isinstance(tpl_path, str) or not tpl_path:
        return None
    if not os.path.isfile(tpl_path):
        return None
    try:
        tpl = cv2.imdecode(
            np.fromfile(tpl_path, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if tpl is None:
            return None
        fh, fw = frame_bgr.shape[:2]
        th, tw = tpl.shape[:2]
        if th > fh or tw > fw:
            return None
        fr_g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        tp_g = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(fr_g, tp_g, cv2.TM_CCOEFF_NORMED)
        _, mx, _, _ = cv2.minMaxLoc(res)
        return float(mx)
    except Exception:
        return None


def _cfg_get(cfg: Any, *path, default=None):
    cur = cfg
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is default:
            return default
    return cur


# ────────────────────────────────────────────────────────────────
# Detect: смерть / экран воскрешения.
# ────────────────────────────────────────────────────────────────


def detect_dead(frame_bgr: np.ndarray, cfg: Dict[str, Any]) -> bool:
    """
    True — если на экране найден шаблон смерти/экрана воскрешения.
    Настройки:
        CFG.RECOVER.DEAD_TPL       — путь к шаблону
        CFG.RECOVER.DEAD_THRESHOLD — порог (default 0.85)
    """
    if frame_bgr is None:
        return False
    tpl_path = _cfg_get(cfg, "RECOVER", "DEAD_TPL")
    if not tpl_path:
        return False
    thr = float(_cfg_get(cfg, "RECOVER", "DEAD_THRESHOLD", default=0.85) or 0.85)
    sc = _match_template_score(frame_bgr, tpl_path)
    return sc is not None and sc >= thr


# ────────────────────────────────────────────────────────────────
# HP-бар по цвету.
# Стандартный UI: полоса HP окрашена от зелёного (полно) до красного
# (мало). Считаем долю красных пикселей относительно всех цветных
# пикселей в ROI. 0.0 — полный HP, 1.0 — почти пусто.
# ────────────────────────────────────────────────────────────────


def compute_hp_ratio(frame_bgr: np.ndarray, cfg: Dict[str, Any]) -> Optional[float]:
    """
    Возвращает долю «красноты» в HP-ROI в диапазоне [0..1], либо None.

    Настройки:
        CFG.RECOVER.HP_ROI  — [x1, y1, x2, y2] в координатах кадра
    """
    if cv2 is None or frame_bgr is None:
        return None
    roi = _cfg_get(cfg, "RECOVER", "HP_ROI")
    if not (isinstance(roi, (list, tuple)) and len(roi) == 4):
        return None
    try:
        x1, y1, x2, y2 = [int(v) for v in roi]
    except Exception:
        return None
    H, W = frame_bgr.shape[:2]
    x1 = max(0, min(W - 1, x1)); x2 = max(x1 + 1, min(W, x2))
    y1 = max(0, min(H - 1, y1)); y2 = max(y1 + 1, min(H, y2))
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # HSV-пороги: зелёный H∈[40,85], красный H∈[0,10] ∪ [170,180],
    # S>60, V>60 — отсекаем серый UI.
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (40, 60, 60), (85, 255, 255))
    red1 = cv2.inRange(hsv, (0, 60, 60), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 60, 60), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)
    gcnt = int(np.count_nonzero(green))
    rcnt = int(np.count_nonzero(red))
    total = gcnt + rcnt
    if total < 20:
        # Полоса не опознана (сильно спрятано / не тот ROI).
        return None
    return float(rcnt) / float(total)


def is_low_hp(frame_bgr: np.ndarray, cfg: Dict[str, Any]) -> Optional[bool]:
    """True — HP ниже порога; False — выше; None — не смогли измерить."""
    ratio = compute_hp_ratio(frame_bgr, cfg)
    if ratio is None:
        return None
    thr = float(_cfg_get(cfg, "RECOVER", "HP_LOW_RATIO", default=0.5) or 0.5)
    return ratio >= thr


# ────────────────────────────────────────────────────────────────
# Поиск кнопки лечения (для overlays/heal.py).
# ────────────────────────────────────────────────────────────────


def find_heal_button(frame_bgr: np.ndarray, cfg: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """
    Возвращает центр кнопки лечения (x, y) в координатах кадра,
    либо None. Шаблон: CFG.RECOVER.HEAL_BTN_TPL, порог HEAL_BTN_THRESHOLD.
    """
    if cv2 is None or frame_bgr is None:
        return None
    tpl_path = _cfg_get(cfg, "RECOVER", "HEAL_BTN_TPL")
    if not tpl_path or not os.path.isfile(tpl_path):
        return None
    thr = float(_cfg_get(cfg, "RECOVER", "HEAL_BTN_THRESHOLD", default=0.85) or 0.85)
    try:
        tpl = cv2.imdecode(
            np.fromfile(tpl_path, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if tpl is None:
            return None
        fh, fw = frame_bgr.shape[:2]
        th, tw = tpl.shape[:2]
        if th > fh or tw > fw:
            return None
        fr_g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        tp_g = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(fr_g, tp_g, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx < thr:
            return None
        x, y = loc
        return (int(x + tw // 2), int(y + th // 2))
    except Exception:
        return None
