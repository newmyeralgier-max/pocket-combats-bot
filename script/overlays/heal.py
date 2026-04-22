# -*- coding: utf-8 -*-
"""
Опциональный «шаг лечения» между STATE_RECOVER и STATE_FIND.

Запускается из fsm_main.py только если CFG.RECOVER.ENABLED = true.
Логика намеренно минимальна:
  • если задан HEAL_TAP_XY (абсолютные координаты) → тапаем туда;
  • иначе если есть шаблон кнопки HEAL_BTN_TPL → тапаем по центру матча;
  • иначе ничего не делаем (возвращаем False).

Не бросает исключений — при любой ошибке возвращает False.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

try:
    from script.detection.player_status import find_heal_button  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    find_heal_button = None  # type: ignore[assignment]


def _cfg_get(cfg: Any, *path, default=None):
    cur = cfg
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is default:
            return default
    return cur


def _coerce_xy(val: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(val, (list, tuple)) or len(val) != 2:
        return None
    try:
        return int(val[0]), int(val[1])
    except Exception:
        return None


def perform_heal(cfg: Dict[str, Any], frame_bgr=None) -> bool:
    """
    Пытается провести heal-тап. Возвращает True, если тап был сделан.

    Args:
      cfg: полный CFG (нужна секция RECOVER).
      frame_bgr: необязательный текущий кадр; если нет — будет запрошен
                 из adb.screenshot_bgr() при поиске шаблона.
    """
    # 1) Явные координаты имеют приоритет.
    xy = _coerce_xy(_cfg_get(cfg, "RECOVER", "HEAL_TAP_XY"))
    if xy is not None:
        try:
            from script.device.adb import tap_raw  # локальный импорт: adb не нужен юнит-тестам
            tap_raw(xy[0], xy[1], reason="fsm_heal_xy")
            time.sleep(float(_cfg_get(cfg, "RECOVER", "HEAL_POST_SLEEP_S", default=0.6) or 0.6))
            return True
        except Exception:
            return False

    # 2) Шаблонный поиск.
    if find_heal_button is None:
        return False
    try:
        if frame_bgr is None:
            from script.device.adb import screenshot_bgr
            frame_bgr = screenshot_bgr()
        if frame_bgr is None:
            return False
        pt = find_heal_button(frame_bgr, cfg)
        if pt is None:
            return False
        from script.device.adb import tap_raw
        tap_raw(pt[0], pt[1], reason="fsm_heal_tpl")
        time.sleep(float(_cfg_get(cfg, "RECOVER", "HEAL_POST_SLEEP_S", default=0.6) or 0.6))
        return True
    except Exception:
        return False
