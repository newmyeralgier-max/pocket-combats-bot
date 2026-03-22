# -*- coding: utf-8 -*-
"""Overlay detection and clearing — chat windows, monster params, etc."""
import time

from script.core.config import CFG
from script.core.logging import structured_log
from script.device.adb import screenshot_bgr, tap_raw
from script.loot.utils import match_template_safe, try_imread, ui_tpl_path, get_thr

# ── Шаблоны ──────────────────────────────────────────────────────────
OVERLAY_TEXTS = [ui_tpl_path("syschat"), ui_tpl_path("monster_params")]
OVERLAY_CLOSES = [ui_tpl_path("close_1"), ui_tpl_path("close_2")]

# ── Состояние антидребезга ───────────────────────────────────────────
_last_overlay_present = False
_last_overlay_ts = 0.0
_last_close_ts = 0.0


def _find(screen, tpl_path: str, thr=0.85):
    tpl = try_imread(tpl_path)
    if tpl is None:
        return None
    H, W = screen.shape[:2]
    return match_template_safe(screen, tpl, (0, 0, W, H), threshold=thr)


def detect_overlays(screen=None, thr_text: float = None):
    """Возвращает {present: bool, matches: [(name, match_dict)], screen}."""
    if thr_text is None:
        thr_text = get_thr("OVERLAY_TEXT", 0.85)
    if screen is None:
        screen = screenshot_bgr()
        if screen is None:
            return {"present": False, "matches": [], "screen": None}
    matches = []
    for name in OVERLAY_TEXTS:
        m = _find(screen, name, thr=thr_text)
        if m:
            matches.append((name, m))
    return {"present": len(matches) > 0, "matches": matches, "screen": screen}


def overlay_present(cfg=CFG) -> bool:
    return detect_overlays()["present"]


def overlay_just_appeared(cfg=CFG) -> bool:
    global _last_overlay_present, _last_overlay_ts
    d = detect_overlays()
    now_present = d["present"]
    just = now_present and not _last_overlay_present
    _last_overlay_present = now_present
    if just:
        _last_overlay_ts = time.time()
        structured_log("overlay_appeared", overlays=[n for n, _ in d["matches"]])
    return just


def clear_overlays(cfg=CFG, cooldown_s: float = 0.0, log_no_overlay: bool = True) -> int:
    """Закрывает оверлей (если есть). Возвращает 2 если закрыл, 0 иначе."""
    global _last_overlay_present, _last_close_ts
    if cooldown_s > 0 and (time.time() - _last_close_ts) < cooldown_s:
        return 0
    d = detect_overlays()
    screen = d["screen"]
    if screen is None:
        return 0
    if not d["present"]:
        if log_no_overlay:
            structured_log("overlay_none")
        _last_overlay_present = False
        return 0
    thr_close = get_thr("OVERLAY_CLOSE", 0.85)
    for cname in OVERLAY_CLOSES:
        m_close = _find(screen, cname, thr=thr_close)
        if m_close and "center" in m_close:
            cx, cy = m_close["center"]
            tap_raw(cx, cy, reason=f"overlay_close:{cname}")
            time.sleep(cfg.get("TIMINGS", {}).get("after_item_click", 0.3))
            _last_close_ts = time.time()
            _last_overlay_present = False
            structured_log("overlay_closed", close_template=cname)
            return 2
    structured_log("overlay_no_close_found", overlays=[n for n, _ in d["matches"]])
    return 0


__all__ = ["OVERLAY_TEXTS", "OVERLAY_CLOSES", "detect_overlays", "overlay_present", "overlay_just_appeared", "clear_overlays"]
