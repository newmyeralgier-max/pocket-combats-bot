# -*- coding: utf-8 -*-
"""Fight helper module."""
import importlib
import os
import re
import random
import time
from typing import Optional

import numpy as np

from script.loot import tpl_loader as TL
from script.loot import utils as loot_utils
from script.loot.utils import ui_tpl_path, BASE_DIR, get_thr
from script.overlays.clear_overlays import clear_overlays, overlay_just_appeared

# ── Runtime bindings from auto_bot ───────────────────────────────────
REG = getattr(TL, "REG", {})
from script.core.config import CFG
from script.core.logging import structured_log, log as Logger
from script.device.adb import screenshot_bgr, tap, tap_raw, device_swipe
from script.loot.auto_bot import ensure_tab_state, detect_tabs_with_occlusion
from script.core import coords as _coords

SCREEN_W = int(CFG.get("SCREEN_W", 1080))
SCREEN_H = int(CFG.get("SCREEN_H", 2460))
RANDOM_PCT = float(CFG.get("RANDOM_PCT", 0.15) or 0.15)


def _thr(name, default=0.82):
    return get_thr(f"FIGHT_{name.upper()}", default)


def rand_xy(x, y, pct=RANDOM_PCT):
    dx, dy = max(1, int(SCREEN_W * pct * 0.02)), max(1, int(SCREEN_H * pct * 0.02))
    rx = max(0, min(SCREEN_W, int(x) + random.randint(-dx, dx)))
    ry = max(0, min(SCREEN_H, int(y) + random.randint(-dy, dy)))
    return rx, ry


def rand_duration(base_sec, pct=RANDOM_PCT):
    delta = base_sec * pct
    return max(0.01, base_sec + random.uniform(-delta, delta))


def find_image_on_screen(image_path: str, screen: Optional[np.ndarray] = None, threshold: float = None):
    """Find template on screen or provided screenshot."""
    if threshold is None:
        threshold = get_thr("FIGHT_IMAGE", 0.82)
    tpl_path = str(image_path)
    tpl_img = loot_utils.try_imread(tpl_path)
    if tpl_img is None:
        Logger(f"[FIGHT] Шаблон не найден: {tpl_path}")
        return None
    if screen is None:
        screen = screenshot_bgr()
    if screen is None:
        return None
    roi = (0, 0, screen.shape[1], screen.shape[0])
    return loot_utils.match_template_safe(screen, tpl_img, roi=roi, threshold=threshold)


def click_if_found(image_path, threshold, reason="", tap_fn=None):
    """Click by found image using provided tap function."""
    if tap_fn is None:
        tap_fn = tap
    frame = screenshot_bgr()
    if frame is None:
        return False
    m = find_image_on_screen(image_path, frame, threshold)
    if m and "center" in m:
        cx, cy = m["center"]
        try:
            tap_fn(cx, cy, reason=reason)
        except Exception:
            tap(cx, cy, reason=reason)
        time.sleep(rand_duration(CFG.get("FIGHT", {}).get("TIMINGS", {}).get("tap_duration", 0.05)))
        structured_log("fight_tap", image=image_path, reason=reason, score=float(m.get("score", 0)))
        time.sleep(rand_duration(CFG.get("FIGHT", {}).get("TIMINGS", {}).get("after_click", 0.4)))
        return True
    return False


def start_fight_sequence():
    ensure_tab_state("монстры", "открыта")

    frame = screenshot_bgr()
    if frame is not None:
        tabs = detect_tabs_with_occlusion(frame)
        if tabs.get("монстры", {}).get("state") == "открыта":
            choice = random.choice([1, 2, 3])
            if choice == 1:
                device_swipe(0, -600, 350)
            elif choice == 2:
                for _ in range(2):
                    device_swipe(0, -300, 250)
            elif choice == 3:
                for _ in range(3):
                    device_swipe(0, -200, 200)

    deadline = time.time() + float(CFG.get("FIGHT", {}).get("TIMEOUTS", {}).get("battle_start", 10))
    while time.time() < deadline:
        scr = screenshot_bgr()
        if scr is None:
            time.sleep(0.2)
            continue
        m = find_image_on_screen(ui_tpl_path("attack"), scr, _thr("attack_btn"))
        if m and "center" in m:
            rx, ry = rand_xy(*m["center"])
            tap(rx, ry, reason="start_attack")
            time.sleep(rand_duration(0.5))
            structured_log("fight_attack_start", found=True)
            return True
        if overlay_just_appeared(CFG):
            clear_overlays(CFG, cooldown_s=0.0, log_no_overlay=False)
        time.sleep(0.2)

    fb_xy = CFG.get("FIGHT", {}).get("ATTACK_FALLBACK_XY")
    if isinstance(fb_xy, (list, tuple)) and len(fb_xy) == 2:
        rx, ry = rand_xy(int(fb_xy[0]), int(fb_xy[1]))
        tap(rx, ry, reason="start_attack_fallback")
        time.sleep(rand_duration(0.5))
        structured_log("fight_attack_start", found=False, fallback=True)
        return True

    structured_log("fight_attack_start", found=False, fallback=False)
    return False


def fight_loop_until_victory():
    """
    Подробный цикл боя с поэтапной диагностикой:
    - Проверка конфигов и путей шаблонов.
    - Очищаем оверлеи, запускаем бой.
    - Ждём и открываем MOVES (с ретраями).
    - Пытаемся нажать навык(и): имя -> иконка.
    - Фолбэк на базовую атаку.
    - Детект победы/поражения, закрытие экрана, возврат результата.
    - Idle guard с детальными логами.
    """

    # --- Параметры из CFG ---
    fight_cfg = CFG.get("FIGHT", {})
    timings = fight_cfg.get("TIMINGS", {})
    skills = fight_cfg.get("SKILLS", ["ледяной_залп"])

    IDLE_LIMIT = fight_cfg.get("IDLE_LIMIT", 10)
    LOOP_DELAY = timings.get("loop_delay", 0.3)
    MOVES_OPEN_RETRIES = timings.get("moves_open_retries", 5)
    MOVES_OPEN_DELAY = timings.get("moves_open_delay", 0.2)

    # --- Пути шаблонов (из верхнего CFG) ---
    moves_path = CFG.get("TEMPLATES_PATHS", {}).get("MOVES")
    attack_path = CFG.get("TEMPLATES_PATHS", {}).get("ATTACK")
    victory_path = CFG.get("TEMPLATES_PATHS", {}).get("VICTORY")
    continue_path = CFG.get("TEMPLATES_PATHS", {}).get("CONTINUE")
    defeat_path = CFG.get("TEMPLATES_PATHS", {}).get("DEFEAT")  # если добавишь

    # --- Диагностика конфигов/путей ---
    structured_log(
        "fight_init",
        idle_limit=IDLE_LIMIT,
        loop_delay=LOOP_DELAY,
        moves_retries=MOVES_OPEN_RETRIES,
        moves_delay=MOVES_OPEN_DELAY,
        skills=skills,
    )

    def _exists(p):
        return bool(p) and os.path.exists(p)

    structured_log(
        "fight_paths",
        MOVES=moves_path,
        MOVES_exists=_exists(moves_path),
        ATTACK=attack_path,
        ATTACK_exists=_exists(attack_path),
        VICTORY=victory_path,
        VICTORY_exists=_exists(victory_path),
        DEFEAT=defeat_path,
        DEFEAT_exists=_exists(defeat_path) if defeat_path else None,
        CONTINUE=continue_path,
        CONTINUE_exists=_exists(continue_path),
    )

    # --- Утилиты диагностики ---
    def thr_value(key):
        try:
            val = _thr(key)
        except Exception as e:
            structured_log("fight_thr_error", key=key, error=str(e))
            val = None
        return val

    def diag_screenshot(tag):
        scr = screenshot_bgr()
        if scr is None:
            structured_log("fight_scr", tag=tag, result="none")
            return None
        try:
            h, w = scr.shape[:2]
        except Exception:
            h = w = None
        structured_log("fight_scr", tag=tag, result="ok", width=w, height=h)
        return scr

    def diag_find(path, thr_key, tag, scr=None):
        if not path:
            structured_log("fight_find", tag=tag, path="None", thr_key=thr_key, result="skip_no_path")
            return False
        thr = thr_value(thr_key)
        if thr is None:
            structured_log("fight_find", tag=tag, path=path, thr_key=thr_key, result="skip_no_thr")
            return False
        if scr is None:
            scr = diag_screenshot(tag=f"{tag}_pre")
        if scr is None:
            structured_log("fight_find", tag=tag, path=path, thr=thr, result="skip_no_scr")
            return False
        # Зависим от реализации find_image_on_screen: возвращает bool.
        t0 = time.time()
        try:
            ok = find_image_on_screen(path, scr, thr)
            dt = round((time.time() - t0) * 1000)
            structured_log("fight_find", tag=tag, path=path, thr=thr, result=bool(ok), dur_ms=dt)
            return bool(ok)
        except Exception as e:
            structured_log("fight_find", tag=tag, path=path, thr=thr, result="error", error=str(e))
            return False

    # Было:
    # def diag_click(path, thr_key, reason, tap_fn=None):

    def diag_click(path, thr_key, reason, tap_fn=None):
        # Если tap_fn не передан — используем tap по умолчанию
        if tap_fn is None:
            tap_fn = tap

        if not path:
            structured_log("fight_click", reason=reason, path="None", thr_key=thr_key, result="skip_no_path")
            return False

        thr = thr_value(thr_key)
        if thr is None:
            structured_log("fight_click", reason=reason, path=path, thr_key=thr_key, result="skip_no_thr")
            return False

        t0 = time.time()
        try:
            ok = click_if_found(path, thr, reason=reason, tap_fn=tap_fn)
            dt = round((time.time() - t0) * 1000)
            structured_log("fight_click", reason=reason, path=path, thr=thr, result=bool(ok), dur_ms=dt)
            return bool(ok)
        except Exception as e:
            structured_log("fight_click", reason=reason, path=path, thr=thr, result="error", error=str(e))
            return False

    def diag_close_result_screen(kind_tag, icon_path, continue_path, continue_thr_key):
        # kind_tag: "victory" | "defeat"
        structured_log("fight_result_detected", kind=kind_tag)
        # Жмём CONTINUE (или тап в центр — фолбэк)
        clicked = False
        if continue_path:
            clicked = diag_click(continue_path, continue_thr_key, reason=f"{kind_tag}_continue", tap_fn=tap_raw)
        if not clicked:
            try:
                fx, fy = _coords.rel_point(CFG, 0.5, 0.9146)  # 0.9146 ≈ 2250/2460
                tap_raw(fx, fy, reason=f"{kind_tag}_continue_fallback")
                time.sleep(rand_duration(0.05))
                structured_log("fight_result_continue_fallback", kind=kind_tag, x=fx, y=fy)
            except Exception as e:
                structured_log("fight_result_continue_fallback", kind=kind_tag, result="error", error=str(e))

        # Проверяем, что экран пропал
        scr2 = diag_screenshot(tag=f"{kind_tag}_post_click")
        if scr2 is None:
            structured_log("fight_result_cleared", kind=kind_tag, reason="no_screen")
            return True
        thr_key = f"{kind_tag}_icon"  # 'victory_icon' / 'defeat_icon'
        still_visible = False
        try:
            thr = thr_value(thr_key)
            if thr is None:
                # Если нет спец-порога — используем тот же ключ, что и раньше
                thr = thr_value("victory_icon") if kind_tag == "victory" else thr_value("defeat_icon")
            still_visible = find_image_on_screen(icon_path, scr2, thr if thr is not None else 0.8)
        except Exception as e:
            structured_log("fight_result_check_error", kind=kind_tag, error=str(e))
        if not still_visible:
            structured_log("fight_result_cleared", kind=kind_tag)
            return True
        else:
            structured_log("fight_result_persist", kind=kind_tag, action="retry_next_loop")
            return False

    # --- Прелюдия боя ---
    if overlay_just_appeared(CFG):
        structured_log("fight_overlay", event="detected_pre")
        clear_overlays(CFG, cooldown_s=0.0, log_no_overlay=False)
        structured_log("fight_overlay", event="cleared_pre")

    structured_log("fight_start_sequence", stage="begin")
    if not start_fight_sequence():
        structured_log("fight_abort", reason="start_failed")
        return "abort"
    structured_log("fight_start_sequence", stage="ok")

    idle_ticks = 0
    loop_idx = 0

    # --- Перед циклом боя ---
    moves_already_opened = False  # флаг: открывали ли приёмы в этом бою

    while True:
        loop_idx += 1
        structured_log("fight_loop_enter", loop=loop_idx)

        clicked = False
        opened_moves = False  # ← добавляем инициализацию здесь

        # --- 1) MOVES ---
        if moves_path and not moves_already_opened:
            for attempt in range(1, MOVES_OPEN_RETRIES + 1):
                if diag_click(moves_path, "moves_btn", reason="open_moves"):
                    opened_moves = True
                    moves_already_opened = True
                    clicked = True
                    break
                time.sleep(MOVES_OPEN_DELAY)
                structured_log("fight_moves_retry_wait", attempt=attempt, delay_s=MOVES_OPEN_DELAY)
        elif not moves_path:
            structured_log("fight_moves_skip", reason="no_moves_path")
        else:
            structured_log("fight_moves_skip", reason="already_opened")

            # --- 2) Навыки: имя -> иконка (автопоиск из реестра) ---
        skill_clicked = False
        if opened_moves:
            # Собираем пары "имя+иконка" по одному идентификатору навыка (без суффиксов)
            skills_map = {}  # skill_id -> {"name_tpl": (group, skill_id), "icon_tpl": (group, skill_id)}

            def skill_id_from_path(p: str) -> str:
                base = os.path.splitext(os.path.basename(p))[0]
                return re.sub(r"_(name|icon)$", "", base, flags=re.IGNORECASE)

            for key, val in REG.items():
                if not isinstance(val, dict):
                    structured_log("reg_entry_not_normalized", key=key, value_type=type(val).__name__)
                    continue

                group = val.get("group")
                tpl_path = val.get("tpl", "")
                if not group or not tpl_path:
                    continue

                # Нормализуем схему групп: skills.name / skills.icon
                group_norm = group.replace("_", ".").lower().strip()

                if group_norm not in ("skills.name", "skills.icon"):
                    continue

                # 1) Пытаемся взять идентификатор навыка из ключа REG (надёжнее)
                skill_id = None
                prefix = group_norm + "."
                k = key.lower().strip()
                if k.startswith(prefix):
                    skill_id = k[len(prefix) :].strip()

                # 2) Фолбэк: из имени файла (срезаем _name/_icon)
                if not skill_id:
                    skill_id = skill_id_from_path(tpl_path)

                if not skill_id:
                    continue

                entry = skills_map.setdefault(skill_id, {})
                if group_norm.endswith(".name"):
                    entry["name_tpl"] = (group_norm, skill_id)
                else:
                    entry["icon_tpl"] = (group_norm, skill_id)

            full_skills = [sid for sid, v in skills_map.items() if "name_tpl" in v and "icon_tpl" in v]
            structured_log("fight_skills_autodetected", count=len(full_skills), skills=full_skills)

            for skill_id in full_skills:
                name_group, name_name = skills_map[skill_id]["name_tpl"]
                icon_group, icon_name = skills_map[skill_id]["icon_tpl"]

                # --- По имени ---
                try:
                    skill_name_tpl = TL.path(name_group, name_name)
                except Exception as e:
                    structured_log("fight_skill_path_error", skill=skill_id, kind="name", error=str(e))
                    skill_name_tpl = None

                if skill_name_tpl and os.path.exists(skill_name_tpl):
                    if diag_click(skill_name_tpl, "skill_text_btn", reason=f"{skill_id}_name"):
                        skill_clicked = True
                        structured_log("fight_skill_clicked", skill=skill_id, kind="name")
                        break
                else:
                    structured_log("fight_skill_missing_tpl", skill=skill_id, kind="name", path=skill_name_tpl)

                # --- По иконке ---
                try:
                    skill_icon_tpl = TL.path(icon_group, icon_name)
                except Exception as e:
                    structured_log("fight_skill_path_error", skill=skill_id, kind="icon", error=str(e))
                    skill_icon_tpl = None

                if skill_icon_tpl and os.path.exists(skill_icon_tpl):
                    if diag_click(skill_icon_tpl, "skill_icon_btn", reason=f"{skill_id}_icon"):
                        skill_clicked = True
                        structured_log("fight_skill_clicked", skill=skill_id, kind="icon")
                        break
                else:
                    structured_log("fight_skill_missing_tpl", skill=skill_id, kind="icon", path=skill_icon_tpl)
        else:
            structured_log("fight_skill_skip", reason="moves_not_opened")

        if skill_clicked:
            clicked = True

        # --- 3) Фолбэк: базовая атака ---
        if not skill_clicked and attack_path:
            if diag_click(attack_path, "attack_btn", reason="basic_attack"):
                clicked = True
        elif not skill_clicked and not attack_path:
            structured_log("fight_attack_skip", reason="no_attack_path")

        # --- 4) Проверка победы / поражения ---
        scr = diag_screenshot(tag="loop")
        # Победа
        if scr is not None and victory_path:
            if diag_find(victory_path, "victory_icon", tag="victory_check", scr=scr):
                leave_open = bool(CFG.get("FIGHT", {}).get("LEAVE_VICTORY_OPEN", True))
                if leave_open:
                    structured_log("fight_victory_detected_leave_open", note="FSM will scan drop and click continue")
                    return "victory"
                else:
                    if diag_close_result_screen(
                        kind_tag="victory",
                        icon_path=victory_path,
                        continue_path=continue_path,
                        continue_thr_key="continue_btn",
                    ):
                        return "victory"

        # Поражение (если конфиг есть)
        if scr is not None and defeat_path:
            if diag_find(defeat_path, "defeat_icon", tag="defeat_check", scr=scr):
                if diag_close_result_screen(
                    kind_tag="defeat",
                    icon_path=defeat_path,
                    continue_path=continue_path,
                    continue_thr_key="continue_btn",
                ):
                    return "defeat"

        # --- 5) Idle guard ---
        idle_ticks = 0 if clicked else idle_ticks + 1
        structured_log("fight_idle_tick", loop=loop_idx, clicked=clicked, idle_ticks=idle_ticks, limit=IDLE_LIMIT)
        if idle_ticks >= IDLE_LIMIT:
            structured_log("fight_abort", reason="idle_guard", loop=loop_idx)
            return "abort"

        # --- 6) Пауза цикла ---
        delay = rand_duration(LOOP_DELAY)
        structured_log("fight_sleep", loop=loop_idx, delay_s=delay)
        time.sleep(delay)


# --- Точка входа ---
if __name__ == "__main__":
    result = fight_loop_until_victory()
    Logger(f"[FIGHT] Результат боя: {result}")
