# -*- coding: utf-8 -*-
"""
FSM: FIND -> FIGHT -> LOOT -> RECOVER
Логи переходов пишутся в тот же лог, что и остальной бот.
"""
import os
import sys
import importlib

# Ensure the root directory (c:\bot) is in sys.path so we can import 'script.*'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time
import cv2  # type: ignore[import-not-found]

from script.loot.matcher import find_all_matches, preprocess_gray, TPL_CACHE  # type: ignore[import-not-found]
from script.overlays.clear_overlays import clear_overlays, overlay_just_appeared  # type: ignore[import-not-found]
from script.fight.fight import fight_loop_until_victory  # type: ignore[import-not-found]

# ── Runtime from auto_bot & other modules ────────────────────────────
auto_bot = importlib.import_module("script.loot.auto_bot")
from script.core.config import CFG, DEBUG_DIR
from script.core.logging import structured_log, log, now_ts
from script.loot.utils import to_gray, frame_hash, imread_u8, ui_tpl_path
from script.device.adb import screenshot_bgr, adb_cmd, tap_raw

ensure_tab_state = auto_bot.ensure_tab_state
auto_loot_once = auto_bot.auto_loot_once

# ── FSM states ───────────────────────────────────────────────────────
STATE_FIND = "find_monster"
STATE_FIGHT = "fight"
STATE_LOOT = "loot"
STATE_RECOVER = "recover"
STATE_UNKNOWN = "unknown_screen"
current_state = STATE_FIND
fsm_context = {"allowed_loot_ids": []}


def _detect_continue_and_wait(CFG, max_wait_s=1.2, settle_s=0.4):
    """
    Ждём появления кнопки CONTINUE и даём кадру 'досинхрониться' для прорисовки иконок.
    Возвращаем последний кадр (frame_bgr) либо None.
    """
    try:
        cont_path = ui_tpl_path("continue")  # без .png
        tpl = imread_u8(cont_path, cv2.IMREAD_COLOR)
        if tpl is None:
            structured_log("fsm_continue_tpl_missing", path=cont_path)
            return None
        scales = list(CFG.get("FIND", {}).get("ITEM_SCALES", [0.9, 0.95, 1.0, 1.05]))
        thr = float(CFG.get("FIGHT", {}).get("THRESHOLDS", {}).get("continue_btn", 0.82))
        min_dy = int(CFG.get("FIND", {}).get("MIN_DY", 10))
        mode = CFG.get("FIND", {}).get("PREPROC_MODE", "gray") or "gray"

        t0 = time.time()
        while time.time() - t0 <= max_wait_s:
            fr = screenshot_bgr()
            if fr is None:
                time.sleep(0.1)
                continue
            fr_p = preprocess_gray(fr, mode=mode)
            tpl_scaled = TPL_CACHE.get_all_scaled(cont_path, tpl, mode, scales)
            hits = find_all_matches(fr_p, tpl_scaled, scales=scales, threshold=thr, min_dy=min_dy)
            if hits:
                time.sleep(settle_s)  # дать дорисоваться иконкам
                return fr
            time.sleep(0.1)
        structured_log("fsm_continue_not_detected_timeout")
        return None
    except Exception as e:
        structured_log("fsm_continue_detect_error", error=str(e))
        return None


def _click_continue(CFG):
    """
    Нажимаем CONTINUE по шаблону; если не нашли — тапаем фолбэком в нижнюю часть экрана.
    """
    try:
        cont_path = ui_tpl_path("continue")
        tpl = imread_u8(cont_path, cv2.IMREAD_COLOR)
        fr = screenshot_bgr()
        if tpl is None or fr is None:
            # Фолбэк: безопасный тап ниже центра
            tap_raw(int(CFG.get("SCREEN_W", 1080)) // 2, 2250, reason="fsm_continue_fallback")
            structured_log("fsm_continue_fallback_tap")
            return

        mode = CFG.get("FIND", {}).get("PREPROC_MODE", "gray") or "gray"
        scales = list(CFG.get("FIND", {}).get("ITEM_SCALES", [0.9, 0.95, 1.0, 1.05]))
        thr = float(CFG.get("FIGHT", {}).get("THRESHOLDS", {}).get("continue_btn", 0.82))
        min_dy = int(CFG.get("FIND", {}).get("MIN_DY", 10))

        fr_p = preprocess_gray(fr, mode=mode)
        tpl_scaled = TPL_CACHE.get_all_scaled(cont_path, tpl, mode, scales)
        hits = find_all_matches(fr_p, tpl_scaled, scales=scales, threshold=thr, min_dy=min_dy)
        if hits:
            mx, my, s, sc = hits[0]
            tpl_gray = preprocess_gray(tpl, mode=mode)
            h, w = tpl_gray.shape[:2]
            tw = max(5, int(round(w * s)))
            th = max(5, int(round(h * s)))
            cx, cy = mx + tw // 2, my + th // 2
            tap_raw(int(cx), int(cy), reason="fsm_continue_click")
            structured_log("fsm_continue_clicked", score=float(sc), scale=float(s), center=[int(cx), int(cy)])
        else:
            tap_raw(int(CFG.get("SCREEN_W", 1080)) // 2, 2250, reason="fsm_continue_fallback_nohit")
            structured_log("fsm_continue_fallback_nohit")
    except Exception as e:
        structured_log("fsm_continue_click_error", error=str(e))


def fsm_loop():
    global current_state
    delay_sec = float(CFG.get("MAIN", {}).get("DELAY_SEC", 0.5))
    max_loops = int(CFG.get("MAIN", {}).get("MAX_LOOPS", 0)) or 999999
    max_state_duration = float(CFG.get("MAX_STATE_DURATION_S", 60.0))
    loops = 0
    total_fights = 0
    victories = 0
    
    state_start_time = time.time()
    last_state = current_state
    unknown_retries = 0

    while loops < max_loops:
        if getattr(auto_bot, "_shutdown_requested", False):
            log("[SHUTDOWN] FSM loop ending cleanly due to shutdown request.")
            from script.core.logging import _logger_v2
            _logger_v2.flush()
            break
            
        loops += 1

        if current_state != last_state:
            state_start_time = time.time()
            last_state = current_state
        elif time.time() - state_start_time > max_state_duration and current_state != STATE_UNKNOWN:
            log(f"[WARN] Stuck in state {current_state} for {time.time() - state_start_time:.1f}s. Entering UNKNOWN.")
            current_state = STATE_UNKNOWN

        # Логим старт шага
        log(f"[FSM] loop={loops} | state={current_state}")

        # Всегда чистим новые оверлеи
        if overlay_just_appeared(CFG):
            log("[FSM] overlay detected — clearing")
            clear_overlays(CFG, cooldown_s=0.0, log_no_overlay=False)

        if current_state == STATE_UNKNOWN:
            screen_dir = os.path.join(DEBUG_DIR, "unknown_screens")
            os.makedirs(screen_dir, exist_ok=True)
            fr = screenshot_bgr()
            if fr is not None:
                cv2.imwrite(os.path.join(screen_dir, f"stuck_{now_ts()}.png"), fr)
                
            clear_overlays(CFG, cooldown_s=0.0, log_no_overlay=False)
            adb_cmd(["shell", "input", "keyevent", "4"])  # Back
            log("[FSM] STATE_UNKNOWN recovery attempt (Back button + clear overlays)")
            time.sleep(3.0)
            
            fr_new = screenshot_bgr()
            if fr is not None and fr_new is not None:
                # Compare frame hashes
                h1 = frame_hash(to_gray(fr))
                h2 = frame_hash(to_gray(fr_new))
                if h1 != h2:
                    log("[FSM] Screen changed. Resuming from FIND.")
                    current_state = STATE_FIND
                    unknown_retries = 0
                    continue
            
            unknown_retries += 1
            if unknown_retries >= int(CFG.get("UNKNOWN_SCREEN_MAX_RETRIES", 3)):
                log("[ERROR] bot_stuck: 3 attempts failed. Waiting 30s before retry FSM.")
                time.sleep(30.0)
                unknown_retries = 0
                current_state = STATE_FIND  # attempt full reset
                
        elif current_state == STATE_FIND:
            # Никаких табов здесь не трогаем — бой сам откроет 'монстры'
            current_state = STATE_FIGHT

        elif current_state == STATE_FIGHT:
            total_fights += 1
            res = fight_loop_until_victory()
            log(f"[FSM] fight result: {res}")

            if res == "victory":
                victories += 1
                try:
                    import script.loot.victory_drop as victory_drop

                    # 1) Ждём CONTINUE и даём кадру стабилизироваться
                    frame_bgr = _detect_continue_and_wait(CFG, max_wait_s=1.2, settle_s=0.4)
                    if frame_bgr is None:
                        structured_log("fsm_victory_skip", reason="no_continue_button_detected")
                        targets = []

                        # важно: даже если не нашли — все равно выходим с экрана победы
                        _click_continue(CFG)
                    else:
                        auto_bot.snap("victory_screen_full", frame_bgr)

                        # 2) Сканируем дроп (ROI 0..1080, 1000..1500 уже есть в victory_drop)
                        drop_info = victory_drop.scan_victory_drop_targets(frame_bgr, CFG)
                        targets = list(drop_info.get("allowed_loot_ids", []))
                        structured_log("fsm_victory_targets_detected", targets=targets, count=len(targets))

                        # 3) Выход с экрана победы
                        _click_continue(CFG)

                    # 4) Сохранить цели в контекст
                    fsm_context["allowed_loot_ids"] = targets

                except Exception as e:
                    structured_log("fsm_victory_drop_error", error=str(e))
                    fsm_context["allowed_loot_ids"] = []
                    targets = []

                # Решение FSM
                if not fsm_context["allowed_loot_ids"]:
                    log("[FSM] no allowed loot detected — skipping loot phase")
                    auto_bot.clear_victory_targets()
                    current_state = STATE_FIND  # сразу новый бой
                else:
                    log(f"[FSM] allowed loot for next phase: {fsm_context['allowed_loot_ids']}")
                    auto_bot.set_victory_targets(fsm_context["allowed_loot_ids"])
                    current_state = STATE_LOOT

            else:
                current_state = STATE_RECOVER

        elif current_state == STATE_LOOT:
            log("[FSM] starting loot")
            ensure_tab_state("монстры", "закрыта")
            ensure_tab_state("вещи", "открыта")
            try:
                if fsm_context["allowed_loot_ids"]:
                    auto_bot.set_victory_targets(fsm_context["allowed_loot_ids"])
                auto_loot_once()
            finally:
                fsm_context["allowed_loot_ids"] = []  # очистка после лутинга
            current_state = STATE_FIND

        elif current_state == STATE_RECOVER:
            log("[FSM] recovering")
            # Никаких табов тут — только быстро прибираем оверлеи и возвращаемся искать
            clear_overlays(CFG, cooldown_s=0.0, log_no_overlay=False)
            current_state = STATE_FIND

        # Логим конец шага
        log(f"[FSM] next_state={current_state}")
        time.sleep(delay_sec)

    # Итог
    if total_fights:
        win_rate = round((victories / total_fights) * 100, 1)
        log(f"[FSM] fights={total_fights}, wins={victories}, win_rate={win_rate}%")
    else:
        log("[FSM] no fights recorded")


if __name__ == "__main__":
    try:
        log("[FSM] boot")
        fsm_loop()
    except KeyboardInterrupt:
        log("[FSM] interrupted by user")
    log("[FSM] exit")
