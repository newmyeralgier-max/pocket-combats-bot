# -*- coding: utf-8 -*-
import os
import sys
import time
import datetime
import random
import signal
import json
from typing import Dict, List, Any, Tuple, Optional, Set
import numpy as np

# Ensure c:\bot is in sys.path when running auto_bot.py directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Imports of separated modules (and re-exports for other scripts that rely on auto_bot.py)
from script.core.config import (
    CFG, ALLOWED_ITEM_NAMES, ALLOW_PICKUP_OTHER, CARD_OPEN_DELAY_MS,
    DEFAULT_CFG, DELAY_BETWEEN_LOOPS, DETECT_ONLY_WHITELIST, EXCLUDE_ZONES,
    FIND, FIND_STABILIZE_FRAMES, ITEM_SCALES, ITEMS_ROI,
    LOGIC, MATCH, MAX_LOOT_ROUNDS, MAX_SCREENSHOT_ATTEMPTS, MAX_SWIPE_ATTEMPTS,
    ORDER, PICKUP_ACTIVE_THRESHOLD, PICKUP_COLOR_RATIO_THRESHOLD, PICKUP_COLOR_SAT_MIN,
    PICKUP_COLOR_VAL_MIN, PICKUP_INACTIVE_THRESHOLD, PICKUP_METHOD_WEIGHTS, PICKUP_REL,
    PICKUP_REL_CARD_FALLBACK, PICKUP_SCALES, PICKUP_TPL_THRESHOLD, POST_SWIPE_DELAY_MS,
    QT, QT_DELAY_BETWEEN, QT_JITTER_PCT, QT_POST_SERIES, QT_X1, QT_X2, QT_Y_OFFSET,
    RESCAN_AFTER_PICKUP, ROI_CFG, SAFE_X1, SAFE_X2, SAFE_Y1, SAFE_Y2,
    SKIP_SUBSTR, SWP, SWIPE_DURATION_MS, SWIPE_PAUSE_MS, SWIPE_SAME_HASH_STOP_N,
    SWIPE_STOP_ON_REPEAT_HASH, SWIPE_VECTORS, TIMINGS, TPL_PICKUP_ACTIVE_LIST,
    TPL_PICKUP_INACTIVE, TPLS, VERIFY1_MS, VERIFY2_MS, ABORT_ON_FIRST_INACTIVE
)
from script.core.logging import (
    log, structured_log, snap, snap_roi, dbg_name, dbg_save, PERF,
    STEP_COUNTER, mark_step, next_step, now_ts
)
from script.device.adb import (
    adb_cmd, device_swipe, get_capture, screenshot_bgr, tap, tap_fast, tap_raw
)
from script.detection.items import (
    _is_service_file, compute_items_visible_roi, find_item_names,
    get_item_name_templates, match_multi_scaled, merge_same_lines,
    IGNORE_SKIP_SUBSTR, IGNORE_WHITELIST, ALLOWED_LOOT_DIRS, FORCE_FS_SCAN
)
from script.detection.pickup import (
    pickup_state, swipe_until_visible_pickup, verify_item_removed
)

from script.loot.utils import clamp, get_thr, page_hash_from_bgr, detect_chat_top_y
from script.loot.tab_detector import detect_tab_states
from script.loot.quick_tap import quick_tap_for_label

_shutdown_requested = False

def _handle_shutdown_signal(sig, frame):
    global _shutdown_requested
    if not _shutdown_requested:
        log("\n[SYSTEM] Received shutdown signal. Will exit after current loot round...")
        _shutdown_requested = True

try:
    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
except ValueError:
    pass  # Not running in main thread

slot_lifecycle: Dict[Tuple[int, int], dict] = {}
_VICTORY_TARGETS: List[str] = []

def set_victory_targets(targets: List[str]):
    global _VICTORY_TARGETS
    _VICTORY_TARGETS = list(targets or [])
    structured_log("victory_targets_set", targets=_VICTORY_TARGETS)

def get_victory_targets() -> List[str]:
    return list(_VICTORY_TARGETS)

def clear_victory_targets():
    global _VICTORY_TARGETS
    _VICTORY_TARGETS = []
    structured_log("victory_targets_cleared")

def update_slot_lifecycle(found_items: list, stage: str) -> None:
    now_ts_str = datetime.datetime.now().isoformat(timespec="seconds")
    seen_keys = set()
    for it in found_items:
        key = int(it["slot_hash"]), int(it["box"][1])
        seen_keys.add(key)
        if key not in slot_lifecycle:
            slot_lifecycle[key] = {
                "name": it["name"],
                "first_seen_ts": now_ts_str,
                "last_seen_ts": now_ts_str,
                "status": stage,
            }
        else:
            slot_lifecycle[key]["last_seen_ts"] = now_ts_str
            if slot_lifecycle[key]["status"] not in ("picked_by_bot", "taken_by_other"):
                slot_lifecycle[key]["status"] = stage
    if stage == "detected":
        for key, data in slot_lifecycle.items():
            if key not in seen_keys and data["status"] not in ("picked_by_bot", "taken_by_other"):
                data["status"] = "taken_by_other"
                data["last_seen_ts"] = now_ts_str

def finalize_slot_lifecycle() -> None:
    structured_log("slot_lifecycle_report", slots=[{"hash": h, "y": y, **data} for (h, y), data in slot_lifecycle.items()])

def dump_queue(tag: str, items_list: list) -> None:
    structured_log("items_queue", tag=tag, items=[{"name": it["name"], "y": int(it["box"][1]), "hash": int(it["slot_hash"])} for it in items_list])

def super_swipe_before_loot():
    choice = random.choice([1, 2, 3])
    if choice == 1:
        device_swipe(0, -600, 350)
    elif choice == 2:
        for _ in range(2):
            device_swipe(0, -400, 300)
    else:
        for _ in range(3):
            device_swipe(0, -250, 250)

def detect_tabs_with_occlusion(frame_bgr: np.ndarray) -> Dict[str, Any]:
    return detect_tab_states(frame_bgr)

def collapse_if_open(panel_open: bool, click_xy: Tuple[int, int], reason: str = "collapse_item_panel") -> bool:
    if not panel_open:
        structured_log("collapse_skip", because="panel_already_closed")
        return False
    x, y = click_xy
    tap(x, y, reason=reason)
    time.sleep(0.3)
    structured_log("collapsed")
    return False

def ensure_tab_state(tab_label: str, target_state: str) -> bool:
    plan_sid = mark_step(f"ensure_tab_{tab_label}_{target_state}")
    frame = screenshot_bgr()
    if frame is None:
        structured_log("ensure_tab_no_frame", step_id=plan_sid)
        return False
    try:
        tabs = detect_tab_states(frame)
    except Exception as e:
        structured_log("ensure_tab_detect_fail", step_id=plan_sid, error=str(e))
        tabs = {}
    cur = tabs.get(tab_label, {}) or {}
    if cur.get("state") == target_state:
        structured_log("ensure_tab_already_ok", step_id=plan_sid, state=target_state)
        return True
    
    band_y = int(cur.get("band_y") or SAFE_Y1)

    def wait_and_check(poll_ms: int = 250, attempts: int = 6) -> bool:
        for _ in range(attempts):
            time.sleep(poll_ms / 1000.0)
            fr = screenshot_bgr()
            if fr is None:
                continue
            try:
                tbs = detect_tab_states(fr)
                st = tbs.get(tab_label, {}).get("state")
                if st == target_state:
                    return True
            except Exception:
                continue
        return False

    if band_y >= SAFE_Y1:
        structured_log("act_tab_toggle", step_id=plan_sid, method="band_line", x=180, y=band_y)
        tap(180, band_y, reason=f"{tab_label}:band_line -> toggle")
        if wait_and_check():
            structured_log("outcome_tab_toggle", step_id=plan_sid, result="success", via="band_line")
            return True
    else:
        structured_log("skip_band_line_tap", step_id=plan_sid, reason="band_y_below_safe", band_y=band_y, safe_y1=SAFE_Y1)

    for cx in (1050, 950):
        if band_y >= SAFE_Y1:
            structured_log("act_tab_toggle", step_id=plan_sid, method="chevron", x=cx, y=band_y)
            tap(cx, band_y, reason=f"{tab_label}:chevron -> toggle")
            if wait_and_check():
                structured_log("outcome_tab_toggle", step_id=plan_sid, result="success", via="chevron", x=cx, y=band_y)
                return True

    structured_log("outcome_tab_toggle", step_id=plan_sid, result="failed", reason="no_state_change")
    log(f"[WARN] {tab_label}: state не меняется после label/band/chevron — пропускаем")
    return False

def auto_loot_once():
    PERF_REPORT_EVERY_N = int(CFG.get("PERF_REPORT_EVERY_N", 10))
    if PERF_REPORT_EVERY_N > 0 and STEP_COUNTER % PERF_REPORT_EVERY_N == 0:
        structured_log("perf_report", **PERF.report())
    log("=== AUTO-LOOT START (strict top→bottom) ===")
    ensure_tab_state("монстры", "закрыта")
    ensure_tab_state("вещи", "открыта")
    super_swipe_before_loot()
    log("[PRE-LOOT] Вкладки в нужном состоянии — делаем стартовый лист")
    structured_log("skip_old_swipe", reason="removed legacy initial loot swipe")

    processed_slots_success: set[tuple[int, int]] = set()
    processed_slots_skipped: set[tuple[int, int]] = set()
    last_page_hash = None
    frame = screenshot_bgr()
    if frame is None:
        return
    page_hash = page_hash_from_bgr(frame)

    def in_set(h, y, S):
        return any(h == ph and abs(y - py) <= int(CFG.get("item_group_threshold", 50)) for ph, py in S)

    def is_processed_success(item):
        return in_set(int(item["slot_hash"]), int(item["box"][1]), processed_slots_success)

    def is_marked_skipped(item):
        return in_set(int(item["slot_hash"]), int(item["box"][1]), processed_slots_skipped)

    tabs = detect_tabs_with_occlusion(frame)
    if tabs.get("occluded_by_extra"):
        fix_sid = mark_step("fights_tab_shift_fix")
        structured_log("plan_fix_fights_shift", step_id=fix_sid, reason="fights_tab_detected")
        log("[TABS] Обнаружена вкладка 'Сражения' — корректирующий свайп вверх >150px")
        dy = -400
        device_swipe(0, dy, SWIPE_DURATION_MS)
        time.sleep(POST_SWIPE_DELAY_MS / 1000.0)
        frame = screenshot_bgr()
        tabs = detect_tabs_with_occlusion(frame)
        page_hash = page_hash_from_bgr(frame)
        structured_log("outcome_fix_fights_shift", step_id=fix_sid, dy=dy, occluded_by_extra=bool(tabs.get("occluded_by_extra")))
    
    raw_items = find_item_names(screenshot_bgr)
    if tabs.get("occluded_by_extra") and len(raw_items) < 3:
        structured_log("occlusion_fix", reason="few_items_visible")
        device_swipe(0, -700, SWIPE_DURATION_MS)
        time.sleep(POST_SWIPE_DELAY_MS / 1000.0)
        frame = screenshot_bgr()
        page_hash = page_hash_from_bgr(frame)
    items = find_item_names(screenshot_bgr) or []
    items.sort(key=lambda d: d["box"][1])
    dump_queue("initial", items)
    structured_log("list_scan", count=len(items))
    if page_hash == last_page_hash:
        items = [it for it in items if not is_processed_success(it) and not is_marked_skipped(it)]
        structured_log("filter_same_page", same=True, kept=len(items))
    else:
        processed_slots_skipped.clear()
        processed_slots_success.clear()
        structured_log("filter_same_page", same=False, reset_sets=True)
    last_page_hash = page_hash

    def quick_double_tap_from_slot(slot):
        label_box_xywh = slot["box"]
        safe_xywh = SAFE_X1, SAFE_Y1, SAFE_X2 - SAFE_X1, SAFE_Y2 - SAFE_Y1
        frame_dbg = screenshot_bgr()
        quick_tap_for_label(
            tap_fn=tap_fast,
            label_box_xywh=label_box_xywh,
            safe_xywh=safe_xywh,
            cfg=CFG,
            frame_bgr=frame_dbg,
            slot_id=None,
            label_text=slot.get("name"),
        )

    while items:
        current = items[0]
        name = current["name"]
        cx, cy = current["center"]
        base_hash = int(current["slot_hash"])
        base_y = int(current["box"][1])
        if is_processed_success(current):
            items.pop(0)
            continue
        log(f"[ITEM] open '{name}' at {current['center']} y={base_y} hash={hex(base_hash)}")
        structured_log("focus_item", item=name, box=current["box"], center=current["center"], base_hash=hex(base_hash))
        if not ALLOW_PICKUP_OTHER and name not in ALLOWED_ITEM_NAMES:
            log(f"[FLOW] '{name}' не в whitelist — пропуск без открытия")
            structured_log("skip_non_whitelist", item=name, box=current["box"])
            processed_slots_success.add((base_hash, base_y))
            items.pop(0)
            continue
        quick_double_tap_from_slot(current)
        if verify_item_removed(name, base_hash, base_y):
            log(f"[FLOW] '{name}' подобран быстрым тапом — подтверждено")
            processed_slots_success.add((base_hash, base_y))
            update_slot_lifecycle([current], stage="picked_by_bot")
            items.pop(0)
            continue
        tap(cx, cy, reason="open_item_card")
        panel_open = True
        time.sleep(max(0.001, CARD_OPEN_DELAY_MS / 1000.0))
        frame_card = screenshot_bgr()
        if frame_card is None:
            panel_open = collapse_if_open(panel_open, (cx, cy))
            processed_slots_skipped.add((base_hash, base_y))
            items.pop(0)
            continue
        H, W = frame_card.shape[:2]
        chat_top = detect_chat_top_y(frame_card)
        top = clamp(cy - 300, SAFE_Y1, SAFE_Y2 - 60)
        bottom = clamp(min(chat_top - 8, SAFE_Y2), top + 60, SAFE_Y2)
        card_rect = int(W * 0.56), top, int(W * 0.42), bottom - top
        st = swipe_until_visible_pickup(screenshot_bgr, base_y=base_y, card_rect=card_rect)
        state, box = st["state"], st.get("box")
        log(f"[PICKUP] state={state} score={st.get('score')} method={st.get('method')}")
        if state == "inactive":
            panel_open = collapse_if_open(panel_open, (cx, cy))
            if ABORT_ON_FIRST_INACTIVE:
                structured_log("inactive_abort_context", current={"name": name, "y": base_y})
                structured_log("exit_reason", reason="pickup_button_inactive")
                finalize_slot_lifecycle()
                ensure_tab_state("вещи", "закрыта")
                log("=== AUTO-LOOT END ===")
                return
            else:
                processed_slots_skipped.add((base_hash, base_y))
                items.pop(0)
                continue
        if state == "active" and box:
            bx, by = box[0] + box[2] // 2, box[1] + box[3] // 2
            tap(bx, by, reason="pickup")
            if verify_item_removed(name, base_hash, base_y):
                log(f"[FLOW] '{name}' подобран — подтверждено")
            else:
                log(f"[WARN] '{name}' не исчез — но помечаем обработанным")
            processed_slots_success.add((base_hash, base_y))
            update_slot_lifecycle([current], stage="picked_by_bot")
        if state == "out_of_view":
            processed_slots_skipped.add((base_hash, base_y))
            update_slot_lifecycle([current], stage="taken_by_other")
        rescan_sid = mark_step("rescan_items")
        structured_log("plan_rescan", step_id=rescan_sid, reason="post_item_action_or_scroll")
        frame = screenshot_bgr()
        page_hash = page_hash_from_bgr(frame)
        structured_log("act_rescan", step_id=rescan_sid, page_hash=int(page_hash))
        all_items = find_item_names(screenshot_bgr) or []
        filtered = [
            it for it in all_items if not in_set(int(it["slot_hash"]), int(it["box"][1]), processed_slots_success)
        ]
        items = sorted(filtered, key=lambda d: d["box"][1])
        dump_queue("after_rescan", items)
        structured_log("outcome_rescan", step_id=rescan_sid, count=len(items), page_hash=int(page_hash))
        last_page_hash = page_hash
        tabs = detect_tabs_with_occlusion(frame)
        if not items and tabs.get("occluded_by_extra"):
            structured_log("exit_reason", reason="items_tab_hidden_until_fight_end")
            log("[FLOW] прекращаем лут — вкладка вещей скрыта сражениями")
            finalize_slot_lifecycle()
            return
        if not items:
            pag_sid = mark_step("pagination")
            structured_log("plan_pagination", step_id=pag_sid, vector=[0, 800], duration_ms=600)
            take_frame = screenshot_bgr()
            before_path = dbg_save("before_pagination_swipe", take_frame)
            structured_log(
                "act_pagination",
                step_id=pag_sid,
                page_hash_before=int(page_hash_from_bgr(take_frame)),
                image_before=before_path,
            )
            device_swipe(0, 800, 600)
            time.sleep(0.5)
            take_frame = screenshot_bgr()
            after_path = dbg_save("after_pagination_swipe", take_frame)
            page_hash = page_hash_from_bgr(take_frame)
            last_page_hash = page_hash
            new_items = find_item_names(screenshot_bgr) or []
            filtered_new = [
                it for it in new_items if not in_set(int(it["slot_hash"]), int(it["box"][1]), processed_slots_success)
            ]
            items = sorted(filtered_new, key=lambda d: d["box"][1])
            structured_log(
                "outcome_pagination",
                step_id=pag_sid,
                count=len(items),
                page_hash_after=int(page_hash),
                image_after=after_path,
            )
            tabs = detect_tabs_with_occlusion(take_frame)
            if tabs.get("occluded_by_extra"):
                structured_log("exit_reason", reason="items_tab_hidden_until_fight_end")
                log("[FLOW] прекращаем лут — вкладка вещей скрыта сражениями")
                finalize_slot_lifecycle()
                return
    finalize_slot_lifecycle()
    ensure_tab_state("вещи", "закрыта")
    log("=== AUTO-LOOT END ===")

def main():
    try:
        adb_cmd(["get-state"])
    except Exception as e:
        log(f"[ERROR] ADB недоступен: {e}")
        return
    rounds = 0
    log("[MAIN] Entering main loop")
    try:
        while True:
            rounds += 1
            if MAX_LOOT_ROUNDS and rounds > MAX_LOOT_ROUNDS:
                log(f"[MAIN] reached max_loot_rounds={MAX_LOOT_ROUNDS}, exiting")
                break
            try:
                if _shutdown_requested:
                    log("[MAIN] Exit loop: shutdown requested")
                    break
                auto_loot_once()
            except Exception as e:
                log(f"[ERROR] Exception in auto_loot_once: {e}")
            log(f"[MAIN] Sleeping {DELAY_BETWEEN_LOOPS}s before next loop")
            time.sleep(DELAY_BETWEEN_LOOPS)
    except KeyboardInterrupt:
        log("[MAIN] Interrupted by user (KeyboardInterrupt)")
    except Exception as e:
        log(f"[MAIN] Unexpected error: {e}")
    finally:
        log("[MAIN] Exiting main")

if __name__ == "__main__":
    main()
