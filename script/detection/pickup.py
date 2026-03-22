# -*- coding: utf-8 -*-
import os
import time
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np

from script.core.config import (
    CFG,
    PICKUP_ACTIVE_THRESHOLD, PICKUP_INACTIVE_THRESHOLD, PICKUP_TPL_THRESHOLD, PICKUP_SCALES,
    PICKUP_COLOR_SAT_MIN, PICKUP_COLOR_VAL_MIN, PICKUP_COLOR_RATIO_THRESHOLD, PICKUP_METHOD_WEIGHTS,
    PICKUP_REL, PICKUP_REL_CARD_FALLBACK,
    TPL_PICKUP_ACTIVE_LIST, TPL_PICKUP_INACTIVE,
    SAFE_Y1, SAFE_Y2, SWIPE_DURATION_MS, POST_SWIPE_DELAY_MS,
    VERIFY1_MS, VERIFY2_MS
)
from script.core.logging import structured_log, snap, snap_roi, mark_step, PERF
from script.loot.utils import to_gray, scharr_mag, match_scaled, rect_from_rel, page_hash_from_bgr, frame_hash, detect_chat_top_y
from script.loot.tpl_loader import _tpl
from script.device.adb import device_swipe, screenshot_bgr
from script.detection.items import find_item_names

@PERF.measure("pickup_detect")
def pickup_state(frame_bgr: np.ndarray, card_rect: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
    if frame_bgr is None:
        return {"state": "out_of_view", "score": 0.0, "box": None, "detail": "no_frame"}
    H, W = frame_bgr.shape[:2]
    if card_rect:
        cx, cy, cw, ch = card_rect
        rx1, ry1, rx2, ry2 = rect_from_rel(cw, ch, PICKUP_REL_CARD_FALLBACK)
        roi_rect = cx + rx1, cy + ry1, rx2 - rx1, ry2 - ry1
        tag = "pickup_card_fallback"
    else:
        rx1, ry1, rx2, ry2 = rect_from_rel(W, H, PICKUP_REL)
        roi_rect = rx1, ry1, rx2 - rx1, ry2 - ry1
        tag = "pickup_global"
    x1, y1, w, h = roi_rect
    x2, y2 = x1 + w, y1 + h
    if w <= 0 or h <= 0 or x2 > W or y2 > H:
        return {"state": "out_of_view", "score": 0.0, "box": None, "detail": "bad_roi_rect"}
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return {"state": "out_of_view", "score": 0.0, "box": None, "detail": "empty_roi"}
    
    snap_roi(f"{tag}", roi, roi_rect)
    gray_roi = to_gray(roi)
    edge_roi = scharr_mag(gray_roi)
    act_tpls = []
    for p in TPL_PICKUP_ACTIVE_LIST:
        img = _tpl(p)
        if img is not None:
            act_tpls.append(img)
    inact_tpl = _tpl(TPL_PICKUP_INACTIVE) if TPL_PICKUP_INACTIVE else None

    best_templ = {"kind": None, "score": -1.0, "box": None}
    for tpl in act_tpls:
        ok, score, (tx, ty, tw, th) = match_scaled(gray_roi, to_gray(tpl), PICKUP_TPL_THRESHOLD, PICKUP_SCALES)
        if score > float(best_templ.get("score", -1.0)):
            best_templ = {"kind": "active", "score": score, "box": (x1 + tx, y1 + ty, tw, th)}

    if inact_tpl is not None:
        ok, score, (tx, ty, tw, th) = match_scaled(gray_roi, to_gray(inact_tpl), PICKUP_TPL_THRESHOLD, PICKUP_SCALES)
        if score > float(best_templ.get("score", -1.0)):
            best_templ = {"kind": "inactive", "score": score, "box": (x1 + tx, y1 + ty, tw, th)}

    templ_score = max(0.0, float(best_templ.get("score", 0.0)))
    edge_score = 0.0
    if best_templ["box"]:
        tpl_src = act_tpls[0] if best_templ["kind"] == "active" and act_tpls else (inact_tpl if inact_tpl is not None else None)
        if tpl_src is not None:
            tpl_e = scharr_mag(to_gray(tpl_src))
            ok, e_score, _ = match_scaled(edge_roi, tpl_e, 0.0, PICKUP_SCALES)
            edge_score = float(e_score)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    color_mask = (sat > PICKUP_COLOR_SAT_MIN) & (val > PICKUP_COLOR_VAL_MIN)
    color_ratio = np.count_nonzero(color_mask) / (w * h) if w * h > 0 else 0.0

    w_t = PICKUP_METHOD_WEIGHTS.get("templ", 0.6)
    w_e = PICKUP_METHOD_WEIGHTS.get("edge", 0.25)
    w_c = PICKUP_METHOD_WEIGHTS.get("color", 0.15)
    color_score = min(1.0, color_ratio / PICKUP_COLOR_RATIO_THRESHOLD)
    final_score = (templ_score * w_t) + (edge_score * w_e) + (color_score * w_c)

    res = {
        "box": best_templ.get("box"),
        "score_templ": templ_score,
        "score_edge": edge_score,
        "score_color": color_score,
        "score_combined": final_score,
    }

    if best_templ["kind"] == "active" and final_score >= PICKUP_ACTIVE_THRESHOLD:
        res["state"] = "active"
        res["score"] = final_score
    elif best_templ["kind"] == "inactive" and final_score >= PICKUP_INACTIVE_THRESHOLD:
        res["state"] = "inactive"
        res["score"] = final_score
    else:
        res["state"] = "out_of_view"
        res["score"] = final_score

    if getattr(CFG, "DEBUG", True):
        structured_log("pickup_detect", result=res["state"], **res)
    return res

def swipe_until_visible_pickup(get_frame_fn, base_y: int, card_rect: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
    from script.core.logging import STEP_COUNTER  # local to avoid circular import if needed (though it's in core)
    frame0 = get_frame_fn()
    st0 = pickup_state(frame0, card_rect=card_rect)
    structured_log("pickup_check_init", step_id=STEP_COUNTER, state=st0["state"])
    snap("pickup_init", frame0, step_id=STEP_COUNTER)
    if st0["state"] in ("active", "inactive"):
        return st0
    safe_top = SAFE_Y1
    safe_bottom = SAFE_Y2
    span = safe_bottom - safe_top
    midline = safe_top + span // 2
    has_box = bool(st0.get("box"))
    cy = st0["box"][1] + st0["box"][3] // 2 if has_box else None
    need_swipe = not has_box or cy is not None and cy > midline
    swipe_sid = mark_step("pickup_swipe_sequence")
    structured_log("plan_swipe_to_pickup", step_id=swipe_sid, midline=int(midline), cy=None if cy is None else int(cy), need_swipe=bool(need_swipe))
    if not need_swipe:
        structured_log("early_exit_pickup_visible_or_upper_half", cy=cy, midline=midline)
        return st0
    dy_strong = -max(300, span // 2)
    structured_log("act_swipe", step_id=swipe_sid, kind="strong_up", dy=int(dy_strong), duration_ms=SWIPE_DURATION_MS)
    device_swipe(0, dy_strong, SWIPE_DURATION_MS)
    time.sleep(POST_SWIPE_DELAY_MS / 1000.0)
    frame1 = get_frame_fn()
    st1 = pickup_state(frame1, card_rect=card_rect)
    structured_log("outcome_swipe_check", step_id=swipe_sid, after="strong_up", state=st1["state"])
    if st1["state"] in ("active", "inactive"):
        return st1
    structured_log("act_swipe", step_id=swipe_sid, kind="micro_up", dy=-150, duration_ms=SWIPE_DURATION_MS)
    device_swipe(0, -150, SWIPE_DURATION_MS)
    time.sleep(POST_SWIPE_DELAY_MS / 1000.0)
    frame2 = get_frame_fn()
    st2 = pickup_state(frame2, card_rect=card_rect)
    structured_log("outcome_swipe_check", step_id=swipe_sid, after="micro_up", state=st2["state"])
    return st2

def verify_item_removed(name: str, base_hash: int, base_y: int) -> bool:
    slot_thr = int(CFG.get("item_group_threshold", 50))
    time.sleep(VERIFY1_MS / 1000.0)
    items1 = find_item_names(screenshot_bgr) or []
    same_line1 = [it for it in items1 if abs(int(it["box"][1]) - int(base_y)) <= slot_thr]
    if not same_line1:
        return True
    if any(int(it["slot_hash"]) != int(base_hash) for it in same_line1):
        return True
    time.sleep(VERIFY2_MS / 1000.0)
    items2 = find_item_names(screenshot_bgr) or []
    same_line2 = [it for it in items2 if abs(int(it["box"][1]) - int(base_y)) <= slot_thr]
    if not same_line2:
        return True
    if any(int(it["slot_hash"]) != int(base_hash) for it in same_line2):
        return True
    return False
