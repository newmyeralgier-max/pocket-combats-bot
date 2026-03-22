import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pickup_detector as ext_pickup
from adb_actions import ADB


class Cfg:
    ASSETS = "C:\\bot\\tpl\\my"
    LOGS_DIR = "C:\\bot\\logs"
    DEBUG_DIR = "C:\\bot\\screens\\_loot_debug"
    LOG_BASENAME = "loot"
    DEBUG_SAVE = True
    DEBUG_SAVE_EVERY_N = 5
    THRESH_STRICT = 0.88
    THRESH_NORMAL = 0.84
    TPL = {
        "items_tab": ["items_tab"],
        "items_label_variants": ["items_label", "items_header", "items_hdr"],
        "monster_tab": ["monster_tab"],
        "monster_hdr_variants": ["monster_hdr"],
        "yantar": ["yantar"],
        "popup_yantar": ["popup_yantar_1"],
        "pickup_active_variants": ["pickup", "pickup_own"],
        "pickup_inactive_variants": ["pickup_other"],
    }
    WAIT_AFTER_TAP = 0.12
    WAIT_AFTER_OPEN = 0.3
    WAIT_AFTER_PICKUP = 0.3
    WAIT_AFTER_CLOSE = 0.2
    START_MONSTERS_OPEN = True
    START_ITEMS_OPEN = False
    SCREEN_W, SCREEN_H = 1080, 2460


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _now_fname():
    import datetime as dt

    return dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


LOG_PATH = None
DBG_SHOT_COUNTER = 0


def init_log():
    global LOG_PATH
    _ensure_dir(Cfg.LOGS_DIR)
    LOG_PATH = os.path.join(Cfg.LOGS_DIR, f"{Cfg.LOG_BASENAME}_{_now_fname()}.txt")
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(f"[init] {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def log(msg: str):
    line = f"{time.strftime('%H:%M:%S')} | {msg}"
    print(line)
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def save_debug(img: np.ndarray, tag: str, rects=None, points=None, force=False):
    global DBG_SHOT_COUNTER
    if not Cfg.DEBUG_SAVE and not force:
        return
    DBG_SHOT_COUNTER += 1
    if DBG_SHOT_COUNTER % Cfg.DEBUG_SAVE_EVERY_N != 0 and not force:
        return
    _ensure_dir(Cfg.DEBUG_DIR)
    vis = img.copy()
    if rects:
        for (x, y, w, h), color, label in rects:
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            if label:
                cv2.putText(vis, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    if points:
        for (cx, cy), color, label in points:
            cv2.circle(vis, (int(cx), int(cy)), 8, color, 2)
            if label:
                cv2.putText(vis, label, (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    path = os.path.join(Cfg.DEBUG_DIR, f"{_now_fname()}_{tag}.png")
    cv2.imwrite(path, vis)


TPL_CACHE: Dict[str, np.ndarray] = {}


def _asset_path(name: str) -> str:
    return os.path.join(Cfg.ASSETS, f"{name}.png")


def load_tpl(name: str) -> Optional[np.ndarray]:
    if name in TPL_CACHE:
        return TPL_CACHE[name]
    p = _asset_path(name)
    tpl = cv2.imread(p, cv2.IMREAD_COLOR)
    if tpl is None or tpl.size == 0:
        log(f"[TPL] not found/empty: {p}")
        return None
    TPL_CACHE[name] = tpl
    return tpl


def match_one(img: np.ndarray, tpl: np.ndarray, thr: float) -> Optional[Dict[str, Any]]:
    res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(res)
    if maxVal < thr:
        return None
    h, w = tpl.shape[:2]
    x, y = maxLoc
    return {"rect": (x, y, w, h), "center": (x + w // 2, y + h // 2), "score": float(maxVal)}


def find_key(img: np.ndarray, name: str, thr: float) -> Optional[Dict[str, Any]]:
    tpl = load_tpl(name)
    if tpl is None:
        return None
    return match_one(img, tpl, thr)


def find_any(img: np.ndarray, names: List[str], thr: float) -> Optional[Tuple[str, Dict[str, Any]]]:
    best = None
    best_key = None
    for n in names:
        hit = find_key(img, n, thr)
        if hit and (best is None or hit["score"] > best["score"]):
            best, best_key = hit, n
    if not best:
        return None
    return best_key, best


def detect_items_open(img: np.ndarray) -> Dict[str, Any]:
    hit = find_any(img, Cfg.TPL["items_label_variants"], Cfg.THRESH_STRICT)
    if hit:
        key, d = hit
        return {"open": True, "matched": key, "score": d["score"], "rect": d["rect"]}
    return {"open": False, "matched": None, "score": None, "rect": None}


def detect_monsters_open(img: np.ndarray) -> Dict[str, Any]:
    hit = find_any(img, Cfg.TPL["monster_hdr_variants"], Cfg.THRESH_NORMAL)
    if hit:
        key, d = hit
        return {"open": True, "matched": key, "score": d["score"], "rect": d["rect"]}
    return {"open": False, "matched": None, "score": None, "rect": None}


def tap_center(adb: ADB, hit: Dict[str, Any], tag: str):
    cx, cy = hit["center"]
    adb.tap(cx, cy)
    time.sleep(Cfg.Wait_AFTER_TAP if hasattr(Cfg, "Wait_AFTER_TAP") else Cfg.WAIT_AFTER_TAP)
    log(f"[tap] {tag} center=({cx},{cy})")


def ensure_tabs_state(adb: ADB, want_items_open: bool, want_monsters_open: bool) -> bool:
    ok = True
    img = adb.screencap_cv()
    if img is None:
        log("[tabs] no screen")
        return False
    items = detect_items_open(img)
    mons = detect_monsters_open(img)
    log(
        f"[tabs] current | items_open={items['open']} via={items['matched']} score={items['score']} rect={items['rect']} | monsters_open={mons['open']} via={mons['matched']} score={mons['score']} rect={mons['rect']}"
    )
    if items["open"] != want_items_open:
        hit = find_any(img, Cfg.TPL["items_tab"], Cfg.THRESH_NORMAL)
        if not hit:
            save_debug(img, "items_tab_not_found", force=True)
            log("[tabs] items_tab not found")
            ok = False
        else:
            _, d = hit
            tap_center(adb, d, "items_tab")
            img2 = adb.screencap_cv()
            if img2 is not None:
                items = detect_items_open(img2)
                log(
                    f"[tabs] after items | items_open={items['open']} via={items['matched']} score={items['score']} rect={items['rect']}"
                )
                img = img2
            if items["open"] != want_items_open:
                ok = False
    img3 = adb.screencap_cv()
    if img3 is not None:
        img = img3
    mons = detect_monsters_open(img)
    if mons["open"] != want_monsters_open:
        hit = find_any(img, Cfg.TPL["monster_tab"], Cfg.THRESH_NORMAL)
        if not hit:
            save_debug(img, "monster_tab_not_found", force=True)
            log("[tabs] monster_tab not found")
            ok = False
        else:
            _, d = hit
            tap_center(adb, d, "monster_tab")
            img4 = adb.screencap_cv()
            if img4 is not None:
                mons = detect_monsters_open(img4)
                log(
                    f"[tabs] after monsters | monsters_open={mons['open']} via={mons['matched']} score={mons['score']} rect={mons['rect']}"
                )
                img = img4
            if mons["open"] != want_monsters_open:
                ok = False
    return ok


def open_yantar_popup(adb: ADB) -> bool:
    img = adb.screencap_cv()
    if img is None:
        log("[yantar] no screen")
        return False
    already = (
        find_any(img, Cfg.TPL["popup_yantar"], Cfg.THRESH_NORMAL)
        or find_any(img, Cfg.TPL["pickup_active_variants"], Cfg.THRESH_NORMAL)
        or find_any(img, Cfg.TPL["pickup_inactive_variants"], Cfg.THRESH_NORMAL)
    )
    if already:
        log("[yantar] popup already open")
        save_debug(img, "popup_already_open")
        return True
    y = find_any(img, Cfg.TPL["yantar"], Cfg.THRESH_NORMAL)
    if not y:
        save_debug(img, "yantar_not_found", force=True)
        log("[yantar] 'yantar' not found (no offsets, as requested)")
        return False
    key, d = y
    save_debug(img, "yantar_click", rects=[(d["rect"], (200, 180, 40), f"{key} {d['score']:.2f}")])
    tap_center(adb, d, "yantar")
    time.sleep(Cfg.WAIT_AFTER_OPEN)
    img2 = adb.screencap_cv()
    if img2 is None:
        log("[yantar] no screen after tap")
        return False
    opened = bool(
        find_any(img2, Cfg.TPL["popup_yantar"], Cfg.THRESH_NORMAL)
        or find_any(img2, Cfg.TPL["pickup_active_variants"], Cfg.THRESH_NORMAL)
        or find_any(img2, Cfg.TPL["pickup_inactive_variants"], Cfg.THRESH_NORMAL)
    )
    save_debug(img2, f"yantar_open_verify_{'ok' if opened else 'fail'}", rects=[(d["rect"], (0, 165, 255), "after")])
    log("[yantar] popup open OK" if opened else "[yantar] popup open FAIL")
    return opened


def detect_pickup_ext(adb: ADB) -> Dict[str, Any]:
    """
    Делает screencap -> отдаёт в твой pickup_detector.detect_pickup_on_image
    Возвращает {'state': 'active'|'inactive'|'not visible', 'box': (x,y,w,h)|None, 'scores': {...}}
    """
    _ensure_dir(Cfg.DEBUG_DIR)
    tmp_path = os.path.join(Cfg.DEBUG_DIR, "_screen_pickup.png")
    img = adb.screencap_cv()
    if img is None:
        return {"state": "unknown", "box": None, "scores": {}}
    cv2.imwrite(tmp_path, img)
    res = ext_pickup.detect_pickup_on_image(tmp_path)
    state = res.get("state", "unknown")
    box = None
    if state == "active" and res["active"]["found"]:
        box = tuple(res["active"]["box"])
    elif state == "inactive" and res["inactive"]["found"]:
        box = tuple(res["inactive"]["box"])
    scores = {"active": res["active"]["score"], "inactive": res["inactive"]["score"]}
    log(
        f"[pickup_ext] state={state} act_found={res['active']['found']} act_score={res['active']['score']} inact_found={res['inactive']['found']} inact_score={res['inactive']['score']} box={box}"
    )
    return {"state": state, "box": box, "scores": scores, "raw": res}


def tap_pickup_if_active(adb: ADB) -> bool:
    det = detect_pickup_ext(adb)
    if det["state"] != "active" or not det["box"]:
        return False
    x, y, w, h = det["box"]
    cx, cy = x + w // 2, y + h // 2
    img = adb.screencap_cv()
    if img is not None:
        save_debug(
            img,
            "pickup_active_click",
            rects=[((x, y, w, h), (0, 255, 0), "active_box")],
            points=[((cx, cy), (0, 255, 0), "tap")],
        )
    adb.tap(cx, cy)
    time.sleep(Cfg.WAIT_AFTER_PICKUP)
    return True


def close_yantar_popup(adb: ADB):
    """
    Закрываем строго по заголовку попапа (popup_yantar) или по имени 'yantar'.
    Нет — BACK.
    """
    img = adb.screencap_cv()
    if img is None:
        adb.key_back()
        time.sleep(Cfg.WAIT_AFTER_CLOSE)
        return
    title = find_any(img, Cfg.TPL["popup_yantar"], Cfg.THRESH_NORMAL)
    if title:
        _, d = title
        save_debug(img, "close_popup_title", rects=[(d["rect"], (180, 120, 255), "popup_title")])
        tap_center(adb, d, "popup_title")
        return
    y = find_any(img, Cfg.TPL["yantar"], Cfg.THRESH_NORMAL)
    if y:
        _, d = y
        save_debug(img, "close_popup_by_name", rects=[(d["rect"], (180, 120, 255), "yantar")])
        tap_center(adb, d, "yantar_close")
        return
    log("[popup] fallback BACK")
    adb.key_back()
    time.sleep(Cfg.WAIT_AFTER_CLOSE)


def loot_yantar_until_inactive(adb: ADB, max_clicks: int = 30) -> Dict[str, Any]:
    """
    1) Привести вкладки к стартовому базлайну (MONSTERS=open, ITEMS=closed).
    2) Открыть ITEMS (явно), затем yantar popup (строго по шаблону).
    3) Жать 'подобрать' (по твоему детектору) до неактивной.
    4) Закрыть попап, вернуть базлайн (MONSTERS=open, ITEMS=closed).
    """
    res = {"ok": False, "picked": 0, "reason": ""}
    if not ensure_tabs_state(adb, want_items_open=Cfg.START_ITEMS_OPEN, want_monsters_open=Cfg.START_MONSTERS_OPEN):
        log("[flow] start baseline not stabilized (continuing)")
    img = adb.screencap_cv()
    if img is None:
        res["reason"] = "no_screen"
        return res
    items = detect_items_open(img)
    if not items["open"]:
        hit = find_any(img, Cfg.TPL["items_tab"], Cfg.THRESH_NORMAL)
        if not hit:
            save_debug(img, "items_tab_not_found_open_items", force=True)
            res["reason"] = "items_tab_not_found"
            return res
        _, d = hit
        tap_center(adb, d, "items_tab_open")
        img2 = adb.screencap_cv()
        if img2 is not None:
            items = detect_items_open(img2)
            log(
                f"[items] after open | open={items['open']} via={items['matched']} score={items['score']} rect={items['rect']}"
            )
            img = img2
        if not items["open"]:
            res["reason"] = "items_open_failed"
            return res
    if not open_yantar_popup(adb):
        res["reason"] = "yantar_popup_open_failed"
        ensure_tabs_state(adb, want_items_open=Cfg.START_ITEMS_OPEN, want_monsters_open=Cfg.START_MONSTERS_OPEN)
        return res
    clicks = 0
    none_count = 0
    while clicks < max_clicks:
        det = detect_pickup_ext(adb)
        log(f"[flow] pickup_state={det['state']} box={det['box']} scores={det.get('scores')}")
        img = adb.screencap_cv()
        if img is not None:
            if det["box"]:
                x, y, w, h = det["box"]
                save_debug(
                    img,
                    f"pickup_state_{det['state']}",
                    rects=[((x, y, w, h), (0, 255, 0) if det["state"] == "active" else (0, 165, 255), det["state"])],
                )
            else:
                save_debug(img, f"pickup_state_{det['state']}")
        if det["state"] == "active":
            none_count = 0
            if tap_pickup_if_active(adb):
                clicks += 1
                continue
            else:
                res["reason"] = "active_but_tap_failed"
                break
        elif det["state"] == "inactive":
            close_yantar_popup(adb)
            res["ok"] = True
            res["picked"] = clicks
            res["reason"] = "stopped_on_inactive"
            ensure_tabs_state(adb, want_items_open=Cfg.START_ITEMS_OPEN, want_monsters_open=Cfg.START_MONSTERS_OPEN)
            return res
        else:
            none_count += 1
            if none_count >= 6:
                log("[flow] pickup none too long → giving up")
                break
            time.sleep(0.25)
    close_yantar_popup(adb)
    res["ok"] = True
    res["picked"] = clicks
    res["reason"] = "max_clicks_or_none_timeout"
    ensure_tabs_state(adb, want_items_open=Cfg.START_ITEMS_OPEN, want_monsters_open=Cfg.START_MONSTERS_OPEN)
    return res


def main():
    init_log()
    adb = ADB()
    w, h = adb.get_size()
    log(f"[adb] screen {w}x{h}")
    if w == 0 or h == 0:
        log("[fatal] no screen via ADB")
        return
    res = loot_yantar_until_inactive(adb, max_clicks=30)
    log(f"[result] {res}")
    log("[done]")


if __name__ == "__main__":
    main()
