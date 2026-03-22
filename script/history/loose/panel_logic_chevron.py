import glob
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class CFG:
    TH_TAB = 0.83
    TH_ARROW = 0.82
    TAB_ITEMS = "tpl\\my\\items_tab.png"
    TAB_MONSTER = "tpl\\my\\monsters_tab.png"
    CHV_ITEMS_OPEN = "tpl\\chevrons\\items_open.png"
    CHV_ITEMS_CLOSE = "tpl\\chevrons\\items_close.png"
    CHV_MON_OPEN = "tpl\\chevrons\\monsters_open.png"
    CHV_MON_CLOSE = "tpl\\chevrons\\monsters_close.png"
    ARROW_OPEN_ANY = "tpl\\my\\open_tab.png"
    ARROW_CLOSE_ANY = "tpl\\my\\close_tab.png"
    ROI_DEFAULTS = {"items_tab": [0.0, 0.18, 0.22, 0.12], "monsters_tab": [0.0, 0.3, 0.22, 0.12]}
    OUT_DIR = os.path.join("diag", "panel_chevron")
    LOG_TXT = os.path.join("diag", "panel_chevron.log")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def ts_now():
    return time.strftime("%Y%m%d_%H%M%S")


def imread_u(path: str, flags=cv2.IMREAD_COLOR):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_u(path: str, img, params=None) -> bool:
    try:
        ensure_dir(os.path.dirname(path))
        ext = os.path.splitext(path)[1]
        ok, buf = cv2.imencode(ext, img, params or [])
        if not ok:
            return False
        with open(path, "wb") as f:
            buf.tofile(f)
        return True
    except Exception:
        return False


def log_line(msg: str):
    ensure_dir(os.path.dirname(CFG.LOG_TXT))
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(CFG.LOG_TXT, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def to_px(rel, W, H):
    x, y, w, h = rel
    if all(isinstance(v, (int, np.integer)) for v in rel):
        return [x, y, w, h]
    return [int(x * W), int(y * H), int(w * W), int(h * H)]


def crop_px(img, r):
    x, y, w, h = r
    return img[y : y + h, x : x + w]


def match_one_safe(img, tpl, roi_px, th, method=cv2.TM_CCOEFF_NORMED):
    if img is None or tpl is None:
        return None
    src = img if roi_px is None else crop_px(img, roi_px)
    if src is None or src.size == 0 or src.shape[0] < tpl.shape[0] or src.shape[1] < tpl.shape[1]:
        return None
    res = cv2.matchTemplate(src, tpl, method)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < th:
        return None
    x0, y0 = max_loc
    tw, thh = tpl.shape[1], tpl.shape[0]
    rx, ry = (0, 0) if roi_px is None else (roi_px[0], roi_px[1])
    rect = [rx + x0, ry + y0, tw, thh]
    cx = rect[0] + tw // 2
    cy = rect[1] + thh // 2
    return {"rect": rect, "center": (cx, cy), "score": float(max_val)}


def calibrate_tabs(screen):
    H, W = screen.shape[:2]
    t_items = imread_u(CFG.TAB_ITEMS)
    t_mons = imread_u(CFG.TAB_MONSTER)
    if t_items is None and t_mons is None:
        return {
            "items_tab_px": to_px(CFG.ROI_DEFAULTS["items_tab"], W, H),
            "monsters_tab_px": to_px(CFG.ROI_DEFAULTS["monsters_tab"], W, H),
            "notes": "fallback ROI (both tab templates missing)",
        }
    m_items = match_one_safe(screen, t_items, None, CFG.TH_TAB) if t_items is not None else None
    m_mons = match_one_safe(screen, t_mons, None, CFG.TH_TAB) if t_mons is not None else None
    if m_items:
        x, y, tw, th = m_items["rect"]
        items_tab_px = [max(0, x - 20), max(0, y - 20), tw + 40, th + 40]
    else:
        items_tab_px = to_px(CFG.ROI_DEFAULTS["items_tab"], W, H)
    if m_mons:
        x, y, tw, th = m_mons["rect"]
        monsters_tab_px = [max(0, x - 20), max(0, y - 20), tw + 40, th + 40]
    else:
        monsters_tab_px = to_px(CFG.ROI_DEFAULTS["monsters_tab"], W, H)
    return {
        "items_tab_px": items_tab_px,
        "monsters_tab_px": monsters_tab_px,
        "notes": f"calibrated: items={'OK' if m_items else 'DEF'}, monsters={'OK' if m_mons else 'DEF'}",
    }


def chevron_in_band(screen, chv_tpl, tab_px, band=140):
    if chv_tpl is None or screen is None or tab_px is None:
        return None
    H, W = screen.shape[:2]
    x, y, w, h = tab_px
    cy = y + h // 2
    y0 = max(0, cy - band // 2)
    roi_band = [0, y0, W, min(band, H - y0)]
    return match_one_safe(screen, chv_tpl, roi_band, CFG.TH_ARROW)


def detect_tab_state(screen, which: str, tab_px) -> Dict:
    if which == "items":
        open_big = imread_u(CFG.CHV_ITEMS_OPEN)
        close_big = imread_u(CFG.CHV_ITEMS_CLOSE)
    else:
        open_big = imread_u(CFG.CHV_MON_OPEN)
        close_big = imread_u(CFG.CHV_MON_CLOSE)
    m_open = chevron_in_band(screen, open_big, tab_px)
    m_close = chevron_in_band(screen, close_big, tab_px)
    verdict = None
    best = None
    best_type = None
    if m_open and (not m_close or m_open["score"] >= m_close["score"]):
        verdict = True
        best = m_open
        best_type = "open"
    elif m_close:
        verdict = False
        best = m_close
        best_type = "close"
    if verdict is None:
        open_sm = imread_u(CFG.ARROW_OPEN_ANY)
        close_sm = imread_u(CFG.ARROW_CLOSE_ANY)
        m_open_s = match_one_safe(screen, open_sm, tab_px, CFG.TH_ARROW) if open_sm is not None else None
        m_close_s = match_one_safe(screen, close_sm, tab_px, CFG.TH_ARROW) if close_sm is not None else None
        if m_open_s and (not m_close_s or m_open_s["score"] >= m_close_s["score"]):
            verdict = True
            best = m_open_s
            best_type = "open_small"
        elif m_close_s:
            verdict = False
            best = m_close_s
            best_type = "close_small"
    return {"open": verdict, "best": best, "best_type": best_type}


def draw_box(img, rect, color, text=None):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    if text:
        cv2.putText(img, text, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def annotate(screen, cal, items_res, mons_res):
    vis = screen.copy()
    for name, r in [("items", cal["items_tab_px"]), ("monsters", cal["monsters_tab_px"])]:
        draw_box(vis, r, (80, 80, 80), f"{name}_tab_roi")
    if items_res["best"]:
        color = (
            (50, 220, 50)
            if items_res["open"] is True
            else (40, 40, 255) if items_res["open"] is False else (180, 180, 40)
        )
        draw_box(
            vis, items_res["best"]["rect"], color, f"items:{items_res['best_type']} {items_res['best']['score']:.3f}"
        )
    if mons_res["best"]:
        color = (
            (50, 220, 50)
            if mons_res["open"] is True
            else (40, 40, 255) if mons_res["open"] is False else (180, 180, 40)
        )
        draw_box(vis, mons_res["best"]["rect"], color, f"mons:{mons_res['best_type']} {mons_res['best']['score']:.3f}")
    y0 = 22
    lines = [
        f"items: {items_res['open']} ({items_res['best_type'] or '-'})",
        f"monsters: {mons_res['open']} ({mons_res['best_type'] or '-'})",
        cal["notes"],
    ]
    for ln in lines:
        cv2.putText(vis, ln, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 230, 20), 2, cv2.LINE_AA)
        y0 += 24
    return vis


def collect_inputs(inp: Optional[str]) -> List[str]:
    if not inp:
        files = sorted(glob.glob(os.path.join("screens", "*.png")), key=os.path.getmtime)
        return files[-1:] if files else []
    if os.path.isdir(inp):
        return sorted(glob.glob(os.path.join(inp, "*.png")))
    if any(ch in inp for ch in ["*", "?", "["]):
        return sorted(glob.glob(inp))
    return [inp] if os.path.isfile(inp) else []


def process_one(path: str, tag: str = ""):
    img = imread_u(path, cv2.IMREAD_COLOR)
    if img is None:
        log_line(f"[{os.path.basename(path)}] ERROR: cannot read image")
        return
    cal = calibrate_tabs(img)
    items_res = detect_tab_state(img, "items", cal["items_tab_px"])
    mons_res = detect_tab_state(img, "monsters", cal["monsters_tab_px"])

    def fmt_res(name, r):
        if r["open"] is True:
            st = "OPEN"
        elif r["open"] is False:
            st = "CLOSE"
        else:
            st = "NONE"
        sc = f"{r['best']['score']:.3f}" if r["best"] else "-"
        bt = r["best_type"] or "-"
        return f"{name}={st} score={sc} type={bt}"

    log_line(
        f"[{os.path.basename(path)}] {fmt_res('items', items_res)} | {fmt_res('monsters', mons_res)} | {cal['notes']}"
    )
    vis = annotate(img, cal, items_res, mons_res)
    base = os.path.splitext(os.path.basename(path))[0]
    tagp = tag + "_" if tag else ""
    out_path = os.path.join(CFG.OUT_DIR, f"{tagp}{base}_chev_{ts_now()}.png")
    imwrite_u(out_path, vis)


def main():
    tag = ""
    inp = None
    args = sys.argv[1:]
    if args:
        if "--tag" in args:
            i = args.index("--tag")
            if i + 1 < len(args):
                tag = args[i + 1]
                del args[i : i + 2]
        if args:
            inp = args[0]
    files = collect_inputs(inp)
    if not files:
        log_line(
            "No input images. Put PNGs in 'screens' or pass a file/dir/glob. Example: python panel_logic_chevron.py screens"
        )
        return
    ensure_dir(CFG.OUT_DIR)
    for p in files:
        process_one(p, tag=tag)


if __name__ == "__main__":
    main()
