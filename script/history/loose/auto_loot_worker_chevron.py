import glob
import os
import time

import cv2
import numpy as np


class CFG:
    TH_TAB = 0.83
    TH_ARROW = 0.82
    TH_PICK = 0.84
    TAB_ITEMS = "tpl\\my\\items_tab.png"
    CHV_ITEMS_OPEN = "tpl\\chevrons\\items_open.png"
    CHV_ITEMS_CLOSE = "tpl\\chevrons\\items_close.png"
    ARROW_OPEN_ANY = "tpl\\my\\open_tab.png"
    ARROW_CLOSE_ANY = "tpl\\my\\close_tab.png"
    PICKUP_ANY = "tpl\\my\\pickup.png"
    PICKUP_OWN = "tpl\\my\\pickup_own.png"
    PICKUP_OTHER = "tpl\\my\\pickup_other.png"
    ROI_DEFAULTS = {"items_tab": [0.0, 0.18, 0.22, 0.12], "loot_hint": [0.22, 0.2, 0.5, 0.68]}
    OUT_LOG = "diag\\loot_run_chevron.txt"
    OUT_IMGDIR = "diag"


def imread_u(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except:
        return None


def save_png(img, path):
    try:
        cv2.imwrite(path, img)
    except:
        pass


def log_line(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(CFG.OUT_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except:
        pass


def to_px(rel, W, H):
    x, y, w, h = rel
    return [int(x * W), int(y * H), int(w * W), int(h * H)]


def crop_px(img, r):
    x, y, w, h = r
    return img[y : y + h, x : x + w]


def match_one_safe(img, tpl, roi, th, method=cv2.TM_CCOEFF_NORMED):
    if img is None or tpl is None:
        return None
    src = img if roi is None else crop_px(img, roi)
    if src.shape[0] < tpl.shape[0] or src.shape[1] < tpl.shape[1]:
        return None
    res = cv2.matchTemplate(src, tpl, method)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < th:
        return None
    x0, y0 = max_loc
    tw, thh = tpl.shape[1], tpl.shape[0]
    rx, ry = (0, 0) if roi is None else (roi[0], roi[1])
    rect = [rx + x0, ry + y0, tw, thh]
    return {"rect": rect, "score": float(max_val)}


def detect_items_state(img, tpl_s_o, tpl_s_c, tab_px):
    m_so = match_one_safe(img, tpl_s_o, tab_px, CFG.TH_ARROW) if tpl_s_o is not None else None
    m_sc = match_one_safe(img, tpl_s_c, tab_px, CFG.TH_ARROW) if tpl_s_c is not None else None
    if m_so:
        verdict, best, label = False, m_so, "OPEN_SMALL"
    elif m_sc:
        verdict, best, label = False, m_sc, "CLOSE_SMALL"
    else:
        verdict, best, label = True, None, "NO_SMALL"
    return verdict, best, label


def find_tab_roi(img):
    tpl = imread_u(CFG.TAB_ITEMS)
    m = match_one_safe(img, tpl, None, CFG.TH_TAB)
    if m:
        x, y, w, h = m["rect"]
        roi = [max(0, x - 20), max(0, y - 20), w + 40, h + 40]
        log_line(f"[calib] items_tab matched score={m['score']:.3f}")
        return roi
    H, W = img.shape[:2]
    roi = to_px(CFG.ROI_DEFAULTS["items_tab"], W, H)
    log_line("[calib] fallback ROI for items_tab")
    return roi


def find_pickup(img):
    picks = [CFG.PICKUP_OWN, CFG.PICKUP_OTHER, CFG.PICKUP_ANY]
    for p in picks:
        tpl = imread_u(p)
        m = match_one_safe(img, tpl, None, CFG.TH_PICK)
        if m:
            x, y, w, h = m["rect"]
            log_line(f"[pickup] found: {os.path.basename(p)} score={m['score']:.3f}")
            return m
    log_line("[pickup] no active pickup found")
    return None


def main():
    files = sorted(glob.glob("screens/*.png"), key=os.path.getmtime)
    if not files:
        log_line("no PNGs in screens/")
        return
    img = imread_u(files[-1])
    if img is None:
        log_line("cannot read image")
        return
    tab_roi = find_tab_roi(img)
    tab_state = detect_items_state(img, tab_roi)
    if not tab_state:
        log_line("items tab is closed — abort")
        return
    pick = find_pickup(img)
    if pick:
        x, y, w, h = pick["rect"]
        color = 50, 220, 50
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        txt = f"pickup {pick['score']:.3f}"
        cv2.putText(img, txt, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        txt = "no pickup found"
        cv2.putText(img, txt, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(files[-1]))[0]
    out_img = os.path.join(CFG.OUT_IMGDIR, f"{base}_chevron_{ts}.png")
    save_png(img, out_img)
    log_line(f"[output] saved annotated image to {out_img}")


if __name__ == "__main__":
    main()
