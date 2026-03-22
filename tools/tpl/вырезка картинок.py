# -*- coding: utf-8 -*-
"""
extract_icons_from_characteristics.py
— Режем иконки из готовых блоков характеристик по списку not_icon.json.

Вход:
  - C:\bot\характеристики2\<имя>_full.png  (исходные блоки)
  - C:\bot\tools\cfg\not_icon.json        (whitelist имён)

Выход:
  - C:\bot\иконки2\<имя>.png              (иконки)
  - C:\bot\иконки2_debug\...              (оверлеи и диагностика)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ======= Пути =======
BOT = Path(r"C:\bot")
SRC_DIR = BOT / "tpl" / "характеристики"
LIST_JSON = BOT / "tools" / "cfg" / "not_icon.json"
OUT_DIR = BOT / "иконки2"
DBG_DIR = BOT / "иконки2_debug"

# ======= Параметры белого фона и геометрии =======
WHITE_S_MAX = 35  # HSV: белый — низкая насыщенность
WHITE_V_MIN = 220  # HSV: белый — высокая яркость
WHITE_BORDER_MIN = 0.88  # расширяем бокс пока >= этой доли белого на границе

MIN_ICON_AREA = 28 * 28  # минимальная площадь компоненты
AR_MIN, AR_MAX = 0.6, 1.6  # допустимое отношение сторон (иконка обычно близка к квадрату)

# Поисковая зона слева от текста (геометрический эвристический ROI, без OCR)
SEARCH_LEFT_MAX_PX = 320  # ширина окна слева от текста (эвристика)
ROI_TOP_FRAC = 0.08  # сверху чуть отступаем (срезать возможные шапки)
ROI_H_FRAC_PRIMARY = 0.55  # первичный ROI по высоте (верхняя половина)
ROI_H_FRAC_FALLBACK = 0.75  # fallback ROI по высоте, если первичный не нашёл


# ======= Юникод-совместимый IO =======
def imread_u(path: Path) -> Optional[np.ndarray]:
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_u(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(".png", img)
    if ok:
        with open(path, "wb") as f:
            enc.tofile(f)


# ======= Утилиты =======
def _hsv_white_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    white = (s <= WHITE_S_MAX) & (v >= WHITE_V_MIN)
    return white.astype(np.uint8)  # 0/1


def _nonwhite_mask(bgr: np.ndarray) -> np.ndarray:
    return (1 - _hsv_white_mask(bgr)).astype(np.uint8)


def _connected_components(mask01: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
    # mask01: 0/1
    m = (mask01 * 255).astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, num):  # 0 — фон
        x, y, w, h, area = stats[i]
        out.append((x, y, w, h, int(area)))
    return out


def _select_icon_blob(
    cands: List[Tuple[int, int, int, int, int]], prefer_right_edge: int
) -> Optional[Tuple[int, int, int, int]]:
    # Выберем крупную, «почти квадратную» и близкую к правому краю ROI (там рядом находится текст)
    scored = []
    for x, y, w, h, area in cands:
        if area < MIN_ICON_AREA:
            continue
        ar = w / max(1, h)
        ar_penalty = 0.0 if AR_MIN <= ar <= AR_MAX else 0.6
        right_dist = abs((x + w) - prefer_right_edge)
        score = area - 220 * ar_penalty - 1.8 * right_dist
        scored.append((score, (x, y, w, h)))
    if not scored:
        return None
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1]


def _expand_to_white_frame(
    roi_bgr: np.ndarray, box_xywh: Tuple[int, int, int, int], max_expand: int = 40
) -> Tuple[int, int, int, int]:
    H, W = roi_bgr.shape[:2]
    x, y, w, h = box_xywh
    white = _hsv_white_mask(roi_bgr)

    def border_ratio(xx, yy, ww, hh) -> float:
        vals = []
        if yy > 0:
            top = white[max(0, yy - 1) : yy, xx : xx + ww]
            vals.append(top.mean() if top.size else 0)
        if yy + hh < H:
            bot = white[yy + hh : min(H, yy + hh + 1), xx : xx + ww]
            vals.append(bot.mean() if bot.size else 0)
        if xx > 0:
            lef = white[yy : yy + hh, max(0, xx - 1) : xx]
            vals.append(lef.mean() if lef.size else 0)
        if xx + ww < W:
            rig = white[yy : yy + hh, xx + ww : min(W, xx + ww + 1)]
            vals.append(rig.mean() if rig.size else 0)
        return sum(vals) / max(1, len(vals))

    best = (x, y, w, h)
    for step in range(1, max_expand + 1):
        xx = max(0, x - 1)
        yy = max(0, y - 1)
        ww = min(W, x + w + 1) - xx
        hh = min(H, y + h + 1) - yy
        if ww <= 0 or hh <= 0:
            break
        if border_ratio(xx, yy, ww, hh) >= WHITE_BORDER_MIN:
            best = (xx, yy, ww, hh)
            x, y, w, h = xx, yy, ww, hh
        else:
            break
    return best


def _pad_to_square(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)  # белый фон
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = img_bgr
    return canvas


def _adaptive_icon_from_block(block_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    H, W = block_bgr.shape[:2]

    def try_roi(top_frac: float, h_frac: float) -> Optional[Tuple[int, int, int, int]]:
        y1 = int(H * top_frac)
        y2 = min(H, int(H * h_frac))
        x2 = min(W, max(64, SEARCH_LEFT_MAX_PX))
        x1 = 0
        if y2 <= y1 or x2 <= x1:
            return None
        roi = block_bgr[y1:y2, x1:x2]
        nonwhite = _nonwhite_mask(roi)
        # сгладим шумы
        nonwhite = cv2.morphologyEx(nonwhite, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        cands = _connected_components(nonwhite)
        if not cands:
            return None
        cand = _select_icon_blob(cands, prefer_right_edge=roi.shape[1])
        if cand is None:
            return None
        cx, cy, cw, ch = cand
        ex, ey, ew, eh = _expand_to_white_frame(roi, (cx, cy, cw, ch), max_expand=40)
        gx1, gy1 = x1 + ex, y1 + ey
        return gx1, gy1, ew, eh

    # 1) Верхняя половина слева
    bx = try_roi(ROI_TOP_FRAC, ROI_H_FRAC_PRIMARY)
    if bx:
        return bx

    # 2) Fallback: побольше по высоте
    bx = try_roi(ROI_TOP_FRAC, ROI_H_FRAC_FALLBACK)
    if bx:
        return bx

    # 3) Анти-залипание: ищем по всей ширине левой половины и по 80% высоты
    y1, y2 = int(H * ROI_TOP_FRAC), min(H, int(H * 0.80))
    x1, x2 = 0, int(W * 0.5)
    roi = block_bgr[y1:y2, x1:x2]
    nonwhite = _nonwhite_mask(roi)
    nonwhite = cv2.morphologyEx(nonwhite, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    cands = _connected_components(nonwhite)
    if not cands:
        return None
    cand = _select_icon_blob(cands, prefer_right_edge=roi.shape[1])
    if cand is None:
        return None
    cx, cy, cw, ch = cand
    ex, ey, ew, eh = _expand_to_white_frame(roi, (cx, cy, cw, ch), max_expand=40)
    gx1, gy1 = x1 + ex, y1 + ey
    return gx1, gy1, ew, eh


# ======= Основная логика =======
def load_whitelist(list_path: Path) -> List[str]:
    with open(list_path, "r", encoding="utf-8") as f:
        arr = json.load(f)
    out = []
    for v in arr:
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
    return out


def process_one_item(name: str) -> bool:
    # Источник: <имя>_full.png в SRC_DIR
    src = SRC_DIR / f"{name}_full.png"
    if not src.exists():
        print(f"[MISS] нет исходника: {src.name}")
        return False
    block = imread_u(src)
    if block is None or block.size == 0:
        print(f"[ERR] не открыть: {src.name}")
        return False

    box = _adaptive_icon_from_block(block)
    overlay = block.copy()
    ok = False

    if box:
        x, y, w, h = box
        x2, y2 = x + w, y + h
        x = max(0, x)
        y = max(0, y)
        x2 = min(block.shape[1], x2)
        y2 = min(block.shape[0], y2)
        if x2 > x and y2 > y:
            icon = block[y:y2, x:x2]
            icon = _pad_to_square(icon)
            out_path = OUT_DIR / f"{name}.png"
            imwrite_u(out_path, icon)
            ok = True
            color = (0, 200, 0)
            label = f"{name} [{w}x{h}]"
            cv2.rectangle(overlay, (x, y), (x2, y2), color, 2)
            cv2.putText(
                overlay,
                label[:48],
                (max(3, x + 3), max(16, y + 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(overlay, "bad box", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(overlay, "icon not found", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    imwrite_u(DBG_DIR / f"{name}__overlay.png", overlay)
    print(f"[{'OK' if ok else 'FAIL'}] {name}")
    return ok


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DBG_DIR.mkdir(parents=True, exist_ok=True)

    names = load_whitelist(LIST_JSON)
    print(f"[INFO] всего по списку not_icon: {len(names)}")

    total_ok = 0
    for nm in names:
        if process_one_item(nm):
            total_ok += 1

    print(f"[DONE] успешно: {total_ok}/{len(names)} → {OUT_DIR}")


if __name__ == "__main__":
    main()
