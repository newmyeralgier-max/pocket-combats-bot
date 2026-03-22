# -*- coding: utf-8 -*-
"""
detect_icons_anywhere_plus.py
— Поиск иконок предметов в безопасной зоне (положение окон НЕ фиксировано).
— Исправляет: гигантские боксы, «грязные» кропы с текстом, низкие CCOEFF на сером.
— Делает: чистую вырезку иконки, цветовую маску «янтаря», грани, CLAHE, маскированный матчинг.
— Полные логи, ROI/маски/кропы/оверлеи, расширенный CSV-аудит.

Вход:
  - C:\bot\back\screens\*.png
  - C:\bot\tools\cfg\config.json (ITEMS_ROI: [x1,y1,x2,y2])
  - C:\bot\tpl\иконки_предметов\*.png  (ищем «янтарь»)

Выход:
  - C:\bot\debug\out_icons_debug\...
"""

import glob
import json
import os
from datetime import datetime

import cv2
import numpy as np

# ==================== Конфиг ====================
with open(r"C:\bot\tools\cfg\config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

ITEMS_X1, SAFE_Y1, ITEMS_X2, SAFE_Y2 = cfg.get("ITEMS_ROI", [0, 0, 0, 0])

# Масштабы шаблона
ICON_SCALES = cfg.get("ICON_SCALES", [0.78, 0.84, 0.88, 0.92, 0.96, 1.00, 1.04, 1.08, 1.12, 1.18])
# Пороги решения (не «жесткая планка» одного метода, а логика принятия)
THR_COLOR = float(cfg.get("THR_COLOR", 0.72))  # TM_CCORR_NORMED с маской по цвету
THR_EDGE = float(cfg.get("THR_EDGE", 0.18))  # TM_CCOEFF_NORMED по Canny
THR_GRAY = float(cfg.get("THR_GRAY", 0.26))  # TM_CCOEFF_NORMED по CLAHE
SHIFT_BATTLE = bool(cfg.get("SHIFT_BATTLE", False))
PREPROC_MODE = str(cfg.get("PREPROC_MODE", "clahe"))  # gray|clahe|sobel|canny

# ==================== Пути ======================
screens_dir = r"C:\bot\back\screens"
tpl_root = r"C:\bot\tpl"
tpl_group = "иконки_предметов"
out_dir = r"C:\bot\debug\out_icons_debug"
os.makedirs(out_dir, exist_ok=True)

roi_dir = os.path.join(out_dir, "roi")
masks_dir = os.path.join(out_dir, "masks")
audit_dir = os.path.join(out_dir, "audit_boxes")
overlays_dir = os.path.join(out_dir, "overlays")
for d in (roi_dir, masks_dir, audit_dir, overlays_dir):
    os.makedirs(d, exist_ok=True)

audit_csv = os.path.join(out_dir, "audit_scores.csv")
if not os.path.exists(audit_csv):
    with open(audit_csv, "w", encoding="utf-8") as fcsv:
        fcsv.write("screen,box_idx,w,h,method,score\n")

# ==================== Цвета/метрики =============
GROUP_COLOR = (0, 255, 255)
HIT_COLOR = (50, 220, 50)
MISS_COLOR = (0, 0, 255)
# -*- coding: utf-8 -*-
import csv
import io
import os
import time
from datetime import datetime


def log_failures(csv_path: str, screen_name: str, idx: int, reasons: list):
    """
    Логирует причины отказа в CSV.
    reasons — список словарей от match_multi с ключами:
      scale, tpl_wh, reason, (опц.) scores
    """
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        if not file_exists:
            writer.writerow(
                ["screen", "roi_idx", "status", "reason", "scale", "tpl_w", "tpl_h", "color", "gray", "edge"]
            )
        for r in reasons:
            scores = r.get("scores", ("", "", ""))
            writer.writerow(
                [
                    screen_name,
                    idx,
                    "FAIL",
                    r["reason"],
                    r["scale"],
                    r["tpl_wh"][0],
                    r["tpl_wh"][1],
                    scores[0],
                    scores[1],
                    scores[2],
                ]
            )


AUDIT_CSV_PATH = os.path.abspath(os.path.join("audit", "audit_scores.csv"))


def _ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def open_audit_csv(path=AUDIT_CSV_PATH):
    _ensure_dir(path)
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    f = open(path, "a", encoding="utf-8-sig", newline="")
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["screen", "box_idx", "w", "h", "method", "score"])
    print(f"[AUDIT] CSV: {path}")
    return f, writer


def write_audit_row(writer, screen, box_idx, w, h, method, score):
    try:
        writer.writerow([screen, box_idx, int(w), int(h), method, float(score)])
        # Важный трейc: всегда логируем сам факт записи строки
        print(f"[CSV] screen={screen} box={box_idx} {w}x{h} {method}={score:.6f}")
    except Exception as e:
        print(f"[ERR] CSV write failed: {e}")


def process_boxes(screen_name, boxes, matcher):
    """
    boxes: iterable of dicts {idx, x, y, w, h, crop}
    matcher: объект/функции, возвращающие скоры по методам
    """
    f, writer = open_audit_csv()
    total_rows = 0
    try:
        print(f"[DETECT] Контуров найдено: {len(boxes)}")

        for b in boxes:
            idx, w, h = b["idx"], b["w"], b["h"]
            print(f"[BOX] idx={idx} size={w}x{h}")

            # Фильтры логируем всегда
            aspect = w / max(h, 1.0)
            area = w * h
            reasons = []
            if w < 20 or h < 20:
                reasons.append("small")
            if aspect < 0.6 or aspect > 1.7:
                reasons.append(f"aspect={aspect:.2f}")
            if area < 30 * 30:
                reasons.append(f"area={area}")
            if reasons:
                print(f"[FILTER] idx={idx} -> skip: {', '.join(reasons)}")
                # ВАЖНО: даже отброшенные — всё равно матчим и пишем скоры для аудита
            try:
                scores = matcher.compute_all(b["crop"])
                # ожидаем dict вида {"color": v1, "gray": v2, "edge": v3}
                for method, score in scores.items():
                    write_audit_row(writer, screen_name, idx, w, h, method, score)
                    total_rows += 1
                print(f"[SCORE] idx={idx} -> " + " ".join(f"{m}={v:.4f}" for m, v in scores.items()))
            except Exception as e:
                print(f"[ERR] MATCH idx={idx}: {e}")
                # фиксируем нули, чтобы видеть частоту сбоев
                for method in ("color", "gray", "edge"):
                    write_audit_row(writer, screen_name, idx, w, h, method, 0.0)
                    total_rows += 1
    finally:
        try:
            f.flush()
            f.close()
        except Exception:
            pass
        print(f"[AUDIT] rows_written={total_rows} -> {AUDIT_CSV_PATH}")


# ==================== Лог =======================
log_path = os.path.join(out_dir, "match_log.txt")


def log(msg: str):
    stamp = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(stamp)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(stamp + "\n")


# ==================== IO c Unicode ==============
def imread_u(path: str, flags=cv2.IMREAD_COLOR):
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def imwrite_u(path: str, img: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, enc = cv2.imencode(".png", img)
    if ok:
        enc.tofile(path)


# ==================== Загрузка шаблона ==========
def load_amber_icon():
    group_dir = os.path.join(tpl_root, tpl_group)
    cand_path, cand_name = None, None
    for ext in ("*.png", "*.PNG"):
        for path in glob.glob(os.path.join(group_dir, ext)):
            name = os.path.basename(path)
            if "янтар" in name.lower():
                cand_path, cand_name = path, name
                break
        if cand_path:
            break
    if not cand_path:
        log("[ERR] Шаблон янтаря не найден!")
        return None, None, None, None

    tpl_bgr = imread_u(cand_path, cv2.IMREAD_COLOR)
    if tpl_bgr is None:
        log("[ERR] Не удалось загрузить шаблон янтаря!")
        return None, None, None, None

    tpl_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
    # HSV маска «янтаря»: оранжево-жёлтые тона с хорошей насыщенностью
    tpl_hsv = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(tpl_hsv)
    # Диапазон для «янтаря» (можно подшлифовать при необходимости)
    # H: 10..30 (из 0..179), S>=70, V>=60
    tpl_mask_color = cv2.inRange(tpl_hsv, (10, 70, 60), (30, 255, 255))

    # Если маска пустая, ослабим S/V
    if cv2.countNonZero(tpl_mask_color) == 0:
        tpl_mask_color = cv2.inRange(tpl_hsv, (8, 50, 40), (35, 255, 255))

    # Гладим маску шаблона
    tpl_mask_color = cv2.morphologyEx(tpl_mask_color, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    tpl_mask_color = cv2.morphologyEx(tpl_mask_color, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    log(f"[TPL] Загружен шаблон янтаря: {cand_name}, размер={tpl_gray.shape}")
    imwrite_u(os.path.join(out_dir, "tpl_amber.png"), tpl_bgr)
    imwrite_u(os.path.join(out_dir, "tpl_amber_mask.png"), tpl_mask_color)
    return tpl_bgr, tpl_gray, tpl_mask_color, cand_name


# ==================== Препроцесс =================
def preprocess_gray(gray: np.ndarray, mode: str) -> np.ndarray:
    if mode == "gray":
        return gray
    elif mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        return clahe.apply(gray)
    elif mode == "sobel":
        g = gray
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.convertScaleAbs(cv2.addWeighted(gx, 0.5, gy, 0.5, 0))
    elif mode == "canny":
        g = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.Canny(g, 40, 120)
    else:
        return gray


def canny_edges(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    e = cv2.Canny(g, 40, 120)
    return e


# ==================== NMS =======================
def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, bx + bh)
    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / float(aw * ah + bw * bh - inter)


def nms(boxes, iou_thr=0.35):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    keep = []
    for b in boxes:
        if all(iou(b, k) < iou_thr for k in keep):
            keep.append(b)
    return keep


# ==================== Маски кандидатов ==========
def build_candidate_masks(roi_bgr: np.ndarray, screen_name: str):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    masks = []

    # Низкая насыщенность (фон карточек/рамок)
    m1 = cv2.inRange(hsv, (0, 0, 70), (180, 85, 255))
    m1 = cv2.medianBlur(m1, 3)
    m1 = cv2.morphologyEx(m1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    m1 = cv2.morphologyEx(m1, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    masks.append(("m1_neutral", m1))

    # Чуть темнее фон
    m2 = cv2.inRange(hsv, (0, 0, 40), (180, 110, 230))
    m2 = cv2.medianBlur(m2, 3)
    m2 = cv2.morphologyEx(m2, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    m2 = cv2.morphologyEx(m2, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    masks.append(("m2_darker", m2))

    # Адаптивная бинаризация
    g3 = cv2.GaussianBlur(gray, (3, 3), 0)
    m3 = cv2.adaptiveThreshold(g3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
    m3 = cv2.medianBlur(m3, 3)
    masks.append(("m3_adapt", m3))

    # Края (иконки имеют плотные границы)
    e = cv2.Canny(gray, 40, 120)
    m4 = cv2.dilate(e, np.ones((3, 3), np.uint8), iterations=1)
    masks.append(("m4_edges", m4))

    # Сохраняем маски (для аудита)
    for tag, m in masks:
        imwrite_u(os.path.join(masks_dir, f"{tag}_{screen_name}"), m)

    return masks


# ==================== Фильтр боксов =============
def filter_boxes(contours, roi_shape):
    H, W = roi_shape[:2]
    roi_area = max(1, H * W)
    out = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area <= 0:
            continue

        # Масштабонезависимые критерии
        ar = w / float(max(1, h))
        if ar < 0.5 or ar > 1.8:
            continue

        frac = area / roi_area
        # Уберём явно мусорные крошки и явно гигантские области
        if frac < 0.0002 or frac > 0.30:
            continue

        out.append((x, y, w, h))

    out = nms(out, iou_thr=0.35)
    out = sorted(out, key=lambda b: (b[1], b[0]))
    return out[:220]


# ==================== Детектор боксов ===========


def detect_icon_boxes(roi_bgr: np.ndarray, screen_name: str):
    masks = build_candidate_masks(roi_bgr, screen_name)
    boxes = []
    for tag, mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes.extend(filter_boxes(contours, roi_bgr.shape))

    if not boxes:
        # fallback: чуть шире диапазон S/V
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        fallback = cv2.inRange(hsv, (0, 0, 55), (180, 120, 255))
        contours, _ = cv2.findContours(fallback, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes.extend(filter_boxes(contours, roi_bgr.shape))
        imwrite_u(os.path.join(masks_dir, f"m_fallback_{screen_name}"), fallback)

    boxes = nms(boxes, iou_thr=0.35)
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    log(f"[DETECT] Контуров найдено: {len(boxes)}")
    return boxes


# ==================== Обрезка до иконки =========
def crop_icon_only(roi_bgr: np.ndarray, bx: int, by: int, bw: int, bh: int, screen_name: str, idx: int):
    """
    Возвращает квадратный кроп с иконкой (без текста), используя:
      1) верхнюю часть вытянутых боксов,
      2) цветовую маску «янтаря» для поиска blob,
      3) доведение до квадрата с границами.
    """
    H, W = roi_bgr.shape[:2]
    # 1) Если бокс вытянут по вертикали — отрезаем низ (текст)
    if bh > bw * 1.3:
        bh = int(bw * 1.05)

    bx = max(0, bx)
    by = max(0, by)
    bw = min(bw, W - bx)
    bh = min(bh, H - by)
    crop = roi_bgr[by : by + bh, bx : bx + bw].copy()

    # 2) Попробуем найти цветной blob «янтаря»
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Янтарь: H 10..30 (0..179), S>=70, V>=60
    mask1 = cv2.inRange(hsv, (10, 70, 60), (30, 255, 255))
    if cv2.countNonZero(mask1) == 0:
        mask1 = cv2.inRange(hsv, (8, 50, 40), (35, 255, 255))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    imwrite_u(os.path.join(masks_dir, f"color_{screen_name}_box{idx}.png"), mask1)

    x1, y1, w1, h1 = 0, 0, bw, bh
    if cv2.countNonZero(mask1) > 0:
        contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # выберем самый крупный цветной blob
        best = None
        best_area = 0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area > best_area:
                best_area = area
                best = (x, y, w, h)
        if best:
            x1, y1, w1, h1 = best

    # 3) Доведём до квадрата и вернём в координаты ROI
    side = max(w1, h1)
    cx = x1 + w1 // 2
    cy = y1 + h1 // 2
    half = side // 2
    sx1 = max(0, cx - half)
    sy1 = max(0, cy - half)
    sx2 = min(crop.shape[1], sx1 + side)
    sy2 = min(crop.shape[0], sy1 + side)
    # если уехали, подравняем
    sx1 = max(0, sx2 - side)
    sy1 = max(0, sy2 - side)

    clean = crop[sy1:sy2, sx1:sx2].copy()
    if clean.size == 0:
        # fallback: верхний квадрат из исходного бокса
        side = min(bw, bh)
        clean = crop[0:side, 0:side].copy()

    # сохраним «чистый» кроп для аудита
    imwrite_u(os.path.join(audit_dir, f"{screen_name}_box{idx}_clean.png"), clean)
    return clean


# ==================== Матчинг ====================
def match_multi(crop_bgr, tpl_bgr, tpl_gray, tpl_mask_color, screen_name, idx):
    """
    Многошкальный матчинг с ап/даунскейлом и полным аудитом:
      - COLOR (HSV) с маской: TM_CCORR_NORMED
      - EDGE (Canny) по CCOEFF_NORMED
      - GRAY (CLAHE/другой PREPROC_MODE) по CCOEFF_NORMED
    Возвращает (metrics, fail_reasons) или (None, fail_reasons).
    metrics содержит best_scale, tpl_w, tpl_h, upscale_k, downscale_k.
    fail_reasons — список словарей с деталями по каждому масштабу.
    """
    THRESH_COLOR, THRESH_GRAY, THRESH_EDGE = 0.6, 0.4, 0.3
    th, tw = tpl_gray.shape[:2]
    ch0, cw0 = crop_bgr.shape[:2]

    upscale_k = 0.0  # глобальный апскейл кропа
    # Минимальный размер шаблона среди заданных масштабов
    tmin_w = max(8, min(int(round(tw * s)) for s in ICON_SCALES))
    tmin_h = max(8, min(int(round(th * s)) for s in ICON_SCALES))

    # Апскейл кропа, если он меньше минимального шаблона
    if cw0 < tmin_w or ch0 < tmin_h:
        kx = (tmin_w + 2) / max(1, cw0)
        ky = (tmin_h + 2) / max(1, ch0)
        upscale_k = min(3.0, max(kx, ky))
        new_w = int(round(cw0 * upscale_k))
        new_h = int(round(ch0 * upscale_k))
        crop_bgr = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        log(f"[UPSCALE] {screen_name} box={idx} {cw0}x{ch0} -> {new_w}x{new_h} (k={upscale_k:.2f})")

    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    crop_proc = preprocess_gray(crop_gray, PREPROC_MODE)
    crop_edge = canny_edges(crop_gray)

    per_scale = []
    fail_reasons = []
    ch, cw = crop_bgr.shape[:2]

    for scale in ICON_SCALES:
        t_w = max(8, int(round(tw * scale)))
        t_h = max(8, int(round(th * scale)))
        local_downscale_k = 0.0

        # Даунскейл шаблона, если он больше кропа
        if t_w > cw or t_h > ch:
            kx = cw / t_w
            ky = ch / t_h
            local_downscale_k = min(kx, ky)
            t_w = max(8, int(round(t_w * local_downscale_k)))
            t_h = max(8, int(round(t_h * local_downscale_k)))
            log(f"[DOWNSCALE] {screen_name} box={idx} scale={scale:.2f} tpl->{t_w}x{t_h} (k={local_downscale_k:.2f})")

        tplg = cv2.resize(tpl_gray, (t_w, t_h), interpolation=cv2.INTER_LINEAR)
        tplc = cv2.resize(tpl_bgr, (t_w, t_h), interpolation=cv2.INTER_LINEAR)
        tmask = cv2.resize(tpl_mask_color, (t_w, t_h), interpolation=cv2.INTER_NEAREST)

        if cv2.countNonZero(tmask) == 0:
            fail_reasons.append(
                {
                    "scale": scale,
                    "tpl_wh": (t_w, t_h),
                    "reason": "mask_empty",
                    "upscale_k": upscale_k,
                    "downscale_k": local_downscale_k,
                }
            )
            continue

        # COLOR
        crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        tpl_hsv = cv2.cvtColor(tplc, cv2.COLOR_BGR2HSV)
        res_col = cv2.matchTemplate(crop_hsv[:, :, 2], tpl_hsv[:, :, 2], cv2.TM_CCORR_NORMED, mask=tmask)
        _, score_col, _, _ = cv2.minMaxLoc(res_col)

        # EDGE
        tpl_edge = canny_edges(tplg)
        res_edge = cv2.matchTemplate(crop_edge, tpl_edge, cv2.TM_CCOEFF_NORMED)
        _, score_edge, _, _ = cv2.minMaxLoc(res_edge)

        # GRAY
        tplp = preprocess_gray(tplg, PREPROC_MODE)
        res_gray = cv2.matchTemplate(crop_proc, tplp, cv2.TM_CCOEFF_NORMED)
        _, score_gray, _, _ = cv2.minMaxLoc(res_gray)

        if score_col < THRESH_COLOR and score_gray < THRESH_GRAY and score_edge < THRESH_EDGE:
            fail_reasons.append(
                {
                    "scale": scale,
                    "tpl_wh": (t_w, t_h),
                    "reason": "low_scores",
                    "scores": (score_col, score_gray, score_edge),
                    "upscale_k": upscale_k,
                    "downscale_k": local_downscale_k,
                }
            )
            continue

        combined = 0.60 * score_col + 0.25 * score_gray + 0.15 * score_edge
        per_scale.append((combined, score_col, score_gray, score_edge, scale, (t_w, t_h), upscale_k, local_downscale_k))

    if not per_scale:
        return None, fail_reasons

    per_scale.sort(key=lambda t: -t[0])
    best = per_scale[0]
    metrics = {
        "combined": best[0],
        "color": best[1],
        "gray": best[2],
        "edge": best[3],
        "scale": best[4],
        "tpl_wh": best[5],
        "upscale_k": best[6],
        "downscale_k": best[7],
    }
    return metrics, fail_reasons


def decision_accept(m):
    """
    Правило принятия:
      - сильный цвет+маска И не совсем пустые грани/gray,
      - либо хороший gray при нормальном edge,
      - либо очень высокий color сам по себе.
    """
    if m is None:
        return False
    color = m["color"]
    gray = m["gray"]
    edge = m["edge"]
    comb = m["combined"]

    # Основной путь: цвет уверенный, остальные «живые»
    if color >= THR_COLOR and (gray >= 0.18 or edge >= THR_EDGE):
        return True

    # Альтернатива: CLAHE-совпадение достаточное и грани не «в ноль»
    if gray >= (THR_GRAY + 0.04) and edge >= 0.10:
        return True

    # Случай: очень высокий цвет сам по себе
    if color >= (THR_COLOR + 0.08):
        return True

    # Комбинированный высок
    if comb >= 0.62 and color >= 0.60:
        return True

    return False


# ==================== Аудит CSV ==================
def write_audit_scores(screen_name: str, idx: int, bw: int, bh: int, metrics, per_scale):

    rows = []
    if metrics is None:
        rows.append(("NO_MATCH", 0.0))
    else:
        rows.append(("COLOR_CCORR_MASKED", float(metrics["color"])))
        rows.append(("GRAY_CCOEFF", float(metrics["gray"])))
        rows.append(("EDGE_CCOEFF", float(metrics["edge"])))
        rows.append(("COMBINED", float(metrics["combined"])))

    # Топ-3 по комбинированному скору
    topk = sorted(per_scale, key=lambda t: -t[0])[:3] if per_scale else []
    # per_scale item: (combined, score_col, score_gray, score_edge, scale, (t_w, t_h))
    for comb, scol, sgray, sedge, sc, (tw, th) in topk:
        rows.append((f"SCALE_COMBINED@{sc:.2f}", float(comb)))
        rows.append((f"SCALE_COLOR@{sc:.2f}", float(scol)))
        rows.append((f"SCALE_GRAY@{sc:.2f}", float(sgray)))
        rows.append((f"SCALE_EDGE@{sc:.2f}", float(sedge)))

    with open(audit_csv, "a", encoding="utf-8") as fcsv:
        for method_name, score in rows:
            fcsv.write(f"{screen_name},{idx},{bw},{bh},{method_name},{score:.4f}\n")


# ==================== Основной прогон ============
def main():
    # --- Перезапуск лога ---
    if os.path.exists(log_path):
        try:
            os.remove(log_path)
        except Exception:
            pass

    tpl_bgr, tpl_gray, tpl_mask_color, amber_name = load_amber_icon()
    if tpl_bgr is None:
        return

    shift_y = 150 if SHIFT_BATTLE else 0

    # Счётчики для статистики
    total_boxes = 0
    total_found = 0
    failed_boxes = 0

    fail_stats_scale = {}
    fail_stats_box = {}
    total_scale_fail_events = 0

    # Для объединённого CSV
    combined_rows = []

    for screen_path in glob.glob(os.path.join(screens_dir, "*.png")):
        screen_name = os.path.basename(screen_path)
        log(f"[SCR] Обработка {screen_name}")

        img = imread_u(screen_path, cv2.IMREAD_COLOR)
        if img is None:
            log(f"[WARN] Не удалось загрузить {screen_path}")
            continue

        y1 = max(0, SAFE_Y1 + shift_y)
        y2 = min(img.shape[0], SAFE_Y2 + shift_y)
        x1 = max(0, ITEMS_X1)
        x2 = min(img.shape[1], ITEMS_X2)
        if y2 <= y1 or x2 <= x1:
            log(f"[ERR] Некорректный ROI из config.json: ({x1},{y1})-({x2},{y2})")
            continue

        roi_bgr = img[y1:y2, x1:x2].copy()
        imwrite_u(os.path.join(roi_dir, f"roi_{screen_name}"), roi_bgr)
        log(f"[ROI] Размер ROI: {roi_bgr.shape}")

        boxes = detect_icon_boxes(roi_bgr, screen_name)
        overlay = img.copy()
        found = 0

        for idx, (bx, by, bw, bh) in enumerate(boxes, start=1):
            total_boxes += 1

            raw = roi_bgr[by : by + bh, bx : bx + bw].copy()
            imwrite_u(os.path.join(audit_dir, f"{screen_name}_box{idx}_raw.png"), raw)

            clean = crop_icon_only(roi_bgr, bx, by, bw, bh, screen_name, idx)
            if clean is None or clean.size == 0:
                log(f"[MATCH] box={idx} пустой чистый кроп")
                write_audit_scores(screen_name, idx, bw, bh, None, None)
                failed_boxes += 1
                fail_stats_box["empty_crop"] = fail_stats_box.get("empty_crop", 0) + 1
                combined_rows.append(
                    {
                        "screen": screen_name,
                        "box_idx": idx,
                        "w": bw,
                        "h": bh,
                        "color": "",
                        "gray": "",
                        "edge": "",
                        "combined": "",
                        "accepted": 0,
                        "box_reason": "empty_crop",
                        "scale_reason": "",
                        "best_scale": "",
                        "tpl_w": "",
                        "tpl_h": "",
                        "upscale_k": "",
                        "downscale_k": "",
                    }
                )
                abs_pt = (x1 + bx, y1 + by)
                cv2.rectangle(overlay, abs_pt, (abs_pt[0] + bw, abs_pt[1] + bh), MISS_COLOR, 1)
                cv2.putText(overlay, "EMPTY", (abs_pt[0], max(0, abs_pt[1] - 3))),
                continue

            ch, cw = clean.shape[:2]
            metrics, fail_reasons = match_multi(clean, tpl_bgr, tpl_gray, tpl_mask_color, screen_name, idx)
            accepted = decision_accept(metrics)

            if metrics is None:
                failed_boxes += 1
                reasons_list = fail_reasons or []
                for r in reasons_list:
                    reason = r.get("reason", "unknown")
                    fail_stats_scale[reason] = fail_stats_scale.get(reason, 0) + 1
                    total_scale_fail_events += 1
                unique_reasons = {r.get("reason", "unknown") for r in reasons_list}
                for reason in unique_reasons:
                    fail_stats_box[reason] = fail_stats_box.get(reason, 0) + 1
                log(f"[MATCH] box={idx} отказ — {', '.join(r['reason'] for r in reasons_list)}")
                write_audit_scores(screen_name, idx, cw, ch, None, None)
                combined_rows.append(
                    {
                        "screen": screen_name,
                        "box_idx": idx,
                        "w": cw,
                        "h": ch,
                        "color": "",
                        "gray": "",
                        "edge": "",
                        "combined": "",
                        "accepted": 0,
                        "box_reason": ",".join(unique_reasons),
                        "scale_reason": json.dumps(reasons_list, ensure_ascii=False),
                        "best_scale": "",
                        "tpl_w": "",
                        "tpl_h": "",
                        "upscale_k": "",
                        "downscale_k": "",
                    }
                )
            else:
                log(
                    f"[MATCH] box={idx} scale={metrics['scale']:.2f} "
                    f"color={metrics['color']:.3f} gray={metrics['gray']:.3f} "
                    f"edge={metrics['edge']:.3f} comb={metrics['combined']:.3f} "
                    f"{'ACCEPT' if accepted else 'REJECT'}"
                )
                write_audit_scores(screen_name, idx, cw, ch, metrics, fail_reasons)
                combined_rows.append(
                    {
                        "screen": screen_name,
                        "box_idx": idx,
                        "w": cw,
                        "h": ch,
                        "color": metrics["color"],
                        "gray": metrics["gray"],
                        "edge": metrics["edge"],
                        "combined": metrics["combined"],
                        "accepted": int(accepted),
                        "box_reason": "" if accepted else "rejected_by_threshold",
                        "scale_reason": json.dumps(fail_reasons, ensure_ascii=False) if fail_reasons else "",
                        "best_scale": metrics.get("scale", ""),
                        "tpl_w": metrics.get("tpl_wh", ("", ""))[0],
                        "tpl_h": metrics.get("tpl_wh", ("", ""))[1],
                        "upscale_k": metrics.get("upscale_k", ""),
                        "downscale_k": metrics.get("downscale_k", ""),
                    }
                )

            abs_pt = (x1 + bx, y1 + by)
            color = HIT_COLOR if accepted else MISS_COLOR
            cv2.rectangle(overlay, abs_pt, (abs_pt[0] + bw, abs_pt[1] + bh), color, 2)
            label_parts = [amber_name]
            if metrics:
                label_parts.append(f"C{metrics['color']:.2f}")
                label_parts.append(f"G{metrics['gray']:.2f}")
                label_parts.append(f"E{metrics['edge']:.2f}")
            label = " ".join(label_parts)
            cv2.putText(
                overlay,
                label,
                (abs_pt[0], max(0, abs_pt[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                color,
                1,
                cv2.LINE_AA,
            )
            if accepted:
                found += 1
                total_found += 1

        imwrite_u(os.path.join(overlays_dir, f"overlay_{screen_name}"), overlay)
        imwrite_u(os.path.join(out_dir, screen_name), overlay)
        log(f"[RES] {screen_name} — всего найдено: {found}")

    # --- Итоговая статистика ---
    log("[SUMMARY] Итоговая статистика:")
    if total_boxes > 0:
        success_pct = (total_found / total_boxes) * 100
        fail_total = total_boxes - total_found
        fail_pct = (fail_total / total_boxes) * 100
        log(f"Всего боксов: {total_boxes}")
        log(f"Успешных матчей: {total_found} ({success_pct:.1f}%)")
        log(f"Отказов: {fail_total} ({fail_pct:.1f}%)")
        if failed_boxes > 0 and fail_stats_box:
            log("[SUMMARY] Причины отказов (по боксам):")
            for reason, count in sorted(fail_stats_box.items(), key=lambda x: -x[1]):
                pct = (count / failed_boxes) * 100
                log(f"  {reason}: {count} ({pct:.1f}%)")
        if total_scale_fail_events > 0 and fail_stats_scale:
            log("[SUMMARY] Причины отказов (по масштабам):")
            for reason, count in sorted(fail_stats_scale.items(), key=lambda x: -x[1]):
                pct = (count / total_scale_fail_events) * 100
                log(f"  {reason}: {count} ({pct:.1f}%)")
    else:
        log("Боксы не найдены ни на одном скрине.")

    # --- Экспорт объединённого CSV ---
    import csv

    combined_csv_path = os.path.join(out_dir, "combined_audit_summary.csv")
    with open(combined_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "screen",
            "box_idx",
            "w",
            "h",
            "color",
            "gray",
            "edge",
            "combined",
            "accepted",
            "box_reason",
            "scale_reason",
            "best_scale",
            "tpl_w",
            "tpl_h",
            "upscale_k",
            "downscale_k",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in combined_rows:
            writer.writerow(row)

    log(f"[SUMMARY] Объединённый CSV сохранён: {combined_csv_path}")
    log("[DONE] Обработка завершена")


if __name__ == "__main__":
    main()
