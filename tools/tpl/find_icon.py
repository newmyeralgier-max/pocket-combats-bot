# -*- coding: utf-8 -*-
r"""
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
import csv
from math import sqrt
import argparse

# ==================== Конфиг ====================
with open(r"C:\bot\tools\cfg\config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)
# Жёсткий минимум для иконки
MIN_W, MIN_H = 45, 50

ITEMS_X1, SAFE_Y1, ITEMS_X2, SAFE_Y2 = cfg.get("ITEMS_ROI", [0, 0, 0, 0])

# Масштабы шаблона
ICON_SCALES = cfg.get("ICON_SCALES", [0.78, 0.84, 0.88, 0.92, 0.96, 1.00, 1.04, 1.08, 1.12, 1.18])
# Пороги решения (не «жесткая планка» одного метода, а логика принятия)
THR_COLOR = float(cfg.get("THR_COLOR", 0.72))  # TM_CCORR_NORMED с маской по цвету
THR_EDGE = float(cfg.get("THR_EDGE", 0.18))  # TM_CCOEFF_NORMED по Canny
THR_GRAY = float(cfg.get("THR_GRAY", 0.26))  # TM_CCOEFF_NORMED по CLAHE
SHIFT_BATTLE = bool(cfg.get("SHIFT_BATTLE", False))
PREPROC_MODE = str(cfg.get("PREPROC_MODE", "clahe"))  # gray|clahe|sobel|canny
# Масочная проверка: допустимая разница по V (0-255)
MASKED_DV = int(cfg.get("MASKED_DV", 20))
# adaptive минимальные доли совпадения маски (малый/средний/большой шаблон)
# Adaptive minimal masked-agreement (small/medium/large). Defaults tuned to 'Balance' set.
MIN_MA_SMALL = float(cfg.get("MIN_MA_SMALL", 0.56))
MIN_MA_MED = float(cfg.get("MIN_MA_MED", 0.52))
MIN_MA_LARGE = float(cfg.get("MIN_MA_LARGE", 0.48))

# Максимальная доля ROI, которую может занимать кандидат (чтобы не брать гигантские квадраты)
MAX_BOX_FRAC = float(cfg.get("MAX_BOX_FRAC", 0.20))

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


def iou_box(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = float(boxAArea + boxBArea - interArea)
    return interArea / unionArea if unionArea > 0 else 0


def center_distance(boxA, boxB):
    cxA = boxA[0] + boxA[2] / 2
    cyA = boxA[1] + boxA[3] / 2
    cxB = boxB[0] + boxB[2] / 2
    cyB = boxB[1] + boxB[3] / 2
    return sqrt((cxA - cxB) ** 2 + (cyA - cyB) ** 2)


def postprocess_matches(raw_matches, tpl_sizes_dict, max_icons=None, iou_thr=0.5, center_thr=0.25, min_w=45, min_h=50):
    """
    Универсальная постобработка матчей:
    - raw_matches: [(tpl_name, x, y, w, h, score), ...]
    - tpl_sizes_dict: {tpl_name: (tpl_w, tpl_h)}
    - max_icons: None = без ограничения, иначе обрезать сверху
    - iou_thr: порог IoU для объединения в один кластер
    - center_thr: порог близости центров (в долях от min(w,h))
    - min_w, min_h: жёсткий минимум размера иконки
    """

    if not raw_matches:
        return []

    # 1. Отбрасываем слишком маленькие боксы
    size_filtered = []
    for m in raw_matches:
        _, _, _, w, h, _ = m
        if w < min_w or h < min_h:
            print(f"[postprocess] drop {m[0]}: too small ({w}x{h})")
            continue
        size_filtered.append(m)

    if not size_filtered:
        return []

    # 2. Сортируем по убыванию score
    sorted_matches = sorted(size_filtered, key=lambda m: m[5], reverse=True)

    # 3. Кластеризация по IoU и центрам
    clusters = []
    for m in sorted_matches:
        tpl_name, x, y, w, h, score = m
        mx, my, mw, mh = x, y, w, h
        mcx, mcy = mx + mw / 2.0, my + mh / 2.0

        placed = False
        for cl in clusters:
            for n in cl:
                nx, ny, nw, nh, _ = n[1], n[2], n[3], n[4], n[5]
                # IoU
                xA = max(mx, nx)
                yA = max(my, ny)
                xB = min(mx + mw, nx + nw)
                yB = min(my + mh, ny + nh)
                iw = max(0, xB - xA)
                ih = max(0, yB - yA)
                inter = iw * ih
                union = mw * mh + nw * nh - inter
                iou = inter / union if union > 0 else 0
                # центр-близость
                ncx, ncy = nx + nw / 2.0, ny + nh / 2.0
                cdist = max(abs(mcx - ncx) / max(mw, 1), abs(mcy - ncy) / max(mh, 1))
                if iou >= iou_thr or cdist <= center_thr:
                    print(f"[postprocess] merge {tpl_name} with {n[0]} " f"(IoU={iou:.2f}, cdist={cdist:.2f})")
                    cl.append(m)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            clusters.append([m])

    # 4. Выбираем лучший матч в каждом кластере
    final = [max(cl, key=lambda t: t[5]) for cl in clusters]

    # 5. Если нужно — ограничиваем сверху
    if max_icons is not None:
        final = sorted(final, key=lambda m: m[5], reverse=True)[:max_icons]

    return final


def cluster_matches(raw_matches, iou_thr=0.50, center_thr=0.25):
    """
    raw_matches: [(tpl_name, x, y, w, h, score, box_id)]
    Кластер: IoU >= iou_thr или близкие центры (<= center_thr от min(w,h)).
    Возвращает репрезентативы кластеров (лучший по score).
    """
    if not raw_matches:
        return []

    def _iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        xA = max(ax, bx)
        yA = max(ay, by)
        xB = min(ax + aw, bx + bw)
        yB = min(ay + ah, bx + bh)
        iw = max(0, xB - xA)
        ih = max(0, yB - yA)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = aw * ah + bw * bh - inter
        return inter / float(union) if union > 0 else 0.0

    clusters = []
    for m in sorted(raw_matches, key=lambda t: -t[5]):
        mx, my, mw, mh = m[1], m[2], m[3], m[4]
        mbox = (mx, my, mw, mh)
        mcx, mcy = mx + mw / 2.0, my + mh / 2.0
        placed = False
        for cl in clusters:
            for n in cl:
                nx, ny, nw, nh = n[1], n[2], n[3], n[4]
                iou = _iou(mbox, (nx, ny, nw, nh))
                ncx, ncy = nx + nw / 2.0, ny + nh / 2.0
                cdist = max(abs(mcx - ncx) / max(mw, 1), abs(mcy - ncy) / max(mh, 1))
                if iou >= iou_thr or cdist <= center_thr:
                    cl.append(m)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            clusters.append([m])

    reps = [max(cl, key=lambda t: t[5]) for cl in clusters]
    return reps


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


# Ensure a single audit CSV path (use the out_dir-based audit_csv defined above)
AUDIT_CSV_PATH = os.path.abspath(audit_csv)


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


# (clean) module-level helpers and imports are defined above

def draw_icon_overlays(overlay, matches, icon_keys=None):
    """
    Рисует зелёные рамки только для иконок из templates_registry.json.
    Фильтрует по ключу (item.icon.*) или имени файла (_icon.png).
    """
    # безопасный доступ к глобальной переменной ICON_KEYS — возможно она ещё не определена
    if icon_keys is None:
        icon_keys = globals().get("ICON_KEYS", set())

    for match in matches:
        # match may be a tuple (tpl_name, x, y, w, h, score) or an object with attrs
        if isinstance(match, (list, tuple)) and len(match) >= 5:
            tpl_key = match[0]
            x1, y1, w, h = int(match[1]), int(match[2]), int(match[3]), int(match[4])
            x2, y2 = x1 + w, y1 + h
        else:
            tpl_key = getattr(match, "tpl_key", "")
            x1 = int(getattr(match, "x1", getattr(match, "x", 0)))
            y1 = int(getattr(match, "y1", getattr(match, "y", 0)))
            x2 = int(getattr(match, "x2", getattr(match, "x", 0) + getattr(match, "w", 0)))
            y2 = int(getattr(match, "y2", getattr(match, "y", 0) + getattr(match, "h", 0)))

        # Фильтр: рисуем только для зарегистрированных item-иконок
        if tpl_key not in icon_keys:
            continue

        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0, 255), 2)


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


# ==================== Быстрый amber-валидатор =============
def amber_stats(bgr: np.ndarray):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (10, 70, 60), (30, 255, 255))
    if cv2.countNonZero(m) == 0:
        m = cv2.inRange(hsv, (8, 50, 40), (35, 255, 255))
    m = cv2.medianBlur(m, 3)
    H, W = m.shape[:2]
    area = max(1, H * W)
    cov = cv2.countNonZero(m) / area
    M = cv2.moments(m)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        off = max(abs(cx - W / 2) / W, abs(cy - H / 2) / H)
    else:
        off = 1.0
    return cov, off


def is_amber_valid(cov, off, cov_min=0.08, cov_max=0.55, off_max=0.18):
    # Жёсткая проверка
    if cov_min <= cov <= cov_max and off <= off_max:
        return True, "amber_valid"

    # Мягкая зона — пропускаем в матчинг, но помечаем
    soft_cov_min, soft_cov_max = 0.05, 0.60
    soft_off_max = 0.22
    if soft_cov_min <= cov <= soft_cov_max and off <= soft_off_max:
        return True, "amber_suspect"

    # Полный отказ
    return False, "amber_invalid"


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


def nms(boxes, iou_thr=0.50):
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
def filter_boxes(contours, roi_shape, debug=False):
    H, W = roi_shape[:2]
    roi_area = max(1, H * W)
    out = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area <= 0:
            continue

        # Жёсткий фильтр по минимальному размеру

        if w < MIN_W or h < MIN_H:
            if debug:
                print(f"[FILTER] skip: too_small ({w}x{h})")
            continue

        # Масштабонезависимые критерии
        ar = w / float(max(1, h))
        if ar < 0.5 or ar > 1.8:
            continue

        frac = area / roi_area
        # Уберём явно мусорные крошки и явно гигантские области
        if frac < 0.0002 or frac > MAX_BOX_FRAC:
            continue

        out.append((x, y, w, h))

    out = nms(out, iou_thr=0.35)
    out = sorted(out, key=lambda b: (b[1], b[0]))
    return out[:220]


def merge_close_boxes(boxes, gap_thr=12, center_ratio=0.6):
    """Объединяет боксы, которые близко друг к другу или небольшим зазором разделены (например, элементы цепи).
    - boxes: list[(x,y,w,h)]
    - gap_thr: максимальный пиксельный зазор между боксами для слияния
    - center_ratio: если расстояние между центрами меньше этой доли от мин размеров — объединяем
    Возвращает список объединённых боксов.
    """
    if not boxes:
        return []

    # Более консервативная логика объединения:
    # - объединяем только если боксы почти соприкасаются по X и Y (gap<=gap_thr)
    # - или если центры близки и размеры соизмеримы (area_ratio в пределах)
    used = [False] * len(boxes)
    merged = []

    for i, a in enumerate(boxes):
        if used[i]:
            continue
        ax, ay, aw, ah = a
        ax2, ay2 = ax + aw, ay + ah
        group = [a]
        used[i] = True
        a_area = aw * ah
        for j, b in enumerate(boxes[i + 1 :], start=i + 1):
            if used[j]:
                continue
            bx, by, bw, bh = b
            bx2, by2 = bx + bw, by + bh
            b_area = bw * bh

            # gap between boxes (if they don't intersect)
            gap_x = max(0, max(bx, ax) - min(bx2, ax2))
            gap_y = max(0, max(by, ay) - min(by2, ay2))

            # center distance normalized by min size
            mcx, mcy = ax + aw / 2.0, ay + ah / 2.0
            ncx, ncy = bx + bw / 2.0, by + bh / 2.0
            cdist_x = abs(mcx - ncx) / max(1.0, min(aw, bw))
            cdist_y = abs(mcy - ncy) / max(1.0, min(ah, bh))
            cdist = max(cdist_x, cdist_y)

            # размеры сравнимы?
            area_ratio = max(a_area / max(1.0, b_area), b_area / max(1.0, a_area))

            should_merge = False
            # 1) почти соприкасаются (gap по обеим осям маленький)
            if gap_x <= gap_thr and gap_y <= gap_thr:
                should_merge = True
            # 2) центры близки И размеры соизмеримы (не сливать мелкие с очень большими)
            elif cdist <= center_ratio and area_ratio <= 4.0:
                should_merge = True

            if should_merge:
                group.append(b)
                used[j] = True

        # объединяем группу в один bounding box
        xs = [g[0] for g in group]
        ys = [g[1] for g in group]
        x2s = [g[0] + g[2] for g in group]
        y2s = [g[1] + g[3] for g in group]
        nx = min(xs)
        ny = min(ys)
        nx2 = max(x2s)
        ny2 = max(y2s)
        merged.append((nx, ny, nx2 - nx, ny2 - ny))

    # сортируем и возвращаем
    merged = sorted(merged, key=lambda b: (b[1], b[0]))
    return merged


# ==================== Детектор боксов ===========


def detect_icon_boxes(roi_bgr: np.ndarray, screen_name: str):
    masks = build_candidate_masks(roi_bgr, screen_name)
    boxes = []
    for tag, mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes.extend(filter_boxes(contours, roi_bgr.shape))
    # Попробуем объединить близкие боксы (чтобы не дробить длинные предметы вроде цепи)
    if boxes:
        log(f"[DETECT] before_merge count={len(boxes)}")
        boxes = merge_close_boxes(boxes, gap_thr=14, center_ratio=0.6)
        log(f"[DETECT] after_merge count={len(boxes)}")

    if not boxes:
        # fallback: чуть шире диапазон S/V
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        fallback = cv2.inRange(hsv, (0, 0, 55), (180, 120, 255))
        contours, _ = cv2.findContours(fallback, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes.extend(filter_boxes(contours, roi_bgr.shape))
        imwrite_u(os.path.join(masks_dir, f"m_fallback_{screen_name}"), fallback)

    # Отбросим слишком большие боксы, которые занимают почти весь ROI
    H, W = roi_bgr.shape[:2]
    filtered_boxes = []
    for x, y, w, h in boxes:
        if (w * h) / float(max(1, H * W)) > MAX_BOX_FRAC:
            # большой бокс — пропускаем
            continue
        filtered_boxes.append((x, y, w, h))
    boxes = nms(filtered_boxes, iou_thr=0.35)
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
      - COLOR (HSV) с маской (если есть): TM_CCORR_NORMED
      - EDGE (Canny) по CCOEFF_NORMED
      - GRAY (CLAHE/другой PREPROC_MODE) по CCOEFF_NORMED
    Возвращает (metrics, fail_reasons) или (None, fail_reasons).
    """
    # более мягкие пороги для предварительного отсева масштабов — позволим больше кандидатов
    THRESH_COLOR, THRESH_GRAY, THRESH_EDGE = 0.5, 0.25, 0.12
    th, tw = tpl_gray.shape[:2]
    ch0, cw0 = crop_bgr.shape[:2]

    upscale_k = 0.0
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
    # доля краёв в кропе — пригодится для фильтрации «плоских» артефактов
    try:
        ch0, cw0 = crop_edge.shape[:2]
        edge_density_global = float(cv2.countNonZero(crop_edge)) / float(max(1, ch0 * cw0))
    except Exception:
        edge_density_global = 0.0

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

        # --- Защита от None/пустой маски ---
        if tpl_mask_color is not None and getattr(tpl_mask_color, 'size', 0) > 0:
            tmask = cv2.resize(tpl_mask_color, (t_w, t_h), interpolation=cv2.INTER_NEAREST)
        else:
            # если у шаблона нет альфа/маски, попробуем сгенерировать её по непустому (не-белому) пикселю
            try:
                tpl_gray_small = cv2.cvtColor(tplc, cv2.COLOR_BGR2GRAY)
                # non-white mask: pixels with brightness < 245
                tmask = (tpl_gray_small < 245).astype('uint8') * 255
                # очистим шум и закроем дырки
                if cv2.countNonZero(tmask) > 0:
                    tmask = cv2.medianBlur(tmask, 3)
                    tmask = cv2.morphologyEx(tmask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
                    tmask = cv2.morphologyEx(tmask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
                else:
                    tmask = None
            except Exception:
                tmask = None

        if tmask is not None and cv2.countNonZero(tmask) == 0:
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
        if tmask is not None:
            res_col = cv2.matchTemplate(crop_hsv[:, :, 2], tpl_hsv[:, :, 2], cv2.TM_CCORR_NORMED, mask=tmask)
        else:
            res_col = cv2.matchTemplate(crop_hsv[:, :, 2], tpl_hsv[:, :, 2], cv2.TM_CCORR_NORMED)
        _, score_col, _, _ = cv2.minMaxLoc(res_col)

        # EDGE
        tpl_edge = canny_edges(tplg)
        res_edge = cv2.matchTemplate(crop_edge, tpl_edge, cv2.TM_CCOEFF_NORMED)
        _, score_edge, _, _ = cv2.minMaxLoc(res_edge)

        # GRAY
        tplp = preprocess_gray(tplg, PREPROC_MODE)
        res_gray = cv2.matchTemplate(crop_proc, tplp, cv2.TM_CCOEFF_NORMED)
        _, score_gray, _, _ = cv2.minMaxLoc(res_gray)

        # Fast-accept: если цветовой скор очень высок (почти точное совпадение по V), берём его
        if score_col >= 0.94:
            combined = 0.80 * score_col + 0.12 * score_gray + 0.08 * score_edge
            per_scale.append((combined, score_col, score_gray, score_edge, scale, (t_w, t_h), upscale_k, local_downscale_k))
            continue

        # Если все три метода сильно ниже порогов — фиксируем отказ, иначе оставляем для дальнейшей оценки
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

        # Сделаем комбинированную метрику чуть более ориентированной на цвет (он даёт наилучшее отличие от фона)
        combined = 0.65 * score_col + 0.22 * score_gray + 0.13 * score_edge
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
    # Дополнительная масочная оценка: насколько пиксели шаблона (mask) совпадают по яркости/цвету с кропом
    try:
        # попытаемся взять альфу/маску из tpl_mask_color или alpha в исходном tpl_bgr
        tpl_mask = None
        try:
            # tpl_mask_color может быть бинарной маской или None
            if tpl_mask_color is not None and getattr(tpl_mask_color, 'size', 0) > 0:
                if len(tpl_mask_color.shape) == 2:
                    tpl_mask = (tpl_mask_color > 0).astype('uint8') * 255
                else:
                    tpl_mask = (cv2.cvtColor(tpl_mask_color, cv2.COLOR_BGR2GRAY) > 0).astype('uint8') * 255
        except Exception:
            tpl_mask = None

        if tpl_mask is not None:
            # позиция лучшего совпадения: найдем локализацию по цветовой карте (v-channel)
            tpl_w, tpl_h = metrics.get('tpl_wh', (0, 0))
            tplc = cv2.resize(tpl_bgr, (tpl_w, tpl_h), interpolation=cv2.INTER_LINEAR)
            tmask = cv2.resize(tpl_mask, (tpl_w, tpl_h), interpolation=cv2.INTER_NEAREST)
            crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
            tpl_hsv = cv2.cvtColor(tplc, cv2.COLOR_BGR2HSV)
            # matchTemplate on V channel to get best location for this scale
            res = cv2.matchTemplate(crop_hsv[:, :, 2], tpl_hsv[:, :, 2], cv2.TM_CCORR_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            mx, my = max_loc
            # extract crop region
            rx1, ry1 = mx, my
            rx2, ry2 = mx + tpl_w, my + tpl_h
            if rx2 <= crop_bgr.shape[1] and ry2 <= crop_bgr.shape[0]:
                sub = crop_bgr[ry1:ry2, rx1:rx2]
                sub_hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
                # сравним V канал по маске: доля пикселей с близкой яркостью
                v_sub = sub_hsv[:, :, 2].astype('int')
                v_tpl = tpl_hsv[:, :, 2].astype('int')
                mask_bool = tmask.astype(bool)
                if mask_bool.sum() > 0:
                    diff = np.abs(v_sub - v_tpl)
                    agree = (diff[mask_bool] <= MASKED_DV).sum()
                    total = mask_bool.sum()
                    masked_agreement = float(agree) / float(total)
                else:
                    masked_agreement = 0.0
            else:
                masked_agreement = 0.0
        else:
            masked_agreement = None
        metrics['masked_agreement'] = masked_agreement
    except Exception:
        metrics['masked_agreement'] = None
    # добавим метрику плотности краёв для дальнейших решений
    metrics['edge_density'] = edge_density_global
    return metrics, fail_reasons


def strict_template_search_on_roi(roi_bgr, screen_name, icon_templates, out_dir, min_combined=0.70, color_weight: float = 0.7):
    """Строгий поиск по шаблонам: для каждого шаблона ищем лучшее совпадение по V-channel (цвет) и по gray.
    Комбинируем скор и возвращаем все совпадения с combined >= min_combined.
    Сохраняем CSV и overlay в папке audit_dir.
    """
    matches = []
    crop_h = roi_bgr.shape[0]
    crop_w = roi_bgr.shape[1]

    # подготовим CSV
    csv_path = os.path.join(out_dir, f"strict_matches_{screen_name}.csv")
    with open(csv_path, "w", encoding="utf-8") as fcsv:
        fcsv.write("tpl_key,x,y,w,h,combined,color,gray,scale\n")

    # use a BGR overlay (no alpha) to draw rectangles/text easily
    overlay = roi_bgr.copy()

    for tpl_key, tpl_bgr, tpl_gray, tpl_mask_color in icon_templates:
        th, tw = tpl_gray.shape[:2]
        best_for_tpl = None
        best_score = -1.0
    # best_loc/best_scale not required here; we only keep best_for_tpl tuple

        for scale in ICON_SCALES:
            t_w = max(8, int(round(tw * scale)))
            t_h = max(8, int(round(th * scale)))
            if t_w >= crop_w or t_h >= crop_h:
                # если шаблон больше ROI — даунскейлим шаблон
                kx = crop_w / max(1, t_w)
                ky = crop_h / max(1, t_h)
                k = min(kx, ky)
                if k <= 0:
                    continue
                t_w = max(8, int(round(t_w * k)))
                t_h = max(8, int(round(t_h * k)))

            tplc = cv2.resize(tpl_bgr, (t_w, t_h), interpolation=cv2.INTER_LINEAR)
            tplg = cv2.resize(tpl_gray, (t_w, t_h), interpolation=cv2.INTER_LINEAR)

            # COLOR (V channel)
            crop_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            tpl_hsv = cv2.cvtColor(tplc, cv2.COLOR_BGR2HSV)
            try:
                if tpl_mask_color is not None and getattr(tpl_mask_color, 'size', 0) > 0:
                    tmask = cv2.resize(tpl_mask_color, (t_w, t_h), interpolation=cv2.INTER_NEAREST)
                    res_col = cv2.matchTemplate(crop_hsv[:, :, 2], tpl_hsv[:, :, 2], cv2.TM_CCORR_NORMED, mask=tmask)
                else:
                    res_col = cv2.matchTemplate(crop_hsv[:, :, 2], tpl_hsv[:, :, 2], cv2.TM_CCORR_NORMED)
                _, score_col, _, max_loc = cv2.minMaxLoc(res_col)
            except Exception:
                score_col = 0.0
                max_loc = (0, 0)

            # GRAY
            crop_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            try:
                res_gray = cv2.matchTemplate(crop_gray, tplg, cv2.TM_CCOEFF_NORMED)
                _, score_gray, _, _ = cv2.minMaxLoc(res_gray)
            except Exception:
                score_gray = 0.0

            # combined (use configurable color_weight)
            gw = 1.0 - float(color_weight)
            combined = float(color_weight) * float(score_col) + gw * float(score_gray)

            if combined > best_score:
                best_score = combined
                best_for_tpl = (combined, float(score_col), float(score_gray), scale, (t_w, t_h), max_loc)

        if best_for_tpl and best_for_tpl[0] >= min_combined:
            comb, scol, sgray, scale, (tw2, th2), (mx, my) = best_for_tpl
            # координаты и размеры в ROI
            x1, y1 = int(mx), int(my)
            x2, y2 = x1 + tw2, y1 + th2
            # вырезаем и записываем
            matches.append((tpl_key, x1, y1, tw2, th2, comb, scol, sgray, scale))
            # draw
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay, f"{tpl_key}:{comb:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            # append CSV
            with open(csv_path, "a", encoding="utf-8") as fcsv:
                fcsv.write(f"{tpl_key},{x1},{y1},{tw2},{th2},{comb:.4f},{scol:.4f},{sgray:.4f},{scale:.2f}\n")

    # save overlay
    overlay_path = os.path.join(out_dir, f"strict_overlay_{screen_name}.png")
    try:
        imwrite_u(overlay_path, overlay)
    except Exception:
        try:
            ok, enc = cv2.imencode('.png', overlay)
            if ok:
                enc.tofile(overlay_path)
        except Exception:
            pass

    # return csv and overlay paths for caller convenience
    return csv_path, overlay_path


def decision_accept(m, cw=None, ch=None, is_suspicious=False):
    """Решение о принятии кандидата на основе метрик.
    is_suspicious — флаг, означающий «плоский/маленький» кроп: в этом случае требуем более высокую
    согласованность маски (masked_agreement) и комбинированный скор.

    """
    if m is None:
        return False
    color = m.get("color", 0.0)
    gray = m.get("gray", 0.0)
    edge = m.get("edge", 0.0)
    comb = m.get("combined", 0.0)

    # Final thresholds (configurable via cfg STRICT section)
    strict_cfg = cfg.get('STRICT', {}) if isinstance(cfg, dict) else {}
    MIN_COMBINED_FINAL = float(strict_cfg.get('MIN_COMBINED_FINAL', strict_cfg.get('MIN_COMBINED', 0.80)))
    MIN_MASKED_AGREE = float(strict_cfg.get('MIN_MASKED_AGREE', 0.60))
    MIN_GRAY_ACCEPT = float(strict_cfg.get('MIN_GRAY_ACCEPT', 0.40))
    MIN_EDGE_ACCEPT = float(strict_cfg.get('MIN_EDGE_ACCEPT', THR_EDGE))
    SOFT_COMBINED = float(strict_cfg.get('SOFT_COMBINED_THRESHOLD', 0.78))
    SOFT_GRAY = float(strict_cfg.get('SOFT_GRAY', 0.44))
    SOFT_MASKED = float(strict_cfg.get('SOFT_MASKED', 0.58))

    # Быстрый accept для очень сильных цветовых совпадений
    ma_quick = m.get('masked_agreement', 0.0)
    if color >= 0.995 and ma_quick >= 0.46:
        print(f"[DECISION] fast-accept by color={color:.3f} MA={ma_quick:.3f}")
        return True

    # Если сразу высокий combined и хотя бы один из вспомогательных сигналов достаточен — принять
    if comb >= MIN_COMBINED_FINAL:
        ma = m.get('masked_agreement', None)
        if (ma is not None and ma >= MIN_MASKED_AGREE) or gray >= MIN_GRAY_ACCEPT or edge >= MIN_EDGE_ACCEPT:
            print(f"[DECISION] accept by final-rule comb={comb:.3f} ma={ma} gray={gray:.3f} edge={edge:.3f}")
            return True

    # Базовые правила
    ok = False
    if color >= THR_COLOR and (gray >= 0.18 or edge >= THR_EDGE):
        ok = True
    elif gray >= (THR_GRAY + 0.04) and edge >= 0.10:
        ok = True
    elif color >= (THR_COLOR + 0.08):
        ok = True
    elif comb >= 0.62 and color >= 0.60:
        ok = True

    # Если базовая проверка не прошла — даём шанс по masked_agreement, но с ужесточением для suspicious
    if not ok:
        ma = m.get('masked_agreement', None)
        if ma is not None:
            tpl_w, tpl_h = m.get('tpl_wh', (0, 0))
            area = max(1, tpl_w * tpl_h)
            if area < 80 * 80:
                min_ma = MIN_MA_SMALL
            elif area < 140 * 140:
                min_ma = MIN_MA_MED
            else:
                min_ma = MIN_MA_LARGE
            # требуем более высокий MA для подозрительных кропов
            required_ma = max(min_ma, 0.60) if is_suspicious else min_ma
            required_comb = 0.60 if is_suspicious else 0.58
            if ma >= required_ma and comb >= required_comb:
                print(f"[DECISION] accept by masked_agreement={ma:.3f} comb={comb:.3f} (req_ma={required_ma:.2f})")
                ok = True
        if not ok:
            return False

    # Физический масштаб шаблона к кропу
    if cw is not None and ch is not None:
        tpl_w, tpl_h = m.get("tpl_wh", (0, 0))
        k_crop = max(cw, ch)
        k_tpl = max(tpl_w, tpl_h)
        if k_crop > 0 and k_tpl > 0:
            ratio = k_tpl / k_crop
            if not (0.55 <= ratio <= 1.8):
                return False

    # Дополнительная масочная проверка: если есть masked_agreement, требуем минимум (второй барьер)
    ma = m.get('masked_agreement', None)
    if ma is not None:
        tpl_w, tpl_h = m.get('tpl_wh', (0, 0))
        area = max(1, tpl_w * tpl_h)
        if area < 80 * 80:
            min_ma = MIN_MA_SMALL
        elif area < 140 * 140:
            min_ma = MIN_MA_MED
        else:
            min_ma = MIN_MA_LARGE
        # для suspicious требуем немного выше
        threshold = max(min_ma, 0.55) if is_suspicious else min_ma
        if ma < threshold:
            print(f"[DECISION] reject by final MA check ma={ma:.3f} threshold={threshold:.3f} suspicious={is_suspicious}")
            return False

    return True


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
        if metrics.get('masked_agreement') is not None:
            rows.append(("MASKED_AGREEMENT", float(metrics.get('masked_agreement'))))

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


def main(strict: bool = False, strict_min_combined: float = 0.70, color_weight: float = 0.7):
    MIN_W, MIN_H = 45, 50  # жёсткий минимум

    # --- Перезапуск лога ---
    if os.path.exists(log_path):
        try:
            os.remove(log_path)
        except Exception:
            pass

    # --- Загружаем все иконки из JSON ---
    with open(r"C:\bot\tools\cfg\templates_registry.json", "r", encoding="utf-8") as f:
        templates_registry = json.load(f)

    # Выбираем только item-иконки из каталога tpl_group
    ICON_KEYS = {k for k, v in templates_registry.items() if k.startswith("item.icon.")}

    icon_templates = []

    for key in sorted(ICON_KEYS):
        path = templates_registry.get(key)
        # дополнительная страховка: только файлы из `tpl_root/tpl_group`
        if not path or tpl_group.lower() not in path.replace('\\', '/').lower():
            continue
        if not os.path.exists(path):
            log(f"[WARN] Файл не найден: {path}")
            continue
        try:
            data = np.fromfile(path, dtype=np.uint8)
            tpl_img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if tpl_img is None:
                log(f"[WARN] Не удалось декодировать: {path}")
                continue
            tpl_mask_color = None
            # Если есть альфа-канал — извлечём маску и оставим BGR
            if tpl_img.ndim == 3 and tpl_img.shape[2] == 4:
                alpha = tpl_img[:, :, 3].copy()
                tpl_mask_color = (alpha > 10).astype('uint8') * 255
                tpl_bgr = tpl_img[:, :, :3].copy()
            else:
                tpl_bgr = tpl_img.copy()
            tpl_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
            icon_templates.append((key, tpl_bgr, tpl_gray, tpl_mask_color))
        except Exception as e:
            log(f"[ERR] Ошибка загрузки {path}: {e}")

    if not icon_templates:
        log("[ERR] Не удалось загрузить ни одного шаблона иконок")
        return

    shift_y = 150 if SHIFT_BATTLE else 0

    # Счётчики
    total_boxes = 0
    total_found = 0
    failed_boxes = 0
    fail_stats_scale = {}
    fail_stats_box = {}
    total_scale_fail_events = 0
    combined_rows = []

    # --- подготовка словаря размеров шаблонов ---
    tpl_sizes_dict = {tpl_name: tpl_gray.shape[::-1] for tpl_name, _, tpl_gray, _ in icon_templates}

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
            log(f"[ERR] Некорректный ROI: ({x1},{y1})-({x2},{y2})")
            continue

        roi_bgr = img[y1:y2, x1:x2].copy()
        imwrite_u(os.path.join(roi_dir, f"roi_{screen_name}"), roi_bgr)
        log(f"[ROI] Размер ROI: {roi_bgr.shape}")

        # Если включён строгий режим — запускаем точный шаблонный поиск по ROI и пропускаем основной flow
        if strict:
            try:
                csv_path, overlay_path = strict_template_search_on_roi(
                    roi_bgr,
                    screen_name,
                    icon_templates,
                    audit_dir,
                    min_combined=strict_min_combined,
                    color_weight=color_weight,
                )
                log(f"[STRICT] strict search results: csv={csv_path} overlay={overlay_path}")
            except Exception as e:
                log(f"[ERR] strict search failed for {screen_name}: {e}")
            continue

        boxes = detect_icon_boxes(roi_bgr, screen_name)
        overlay = img.copy()
        raw_matches = []  # сюда собираем все принятые матчи

        MIN_COMBINED = 0.68

        for idx, (bx, by, bw, bh) in enumerate(boxes, start=1):
            total_boxes += 1
            raw = roi_bgr[by : by + bh, bx : bx + bw].copy()
            imwrite_u(os.path.join(audit_dir, f"{screen_name}_box{idx}_raw.png"), raw)
            # --- Amber‑валидация с мягким режимом ---
            cov, off = amber_stats(raw)
            ok_amber, amber_status = is_amber_valid(cov, off)

            # Логируем в combined_rows (даже если потом бокс отвалится)
            base_row = {
                "screen": screen_name,
                "box_idx": idx,
                "w": bw,
                "h": bh,
                "cov": round(cov, 3),
                "off": round(off, 3),
                "amber_status": amber_status,
                "color": "",
                "gray": "",
                "edge": "",
                "combined": "",
                "accepted": 0,
                "box_reason": "",
                "scale_reason": "",
                "best_scale": "",
                "tpl_w": "",
                "tpl_h": "",
                "upscale_k": "",
                "downscale_k": "",
            }

            if not ok_amber:
                # Помечаем статус янтарности, но НЕ пропускаем матчинг — это мешало находить
                # зеленые/не-янтарные предметы (например, ягоды). Будем матчить, но пометим.
                log(f"[FILTER] box={idx} {amber_status} cov={cov:.3f} off={off:.3f}")
                base_row["box_reason"] = amber_status
                # Нарисуем пометку на оверлее, но продолжим матчинг
                abs_pt = (x1 + bx, y1 + by)
                color_rect = (0, 0, 255) if amber_status == "amber_invalid" else (0, 255, 255)
                cv2.rectangle(overlay, abs_pt, (abs_pt[0] + bw, abs_pt[1] + bh), color_rect, 1)
                cv2.putText(
                    overlay,
                    amber_status.upper(),
                    (abs_pt[0], max(0, abs_pt[1] - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color_rect,
                    1,
                    cv2.LINE_AA,
                )

            # --- Жёсткий фильтр по исходному боксу ---
            if bw < MIN_W or bh < MIN_H:
                log(f"[MATCH] box={idx} отклонён — too_small_box ({bw}x{bh})")
                write_audit_scores(screen_name, idx, bw, bh, None, None)
                failed_boxes += 1
                fail_stats_box["too_small_box"] = fail_stats_box.get("too_small_box", 0) + 1
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
                        "box_reason": "too_small_box",
                        "scale_reason": "",
                        "best_scale": "",
                        "tpl_w": "",
                        "tpl_h": "",
                        "upscale_k": "",
                        "downscale_k": "",
                        "masked_agreement": "",
                        "edge_density": "",
                    }
                )
                abs_pt = (x1 + bx, y1 + by)
                cv2.rectangle(overlay, abs_pt, (abs_pt[0] + bw, abs_pt[1] + bh), MISS_COLOR, 1)
                cv2.putText(
                    overlay,
                    "MIN_SIZE",
                    (abs_pt[0], max(0, abs_pt[1] - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    MISS_COLOR,
                    1,
                    cv2.LINE_AA,
                )
                continue

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
                        "edge_density": "",
                    }
                )
                abs_pt = (x1 + bx, y1 + by)
                cv2.rectangle(overlay, abs_pt, (abs_pt[0] + bw, abs_pt[1] + bh), MISS_COLOR, 1)
                cv2.putText(
                    overlay,
                    "EMPTY",
                    (abs_pt[0], max(0, abs_pt[1] - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    MISS_COLOR,
                    1,
                    cv2.LINE_AA,
                )
                continue

            ch, cw = clean.shape[:2]
            if cw < MIN_W or ch < MIN_H:
                log(f"[MATCH] box={idx} подозрительный кроп — suspicious_crop ({cw}x{ch}, box={bw}x{bh})")
                fail_stats_box["suspicious_crop"] = fail_stats_box.get("suspicious_crop", 0) + 1

            accepted = False
            best_match = None
            best_metrics = None
            best_fail_reasons = None
            all_candidates = []

            # флаг suspicious: если чистый кроп маленький или мало краёв
            ch, cw = clean.shape[:2]
            # порог плотности краёв ниже которого считаем кроп 'плоским'
            SUSP_EDGE_DENSITY = 0.008
            # получим метрику edge_density из match_multi позже; пока предварительный флаг
            pre_suspicious = True if (cw < MIN_W or ch < MIN_H) else False

            for tpl_name, tpl_bgr, tpl_gray, tpl_mask_color in icon_templates:
                metrics, fail_reasons = match_multi(clean, tpl_bgr, tpl_gray, tpl_mask_color, screen_name, idx)
                if metrics:
                    all_candidates.append((tpl_name, templates_registry[tpl_name], metrics["combined"], metrics))
                    # окончательный suspicious: учитываем edge_density, если она есть в метриках
                    md = metrics.get('edge_density', 0.0)
                    is_suspicious = pre_suspicious or (md < SUSP_EDGE_DENSITY)
                    if (
                        decision_accept(metrics, cw=clean.shape[1], ch=clean.shape[0], is_suspicious=is_suspicious)
                        and metrics["combined"] >= MIN_COMBINED
                    ):

                        accepted = True
                        best_match = tpl_name
                        best_metrics = metrics
                        best_fail_reasons = fail_reasons
                        break

            if not accepted:
                failed_boxes += 1
                reasons_list = best_fail_reasons or [{"reason": "no_match"}]
                for r in reasons_list:
                    reason = r.get("reason", "unknown")
                    fail_stats_scale[reason] = fail_stats_scale.get(reason, 0) + 1
                    total_scale_fail_events += 1
                unique_reasons = {r.get("reason", "unknown") for r in reasons_list}
                for reason in unique_reasons:
                    fail_stats_box[reason] = fail_stats_box.get(reason, 0) + 1
                log(f"[MATCH] box={idx} отказ — {', '.join(unique_reasons)}")
                write_audit_scores(screen_name, idx, cw, ch, None, None)

                # Логируем топ-3 кандидата для отладки
                if all_candidates:
                    top3 = sorted(all_candidates, key=lambda x: x[2], reverse=True)[:3]
                    log("[MATCH] box={0} TOP3 candidates: " + ", ".join([f"{t[0]}({t[2]:.3f}) MA={t[3].get('masked_agreement') if t[3] else ''}" for t in top3]).format(idx))
                    # используем лучший кандидат для записи masked_agreement в combined_rows
                    best_cand = top3[0]
                    best_ma = best_cand[3].get('masked_agreement') if best_cand and best_cand[3] else ""
                else:
                    best_ma = ""

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
                        "masked_agreement": round(best_ma, 3) if isinstance(best_ma, float) else "",
                        "edge_density": "",
                    }
                )
            else:
                top3 = sorted(all_candidates, key=lambda x: x[2], reverse=True)[:3]
                ma_str = (
                    f" MA={best_metrics['masked_agreement']:.3f}" if best_metrics and best_metrics.get('masked_agreement') is not None else ""
                )
                log(
                    f"[MATCH] box={idx} ACCEPT: {best_match} ({templates_registry[best_match]}) "
                    f"scale={best_metrics['scale']:.2f} "
                    f"color={best_metrics['color']:.3f} gray={best_metrics['gray']:.3f} "
                    f"edge={best_metrics['edge']:.3f} comb={best_metrics['combined']:.3f}{ma_str}"
                )
                log("       top3: " + ", ".join(f"{n}({c:.3f})" for n, _, c, _ in top3))

                write_audit_scores(screen_name, idx, cw, ch, best_metrics, best_fail_reasons)
                combined_rows.append(
                    {
                        "screen": screen_name,
                        "box_idx": idx,
                        "w": cw,
                        "h": ch,
                        "color": best_metrics["color"],
                        "gray": best_metrics["gray"],
                        "edge": best_metrics["edge"],
                        "combined": best_metrics["combined"],
                        "accepted": 1,
                        "box_reason": "",
                        "scale_reason": json.dumps(best_fail_reasons, ensure_ascii=False) if best_fail_reasons else "",
                        "best_scale": best_metrics.get("scale", ""),
                        "tpl_w": best_metrics.get("tpl_wh", ("", ""))[0],
                        "tpl_h": best_metrics.get("tpl_wh", ("", ""))[1],
                        "upscale_k": best_metrics.get("upscale_k", ""),
                        "downscale_k": best_metrics.get("downscale_k", ""),
                        "masked_agreement": round(best_metrics.get("masked_agreement", 0.0), 3) if best_metrics and best_metrics.get("masked_agreement") is not None else "",
                        "edge_density": round(best_metrics.get("edge_density", 0.0), 4) if best_metrics and best_metrics.get("edge_density") is not None else "",
                    }
                )
                # вместо found += 1 — добавляем в raw_matches
                raw_matches.append((best_match, x1 + bx, y1 + by, bw, bh, best_metrics["combined"]))

            abs_pt = (x1 + bx, y1 + by)
            color = HIT_COLOR if accepted else MISS_COLOR
            cv2.rectangle(overlay, abs_pt, (abs_pt[0] + bw, abs_pt[1] + bh), color, 2)
            label_parts = [best_match or "NO_MATCH"]
            if best_metrics:
                label_parts.append(f"C{best_metrics['color']:.2f}")
                label_parts.append(f"G{best_metrics['gray']:.2f}")
                label_parts.append(f"E{best_metrics['edge']:.2f}")
            label = " ".join(label_parts)
            cv2.putText(
                overlay, label, (abs_pt[0], max(0, abs_pt[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
            )

        # --- пост‑обработка: оставляем максимум 5 уникальных иконок ---
        final_matches = postprocess_matches(
            raw_matches,
            tpl_sizes_dict,
            max_icons=None,  # или число, если хочешь ограничить
            iou_thr=0.5,
            center_thr=0.25,
        )

        # Дедупликация финальных матчей: если два матча пересекаются сильно — оставляем лучший
        deduped = []
        for m in sorted(final_matches, key=lambda t: t[5], reverse=True):
            keep = True
            for k in deduped:
                if iou_box((m[1], m[2], m[3], m[4]), (k[1], k[2], k[3], k[4])) > 0.85:
                    keep = False
                    break
            if keep:
                deduped.append(m)
        final_matches = deduped

        found = len(final_matches)
        total_found += found

        # Отрисовываем финальные боксы поверх оверлея
        for tpl_name, fx, fy, fw, fh, score in final_matches:
            cv2.rectangle(overlay, (fx, fy), (fx + fw, fy + fh), HIT_COLOR, 2)
            cv2.putText(
                overlay,
                f"{tpl_name} {score:.2f}",
                (fx, max(0, fy - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                HIT_COLOR,
                1,
                cv2.LINE_AA,
            )

        # --- сохраняем оверлей ---
        overlay_path = os.path.join(audit_dir, f"overlay_{screen_name}")
        imwrite_u(overlay_path, overlay)
        log(f"[SAVE] Overlay сохранён: {overlay_path}")

        log(f"[STAT] {screen_name}: найдено {found} из {len(boxes)}")

    # --- Итоговая статистика ---
    log(f"[TOTAL] Всего боксов: {total_boxes}")
    log(f"[TOTAL] Найдено: {total_found}")
    log(f"[TOTAL] Не найдено: {failed_boxes}")

    log("[FAIL-BOX] Причины отказа по боксам:")
    for reason, cnt in fail_stats_box.items():
        log(f"  {reason}: {cnt}")

    log("[FAIL-SCALE] Причины отказа по масштабам:")
    for reason, cnt in fail_stats_scale.items():
        log(f"  {reason}: {cnt}")
    log(f"[FAIL-SCALE] Всего событий: {total_scale_fail_events}")

    # --- Сохраняем комбинированный CSV ---
    csv_path = os.path.join(audit_dir, "combined_audit.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "screen",
            "box_idx",
            "w",
            "h",
            "color",
            "gray",
            "edge",
            "combined",
            "edge_density",
            "accepted",
            "box_reason",
            "scale_reason",
            "best_scale",
            "tpl_w",
            "tpl_h",
            "upscale_k",
            "downscale_k",
            "masked_agreement",
            "cov",
            "off",
            "amber_status",  # ← добавили новые поля
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in combined_rows:
            writer.writerow(row)

    log(f"[SAVE] CSV сохранён: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find icons in screens")
    parser.add_argument("--strict", action="store_true", help="Run strict template-only search on each ROI")
    # defaults from config
    strict_cfg = cfg.get('STRICT', {}) if isinstance(globals().get('cfg', None), dict) else {}
    default_min_combined = float(strict_cfg.get('MIN_COMBINED', 0.70))
    default_color_weight = float(strict_cfg.get('COLOR_WEIGHT', 0.7))
    parser.add_argument("--min-combined", type=float, default=default_min_combined, help="Min combined score for strict mode")
    parser.add_argument("--color-weight", type=float, default=default_color_weight, help="Weight for color (V) when computing combined score; gray_weight = 1-color_weight")
    args = parser.parse_args()
    main(strict=args.strict, strict_min_combined=args.min_combined, color_weight=args.color_weight)
