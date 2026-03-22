# -*- coding: utf-8 -*-
"""
Универсальный преобразователь шаблонов
Moon Edition — чисто-белый фон без полос, выборка по JSON
"""

import csv
import hashlib
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")

# ======= ПАРАМЕТРЫ =======
SRC_DIR = Path(r"C:\bot\tpl\иконки_предметов")
OUT_DIRS = {
    "alpha": SRC_DIR.parent / "alpha",
    "edges": SRC_DIR.parent / "edges",
    "bw": SRC_DIR.parent / "bw",
    "bw_inv": SRC_DIR.parent / "bw_inv",
}
MANIFEST_PATH = SRC_DIR.parent / ".preprocess_manifest.json"
LOG_PATH = SRC_DIR.parent / "preprocess.txt"
NOT_ALPHA_PATH = Path(r"C:\bot\tools\cfg\not_alpha.json")
ZALPHA_DIR = SRC_DIR.parent / "zalpha"

# CSV-аудит (UTF-8-SIG для Excel)
AUDIT_SUMMARY_CSV = SRC_DIR.parent / "preprocess_summary.csv"
AUDIT_DETAIL_CSV = SRC_DIR.parent / "preprocess_detail.csv"
# ======= ТРЕЙС/РЕЖИМЫ =======
TRACE_HEAVY = True  # сохранять все промежуточные артефакты и метрики
PREFILTER_MODE = "dry"  # off | dry | block
PREFILTER_GROUPS = ("icons",)  # какие группы пускать в обработку при mode="block"

STEPS = {"remove_bg": True, "edges": True, "bw": True, "bw_inv": True}

# Режим: продвинутая вырезка чисто белого фона без полос (unmatte)
# По умолчанию включаем строгий режим удаления белого (пользователь попросил)
BG_MODE = "pure_white_strict"
# Альтернативный режим: удаляем строго белый цвет (255,255,255) без блюра/инпейнта
BG_MODE_STRICT = "pure_white_strict"

# Порог для режима "мягкого" строгого удаления (если понадобится). 255 = ровно белый.
WHITE_MIN = 255


def remove_bg_unmatte_white(src_path: Path, dst_path: Path, white_min: int = 250):
    """
    Compute alpha from how close a pixel is to white (threshold white_min).
    alpha = 1 - min((b-wm)/(255-wm), (g-wm)/(255-wm), (r-wm)/(255-wm)) clipped [0,1].
    Then unmatte colors by dividing out white background using that alpha.
    No Gaussian blur or inpaint — produces sharp edges.
    """
    img = cv2_imread_unicode(src_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        logging.error(f"[unmatte] Не удалось загрузить {src_path}")
        return False, "file_read_error", 0, {}
    # Приведение к 3 каналам
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    b, g, r = cv2.split(img)
    # Авто-порог по LAB (по рамке)
    border_px = 4
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    Lbg = max(L_MIN_CLAMP, int(np.percentile(L[_border_mask(*img.shape[:2], border_px)], 10)))
    wm = int(max(WHITE_MIN, Lbg))
    denom = float(max(1, 255 - wm))
    wb = np.clip((b.astype(np.float32) - wm) / denom, 0.0, 1.0)
    wg = np.clip((g.astype(np.float32) - wm) / denom, 0.0, 1.0)
    wr = np.clip((r.astype(np.float32) - wm) / denom, 0.0, 1.0)
    white_strength = np.minimum(np.minimum(wb, wg), wr)
    alpha_f = 1.0 - white_strength
    # Мягкая альфа, не обрезаем детали
    alpha_safe = np.clip(alpha_f, 1e-2, 1.0)
    rgb = np.stack((b, g, r), axis=2).astype(np.float32)
    out_rgb = (rgb - (1.0 - alpha_safe)[..., None] * 255.0) / alpha_safe[..., None]
    out_rgb = np.clip(out_rgb, 0, 255).astype(np.uint8)
    fully_transparent = (alpha_f <= 1e-2)
    out_rgb[fully_transparent] = 0
    alpha_u8 = (np.clip(alpha_f, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgba = cv2.merge((out_rgb[:, :, 0], out_rgb[:, :, 1], out_rgb[:, :, 2], alpha_u8))
    tpix = int(np.sum(alpha_u8 == 0))
    if not cv2_imwrite_unicode(dst_path, rgba):
        logging.error(f"[unmatte] Не удалось сохранить {dst_path}")
        return False, "write_error", tpix, {"white_min": wm}
    if tpix == 0:
        logging.warning(f"[unmatte] Нет прозрачных пикселей: {src_path}")
        return False, "no_transparency", tpix, {"white_min": wm}
    # Debug-вывод спорных случаев
    if tpix < 50 or tpix > 0.9 * img.size:
        debug_path = dst_path.parent / f"debug_{dst_path.name}"
        cv2_imwrite_unicode(debug_path, rgba)
    return True, "ok", tpix, {"white_min": wm}
# Альтернативный режим: удаляем строго белый цвет (255,255,255) без блюра/инпейнта
# полезно когда нужно просто убрать белый фон ровно
BG_MODE_STRICT = "pure_white_strict"

# Настройки порогов и инструментов (None = авто)
BG_WHITE_THRESH = None  # было 250 — фикс; теперь авто по умолчанию
BG_WHITE_EXPAND = None  # авто
ERODE_PX = None  # авто по размеру иконки
BLUR_MASK = None  # авто, нечётный размер ядра

CANNY_T1 = 80
CANNY_T2 = 150

# Жёсткий минимум размеров иконок
MIN_W, MIN_H = 45, 50
# ======= БЕЛЫЙ ФОН (ЦВЕТОБЕЗОПАСНО) =======
COLOR_SAFE_WHITE = True  # Жёстко защищать цвет при вырезке белого
BORDER_PX = 4  # Толщина рамки для оценки фона
L_MIN_CLAMP = 238  # Нижняя граница порога L для "белого" (если авто)
C_MAX_CLAMP = (8, 22)  # Диапазон допустимой хроматичности для "белого"
S_MAX_CLAMP = (0, 28)  # Диапазон допустимой насыщенности для "белого" (HSV.S 0..255)

# ======= ЛОГИ =======
os.makedirs(LOG_PATH.parent, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("tpl_preprocess")
import platform
import time


def _fmt_kv(**kv):
    return " ".join(f"{k}={v}" for k, v in kv.items())


def trace_env_header():
    log.info(
        "[env] "
        + _fmt_kv(
            py=platform.python_version(),
            cv2=cv2.__version__,
            numpy=np.__version__,
            cwd=os.getcwd(),
            src=str(SRC_DIR),
            out=str(SRC_DIR.parent),
            min_w=MIN_W,
            min_h=MIN_H,
            bg_mode=BG_MODE,
            white_thresh=str(BG_WHITE_THRESH),
            expand=str(BG_WHITE_EXPAND),
            erode=str(ERODE_PX),
            blur=str(BLUR_MASK),
            prefilter=PREFILTER_MODE,
        )
    )


import shutil

# ======= УТИЛИТЫ =======
from typing import Dict, List

# Константы
REJECTS_DIR = Path("prefilter_rejects")
ALLOWED_EXT = {".png", ".jpg", ".jpeg"}


def _border_mask(h, w, px=4):
    m = np.zeros((h, w), np.uint8)
    m[:px, :] = 1
    m[-px:, :] = 1
    m[:, :px] = 1
    m[:, -px:] = 1
    return m.astype(bool)


def _auto_white_thresholds_from_border(bgr, border_px=4):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.uint8)
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)
    C = np.sqrt((a - 128.0) ** 2 + (b - 128.0) ** 2).astype(np.float32)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1].astype(np.uint8)

    h, w = L.shape
    mb = _border_mask(h, w, border_px)
    Lb = L[mb]
    Cb = C[mb]
    Sb = S[mb]
    # Перцентили: адаптивно, но зажатые к клампам
    Lbg = max(L_MIN_CLAMP, int(np.percentile(Lb, 10)))
    Cmax = int(np.percentile(Cb, 95))
    Smax = int(np.percentile(Sb, 95))
    # Клампы для стабильности
    Cmax = int(np.clip(Cmax, C_MAX_CLAMP[0], C_MAX_CLAMP[1]))
    Smax = int(np.clip(Smax, S_MAX_CLAMP[0], S_MAX_CLAMP[1]))
    return Lbg, Cmax, Smax


def build_bg_mask_color_safe(img_bgr, frame_px=2, Lmin=240, Cmax=15, Smax=20):
    import colorsys

    h, w = img_bgr.shape[:2]
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(img_lab)

    # --- рамка для оценки фона ---
    frame_mask = np.zeros((h, w), np.uint8)
    frame_mask[:frame_px, :] = 1
    frame_mask[-frame_px:, :] = 1
    frame_mask[:, :frame_px] = 1
    frame_mask[:, -frame_px:] = 1

    # --- перевод в HSV для цветового анализа ---
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(img_hsv)

    # --- считаем L только по реально белым пикселям рамки ---
    white_frame_mask = (S <= Smax) & (np.abs(A.astype(int) - 128) <= Cmax) & (np.abs(B.astype(int) - 128) <= Cmax)
    white_frame_mask &= frame_mask.astype(bool)

    if np.any(white_frame_mask):
        white_L = np.percentile(L[white_frame_mask], 10)
    else:
        white_L = 255

    # --- жёсткий минимум ---
    white_L = max(white_L, Lmin)

    # --- маска фона: и по L, и по цвету ---
    mask_bg = (
        (L >= white_L) & (np.abs(A.astype(int) - 128) <= Cmax) & (np.abs(B.astype(int) - 128) <= Cmax) & (S <= Smax)
    )

    # --- защита насыщенных пикселей ---
    protect_mask = (S > Smax) | (np.abs(A.astype(int) - 128) > Cmax) | (np.abs(B.astype(int) - 128) > Cmax)
    mask_bg[protect_mask] = False

    return mask_bg.astype(np.uint8) * 255


def apply_prefilter(files, mode="off", groups=("icons",)):
    if mode == "off":
        return files, None
    groups_dict = prefilter_images(files)
    # Сводка по группам в лог
    if groups_dict:
        summary = {k: len(v) for k, v in groups_dict.items()}
        log.info(f"[prefilter] summary {summary}")
    if mode == "dry":
        # Логируем/копируем rejects, но пропускаем в пайплайн все файлы
        return files, groups_dict
    # "block": пускаем только выбранные группы
    selected = []
    for g in groups:
        selected.extend(groups_dict.get(g, []))
    return selected, groups_dict


def prefilter_images(files: List[Path]) -> Dict[str, List[Path]]:
    """
    Предфильтрация входных файлов:
    - Проверка расширения
    - Проверка читаемости
    - Проверка минимального размера
    - (опционально) проверка фона
    Возвращает словарь с группами прошедших файлов.
    """
    passed_groups = {"icons": [], "art": [], "maps": [], "items": []}
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for f in files:
        name = f.name
        ext = f.suffix.lower()

        # 1. Проверка расширения
        if ext not in ALLOWED_EXT:
            reject_file(f, "bad_ext")
            log_reject(ts, name, "bad_ext")
            continue

        # 2. Читаемость
        meta = probe_image_meta(f)
        if meta is None:
            reject_file(f, "file_read_error")
            log_reject(ts, name, "file_read_error")
            continue

        # 3. Минимальный размер
        if meta["w"] < MIN_W or meta["h"] < MIN_H:
            reason = "too_small_%dx%d" % (meta["w"], meta["h"])
            reject_file(f, reason)
            log_reject(ts, name, "too_small(%dx%d)" % (meta["w"], meta["h"]))
            continue

        # 4. (опционально) проверка фона
        # if not has_expected_background(f):
        #     reject_file(f, "bad_background")
        #     log_reject(ts, name, "bad_background")
        #     continue

        # 5. Группировка по типам
        group = detect_group(name)
        passed_groups[group].append(f)

    return passed_groups


def reject_file(f: Path, reason: str) -> None:
    """Сохраняет файл в папку rejects/<reason>/"""
    target_dir = REJECTS_DIR / reason
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(f), str(target_dir / f.name))


def log_reject(ts: str, name: str, reason: str) -> None:
    """Логирует причину отсева в CSV аудит"""
    audit_write_details([{"timestamp": ts, "file": name, "stage": "prefilter", "message": reason}])


def detect_group(filename: str) -> str:
    """Простейшая классификация по имени файла"""
    low = filename.lower()
    if "_icon" in low or "icon_" in low:
        return "icons"
    elif "map" in low:
        return "maps"
    elif "item" in low or "weapon" in low:
        return "items"
    else:
        return "art"


def file_md5(path: Path) -> str:
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_manifest(m: dict) -> None:
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)


def load_not_alpha_list() -> set:
    if NOT_ALPHA_PATH.exists():
        try:
            with open(NOT_ALPHA_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return set(data)
                if isinstance(data, dict):
                    items = set()
                    for v in data.values():
                        if isinstance(v, list):
                            items.update(v)
                    return items
        except Exception as e:
            log.warning(f"[warn] Не удалось загрузить not_alpha.json — {e}")
    return set()


# Юникод-совместимые I/O
def cv2_imread_unicode(path, flags=cv2.IMREAD_UNCHANGED):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            log.warning(f"[read_fail] Пустой поток данных: {path}")
            return None
        img = cv2.imdecode(data, flags)
        if img is None:
            log.warning(f"[read_fail] Не удалось декодировать: {path}")
        return img
    except Exception as e:
        log.error(f"[read_err] {path}: {e}")
        return None


def cv2_imwrite_unicode(path, img) -> bool:
    try:
        ext = Path(path).suffix or ".png"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            log.warning(f"[write_fail] Не удалось закодировать: {path}")
            return False
        buf.tofile(str(path))
        return True
    except Exception as e:
        log.warning(f"[write_err] {path}: {e}")
        return False


def probe_image_meta(path: Path):
    img = cv2_imread_unicode(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    h, w = img.shape[:2]
    c = 1 if img.ndim == 2 else img.shape[2]
    return {"w": w, "h": h, "c": c}


# ======= CSV-АУДИТ =======
def safe_val(v):
    """Возвращает значение, безопасное для записи в CSV."""
    return v if isinstance(v, (int, str)) else ""


def audit_write_csv(path, fieldnames, rows):
    """Универсальная функция записи в CSV с безопасной обработкой значений."""
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        if write_header:
            w.writeheader()
        for r in rows:
            safe_row = {k: safe_val(r.get(k, "")) for k in fieldnames}
            w.writerow(safe_row)


def audit_write_summary(row: dict):
    """Запись строки в сводный CSV-аудит."""
    fieldnames = [
        "timestamp",
        "file",
        "status",
        "mode",
        "transparent_pixels",
        "w",
        "h",
        "white_thresh",
        "expand",
        "erode_px",
        "blur_mask",
        "alpha_path",
        "debug_dir",
    ]
    audit_write_csv(AUDIT_SUMMARY_CSV, fieldnames, [row])


def audit_write_details(rows: list):
    """Запись детализированных строк аудита."""
    fieldnames = ["timestamp", "file", "stage", "message"]
    audit_write_csv(AUDIT_DETAIL_CSV, fieldnames, rows)


# ======= ПРОВЕРКА ФОНА (для сообщений) =======
def check_alpha_transparency(path: Path):
    img = cv2_imread_unicode(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        log.warning(f"[warn] Не удалось проверить фон: {path}")
        return 0
    if img.ndim < 3 or img.shape[2] < 4:
        log.warning(f"[warn] {path.name}: нет альфа-канала — фон остался")
        return 0
    alpha = img[:, :, 3]
    transparent_pixels = int(np.sum(alpha == 0))
    if transparent_pixels > 0:
        log.info(f"[ok] {path.name}: фон вырезан, прозрачных пикселей {transparent_pixels}")
    else:
        log.warning(f"[warn] {path.name}: альфа есть, но прозрачных пикселей нет — фон не убран")
    return transparent_pixels


# ======= ВЫРЕЗКА ЧИСТО-БЕЛОГО (LAB, мягкая альфа) =======
def _auto_params(w: int, h: int, white_thresh):
    diag = (w * w + h * h) ** 0.5
    # erode/blur авто
    if diag < 256:
        erode_px = 1
        blur_mask = 3
    elif diag < 512:
        erode_px = 2
        blur_mask = 5
    else:
        erode_px = 3
        blur_mask = 7
    # expand авто (если нужен)
    if white_thresh is None:
        expand = None
    else:
        if white_thresh >= 247:
            expand = 2
        elif white_thresh >= 240:
            expand = 3
        else:
            expand = 4
    return erode_px, blur_mask, expand


def remove_bg_pure_white(
    src_path: Path,
    dst_path: Path,
    white_thresh: Optional[int] = None,
    expand: Optional[int] = None,
    erode_px: Optional[int] = None,
    blur_mask: Optional[int] = None,
    save_debug: bool = True,
    radius_factor: float = 0.08,
):
    """
    Возвращает: (ok: bool, status: str, transparent_pixels: int, used_params: dict)
    """
    bgr = cv2_imread_unicode(src_path, cv2.IMREAD_COLOR)
    if bgr is None:
        return False, "file_read_error", 0, {"exclude_reason": "file_read_error"}

    h, w = bgr.shape[:2]
    if w < 45 or h < 50:
        return False, "too_small", 0, {"exclude_reason": f"size<{w}x{h}"}

    # --- LAB + HSV (используем адаптивный порог по рамке) ---
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L_chan = lab[:, :, 0].astype(np.float32)
    A = lab[:, :, 1].astype(np.float32)
    B = lab[:, :, 2].astype(np.float32)
    C = np.sqrt((A - 128.0) ** 2 + (B - 128.0) ** 2)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1].astype(np.float32)

    # адаптивные параметры по рамке — стабильнее для разных размеров
    try:
        Lbg_auto, Cmax_auto, Smax_auto = _auto_white_thresholds_from_border(bgr, border_px=BORDER_PX)
    except Exception:
        Lbg_auto, Cmax_auto, Smax_auto = L_MIN_CLAMP, C_MAX_CLAMP[1], S_MAX_CLAMP[1]

    # Используем переданные/глобальные значения как оверрайд
    Lbg = int(white_thresh) if (white_thresh is not None) else int(max(L_MIN_CLAMP, Lbg_auto))
    Cmax = C_MAX_CLAMP[1] if (BG_WHITE_THRESH is None) else C_MAX_CLAMP[1]
    Smax = S_MAX_CLAMP[1] if (BG_WHITE_THRESH is None) else S_MAX_CLAMP[1]

    used_thresh = f"L={Lbg},Cmax={Cmax},Smax={Smax}"

    # начальная маска фона — L высокий и малые отклонения по цвету/насыщенности
    mask_bg = (L_chan >= Lbg) & (np.abs(A - 128) <= Cmax) & (np.abs(B - 128) <= Cmax) & (S <= Smax)

    # Удаляем мелкие полосы/шумы в маске фона: медианный фильтр + морфология
    mask_bg_u8 = (mask_bg.astype(np.uint8) * 255)
    k_med = max(3, int(min(w, h) / 100) | 1)
    try:
        mask_bg_u8 = cv2.medianBlur(mask_bg_u8, k_med)
    except Exception:
        pass

    # Морфология для удаления полос: сначала закрытие, затем открытие с адаптивными ядрами
    kx = max(3, w // 80)
    ky = max(3, h // 80)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kx | 1, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky | 1))
    kernel_sq = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_bg_u8 = cv2.morphologyEx(mask_bg_u8, cv2.MORPH_CLOSE, kernel_h, iterations=1)
    mask_bg_u8 = cv2.morphologyEx(mask_bg_u8, cv2.MORPH_CLOSE, kernel_v, iterations=1)
    mask_bg_u8 = cv2.morphologyEx(mask_bg_u8, cv2.MORPH_OPEN, kernel_sq, iterations=1)

    # --- Защита насыщенных/цветных пикселей: исключаем их из фона ---
    protect_mask = (S > Smax) | (np.abs(A - 128) > Cmax) | (np.abs(B - 128) > Cmax)
    mask_bg_u8[protect_mask] = 0

    mask_bg = (mask_bg_u8 > 0).astype(np.uint8) * 255

    # --- Автопараметры морфологии ---
    auto_erode, auto_blur, auto_expand = _auto_params(w, h, Lbg)
    if erode_px is None:
        erode_px = auto_erode
    if blur_mask is None:
        blur_mask = auto_blur
    if expand is None:
        expand = auto_expand or 0
    if blur_mask < 1:
        blur_mask = 1
    if blur_mask % 2 == 0:
        blur_mask += 1

    # Расширение/сглаживание маски фона
    if expand > 0:
        mask_bg = cv2.dilate(mask_bg, np.ones((3, 3), np.uint8), iterations=int(expand))

    # маска переднего плана
    mask_fg = cv2.bitwise_not(mask_bg)
    if erode_px > 0:
        mask_fg = cv2.erode(mask_fg, np.ones((3, 3), np.uint8), iterations=int(erode_px))
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    # Мягкая альфа через distance transform, но делаем внутренность предмета непрозрачной
    mask_bin = (mask_fg > 0).astype(np.uint8)
    dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
    # radius — относительный размер области, внутри которой альфа должна быть ~1
    radius = max(2, int(min(w, h) * float(radius_factor)))
    if radius < 2:
        radius = 2
    if dist.max() > 0:
        alpha = np.clip(dist / float(radius), 0.0, 1.0)
    else:
        alpha = mask_bin.astype(np.float32)
    # небольшое сглаживание края — не больше 3
    small_blur = 3
    try:
        alpha = cv2.GaussianBlur(alpha, (small_blur, small_blur), 0)
    except Exception:
        pass
    alpha = np.clip(alpha, 0.0, 1.0)

    # Unmatte: восстанавливаем цвета под альфой (white background)
    rgb = bgr.astype(np.float32)
    a_ch = alpha[..., None]
    eps = 1e-6
    out_rgb = (rgb - (1.0 - a_ch) * 255.0) / np.maximum(a_ch, eps)
    out_rgb[a_ch.squeeze() < 0.02] = 0
    out_rgb = np.clip(out_rgb, 0, 255).astype(np.uint8)

    # местный inpaint по краю для удаления артефактов
    edge_mask = ((alpha > 0) & (alpha < 1)).astype(np.uint8) * 255
    edge_mask = cv2.dilate(edge_mask, np.ones((3, 3), np.uint8), iterations=1)
    try:
        # меньший радиус инпейнта уменьшает размытие
        out_rgb = cv2.inpaint(out_rgb, edge_mask, 1, cv2.INPAINT_TELEA)
    except Exception:
        pass

    alpha_u8 = (alpha * 255.0).astype(np.uint8)
    rgba = cv2.merge((out_rgb[:, :, 0], out_rgb[:, :, 1], out_rgb[:, :, 2], alpha_u8))

    # --- Дебаг ---
    dbg_dir = (Path(dst_path).parent / "_debug_white") if save_debug else None
    if dbg_dir:
        try:
            dbg_dir.mkdir(parents=True, exist_ok=True)
            cv2_imwrite_unicode(dbg_dir / f"{src_path.stem}_mask_fg.png", mask_fg)
            cv2_imwrite_unicode(dbg_dir / f"{src_path.stem}_alpha.png", alpha_u8)
            overlay = bgr.copy()
            overlay[mask_fg == 0] = (overlay[mask_fg == 0] * 0.3 + np.array([0, 0, 255]) * 0.7).astype(np.uint8)
            cv2_imwrite_unicode(dbg_dir / f"{src_path.stem}_overlay.png", overlay)
        except Exception as e:
            log.warning(f"[debug_write_err] {src_path} debug write failed: {e}")

    # --- Итог ---
    tpix = int(np.sum(alpha_u8 == 0))
    if not cv2_imwrite_unicode(dst_path, rgba):
        return False, "write_error", tpix, {"exclude_reason": "write_error"}

    if tpix == 0:
        return False, "no_transparency", tpix, {"exclude_reason": "no_transparency"}

    return (
        True,
        "ok",
        tpix,
        {
            "white_thresh": used_thresh,
            "white_thresh_val": int(Lbg),
            "expand": expand,
            "erode_px": erode_px,
            "blur_mask": blur_mask,
            "debug_dir": str(dbg_dir) if dbg_dir else "",
        },
    )


def remove_bg_fast(src_path: Path, dst_path: Path):
    img = cv2_imread_unicode(src_path, cv2.IMREAD_COLOR)
    if img is None:
        return False, "file_read_error", 0, {}
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
    alpha = cv2.bitwise_not(mask)
    b, g, r = cv2.split(img)
    rgba = cv2.merge((b, g, r, alpha))
    tpix = int(np.sum(alpha == 0))
    if not cv2_imwrite_unicode(dst_path, rgba):
        return False, "write_error", tpix, {}
    if tpix == 0:
        return False, "no_transparency", tpix, {}
    return True, "ok", tpix, {}


def remove_bg_strict_white(src_path: Path, dst_path: Path):
    """
    Простое удаление чисто-белого фона: любые пиксели ровно (255,255,255)
    считаются фоном и переводятся в полностью прозрачные. Никакого блюра,
    inpaint или unmatte — просто жесткое обнуление альфа для белых пикселей.
    """
    img = cv2_imread_unicode(src_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return False, "file_read_error", 0, {}

    # Если есть уже альфа — удаляем фон по RGB каналам и обновляем альфа
    if img.ndim == 2:
        # одноканальное — считаем как градация серого, белый -> прозрачный
        b = img
        alpha = np.where(b == 255, 0, 255).astype(np.uint8)
        rgba = cv2.merge((b, b, b, alpha))
    else:
        # color image
        if img.shape[2] == 4:
            b, g, r, a = cv2.split(img)
        else:
            b, g, r = cv2.split(img)
            a = np.full_like(b, 255)

        # Используем глобальный WHITE_MIN если он задан (модульный атрибут)
        white_min = globals().get("WHITE_MIN", 255)
        if white_min is None:
            white_min = 255
        # >= white_min считается белым
        white_mask = (b >= white_min) & (g >= white_min) & (r >= white_min)
        # Новая альфа: белые пиксели -> 0, остальные сохраняют существующую альфу (если была) или 255
        new_a = a.copy()
        new_a[white_mask] = 0
        rgba = cv2.merge((b, g, r, new_a))

    tpix = int(np.sum(rgba[:, :, 3] == 0))
    if not cv2_imwrite_unicode(dst_path, rgba):
        return False, "write_error", tpix, {"white_min": int(white_min)}
    if tpix == 0:
        return False, "no_transparency", tpix, {"white_min": int(white_min)}
    return True, "ok", tpix, {"white_min": int(white_min)}


def remove_bg_grabcut(src_path: Path, dst_path: Path):
    img = cv2_imread_unicode(src_path, cv2.IMREAD_COLOR)
    if img is None:
        return False, "file_read_error", 0, {}
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (1, 1, img.shape[1] - 2, img.shape[0] - 2)
    try:
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    except Exception as e:
        log.warning(f"[warn] GrabCut ошибка: {src_path} — {e}")
        return False, "algo_error", 0, {}
    mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")
    mask2 = cv2.GaussianBlur(mask2, (3, 3), 0)
    b, g, r = cv2.split(img)
    rgba = cv2.merge((b, g, r, mask2))
    tpix = int(np.sum(mask2 == 0))
    if not cv2_imwrite_unicode(dst_path, rgba):
        return False, "write_error", tpix, {}
    if tpix == 0:
        return False, "no_transparency", tpix, {}
    return True, "ok", tpix, {}


# ======= КОНТУРЫ/ЧБ (для аудита) =======
def edges_canny(src_path, dst_path):
    img = cv2_imread_unicode(src_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    edges = cv2.Canny(img, CANNY_T1, CANNY_T2)
    return cv2_imwrite_unicode(dst_path, edges)


def bw_threshold(src_path, dst_path, invert=False):
    img = cv2_imread_unicode(src_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert:
        th = cv2.bitwise_not(th)
    return cv2_imwrite_unicode(dst_path, th)


def safe_param(params: dict, key: str):
    val = params.get(key, "")
    return val if isinstance(val, (int, str)) else ""


# ======= ОСНОВНОЙ ПРОЦЕСС =======
def process_one(src: Path) -> bool:
    name = src.name
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alpha_path = OUT_DIRS["alpha"] / name
    detail_rows = []
    t0 = time.perf_counter()

    def _detail(stage, **kv):
        detail_rows.append({"timestamp": ts, "file": name, "stage": stage, "message": _fmt_kv(**kv)})

    # probe
    meta = probe_image_meta(src)
    if meta is None:
        msg = "file_read_error"
        log.warning(f"[warn] Не удалось прочитать: {name}")
        _detail("probe", status=msg)
        audit_write_details(detail_rows)
        audit_write_summary(
            {
                "timestamp": ts,
                "file": name,
                "status": msg,
                "mode": BG_MODE,
                "transparent_pixels": 0,
                "w": 0,
                "h": 0,
                "white_thresh": "",
                "expand": "",
                "erode_px": "",
                "blur_mask": "",
                "alpha_path": str(alpha_path),
                "debug_dir": "",
            }
        )
        return False
    _detail("probe", w=meta["w"], h=meta["h"], c=meta["c"])

    if meta["w"] < MIN_W or meta["h"] < MIN_H:
        msg = f"too_small({meta['w']}x{meta['h']})"
        log.warning(f"[warn] {name}: меньше минимума {MIN_W}x{MIN_H} — исключаю")
        _detail("probe", status="too_small", w=meta["w"], h=meta["h"])
        audit_write_details(detail_rows)
        audit_write_summary(
            {
                "timestamp": ts,
                "file": name,
                "status": "too_small",
                "mode": BG_MODE,
                "transparent_pixels": 0,
                "w": meta["w"],
                "h": meta["h"],
                "white_thresh": "",
                "expand": "",
                "erode_px": "",
                "blur_mask": "",
                "alpha_path": str(alpha_path),
                "debug_dir": "",
            }
        )
        return False

    # Удаление фона
    stage_start = time.perf_counter()
    ok = False
    status = "remove_bg_skipped"
    tpix = 0
    params = {}
    dbg_dir = None

    if STEPS["remove_bg"]:
        try:
            if BG_MODE == "pure_white":
                ok, status, tpix, params = remove_bg_pure_white(
                    src,
                    alpha_path,
                    white_thresh=(None if BG_WHITE_THRESH is None else int(BG_WHITE_THRESH)),
                    expand=(None if BG_WHITE_EXPAND is None else int(BG_WHITE_EXPAND)),
                    erode_px=(None if ERODE_PX is None else int(ERODE_PX)),
                    blur_mask=(None if BLUR_MASK is None else int(BLUR_MASK)),
                    save_debug=True,
                )
            elif BG_MODE == BG_MODE_STRICT or BG_MODE == "pure_white_strict":
                ok, status, tpix, params = remove_bg_strict_white(src, alpha_path)
            elif BG_MODE == "pure_white_unmatte":
                wm = globals().get("WHITE_MIN", 255)
                ok, status, tpix, params = remove_bg_unmatte_white(src, alpha_path, white_min=wm)
            elif BG_MODE == "fast":
                ok, status, tpix, params = remove_bg_fast(src, alpha_path)
            elif BG_MODE == "grabcut":
                ok, status, tpix, params = remove_bg_grabcut(src, alpha_path)
            else:
                ok, status, tpix, params = False, "unknown_mode", 0, {}

            # Трассировка параметров и метрик
            used_thresh = params.get("white_thresh", "")
            expand_v = params.get("expand", "")
            erode_v = params.get("erode_px", "")
            blur_v = params.get("blur_mask", "")
            dbg_dir = params.get("debug_dir", "")

            _detail(
                "remove_bg",
                status=status,
                ok=int(ok),
                tpix=tpix,
                white_thresh=used_thresh,
                expand=expand_v,
                erode_px=erode_v,
                blur_mask=blur_v,
            )

            # Доп. артефакты и метрики
            if TRACE_HEAVY and str(dbg_dir):
                try:
                    # Сохраняем L-канал и mask_bg (пересчёт тут же, дешёвый)
                    bgr = cv2_imread_unicode(src, cv2.IMREAD_COLOR)
                    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
                    L = lab[:, :, 0]
                    thr_val, mask_bg = cv2.threshold(
                        L,
                        0 if used_thresh in ("", "otsu") else int(used_thresh),
                        255,
                        cv2.THRESH_BINARY + (cv2.THRESH_OTSU if used_thresh in ("", "otsu") else 0),
                    )
                    cv2_imwrite_unicode(Path(dbg_dir) / f"{Path(src).stem}_L.png", L)
                    cv2_imwrite_unicode(Path(dbg_dir) / f"{Path(src).stem}_mask_bg.png", mask_bg)

                    # Быстрые метрики
                    h, w = L.shape[:2]
                    bg_cov = round(float(np.count_nonzero(mask_bg)) / max(1, h * w), 4)
                    # Простейшая статистика альфы, если записана
                    alpha_path_tmp = OUT_DIRS["alpha"] / name
                    alpha_stat = {}
                    rgba = cv2_imread_unicode(alpha_path_tmp, cv2.IMREAD_UNCHANGED)
                    if rgba is not None and rgba.ndim == 3 and rgba.shape[2] == 4:
                        a = rgba[:, :, 3]
                        alpha_stat = dict(
                            a_min=int(a.min()),
                            a_max=int(a.max()),
                            a_mean=round(float(a.mean()), 2),
                            a_zeros=int(np.sum(a == 0)),
                        )
                    # JSON метрики
                    metrics = dict(
                        file=name,
                        w=w,
                        h=h,
                        status=status,
                        tpix=tpix,
                        used_thresh=used_thresh,
                        thr_val=float(thr_val),
                        expand=expand_v,
                        erode_px=erode_v,
                        blur_mask=blur_v,
                        bg_cov=bg_cov,
                        alpha=alpha_stat,
                    )
                    with open(Path(dbg_dir) / f"{Path(src).stem}_metrics.json", "w", encoding="utf-8") as jf:
                        json.dump(metrics, jf, ensure_ascii=False, indent=2)
                except Exception as e:
                    _detail("remove_bg_trace_error", err=str(e))
        except Exception as e:
            ok, status, tpix, params = False, "exception", 0, {}
            _detail("remove_bg_exception", err=str(e))

        # Человеческий лог
        if ok:
            log.info(f"[ok] {name}: фон вырезан, прозрачных пикселей {tpix}")
        else:
            if status == "no_transparency":
                log.warning(f"[warn] {name}: альфа есть, но прозрачных пикселей нет — фон не убран")
            else:
                log.warning(f"[warn] {name}: {status}")

    _detail("timing_remove_bg", ms=int(1000 * (time.perf_counter() - stage_start)))


    # ===== Автотюнинг по эталонам из ZALPHA_DIR =====
    def autotune_against_zalpha(src: Path, out_path: Path, dbg_dir: Optional[str], params: dict):
        """Если в ZALPHA_DIR есть изображение с таким же именем — перебираем radius_factor и expand, выбираем лучший по IoU."""
        try:
            zalpha_file = ZALPHA_DIR / src.name
            if not zalpha_file.exists():
                return params, False
            zalpha = cv2_imread_unicode(zalpha_file, cv2.IMREAD_UNCHANGED)
            if zalpha is None or zalpha.ndim < 3 or zalpha.shape[2] < 4:
                return params, False
            zalpha_a = (zalpha[:, :, 3] > 128).astype(np.uint8)

            best = (None, -1.0, None)
            radius_candidates = [0.04, 0.06, 0.08, 0.10, 0.14]
            expand_candidates = [0, 1, 2, 3, 4]
            # корректно извлекаем white_thresh_val
            wtv = params.get("white_thresh_val", None)
            if isinstance(wtv, str):
                try:
                    wtv = int(wtv)
                except Exception:
                    wtv = None
            for rf in radius_candidates:
                for ex in expand_candidates:
                    ok_t, status_t, tpix_t, params_t = remove_bg_pure_white(
                        src, out_path, white_thresh=wtv, expand=ex, erode_px=params.get("erode_px", None), blur_mask=params.get("blur_mask", None), save_debug=False, radius_factor=rf
                    )
                    if not ok_t:
                        continue
                    res = cv2_imread_unicode(out_path, cv2.IMREAD_UNCHANGED)
                    if res is None or res.ndim < 3 or res.shape[2] < 4:
                        continue
                    a = (res[:, :, 3] > 128).astype(np.uint8)
                    inter = int(np.sum((a & zalpha_a) > 0))
                    union = int(np.sum((a | zalpha_a) > 0))
                    iou = float(inter) / union if union > 0 else 0.0
                    if iou > best[1]:
                        best = (params_t, iou, res)
            if best[0] is not None and best[1] > 0.75:
                params.update(best[0])
                if dbg_dir:
                    try:
                        with open(Path(dbg_dir) / f"{src.stem}_autotune_iou.txt", "w", encoding="utf-8") as f:
                            f.write(f"IoU={best[1]:.4f}\n")
                    except Exception:
                        pass
                return params, True
        except Exception:
            return params, False
        return params, False

    # если есть эталон — применим автотюнинг
    try:
        if ZALPHA_DIR.exists() and ZALPHA_DIR.is_dir():
            params, tuned = autotune_against_zalpha(src, alpha_path, dbg_dir, params)
            if tuned:
                _detail("autotune", tuned=1)
            else:
                _detail("autotune", tuned=0)
    except Exception:
        pass

    # Контуры/ЧБ (аудит, не влияет на статус)
    if STEPS["edges"]:
        t_edges = time.perf_counter()
        e_ok = edges_canny(src, OUT_DIRS["edges"] / name)
        _detail("edges", status=("ok" if e_ok else "fail"))
        _detail("timing_edges", ms=int(1000 * (time.perf_counter() - t_edges)))
    if STEPS["bw"]:
        t_bw = time.perf_counter()
        b_ok = bw_threshold(src, OUT_DIRS["bw"] / name, invert=False)
        _detail("bw", status=("ok" if b_ok else "fail"))
        _detail("timing_bw", ms=int(1000 * (time.perf_counter() - t_bw)))
    if STEPS["bw_inv"]:
        t_bwi = time.perf_counter()
        bi_ok = bw_threshold(src, OUT_DIRS["bw_inv"] / name, invert=True)
        _detail("bw_inv", status=("ok" if bi_ok else "fail"))
        _detail("timing_bw_inv", ms=int(1000 * (time.perf_counter() - t_bwi)))

    # CSV аудит (детали + сводка)
    audit_write_details(detail_rows)
    audit_write_summary(
        {
            "timestamp": ts,
            "file": name,
            "status": status,
            "mode": BG_MODE,
            "transparent_pixels": tpix,
            "w": meta["w"],
            "h": meta["h"],
            "white_thresh": safe_param(params, "white_thresh"),
            "expand": safe_param(params, "expand"),
            "erode_px": safe_param(params, "erode_px"),
            "blur_mask": safe_param(params, "blur_mask"),
            "alpha_path": str(alpha_path),
            "debug_dir": params.get("debug_dir", ""),
        }
    )

    _detail("timing_total", ms=int(1000 * (time.perf_counter() - t0)))
    return ok


# ======= ОСНОВНОЙ ЗАПУСК =======
if __name__ == "__main__":
    # Папки
    for d in OUT_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    trace_env_header()

    manifest = load_manifest()
    not_alpha_set = load_not_alpha_list()

    processed_count = 0
    skipped_count = 0

    all_files = sorted(SRC_DIR.glob("*.png"))
    log.info(f"[input] найдено файлов: {len(all_files)}")

    # Префильтр: off | dry | block
    to_process, groups_info = apply_prefilter(all_files, PREFILTER_MODE, PREFILTER_GROUPS)
    log.info(f"[prefilter] режим={PREFILTER_MODE} -> к обработке: {len(to_process)}")

    for file in to_process:
        # Пропуск из not_alpha.json
        if file.name in not_alpha_set:
            log.info(f"[skip] {file.name} в not_alpha.json")
            skipped_count += 1
            continue

        # Манифест (хэш)
        md5_now = file_md5(file)
        if manifest.get(file.name) == md5_now:
            log.info(f"[skip] {file.name} без изменений")
            skipped_count += 1
            continue

        # Обработка
        if process_one(file):
            manifest[file.name] = md5_now
            processed_count += 1

    save_manifest(manifest)
    log.info(f"Готово. Новых/обновлённых: {processed_count}, пропущено: {skipped_count}")
