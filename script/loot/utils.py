# -*- coding: utf-8 -*-
"""
Центральный модуль утилит бота.
Единственный источник истины для: путей, логирования, imread, clamp,
match_scaled, to_gray, scharr_mag, rect_from_rel, ui_tpl_path.
"""

import datetime
import json
import os

import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

# ── Корневой путь проекта ────────────────────────────────────────────
BASE_DIR = "C:/bot"


def bot_root():
    """
    Определяет корень проекта (C:\\bot) по расположению utils.py.
    Возвращает путь, где есть папка tpl/templates и tools/cfg.
    """
    here = os.path.abspath(os.path.dirname(__file__))  # .../script/loot
    candidates = [
        os.path.abspath(os.path.join(here, "..", "..")),  # C:\bot
        os.path.abspath(os.path.join(here, "..")),        # C:\bot\script
    ]
    for c in candidates:
        tpl_ok = os.path.isdir(os.path.join(c, "tpl", "имя_предметов"))
        cfg_ok = os.path.isfile(os.path.join(c, "tools", "cfg", "config.json"))
        if tpl_ok and (cfg_ok or os.path.isdir(os.path.join(c, "tools", "cfg"))):
            return c
    return os.path.abspath(os.path.join(here, "..", ".."))


# ── Пути ─────────────────────────────────────────────────────────────

def path_cfg():
    return os.path.join(bot_root(), "tools", "cfg", "config.json")


def path_log():
    return os.path.join(bot_root(), "debug", "log", "run_log.txt")


def path_tpl_my():
    return os.path.join(bot_root(), "tpl", "имя_предметов")


def path_tpl_chevrons():
    return os.path.join(bot_root(), "tpl", "служебные")


# ── Каталоги ─────────────────────────────────────────────────────────

def ensure_dirs():
    os.makedirs(os.path.dirname(path_log()), exist_ok=True)


# ── Логгер ───────────────────────────────────────────────────────────

class Logger:
    @staticmethod
    def ts():
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # type: ignore[index]

    @staticmethod
    def log(msg):
        ensure_dirs()
        line = f"[{Logger.ts()}] {msg}"
        print(line)
        with open(path_log(), "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ── Базовые утилиты ──────────────────────────────────────────────────

def clamp(v, lo, hi):
    """Ограничить v в диапазон [lo, hi]."""
    return max(lo, min(hi, v))


def read_config():
    cfg_path = path_cfg()
    if not os.path.isfile(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

_CFG_CACHE = None
def get_thr(name: str, default: float) -> float:
    """Получает порог из секции THRESHOLDS в config.json."""
    global _CFG_CACHE
    if _CFG_CACHE is None:
        _CFG_CACHE = read_config()
    return float(_CFG_CACHE.get("THRESHOLDS", {}).get(name, default))


# ── Чтение изображений ──────────────────────────────────────────────

def np_imread_unicode(path):
    """Чтение изображения через numpy (поддержка Unicode путей)."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def try_imread(path):
    try:
        return np_imread_unicode(path)
    except Exception as e:
        Logger.log(f"ERROR imread: {path} -> {e}")
        return None


def imread_u8(path, flags=cv2.IMREAD_COLOR):
    """
    Чтение через open()+imdecode (поддержка кириллицы).
    Логирует ошибки, возвращает None при неудаче.
    """
    try:
        if not os.path.exists(path):
            print(f"[ERROR] imread_u8 file not found: {path!r}")
            return None
        with open(path, "rb") as f:
            data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, flags)
        if img is None:
            print(f"[ERROR] imread_u8 failed to decode: {path!r}")
        return img
    except Exception as e:
        print(f"[ERROR] imread_u8 exception: {path!r} — {e}")
        return None


# ── Работа со списками файлов ────────────────────────────────────────

def list_pngs(folder: str):
    if not os.path.isdir(folder):
        return []
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".png") and os.path.isfile(os.path.join(folder, f))
    ]


def load_first_existing(folder: str, names):
    for name in names:
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            img = try_imread(p)
            if img is not None:
                return img, p
    return None, None


# ── Шаблонный resolver ───────────────────────────────────────────────

def ui_tpl_path(name: str) -> str:
    """
    Единый resolver пути к UI-шаблону по имени (без расширения).
    Сначала пробует tpl_loader.ui_path, потом fallback в tpl/служебные.
    """
    try:
        from script.loot import tpl_loader as TL  # type: ignore[import-not-found]
        return TL.ui_path(name)
    except Exception:
        pass
    return os.path.join(BASE_DIR, "tpl", "служебные", f"{name}.png").replace("\\", "/")


# ── Препроцессинг изображений ────────────────────────────────────────

def to_gray(img: np.ndarray) -> np.ndarray:
    """Перевод в gray + bilateralFilter + CLAHE."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.bilateralFilter(g, 5, 35, 35)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    return g


def scharr_mag(gray: np.ndarray) -> np.ndarray:
    """Градиент Шарра (magnitude), нормализованный в 0..255."""
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# ── Template matching ────────────────────────────────────────────────

def match_scaled(gray_img, tpl_gray_full, threshold, scales=None):
    """
    Масштабированный template matching.
    Возвращает (found: bool, score: float, box: (x, y, w, h)).
    """
    if scales is None:
        scales = [0.9, 0.95, 1.0, 1.05, 1.1]
    best_ok = False
    best_score = 0.0
    best_box = (0, 0, 0, 0)
    th, tw = tpl_gray_full.shape[:2]
    H, W = gray_img.shape[:2]
    for s in scales:
        tws = max(5, int(tw * s))
        ths = max(5, int(th * s))
        if H < ths or W < tws:
            continue
        interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
        tpl_s = cv2.resize(tpl_gray_full, (tws, ths), interpolation=interp)
        res = cv2.matchTemplate(gray_img, tpl_s, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if float(max_val) > best_score:
            best_score = float(max_val)
            best_ok = best_score >= threshold
            best_box = (int(max_loc[0]), int(max_loc[1]), tws, ths)
    return best_ok, best_score, best_box


def rect_from_rel(W: int, H: int, rel):
    """Конвертация относительных координат [x1_rel, y1_rel, x2_rel, y2_rel] в пиксели."""
    return int(rel[0] * W), int(rel[1] * H), int(rel[2] * W), int(rel[3] * H)


# ── OpenCV safe wrappers ─────────────────────────────────────────────

def crop_roi(screen, roi):
    H, W = screen.shape[:2]
    x1, y1, x2, y2 = roi
    x1, x2 = clamp(x1, 0, W - 1), clamp(x2, 0, W)
    y1, y2 = clamp(y1, 0, H - 1), clamp(y2, 0, H)
    if x2 <= x1 or y2 <= y1:
        return None, None
    return screen[y1:y2, x1:x2], (x1, y1, x2, y2)


def expand_roi_to_fit(roi, tpl_w, tpl_h, W, H):
    x1, y1, x2, y2 = roi
    rw, rh = x2 - x1, y2 - y1
    need_w = max(0, tpl_w - rw)
    need_h = max(0, tpl_h - rh)
    if need_w == 0 and need_h == 0:
        return x1, y1, x2, y2
    x1 = clamp(x1 - need_w // 2 - 1, 0, W - 1)
    y1 = clamp(y1 - need_h // 2 - 1, 0, H - 1)
    x2 = clamp(x2 + (need_w - need_w // 2) + 1, 0, W)
    y2 = clamp(y2 + (need_h - need_h // 2) + 1, 0, H)
    return x1, y1, x2, y2


def match_template_safe(screen, tpl, roi, threshold=0.82, method=cv2.TM_CCOEFF_NORMED):
    if screen is None or tpl is None or roi is None:
        return None
    H, W = screen.shape[:2]
    th, tw = tpl.shape[:2]
    roi = expand_roi_to_fit(roi, tw, th, W, H)
    roi_img, roi = crop_roi(screen, roi)
    if roi_img is None or roi is None:
        return None
    rh, rw = roi_img.shape[:2]
    if rh < th or rw < tw:
        return None
    res = cv2.matchTemplate(roi_img, tpl, method)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < threshold:
        return None
    rx1, ry1, rx2, ry2 = roi
    top_left = rx1 + max_loc[0], ry1 + max_loc[1]
    center = top_left[0] + tw // 2, top_left[1] + th // 2
    return {"pt": top_left, "center": center, "score": max_val, "tpl_size": (tw, th), "roi": roi}


# ── Метки времени ────────────────────────────────────────────────────

def full_timestamp():
    return datetime.datetime.now().isoformat(timespec="milliseconds")


def short_time_tag(include_seconds=False):
    fmt = "%H%M%S" if include_seconds else "%H%M"
    return datetime.datetime.now().strftime(fmt)


# ── Хеширование кадров ──────────────────────────────────────────────

def frame_hash(gray: np.ndarray) -> int:
    """
    Perceptual hash кадра (dHash, 8×8 → 64 bits).
    Оптимизировано через np.packbits (план: пункт 9).
    """
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = np.packbits(diff.flatten(), bitorder='little')
    return int.from_bytes(bits.tobytes(), byteorder='little')


def page_hash_from_bgr(frame_bgr: np.ndarray) -> int:
    """Хеш страницы из BGR-кадра."""
    return frame_hash(to_gray(frame_bgr))


def detect_chat_top_y(frame_bgr: np.ndarray) -> int:
    """Детектирует верхнюю границу системного чата; возвращает Y-координату."""
    syschat_path = ui_tpl_path("syschat")
    H, W = frame_bgr.shape[:2]
    tpl = imread_u8(syschat_path, cv2.IMREAD_COLOR)
    if tpl is not None:
        ok, score, (x, y, w, h) = match_scaled(
            to_gray(frame_bgr), to_gray(tpl), 0.85, [0.9, 1.0, 1.1]
        )
        if ok:
            return y
    return int(H * 0.88)
