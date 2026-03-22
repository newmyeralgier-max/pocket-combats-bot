# -*- coding: utf-8 -*-
import json
import re
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pytesseract

# ======= ПУТИ =======
INPUT_DIR = Path(r"C:\bot\tpl\монстры_full")
TARGETS_JSON = Path(r"C:\bot\tools\cfg\not_full.json")
OUTPUT_DIR = Path(r"C:\bot\характеристики2")
DEBUG_DIR = Path(r"C:\bot\характеристики2_debug")

# ======= ПАРАМЕТРЫ =======
SEPARATOR_RGB = 225
SEPARATOR_TOL = 14
ROW_COVER_FRAC = 0.65
MIN_BAND_PX = 1

TITLE_SLICE_FRAC = 0.5
TITLE_SLICE_MAX = 250
TITLE_SLICE_MIN = 32

STOP_WORDS = {
    "маг",
    "воин",
    "требуется",
    "уровень",
    "масса",
    "гнёзда",
    "броня",
    "защита",
    "шанс",
    "выпадения",
    "не",
    "подходит",
    "профессия",
}


# ======= IO =======
def imread_u(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_u(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    if ok:
        with open(path, "wb") as f:
            buf.tofile(f)


# ======= НОРМАЛИЗАЦИЯ =======
LAT2CYR = str.maketrans(
    {
        "a": "а",
        "b": "Ь",
        "c": "с",
        "e": "е",
        "h": "һ",
        "k": "к",
        "m": "м",
        "o": "о",
        "p": "р",
        "s": "ѕ",
        "t": "т",
        "x": "х",
        "y": "у",
        "A": "а",
        "B": "Ь",
        "C": "с",
        "E": "е",
        "H": "һ",
        "K": "к",
        "M": "м",
        "O": "о",
        "P": "р",
        "S": "ѕ",
        "T": "т",
        "X": "х",
        "Y": "у",
        "u": "и",
        "U": "и",
        "v": "у",
        "V": "у",
    }
)
DIGIT2CYR = str.maketrans(
    {
        "0": "о",
        "3": "з",
        "4": "ч",
        "6": "б",
        "8": "в",
    }
)


def normalize_soft(s):
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("ё", "е")
    s = s.translate(LAT2CYR)
    s = s.translate(DIGIT2CYR)
    s = s.lower()
    s = re.sub(r"[^0-9a-zа-я _\-]", " ", s)
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s):
    return [w for w in re.split(r"[_\s]+", s) if w]


def clean_title(text):
    # первая непустая строка
    first_line = next((l for l in text.splitlines() if l.strip()), "")
    tokens = [t for t in tokenize(normalize_soft(first_line)) if t not in STOP_WORDS]
    return tokens


# ======= ЗАГРУЗКА ЦЕЛЕЙ =======
def load_targets_list(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping = {}
    for raw in data:
        if isinstance(raw, str):
            tokens = [t for t in tokenize(normalize_soft(raw)) if t not in STOP_WORDS]
            mapping["_".join(tokens)] = raw
    return mapping


# ======= ПОИСК РАЗДЕЛИТЕЛЕЙ =======
def find_separator_y_positions(img_bgr):
    H, W = img_bgr.shape[:2]
    b, g, r = cv2.split(img_bgr)
    m_all = (
        (np.abs(b.astype(np.int16) - SEPARATOR_RGB) <= SEPARATOR_TOL)
        & (np.abs(g.astype(np.int16) - SEPARATOR_RGB) <= SEPARATOR_TOL)
        & (np.abs(r.astype(np.int16) - SEPARATOR_RGB) <= SEPARATOR_TOL)
    ).astype(np.uint8)
    row_frac = m_all.sum(axis=1) / float(W)
    rows_bin = (row_frac >= ROW_COVER_FRAC).astype(np.uint8)
    ys, in_band, start = [], False, 0
    for y in range(H):
        if rows_bin[y] and not in_band:
            in_band, start = True, y
        elif not rows_bin[y] and in_band:
            in_band = False
            if (y - start) >= MIN_BAND_PX:
                ys.append((start + y) // 2)
    if in_band and (H - start) >= MIN_BAND_PX:
        ys.append((start + H) // 2)
    dedup = []
    for y in sorted(ys):
        if not dedup or abs(y - dedup[-1]) > 2:
            dedup.append(y)
    return dedup


def blocks_from_separators(H, sep_ys):
    cuts = [0] + sep_ys + [H]
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] - cuts[i] >= 12]


# ======= OCR =======
def preprocess_for_ocr(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(bw) > 127:
        bw = 255 - bw
    return bw


def try_ocr(img_bgr, psm):
    bw = preprocess_for_ocr(img_bgr)
    txt = pytesseract.image_to_string(bw, lang="rus", config="--oem 3 --psm %d" % psm)
    return txt


def ocr_block_title(block):
    H = block.shape[0]
    h_title = int(max(TITLE_SLICE_MIN, min(TITLE_SLICE_MAX, TITLE_SLICE_FRAC * H)))
    roi_top = block[0:h_title, :]
    for roi, psm in [(roi_top, 6), (block, 6), (roi_top, 7), (block, 7)]:
        t = try_ocr(roi, psm)
        if t.strip():
            return t
    return ""


# ======= ОБРАБОТКА =======
def process_image(path, targets_map):
    img = imread_u(path)
    if img is None:
        print("[ERR] Не удалось открыть %s" % path)
        return 0

    sep_ys = find_separator_y_positions(img)
    blocks = blocks_from_separators(img.shape[0], sep_ys)
    overlay = img.copy()
    saved = 0

    for idx, (y0, y1) in enumerate(blocks):
        block = img[y0:y1, :]
        ocr_text = ocr_block_title(block)
        ocr_tokens = clean_title(ocr_text)
        key_str = "_".join(ocr_tokens)

        color = (0, 0, 255)
        label = key_str or "?"

        if key_str in targets_map:
            save_name = "%s_full.png" % targets_map[key_str]
            imwrite_u(OUTPUT_DIR / save_name, block)
            saved += 1
            color = (0, 200, 0)
            label = targets_map[key_str]

        cv2.rectangle(overlay, (0, y0), (img.shape[1] - 1, y1), color, 2)
        cv2.putText(overlay, label[:48], (6, max(18, y0 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    for y in sep_ys:
        cv2.line(overlay, (0, y), (img.shape[1] - 1, y), (255, 0, 0), 1)

        # Сохраняем отладочный оверлей
    imwrite_u(DEBUG_DIR / ("%s__overlay.png" % path.stem), overlay)
    print("[OK] %s: сохранено %d/%d" % (path.name, saved, len(blocks)))
    return saved


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    targets_map = load_targets_list(TARGETS_JSON)
    print("[INFO] Загружено %d целевых имён" % len(targets_map))

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    sources = [p for p in sorted(INPUT_DIR.rglob("*")) if p.suffix.lower() in exts]

    total_saved = 0
    for p in sources:
        total_saved += process_image(p, targets_map)

    print("[DONE] Всего сохранено: %d" % total_saved)


if __name__ == "__main__":
    main()
