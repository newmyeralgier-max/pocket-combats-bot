import json
import os
import shutil

import cv2
import numpy as np

# ===== НАСТРОЙКИ =====
SRC_DIR = r"c:\bot\tpl\монстры"  # Папка-источник
DST_DIR = r"c:\bot\tpl\монстры_full"  # Папка-назначение
JSON_LIST = r"c:\bot\tools\cfg\not_монстры_full.json"  # JSON со списком недостающих файлов

MIN_WIDTH = 800  # Минимальная ширина для копирования
CHECK_WIDTH = True  # True — проверять ширину, False — игнорировать
IGNORE_PREFIXES = ["_diag"]  # Префиксы, которые пропускаем
COPY_ONLY_FORMAT = "all"  # "all" или, например, "png", "jpg"


# ===== Unicode I/O для Windows =====
def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


# ===== ЛОГИКА =====
def copy_missing():
    if not os.path.isdir(SRC_DIR):
        print(f"[ERR] Папка-источник не найдена: {SRC_DIR}")
        return
    os.makedirs(DST_DIR, exist_ok=True)

    # Загружаем список недостающих файлов
    try:
        with open(JSON_LIST, "r", encoding="utf-8") as f:
            missing_files = set(json.load(f))
    except Exception as e:
        print(f"[ERR] Не удалось прочитать JSON: {e}")
        return

    copied = 0
    skipped = 0

    for fname in missing_files:
        src_path = os.path.join(SRC_DIR, fname)
        if not os.path.isfile(src_path):
            print(f"[MISS] Нет в источнике: {fname}")
            skipped += 1
            continue

        # Игнор по префиксам
        if any(fname.lower().startswith(pref.lower()) for pref in IGNORE_PREFIXES):
            print(f"[SKIP] {fname} — префикс в списке игнора")
            skipped += 1
            continue

        # Игнор по формату
        if COPY_ONLY_FORMAT.lower() != "all":
            if not fname.lower().endswith("." + COPY_ONLY_FORMAT.lower()):
                print(f"[SKIP] {fname} — не тот формат")
                skipped += 1
                continue

        # Проверка ширины
        if CHECK_WIDTH:
            img = imread_unicode(src_path)
            if img is None:
                print(f"[FAIL] {fname} — не удалось прочитать для проверки ширины")
                skipped += 1
                continue
            h, w = img.shape[:2]
            if w < MIN_WIDTH:
                print(f"[SKIP] {fname} — ширина {w}px < {MIN_WIDTH}px")
                skipped += 1
                continue

        # Копирование
        dst_path = os.path.join(DST_DIR, fname)
        try:
            shutil.copy2(src_path, dst_path)
            print(f"[COPY] {fname} → {dst_path}")
            copied += 1
        except Exception as e:
            print(f"[ERR] {fname} — ошибка копирования: {e}")
            skipped += 1

    print(f"\n[DONE] Скопировано: {copied}, Пропущено: {skipped}, Всего в списке: {len(missing_files)}")


# ===== ЗАПУСК =====
if __name__ == "__main__":
    copy_missing()
