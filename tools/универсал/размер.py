import os

import cv2
import numpy as np

# Папка с иконками
icons_dir = r"C:\bot\tpl\имя_предметов"  # <-- поменяй на свою

min_w = None
min_h = None
min_file = None

for root, _, files in os.walk(icons_dir):
    for fname in files:
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(root, fname)
            try:
                data = np.fromfile(path, dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            except Exception as e:
                print(f"[ERR] {path}: {e}")
                continue

            if img is None:
                print(f"[WARN] Не удалось открыть {path}")
                continue

            h, w = img.shape[:2]
            if min_w is None or w < min_w or h < min_h:
                min_w = w if min_w is None or w < min_w else min_w
                min_h = h if min_h is None or h < min_h else min_h
                min_file = path

print("=== РЕЗУЛЬТАТ ===")
if min_file:
    print(f"Самая маленькая иконка: {min_file}")
    print(f"Размер: {min_w} x {min_h}")
else:
    print("Иконки не найдены")
