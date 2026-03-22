import hashlib
import json
import logging
import os
import sys

import cv2
import numpy as np
from config_bot import (
    PREP_LOG,
    TPL_ALPHA,
    TPL_BW,
    TPL_BW_INV,
    TPL_DIR,
    TPL_EDGES,
    TPL_MY,
)
from PIL import Image

os.makedirs(TPL_ALPHA, exist_ok=True)
os.makedirs(TPL_EDGES, exist_ok=True)
os.makedirs(TPL_BW, exist_ok=True)
os.makedirs(TPL_BW_INV, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(PREP_LOG, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("preprocess")
MANIFEST_PATH = os.path.join(TPL_DIR, ".preprocess_manifest.json")


def file_md5(path):
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_manifest(m):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)


def remove_white_bg_to_alpha(src_path, dst_path):
    img = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if img is None:
        return False
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
    b, g, r = cv2.split(img)
    a = cv2.bitwise_not(mask)
    rgba = cv2.merge((b, g, r, a))
    cv2.imwrite(dst_path, rgba)
    return True


def edges_canny(src_path, dst_path):
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    edges = cv2.Canny(img, 80, 150)
    cv2.imwrite(dst_path, edges)
    return True


def bw_threshold(src_path, dst_path, invert=False):
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert:
        th = cv2.bitwise_not(th)
    cv2.imwrite(dst_path, th)
    return True


def process_one(src):
    name = os.path.basename(src)
    alpha_dst = os.path.join(TPL_ALPHA, name)
    edges_dst = os.path.join(TPL_EDGES, name)
    bw_dst = os.path.join(TPL_BW, name)
    inv_dst = os.path.join(TPL_BW_INV, name)
    ok1 = remove_white_bg_to_alpha(src, alpha_dst)
    ok2 = edges_canny(src, edges_dst)
    ok3 = bw_threshold(src, bw_dst, invert=False)
    ok4 = bw_threshold(src, inv_dst, invert=True)
    return ok1 and ok2 and ok3 and ok4


def main():
    manifest = load_manifest()
    changed = 0
    for fname in os.listdir(TPL_MY):
        if not fname.lower().endswith(".png"):
            continue
        src = os.path.join(TPL_MY, fname)
        h = file_md5(src)
        if manifest.get(fname) == h:
            log.info(f"[skip] Уже обработан: {fname}")
            continue
        log.info(f"[do] Обрабатываю: {fname}")
        ok = process_one(src)
        if ok:
            manifest[fname] = h
            changed += 1
        else:
            log.warning(f"[warn] Проблема при обработке: {fname}")
    save_manifest(manifest)
    log.info(f"Готово. Новых/обновлённых файлов: {changed}")


if __name__ == "__main__":
    main()
