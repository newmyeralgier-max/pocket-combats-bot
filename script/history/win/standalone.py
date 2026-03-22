# -*- coding: utf-8 -*-
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pytesseract

# ===== ТВОЯ КОНФИГА =====
IN_DIR = r"C:\bot\tpl\монстры"
OUT_DIR = r"C:\bot\out"
LOG_FILE = r"C:\bot\script\debug\log\live_log.txt"

TEMPLATES = {
    "icon": r"C:\bot\tpl\иконки предметов",
    "stats": r"C:\bot\tpl\характеристики",
    "title": r"C:\bot\tpl\templates",
}

# Порог совпадения (чуть ослаблен под «разношерстные»)
TEMPLATE_THR_ICON = 0.70
TEMPLATE_THR_STATS = 0.72
TEMPLATE_THR_TITLE = 0.80  # названия обычно точные; можно ослабить до 0.75

# OCR
OCR_LANG = "rus"


# ===== УТИЛИТЫ: чтение/запись с кириллицей =====
def imread_unicode(path: Path, flags=cv2.IMREAD_COLOR):
    arr = np.fromfile(str(path), dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, flags)


def imwrite_unicode(path: Path, img) -> bool:
    ensure_dir(path.parent)
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def list_images(dir_or_file: Path):
    p = Path(dir_or_file)
    if p.is_file():
        return [p]
    if not p.exists():
        return []
    files = []
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.tif", "*.tiff")
    for ext in exts:
        files.extend(p.glob(ext))
    return sorted(files)


# ===== ЛОГГЕР =====
class LiveLogger:
    def __init__(self, path: Path):
        ensure_dir(path.parent)
        if path.exists():
            try:
                path.replace(path.with_suffix(path.suffix + ".prev"))
            except:
                pass
        self.f = open(path, "w", encoding="utf-8", buffering=1)

    def log(self, msg: str):
        t = time.strftime("%H:%M:%S")
        line = f"[{t}] {msg}"
        print(line, flush=True)
        try:
            self.f.write(line + "\n")
            self.f.flush()
        except:
            pass

    def close(self):
        try:
            self.f.close()
        except:
            pass


LOGGER = LiveLogger(Path(LOG_FILE))


# ===== НОРМАЛИЗАЦИЯ ИМЕНИ ПАПКИ (сохраняем кириллицу) =====
def clean_fs_name(s: str) -> str:
    # Удаляем недопустимые символы, пробелы триммим
    s = s.strip()
    s = s.replace(":", "․").replace("|", "¦").replace("/", "／").replace("\\", "＼")
    s = s.replace("?", "？").replace("*", "＊").replace("<", "‹").replace(">", "›")
    return s or "элемент"


# ===== OCR заголовка =====
def ocr_title(img_bgr):
    h, w = img_bgr.shape[:2]
    # Верхняя треть — там обычно заголовок
    x, y, ww, hh = 0, 0, w, int(h * 0.33)
    roi = img_bgr[y : y + hh, x : x + ww]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    txt = pytesseract.image_to_string(gray, lang=OCR_LANG) or ""
    clean = " ".join(txt.split())
    # Возвращаем текст и bbox в координатах исходника
    return (clean if clean else None), (x, y, ww, hh)


# ===== Template matching по набору эталонов =====
def match_templates(img_bgr, tmpl_dir_or_file: Path, thr=0.75, restrict_roi=None):
    """
    img_bgr: исходное BGR
    tmpl_dir_or_file: папка или файл-шаблон(ы)
    restrict_roi: (x, y, w, h) — ограничить область поиска по исходному
    Возвращает лучшую находку: dict(x,y,w,h,score,template) или None
    """
    H, W = img_bgr.shape[:2]
    if restrict_roi:
        rx, ry, rw, rh = restrict_roi
        rx2, ry2 = min(W, rx + rw), min(H, ry + rh)
        rx, ry = max(0, rx), max(0, ry)
        if rx >= rx2 or ry >= ry2:
            search = img_bgr
            offx, offy = 0, 0
        else:
            search = img_bgr[ry:ry2, rx:rx2]
            offx, offy = rx, ry
    else:
        search = img_bgr
        offx, offy = 0, 0

    best = None
    tmpl_list = list_images(tmpl_dir_or_file)
    for tpath in tmpl_list:
        tpl = imread_unicode(tpath, cv2.IMREAD_COLOR)
        if tpl is None:
            LOGGER.log(f"[tpl] пропуск (не прочёлся): {tpath}")
            continue
        th, tw = tpl.shape[:2]
        # Набор масштабов
        scales = [1.0, 0.9, 1.1, 0.8, 1.2, 1.3, 0.7]
        for s in scales:
            tw2, th2 = int(tw * s), int(th * s)
            if tw2 < 8 or th2 < 8:
                continue
            if tw2 >= search.shape[1] or th2 >= search.shape[0]:
                continue
            tpl_s = cv2.resize(tpl, (tw2, th2), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(search, tpl_s, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val >= thr:
                cand = {
                    "x": int(offx + max_loc[0]),
                    "y": int(offy + max_loc[1]),
                    "w": int(tw2),
                    "h": int(th2),
                    "score": float(max_val),
                    "template": str(tpath.name),
                }
                if best is None or cand["score"] > best["score"]:
                    best = cand
    return best


# ===== Основной проход =====
def main():
    try:
        ensure_dir(Path(OUT_DIR))
        files = list_images(Path(IN_DIR))
        LOGGER.log(f"Входных файлов: {len(files)} в {IN_DIR}")
        if not files:
            LOGGER.log("Пусто на входе. Проверь папку C:\\bot\\tpl\\монстры")
            return

        # Сколько эталонов в папках
        for k, p in TEMPLATES.items():
            cnt = len(list_images(Path(p)))
            LOGGER.log(f"Шаблонов {k}: {cnt} из {p}")

        report = {}
        for fp in files:
            name_raw = Path(fp).stem
            img = imread_unicode(fp)
            if img is None:
                LOGGER.log(f"{name_raw}: [ERR] не открылся (путь/кодировка/битый файл)")
                continue

            H, W = img.shape[:2]
            dbg = img.copy()
            report[name_raw] = {}

            # 1) OCR названия
            title_text, title_box = ocr_title(img)
            if title_text:
                folder_name = clean_fs_name(title_text)
                LOGGER.log(f"{name_raw}: title OCR OK -> '{title_text}' -> папка '{folder_name}'")
            else:
                folder_name = clean_fs_name(name_raw)
                LOGGER.log(f"{name_raw}: title OCR MISS -> папка по имени файла '{folder_name}'")

            out_mon = Path(OUT_DIR) / folder_name
            ensure_dir(out_mon)

            # Доп. попытка: матчинговый поиск заголовка по templates (в верхней зоне)
            title_match = match_templates(
                img, Path(TEMPLATES["title"]), thr=TEMPLATE_THR_TITLE, restrict_roi=(0, 0, W, int(H * 0.5))
            )
            # Сохраняем title ROI: приоритет — OCR зона, иначе матчинговая
            if title_text:
                x, y, w, h = title_box
                title_roi = img[y : y + h, x : x + w]
                imwrite_unicode(out_mon / "title.png", title_roi)
                cv2.rectangle(dbg, (x, y), (x + w, y + h), (255, 0, 0), 2)
                report[name_raw]["title"] = {"found": True, "text": title_text, "x": x, "y": y, "w": w, "h": h}
            elif title_match:
                x, y, w, h = title_match["x"], title_match["y"], title_match["w"], title_match["h"]
                title_roi = img[y : y + h, x : x + w]
                imwrite_unicode(out_mon / "title.png", title_roi)
                cv2.rectangle(dbg, (x, y), (x + w, y + h), (255, 0, 255), 2)
                report[name_raw]["title"] = {"found": True, "by": "template", **title_match}
                LOGGER.log(
                    f"{name_raw}: title template OK score={title_match['score']:.3f} ({title_match['template']})"
                )
            else:
                report[name_raw]["title"] = {"found": False, "reason": "ocr_and_template_miss"}
                LOGGER.log(f"{name_raw}: title окончательно MISS")

            # 2) ICON
            res_icon = match_templates(img, Path(TEMPLATES["icon"]), thr=TEMPLATE_THR_ICON)
            if res_icon:
                xi, yi, wi, hi = res_icon["x"], res_icon["y"], res_icon["w"], res_icon["h"]
                icon_crop = img[yi : yi + hi, xi : xi + wi]
                imwrite_unicode(out_mon / "icon.png", icon_crop)
                cv2.rectangle(dbg, (xi, yi), (xi + wi, yi + hi), (0, 255, 0), 2)
                report[name_raw]["icon"] = {"found": True, **res_icon}
                LOGGER.log(f"{name_raw}: icon OK score={res_icon['score']:.3f}")
            else:
                report[name_raw]["icon"] = {"found": False, "reason": "no_match"}
                LOGGER.log(f"{name_raw}: icon MISS")

            # 3) STATS
            res_stats = match_templates(img, Path(TEMPLATES["stats"]), thr=TEMPLATE_THR_STATS)
            if res_stats:
                xs, ys, ws, hs = res_stats["x"], res_stats["y"], res_stats["w"], res_stats["h"]
                stats_crop = img[ys : ys + hs, xs : xs + ws]
                imwrite_unicode(out_mon / "stats.png", stats_crop)
                cv2.rectangle(dbg, (xs, ys), (xs + ws, ys + hs), (0, 255, 255), 2)
                report[name_raw]["stats"] = {"found": True, **res_stats}
                LOGGER.log(f"{name_raw}: stats OK score={res_stats['score']:.3f}")
            else:
                report[name_raw]["stats"] = {"found": False, "reason": "no_match"}
                LOGGER.log(f"{name_raw}: stats MISS")

            # 4) DIAG
            imwrite_unicode(out_mon / "diag.png", dbg)
            LOGGER.log(f"{name_raw}: diag сохранён -> {out_mon}\\diag.png\n")

        # Итоговый отчёт
        report_path = Path(OUT_DIR) / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        LOGGER.log(f"Готово. Отчёт: {report_path}")
    finally:
        LOGGER.close()


if __name__ == "__main__":
    main()
