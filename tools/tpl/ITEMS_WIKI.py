# pc_wiki_export_unified.py
# -*- coding: utf-8 -*-
"""
Выгрузчик предметов Pocket Combats Wiki — иконки и JSON.
- Сохраняет иконки в C:\bot\tpl\templates\icons
- Сохраняет JSON в C:\bot\tpl\templates\data и общий index.json
- Имена файлов: пробелы -> "_" ; если есть большой скриншот/JPG -> сохраняем <name>_full.png
  а иконку (если есть) сохраняем <name>.png
- Конвертирует загруженные изображения в PNG.
"""


import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup
from PIL import Image
from requests.utils import requote_uri

# ========== Настройки ==========
API_ROOT = "https://pocketcombats.fandom.com/ru/api.php"
MAIN_PAGE = "Предметы_и_лут"
OUT_BASE = Path(r"C:\bot\tpl\templates")
ICONS_DIR = OUT_BASE / "icons"
DATA_DIR = OUT_BASE / "data"
INDEX_PATH = OUT_BASE / "index.json"


HEADERS = {"User-Agent": "pc-wiki-export-unified/1.0 (+https://example.com)"}
REQUEST_DELAY = 0.35  # пауза между запросами (сек)
MIN_FULL_WIDTH = 300  # порог ширины, при котором файл считается 'full' (screenshot/large)


# ========== Утилиты ==========
def ensure_dirs():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    ICONS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_name(s: str) -> str:
    # Заменяет запрещённые символы и пробелы -> подчёркивания
    if not s:
        s = "item"
    s = s.strip()
    # replace whitespace -> underscore
    s = re.sub(r"\s+", "_", s)
    # remove invalid filename chars
    s = re.sub(r'[\\/:*?"<>|]', "_", s)
    return s[:160]


def api_get(params: dict) -> dict:
    p = dict(params)
    p.setdefault("format", "json")
    r = requests.get(API_ROOT, params=p, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def polite_sleep():
    time.sleep(REQUEST_DELAY)


def safe_download_to_temp(url: str, referer: Optional[str], dst_temp: Path) -> bool:
    """Скачивает URL в временный файл, безопасно кодируя URL/Referer."""
    try:
        url_safe = requote_uri(url)
    except Exception:
        url_safe = url
    headers = dict(HEADERS)
    if referer:
        try:
            headers["Referer"] = requote_uri(referer)
        except Exception:
            headers["Referer"] = referer
    try:
        with requests.get(url_safe, headers=headers, stream=True, timeout=40) as r:
            r.raise_for_status()
            with open(dst_temp, "wb") as f:
                for chunk in r.iter_content(8192):
                    if not chunk:
                        break
                    f.write(chunk)
        return True
    except Exception as e:
        print("    [download error]", e)
        return False


def convert_to_png(src: Path, dst: Path) -> bool:
    """Открывает изображение и сохраняет как PNG (перезаписывает, если нужно)."""
    try:
        im = Image.open(src).convert("RGBA")
        im.save(dst, format="PNG")
        return True
    except Exception as e:
        print("    [convert error]", e)
        return False


# ========== Логика выбора изображений ==========
def get_pages_from_main(page_title: str) -> List[str]:
    titles = []
    cont = {}
    while True:
        params = {"action": "query", "prop": "links", "titles": page_title, "pllimit": "max"}
        params.update(cont)
        res = api_get(params)
        pages = res.get("query", {}).get("pages", {})
        for p in pages.values():
            for l in p.get("links", []):
                t = l.get("title")
                if t and ":" not in t:
                    titles.append(t)
        if "continue" in res:
            cont = res["continue"]
            polite_sleep()
            continue
        break
    # уникальные сохраняем порядок
    seen = set()
    uniq = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def get_images_list(page_title: str) -> List[str]:
    """Возвращает список 'File:...' имён, если есть."""
    params = {"action": "query", "titles": page_title, "prop": "images", "imlimit": "max"}
    res = api_get(params)
    pages = res.get("query", {}).get("pages", {})
    files = []
    for p in pages.values():
        for im in p.get("images", []):
            name = im.get("title")
            if name:
                files.append(name)
    return files


def get_imageinfo(file_title: str, iiurlwidth: Optional[int] = None) -> Optional[Dict]:
    params = {"action": "query", "titles": file_title, "prop": "imageinfo", "iiprop": "url|size|mime"}
    if iiurlwidth:
        params["iiurlwidth"] = str(iiurlwidth)
    res = api_get(params)
    pages = res.get("query", {}).get("pages", {})
    for p in pages.values():
        ii = p.get("imageinfo")
        if ii and isinstance(ii, list):
            return ii[0]
    return None


# ========== Основной обработчик страницы ==========
def process_page(page_title: str) -> Dict:
    """
    Возвращает словарь item с полями:
    display_name, page_title, icon_path (основной .png), full_path (опционально), chosen_candidates...
    """
    print("Processing page:", page_title)
    polite_sleep()
    # Получаем HTML заголовка через parse, чтобы точнее получить display_name (H1)
    display_name = page_title
    try:
        res = api_get({"action": "parse", "page": page_title, "prop": "text"})
        display_html = res.get("parse", {}).get("text", {}).get("*")
        if display_html:
            soup = BeautifulSoup(display_html, "html.parser")
            h1 = soup.find("h1")
            if h1 and h1.get_text(strip=True):
                display_name = h1.get_text(strip=True)
    except Exception:
        # fallback keep page_title
        pass

    files = get_images_list(page_title)
    if not files:
        print("  no image files found")
        return {"page_title": page_title, "display_name": display_name}

    candidates = []
    for f in files:
        polite_sleep()
        info = get_imageinfo(f)
        if not info:
            continue
        url = info.get("url")
        mime = info.get("mime")
        width = info.get("width") or 0
        height = info.get("height") or 0
        size_bytes = info.get("size") or 0
        candidates.append(
            {"file_title": f, "url": url, "mime": mime, "width": width, "height": height, "size": size_bytes}
        )

        # if small width, try iiurlwidth to ask for larger (some MW return thumb even for imageinfo)
        if width and width < MIN_FULL_WIDTH:
            polite_sleep()
            info2 = get_imageinfo(f, iiurlwidth=2048)
            if info2:
                # override if returned
                url2 = info2.get("url")
                w2 = info2.get("width") or 0
                h2 = info2.get("height") or 0
                if url2:
                    candidates[-1].update({"url": url2, "width": w2, "height": h2})

    if not candidates:
        print("  no imageinfo candidates")
        return {"page_title": page_title, "display_name": display_name}

    # Выбираем самый широкий (largest width)
    candidates_sorted = sorted(candidates, key=lambda x: (x.get("width") or 0), reverse=True)
    best = candidates_sorted[0]
    # Также ищем первые PNG candidate (иконка) отдельно, если есть
    png_candidate = None
    for c in candidates_sorted:
        if c.get("mime") and "png" in c.get("mime").lower():
            png_candidate = c
            break

    # Определяем, считать ли best как full:
    is_full = False
    fname_lower = (best.get("file_title") or "").lower()
    if (best.get("width") or 0) >= MIN_FULL_WIDTH:
        is_full = True
    if best.get("mime") and "jpeg" in best.get("mime").lower():
        is_full = True
    if "скриншот" in fname_lower or "screenshot" in fname_lower:
        is_full = True

    safe = sanitize_name(display_name)

    result = {
        "page_title": page_title,
        "display_name": display_name,
        "candidates": candidates_sorted,
        "icon_path": None,
        "full_path": None,
    }

    # Temporary folder for downloads
    tmp_dir = OUT_BASE / "_tmp_downloads"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # If there is a png_candidate, save it as base icon (safe.png)
    if png_candidate:
        url = png_candidate.get("url")
        if url:
            ext = os.path.splitext(url.split("?")[0])[1] or ".png"
            tmp_dst = tmp_dir / (safe + "_tmp" + ext)
            file_page = "https://pocketcombats.fandom.com/ru/wiki/" + png_candidate.get("file_title", "").replace(
                " ", "_"
            )
            print("  trying PNG icon:", url, " size:", png_candidate.get("width"), "x", png_candidate.get("height"))
            ok = safe_download_to_temp(url, referer=file_page, dst_temp=tmp_dst)
            if ok:
                final_icon = ICONS_DIR / (safe + ".png")
                if convert_to_png(tmp_dst, final_icon):
                    result["icon_path"] = str(final_icon)
                else:
                    # fallback: move original file (if convert failed)
                    tmp_dst.replace(final_icon)
                    result["icon_path"] = str(final_icon)
            else:
                print("   PNG icon download failed")
            try:
                tmp_dst.unlink(missing_ok=True)
            except Exception:
                pass

    # If best is full (large or jpeg), save as _full.png
    if is_full:
        url = best.get("url")
        if url:
            ext = os.path.splitext(url.split("?")[0])[1] or ".png"
            tmp_dst = tmp_dir / (safe + "_full_tmp" + ext)
            file_page = "https://pocketcombats.fandom.com/ru/wiki/" + best.get("file_title", "").replace(" ", "_")
            print("  trying FULL image:", url, " size:", best.get("width"), "x", best.get("height"))
            ok = safe_download_to_temp(url, referer=file_page, dst_temp=tmp_dst)
            if ok:
                final_full = ICONS_DIR / (safe + "_full.png")
                if convert_to_png(tmp_dst, final_full):
                    result["full_path"] = str(final_full)
                else:
                    tmp_dst.replace(final_full)
                    result["full_path"] = str(final_full)
            else:
                print("   FULL download failed")
            try:
                tmp_dst.unlink(missing_ok=True)
            except Exception:
                pass
    else:
        # If best is not considered full but we don't have icon yet, use best as icon
        if not result["icon_path"]:
            url = best.get("url")
            if url:
                ext = os.path.splitext(url.split("?")[0])[1] or ".png"
                tmp_dst = tmp_dir / (safe + "_tmp" + ext)
                file_page = "https://pocketcombats.fandom.com/ru/wiki/" + best.get("file_title", "").replace(" ", "_")
                print("  trying BEST as icon:", url, " size:", best.get("width"), "x", best.get("height"))
                ok = safe_download_to_temp(url, referer=file_page, dst_temp=tmp_dst)
                if ok:
                    final_icon = ICONS_DIR / (safe + ".png")
                    if convert_to_png(tmp_dst, final_icon):
                        result["icon_path"] = str(final_icon)
                    else:
                        tmp_dst.replace(final_icon)
                        result["icon_path"] = str(final_icon)
                else:
                    print("   BEST download failed")
                try:
                    tmp_dst.unlink(missing_ok=True)
                except Exception:
                    pass

    # cleanup tmp_dir if empty
    try:
        if tmp_dir.exists() and not any(tmp_dir.iterdir()):
            tmp_dir.rmdir()
    except Exception:
        pass

    return result


# ========== Главная функция ==========
def main(limit: Optional[int] = None):
    ensure_dirs()
    pages = get_pages_from_main(MAIN_PAGE)
    if limit and limit > 0:
        pages = pages[:limit]
    print("Pages to process:", len(pages))
    index = []
    for i, p in enumerate(pages, 1):
        print(f"[{i}/{len(pages)}] {p}")
        try:
            item = process_page(p)
            # save per-item JSON
            safe = sanitize_name(item.get("display_name") or p)
            jpath = DATA_DIR / (safe + ".json")
            with open(jpath, "w", encoding="utf-8") as jf:
                json.dump(item, jf, ensure_ascii=False, indent=2)
            index.append(item)
        except Exception as e:
            print("  [!] error processing page:", e)
        polite_sleep()

    # save global index
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print("Done. Items processed:", len(index))
    print("Icons directory:", ICONS_DIR)
    print("Data directory:", DATA_DIR)


# ========== CLI ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Limit number of pages (0 = all)")
    args = parser.parse_args()
    LIM = args.limit if args.limit and args.limit > 0 else None
    main(limit=LIM)
