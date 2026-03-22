# pc_wiki_full_export.py
# -*- coding: utf-8 -*-
"""
Full exporter: monsters + items (2-in-1) with improved rune handling and verbose logging.
- Processes the bestiary (monsters), collects loot items.
- Then processes the main items page to pick up items not in monsters.
- Saves icons and full images (for runes full screenshot is always forced).
- Robust: converts to PNG when possible; if conversion fails, keeps original ext.
- Outputs saved in C:\bot\tpl\templates\icons and C:\bot\tpl\templates\data


Changes vs original:
- Stronger rune detection (RU/EN + safe_name patterns).
- Force saving _full for runes: icon -> "<name>.png", full -> "<name>_full.png".
- Much more detailed logging: each step, paths, errors, totals.
"""


import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from PIL import Image, UnidentifiedImageError
from requests.utils import requote_uri

# ------------- config -------------
API_ROOT = "https://pocketcombats.fandom.com/ru/api.php"
BESTIARY_PAGE = "Бестиарий"
ITEMS_PAGE = "Предметы_и_лут"
OUT_BASE = Path(r"C:\bot\tpl\templates")
ICONS_DIR = OUT_BASE / "icons"
DATA_DIR = OUT_BASE / "data"


HEADERS = {"User-Agent": "pc-wiki-full-export/1.1"}
REQUEST_DELAY = 0.30
MIN_FULL_WIDTH = 300


# ------------- logger -------------
class Logger:
    def __init__(self):
        self.lines: List[str] = []

    def _ts(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def log(self, msg: str):
        line = f"[{self._ts()}] {msg}"
        self.lines.append(line)
        print(line, flush=True)

    def error(self, msg: str):
        self.log("ERROR: " + msg)

    def dump_to(self, path: Path):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.lines))
        except Exception as e:
            print("Failed to write log:", e)


LOG = Logger()


# ------------- helpers -------------
def ensure_dirs():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    ICONS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_name(s: str) -> str:
    if not s:
        s = "item"
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r'[\\/:*?"<>|]', "_", s)
    return s[:180]


def api_get(params: dict) -> dict:
    p = dict(params)
    p.setdefault("format", "json")
    LOG.log(f"API GET: {p}")
    r = requests.get(API_ROOT, params=p, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def polite_sleep():
    time.sleep(REQUEST_DELAY)


def download_to_path(url: str, referer: Optional[str], dst: Path, timeout=60) -> bool:
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

    LOG.log(f"Download start: {url_safe} -> {dst}")
    try:
        with requests.get(url_safe, headers=headers, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "wb") as f:
                for chunk in r.iter_content(8192):
                    if not chunk:
                        break
                    f.write(chunk)
        LOG.log(f"Download OK: {dst} ({dst.stat().st_size} bytes)")
        return True
    except Exception as e:
        LOG.error(f"Download failed: {url_safe} -> {dst} :: {e}")
        return False


def safe_convert_to_png(src: Path, dst_png: Path) -> bool:
    try:
        with Image.open(src) as im:
            im = im.convert("RGBA")
            dst_png.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst_png, format="PNG")
        LOG.log(f"Convert to PNG OK: {src.name} -> {dst_png.name}")
        return True
    except (UnidentifiedImageError, OSError, Exception) as e:
        LOG.error(f"Convert to PNG failed: {src} -> {dst_png} :: {repr(e)}")
        return False


# ------------- MediaWiki helpers -------------
def get_links_from_page(page_title: str) -> List[str]:
    out = []
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
                    out.append(t)
        if "continue" in res:
            cont = res["continue"]
            polite_sleep()
            continue
        break
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    LOG.log(f"Found {len(uniq)} links on page '{page_title}'")
    return uniq


def get_page_html(title: str) -> Optional[str]:
    try:
        res = api_get({"action": "parse", "page": title, "prop": "text"})
        return res.get("parse", {}).get("text", {}).get("*")
    except Exception as e:
        LOG.error(f"Parse failed for page '{title}': {e}")
        return None


def get_images_on_page(page_title: str) -> List[str]:
    try:
        res = api_get({"action": "query", "titles": page_title, "prop": "images", "imlimit": "max"})
    except Exception as e:
        LOG.error(f"get_images failed for '{page_title}': {e}")
        return []
    pages = res.get("query", {}).get("pages", {})
    files = []
    for p in pages.values():
        for im in p.get("images", []):
            name = im.get("title")
            if name:
                files.append(name)
    LOG.log(f"Images on '{page_title}': {len(files)}")
    return files


def get_imageinfo(file_title: str, iiurlwidth: Optional[int] = None) -> Optional[dict]:
    params = {"action": "query", "titles": file_title, "prop": "imageinfo", "iiprop": "url|size|mime"}
    if iiurlwidth:
        params["iiurlwidth"] = str(iiurlwidth)
    try:
        res = api_get(params)
    except Exception as e:
        LOG.error(f"imageinfo failed for '{file_title}': {e}")
        return None
    pages = res.get("query", {}).get("pages", {})
    for p in pages.values():
        ii = p.get("imageinfo")
        if ii and isinstance(ii, list):
            return ii[0]
    return None


# ------------- rune detection -------------
_RUNE_PATTERNS = [
    r"\bруна\b",
    r"\bруны\b",
    r"\bруной\b",
    r"\bruna\b",  # латиница
    r"_runa\b",
    r"\bruna_",  # префикс/суффикс в safe_name
    r"\b_runa_?\b",
]


def looks_like_rune(
    display_name: str, page_title: str, page_html: Optional[str], safe_name: Optional[str] = None
) -> bool:
    def _match_any(text: Optional[str]) -> bool:
        if not text:
            return False
        t = text.lower()
        for pat in _RUNE_PATTERNS:
            if re.search(pat, t):
                return True
        return False

    # самые надёжные источники
    if _match_any(display_name) or _match_any(page_title) or _match_any(page_html):
        return True
    if safe_name and _match_any(safe_name):
        return True
    # эвристика: многие страницы рун содержат слово "руна" в любом месте html
    if page_html and "руна" in page_html.lower():
        return True
    return False


# ------------- choose & download -------------
def choose_and_save(
    file_titles: List[str], safe_basename: str, save_icon: bool = True, save_full: bool = True, force_full: bool = False
) -> Dict[str, Any]:
    """
    - icon -> ICONS_DIR / "<safe_basename>.png"
    - full -> ICONS_DIR / "<safe_basename>_full.png"
    Returns dict {icon_path, full_path, candidates}
    """
    LOG.log(f"choose_and_save for '{safe_basename}' | files: {len(file_titles)} | force_full={force_full}")
    candidates = []
    for f in file_titles:
        polite_sleep()
        info = get_imageinfo(f)
        if not info:
            continue
        url = info.get("url")
        mime = info.get("mime")
        w = info.get("width") or 0
        h = info.get("height") or 0
        cand = {"file_title": f, "url": url, "mime": mime, "width": w, "height": h}
        if w and w < MIN_FULL_WIDTH:
            polite_sleep()
            info2 = get_imageinfo(f, iiurlwidth=2048)
            if info2 and info2.get("url"):
                cand.update(
                    {"url": info2.get("url"), "width": info2.get("width") or w, "height": info2.get("height") or h}
                )
        candidates.append(cand)

    if not candidates:
        LOG.error("No image candidates")
        return {"icon_path": None, "full_path": None, "candidates": []}

    candidates_sorted = sorted(candidates, key=lambda x: (x.get("width") or 0), reverse=True)
    best = candidates_sorted[0]
    png_candidate = next(
        (c for c in candidates_sorted if c.get("mime") and "png" in (c.get("mime") or "").lower()), None
    )
    result = {"icon_path": None, "full_path": None, "candidates": candidates_sorted}
    tmp_dir = OUT_BASE / "_tmp_downloads"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    def _download_and_handle(url, filepage, target_png_path, fallback_ext_allowed=True):
        try:
            ext = os.path.splitext(url.split("?")[0])[1] or ".bin"
            tmp = tmp_dir / ("tmpfile" + ext)
            ok = download_to_path(url, referer=filepage, dst=tmp)
            if not ok:
                return None
            # if PNG originally, move directly
            if ext.lower() == ".png":
                final = (
                    target_png_path
                    if str(target_png_path).lower().endswith(".png")
                    else Path(str(target_png_path) + ".png")
                )
                tmp.replace(final)
                LOG.log(f"Saved PNG directly: {final}")
                return str(final)
            # try convert to PNG
            final_png = (
                target_png_path
                if str(target_png_path).lower().endswith(".png")
                else Path(str(target_png_path) + ".png")
            )
            conv_ok = safe_convert_to_png(tmp, final_png)
            if conv_ok:
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass
                LOG.log(f"Saved converted PNG: {final_png}")
                return str(final_png)
            else:
                if fallback_ext_allowed:
                    final_orig = ICONS_DIR / (target_png_path.stem + ext)
                    tmp.replace(final_orig)
                    LOG.log(f"Saved as original ext (no convert): {final_orig}")
                    return str(final_orig)
                else:
                    tmp.unlink(missing_ok=True)
                    LOG.error("Could not save file (no fallback allowed)")
                    return None
        except Exception as e:
            LOG.error(f"_download_and_handle error: {e}")
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            return None

    # save icon if exists
    if save_icon and png_candidate:
        url = png_candidate.get("url")
        filepage = "https://pocketcombats.fandom.com/ru/wiki/" + png_candidate.get("file_title", "").replace(" ", "_")
        LOG.log(f"[icon] downloading {url}")
        target_png = ICONS_DIR / (safe_basename + ".png")
        p = _download_and_handle(url, filepage, target_png, fallback_ext_allowed=True)
        if p:
            result["icon_path"] = p

    # decide full
    fname_lower = (best.get("file_title") or "").lower()
    is_full = False
    if (best.get("width") or 0) >= MIN_FULL_WIDTH:
        is_full = True
    if best.get("mime") and "jpeg" in (best.get("mime") or "").lower():
        is_full = True
    if "скриншот" in fname_lower or "screenshot" in fname_lower:
        is_full = True

    if save_full and (is_full or force_full):
        url = best.get("url")
        filepage = "https://pocketcombats.fandom.com/ru/wiki/" + best.get("file_title", "").replace(" ", "_")
        LOG.log(f"[full] downloading {url} (force_full={force_full})")
        target_full_png = ICONS_DIR / (safe_basename + "_full.png")
        p = _download_and_handle(url, filepage, target_full_png, fallback_ext_allowed=True)
        if p:
            result["full_path"] = p
    else:
        # fallback: if no icon yet and allowed, use best as icon
        if save_icon and not result["icon_path"]:
            url = best.get("url")
            filepage = "https://pocketcombats.fandom.com/ru/wiki/" + best.get("file_title", "").replace(" ", "_")
            LOG.log(f"[fallback-icon] downloading {url}")
            target_png = ICONS_DIR / (safe_basename + ".png")
            p = _download_and_handle(url, filepage, target_png, fallback_ext_allowed=True)
            if p:
                result["icon_path"] = p

    try:
        if tmp_dir.exists() and not any(tmp_dir.iterdir()):
            tmp_dir.rmdir()
    except Exception:
        pass

    LOG.log(f"Result for '{safe_basename}': icon={bool(result['icon_path'])}, full={bool(result['full_path'])}")
    return result


# ------------- main flow -------------
# ------------- main flow -------------
def run_full(
    limit_monsters: Optional[int] = None,
    limit_items: Optional[int] = None,
    only_monsters: bool = False,
    only_items: bool = False,
    list_file: Optional[str] = None,
    use_list_for_monsters: bool = False,
):
    ensure_dirs()
    monsters: List[Dict[str, Any]] = []
    items_registry: Dict[str, Dict[str, Any]] = []

    LOG.log("=== START EXPORT ===")
    LOG.log(f"Config: OUT_BASE='{OUT_BASE}', ICONS='{ICONS_DIR}', DATA='{DATA_DIR}'")
    LOG.log(
        f"Limits: monsters={limit_monsters}, items={limit_items}, only_monsters={only_monsters}, only_items={only_items}"
    )

    # 1) Грузим ЕДИНСТВЕННЫЙ список целей (твой вайтлист). НИКАКИХ пересечений.
    DEFAULT_LIST = r"C:\bot\tools\cfg\not_items.json"
    list_path = Path(list_file or DEFAULT_LIST)

    try:
        with open(list_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        LOG.error(f"Failed to read list_file='{list_path}': {e}")
        raw = []

    # Поддерживаем любые форматы: список строк, список объектов, словарь ключей
    target_items: List[str] = []

    if isinstance(raw, dict):
        # берем ключи словаря как названия страниц
        target_items = [str(k) for k in raw.keys()]
    elif isinstance(raw, list):
        for x in raw:
            if isinstance(x, str):
                target_items.append(x)
            elif isinstance(x, dict):
                # предпочитаем явные поля, иначе берём первое текстовое значение
                if "page_title" in x and isinstance(x["page_title"], str):
                    target_items.append(x["page_title"])
                elif "display_name" in x and isinstance(x["display_name"], str):
                    target_items.append(x["display_name"])
                elif "name" in x and isinstance(x["name"], str):
                    target_items.append(x["name"])
                else:
                    for v in x.values():
                        if isinstance(v, str):
                            target_items.append(v)
                            break
    else:
        target_items = []

    # Очистка и стабилизация
    target_items = [t.strip() for t in target_items if isinstance(t, str) and t.strip()]
    # Лимиты (если заданы)
    if limit_items and limit_items > 0:
        target_items = target_items[:limit_items]

    LOG.log(f"Target list loaded from '{list_path}': {len(target_items)}")

    processed_items = 0
    processed_monsters = 0

    try:
        # ---- PART A: monsters (только если явно сказано брать из списка) ----
        if not only_items and use_list_for_monsters:
            monster_pages = list(target_items)
            if limit_monsters and limit_monsters > 0:
                monster_pages = monster_pages[:limit_monsters]
            LOG.log(f"Monsters to process (from list): {len(monster_pages)}")

            for idx, mpage in enumerate(monster_pages, 1):
                LOG.log(f"[M {idx}/{len(monster_pages)}] {mpage}")
                polite_sleep()
                html = get_page_html(mpage)
                if not html:
                    LOG.error(f"{mpage}: no html")
                    continue

                soup = BeautifulSoup(html, "html.parser")
                h1 = soup.find("h1")
                display_name = h1.get_text(strip=True) if h1 and h1.get_text(strip=True) else mpage
                safe_mon = sanitize_name(display_name)
                file_titles = get_images_on_page(mpage)

                # monsters: icon only
                mon_res = choose_and_save(file_titles, safe_mon, save_icon=True, save_full=False)
                monster_icon = mon_res.get("icon_path")

                # Лут: здесь вообще не фильтруем — ты хотел "только список", значит мы лут не собираем по ссылкам
                monsters.append(
                    {
                        "display_name": display_name,
                        "page_title": mpage,
                        "safe_name": safe_mon,
                        "icon_path": monster_icon,
                        "loot": [],
                    }
                )
                processed_monsters += 1
        else:
            LOG.log("Monsters skipped (use_list_for_monsters=False or only_items=True)")

        # ---- PART B: items (строго по списку) ----
        if not only_monsters:
            LOG.log(f"Items to process (from list): {len(target_items)}")

            for idx, it_page in enumerate(target_items, 1):
                safe_item = sanitize_name(it_page)
                if safe_item in items_registry:
                    continue
                LOG.log(f"[I {idx}/{len(target_items)}] {it_page}")

                polite_sleep()
                page_html_item = get_page_html(it_page)
                file_titles_item = get_images_on_page(it_page)

                rune_flag = looks_like_rune(it_page, it_page, page_html_item, safe_item)
                imres = choose_and_save(
                    file_titles_item, safe_item, save_icon=True, save_full=True, force_full=rune_flag
                )

                items_registry.append(
                    {
                        "display_name": it_page,
                        "page_title": it_page,
                        "safe_name": safe_item,
                        "icon_path": imres.get("icon_path"),
                        "full_path": imres.get("full_path"),
                        "page_url": "https://pocketcombats.fandom.com/ru/wiki/" + it_page.replace(" ", "_"),
                        "is_rune": rune_flag,
                    }
                )
                processed_items += 1

    finally:
        with open(DATA_DIR / "monsters.json", "w", encoding="utf-8") as f:
            json.dump(monsters, f, ensure_ascii=False, indent=2)
        with open(DATA_DIR / "items.json", "w", encoding="utf-8") as f:
            json.dump(items_registry, f, ensure_ascii=False, indent=2)

        LOG.log(f"Totals: monsters={processed_monsters}, items={processed_items}")


# ------------- CLI -------------
# ------------- CLI -------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-monsters", type=int, default=0, help="Limit monsters from list (0 = all)")
    parser.add_argument("--limit-items", type=int, default=0, help="Limit items from list (0 = all)")
    parser.add_argument("--only-monsters", action="store_true", help="Process only monsters")
    parser.add_argument("--only-items", action="store_true", help="Process only items")
    parser.add_argument(
        "--list-file", type=str, default=r"C:\bot\tools\cfg\not_items.json", help="Path to target list JSON (titles)"
    )
    parser.add_argument("--use-list-for-monsters", action="store_true", help="Process monsters from the same list")
    args = parser.parse_args()

    LIM_M = args.limit_monsters if args.limit_monsters and args.limit_monsters > 0 else None
    LIM_I = args.limit_items if args.limit_items and args.limit_items > 0 else None

    run_full(
        limit_monsters=LIM_M,
        limit_items=LIM_I,
        only_monsters=args.only_monsters,
        only_items=args.only_items,
        list_file=args.list_file,
        use_list_for_monsters=args.use_list_for_monsters,
    )
