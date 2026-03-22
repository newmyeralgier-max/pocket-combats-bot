# collect_wiki_names.py
# -*- coding: utf-8 -*-
"""
Собрать все названия предметов и лута с вики PocketCombats в один JSON.
Выход: C:\bot\tpl\templates\data\all_wiki_names.json
Каждый объект:
{
  "display_name": "Янтарь",
  "page_title": "Янтарь",
  "safe_name": "yantar",
  "sources": ["items_page", "loot_of:Гоблин"],
  "page_url": "https://pocketcombats.fandom.com/ru/wiki/Янтарь"
}
"""
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import requests
from bs4 import BeautifulSoup
from requests.utils import requote_uri

API_ROOT = "https://pocketcombats.fandom.com/ru/api.php"
BESTIARY_PAGE = "Бестиарий"
ITEMS_PAGE = "Предметы_и_лут"
OUT_BASE = Path(r"C:\bot\tpl\templates")
DATA_DIR = OUT_BASE / "data"
OUT_FILE = DATA_DIR / "all_wiki_names.json"
USER_AGENT = "pc-wiki-collect-names/1.0"
REQUEST_DELAY = 0.25


HEADERS = {"User-Agent": USER_AGENT}


def sanitize_name(s: str) -> str:
    if not s:
        return "item"
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r'[\\/:*?"<>|]', "_", s)
    return s[:180]


def api_get(params: dict) -> dict:
    p = dict(params)
    p.setdefault("format", "json")
    r = requests.get(API_ROOT, params=p, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def polite_sleep():
    time.sleep(REQUEST_DELAY)


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
    # unique preserve order
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def get_page_html(title: str) -> Optional[str]:
    try:
        res = api_get({"action": "parse", "page": title, "prop": "text"})
        return res.get("parse", {}).get("text", {}).get("*")
    except Exception:
        return None


def extract_loot_from_monster_html(html: str) -> List[Dict[str, str]]:
    """Парсим html страницы монстра и возвращаем список ссылок на предметы: [{'display_name':..., 'page_title':...}, ...]"""
    out = []
    if not html:
        return out
    soup = BeautifulSoup(html, "html.parser")
    # ищем разделы с "добыч", "шанс" и т.п.
    loot_html = None
    for h in soup.find_all(re.compile("^h[1-6]$")):
        t = h.get_text(" ", strip=True).lower()
        if "добыч" in t or "шанс" in t:
            sib = h.find_next_sibling()
            if sib:
                loot_html = str(sib)
                break
    if loot_html:
        lsoup = BeautifulSoup(loot_html, "html.parser")
        for a in lsoup.find_all("a", href=True):
            href = a["href"]
            if "/wiki/" in href and "/wiki/Файл:" not in href and "/wiki/File:" not in href:
                title = a.get_text(strip=True)
                page_title = href.split("/wiki/")[-1]
                try:
                    page_title = requote_uri(page_title)
                except Exception:
                    pass
                page_title = requests.utils.unquote(page_title).replace("_", " ")
                if title:
                    out.append({"display_name": title, "page_title": page_title})
    else:
        # fallback: проход по всем li на странице (как в старом скрипте)
        s_all = BeautifulSoup(html, "html.parser")
        for li in s_all.find_all("li"):
            a = li.find("a", href=True)
            if a and "/wiki/" in a["href"] and "/wiki/Файл:" not in a["href"] and "/wiki/File:" not in a["href"]:
                title = a.get_text(strip=True)
                page_title = a["href"].split("/wiki/")[-1]
                page_title = requests.utils.unquote(page_title).replace("_", " ")
                if title:
                    out.append({"display_name": title, "page_title": page_title})
    return out


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    entries: Dict[str, Dict] = {}  # key by page_title (normalized) to store unique items

    # 1) items from ITEMS_PAGE
    try:
        print("Collecting item links from page:", ITEMS_PAGE)
        items = get_links_from_page(ITEMS_PAGE)
        for t in items:
            page_title = t
            key = page_title.strip()
            if not key:
                continue
            safe = sanitize_name(page_title)
            if key not in entries:
                entries[key] = {
                    "display_name": page_title,
                    "page_title": page_title,
                    "safe_name": safe,
                    "sources": ["items_page"],
                    "page_url": "https://pocketcombats.fandom.com/ru/wiki/" + page_title.replace(" ", "_"),
                }
    except Exception as e:
        print("Failed to collect items page links:", e)

    # 2) monster pages -> extract loot links
    try:
        print("Collecting monster pages from:", BESTIARY_PAGE)
        monsters = get_links_from_page(BESTIARY_PAGE)
        print(f"Found monsters: {len(monsters)} -- extracting loot from each")
        for m in monsters:
            polite_sleep()
            html = get_page_html(m)
            loot = extract_loot_from_monster_html(html)
            for li in loot:
                pt = li.get("page_title")
                if not pt:
                    continue
                key = pt.strip()
                safe = sanitize_name(li.get("display_name") or pt)
                source_label = f"loot_of:{m}"
                if key in entries:
                    # добавляем источник, если ещё нет
                    if source_label not in entries[key]["sources"]:
                        entries[key]["sources"].append(source_label)
                else:
                    entries[key] = {
                        "display_name": li.get("display_name") or pt,
                        "page_title": pt,
                        "safe_name": safe,
                        "sources": [source_label],
                        "page_url": "https://pocketcombats.fandom.com/ru/wiki/" + pt.replace(" ", "_"),
                    }
    except Exception as e:
        print("Failed to process monsters:", e)

    # Final: write to json (unique by page_title)
    all_list = list(entries.values())
    # sort alphabetically by display_name for readability
    all_list.sort(key=lambda x: x.get("display_name", "").lower())
    out_path = OUT_FILE
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_list, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_list)} entries -> {out_path}")


if __name__ == "__main__":
    main()
