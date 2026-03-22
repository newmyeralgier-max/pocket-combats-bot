import json
import os
from pathlib import Path
from typing import Dict

ASSETS_DIR = Path("assets/icons")


def ensure_assets_dir():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").strip()


def save_icon_bytes(item_name: str, content: bytes, ext: str = "png"):
    ensure_assets_dir()
    fname = ASSETS_DIR / f"{sanitize_name(item_name)}.{ext}"
    with open(fname, "wb") as f:
        f.write(content)
    return str(fname)


def update_items_db_from_wiki(raw_meta: Dict[str, dict], db_path: str = "items_db.json"):
    with open(db_path, "r", encoding="utf-8") as f:
        db = json.load(f)
    for name, meta in raw_meta.items():
        if name not in db:
            db[name] = {}
        for k, v in meta.items():
            db[name][k] = v
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
