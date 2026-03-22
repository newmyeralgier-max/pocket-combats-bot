import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Set

# === Пути ===
ITEMS_JSON = Path(r"C:\bot\tools\cfg\items.json")
TPL = Path(r"C:\bot\tpl")
DIRS = {
    "название": TPL / "имя_предметов",
    "руны": TPL / "иконки_рун",
    "руны_full": TPL / "описание_рун",
    "монстры": TPL / "монстры",
}

OUT_NEED_TO_MAKE = ITEMS_JSON.parent / "not_name.json"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


# === Канон имён ===
def canon_name(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).strip().lower().replace("ё", "е")
    s = re.sub(r"\.(png|jpg|jpeg|webp)$", "", s, flags=re.I)
    s = re.sub(r"_full$", "", s)
    s = re.sub(r"[\s\-–—]+", "_", s)
    s = re.sub(r"[^a-z0-9а-я_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# === Загрузка items.json ===
def load_item_names(p: Path) -> List[str]:
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    CAND_KEYS = ("name", "title", "display", "displayName", "name_ru", "title_ru")
    names: List[str] = []
    if isinstance(data, list):
        for it in data:
            if isinstance(it, str):
                names.append(it)
            elif isinstance(it, dict):
                val = next((it[k] for k in CAND_KEYS if k in it and isinstance(it[k], str) and it[k].strip()), None)
                if val:
                    names.append(val)
    elif isinstance(data, dict):
        if any(isinstance(v, dict) for v in data.values()):
            for k, v in data.items():
                if isinstance(v, dict):
                    val = next(
                        (v[k2] for k2 in CAND_KEYS if k2 in v and isinstance(v[k2], str) and v[k2].strip()), None
                    )
                    names.append(val if val else k)
                else:
                    names.append(k)
        else:
            names = list(data.keys())
    return [n for n in names if isinstance(n, str) and n.strip()]


# === Скан папок ===
def scan_dir_canon(path: Path) -> Set[str]:
    out: Set[str] = set()
    if not path.exists():
        return out
    for f in path.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            out.add(canon_name(f.stem))
    return out


# === Основной поток ===
item_names = load_item_names(ITEMS_JSON)
items_canon_to_original: Dict[str, str] = {canon_name(n): n for n in item_names}
items_canon = set(items_canon_to_original.keys())

# Собираем готовые ассеты из 4 папок
ready_assets: Set[str] = set()
for p in DIRS.values():
    ready_assets |= scan_dir_canon(p)

# Остаток = всё из items.json минус готовые ассеты
need_to_make = sorted(items_canon_to_original[c] for c in items_canon if c not in ready_assets)

# === Сохраняем ===
with open(OUT_NEED_TO_MAKE, "w", encoding="utf-8") as f:
    json.dump(need_to_make, f, ensure_ascii=False, indent=2)

print(f"Готово. Остаток ({len(need_to_make)} шт.) сохранён в {OUT_NEED_TO_MAKE}")
