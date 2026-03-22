import json
import re
from pathlib import Path

CFG_DIR = Path(__file__).resolve().parent.parent / "cfg"
CONFIG_PATH = CFG_DIR / "config.json"
ITEMS_PATH = CFG_DIR / "items.json"
RU_TO_LAT = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "sch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}


def normalize(s):
    s = s.strip().lower().replace("ё", "е")
    s = s.replace(" ", "_")
    s = re.sub("[^a-z0-9а-я_]+", "", s)
    return s


def translit_ru_to_lat(s):
    return "".join(RU_TO_LAT.get(ch, ch) for ch in s)


def to_canon(s):
    return translit_ru_to_lat(normalize(s))


def base_without_index(fname):
    name = fname[:-4] if fname.lower().endswith(".png") else fname
    name = re.sub("_(\\d+)$", "", name)
    return normalize(name)


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_items():
    with ITEMS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "items" in data:
        data = data["items"]
    return [str(x) for x in data]


def infer_tpl_dirs(cfg):
    dirs = set()
    wlm = cfg.get("WHITELIST_MANAGER", {})
    if "TPL_DIR" in wlm and wlm["TPL_DIR"]:
        dirs.add(Path(wlm["TPL_DIR"]).resolve())
    if "TPL_DIRS" in wlm:
        for d in wlm["TPL_DIRS"]:
            if d:
                dirs.add(Path(d).resolve())
    return sorted(dirs)


def list_png(dirs):
    files = []
    for d in dirs:
        if d.exists():
            files += [p.name for p in d.glob("*.png")]
    return sorted(set(files))


def main():
    cfg = load_config()
    items = load_items()
    tpl_dirs = infer_tpl_dirs(cfg)
    files = list_png(tpl_dirs)
    allowed = set(cfg.get("ALLOWED_ITEM_NAMES", []))
    file_map = {}
    for fname in files:
        canon = to_canon(base_without_index(fname))
        file_map.setdefault(canon, set()).add(fname)
    ok, partial, missing = [], [], []
    for item in items:
        canon = to_canon(item)
        variants = file_map.get(canon, set())
        if not variants:
            missing.append(item)
        elif all(f in allowed for f in variants):
            ok.append(item)
        elif any(f in allowed for f in variants):
            partial.append((item, variants - allowed))
        else:
            partial.append((item, variants))
    print("\n=== OK (полностью в whitelist) ===")
    for it in ok:
        print(it)
    print("\n=== PARTIAL / NEEDS ATTENTION ===")
    for it, not_allowed in partial:
        print(f"{it}  -> не в ALLOWED: {', '.join(sorted(not_allowed))}")
    print("\n=== MISSING PNG ===")
    for it in missing:
        print(it)


if __name__ == "__main__":
    main()
