import argparse
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path


def ru_to_lat(text: str) -> str:
    tbl = {
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
    s = text.strip().lower()
    out = []
    for ch in s:
        if ch in tbl:
            out.append(tbl[ch])
        elif ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "—", "‑"):
            out.append("_")
        else:
            out.append("_")
    lat = "".join(out)
    lat = re.sub("_+", "_", lat).strip("_")
    return lat


def with_png(name: str) -> str:
    return f"{name}.png"


def find_existing(path_no_ext: Path) -> Path | None:
    for ext in (".png", ".PNG"):
        p = Path(str(path_no_ext) + ext)
        if p.exists():
            return p
    return None


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    parser = argparse.ArgumentParser(description="Создаёт недостающие пары рус/лат имён шаблонов в одной папке.")
    parser.add_argument("--json", required=True, help="Путь к items.json с игровыми именами.")
    parser.add_argument("--dir", required=True, help="Путь к папке my с вырезками (например, C:/bot/tpl/my).")
    parser.add_argument("--dry-run", action="store_true", help="Только показать, что будет сделано, без копирования.")
    parser.add_argument("--log", default=None, help="Путь к лог-файлу (по умолчанию C:/bot/logs/dual_namer.log).")
    args = parser.parse_args()
    items_json = Path(args.json)
    my_dir = Path(args.dir)
    log_path = Path(args.log) if args.log else Path("C:/bot/logs/dual_namer.log")
    my_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    items = json.loads(items_json.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("items.json должен быть массивом строк с именами предметов.")
    items = [str(x).strip() for x in items if str(x).strip()]
    created = 0
    skipped_both_absent = 0
    already_ok = 0
    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"""[{timestamp()}] Старт dual_namer | dir={my_dir} | json={items_json} | dry_run={args.dry_run}
"""
        )
        for name in items:
            rus_no_ext = my_dir / name
            lat_no_ext = my_dir / ru_to_lat(name)
            rus_path = find_existing(rus_no_ext)
            lat_path = find_existing(lat_no_ext)
            if rus_path and lat_path:
                already_ok += 1
                log.write(
                    f"""[{timestamp()}] OK: обе версии есть | rus={rus_path.name} | lat={lat_path.name}
"""
                )
                continue
            if rus_path and not lat_path:
                target = Path(str(lat_no_ext) + ".png")
                if args.dry_run:
                    log.write(
                        f"""[{timestamp()}] DRY: создал бы латиницу из русской | {rus_path.name} -> {target.name}
"""
                    )
                else:
                    shutil.copy2(rus_path, target)
                    log.write(
                        f"""[{timestamp()}] CREATE: латиница из русской | {rus_path.name} -> {target.name}
"""
                    )
                created += 1
                continue
            if lat_path and not rus_path:
                target = Path(str(rus_no_ext) + ".png")
                if args.dry_run:
                    log.write(
                        f"""[{timestamp()}] DRY: создал бы русскую из латиницы | {lat_path.name} -> {target.name}
"""
                    )
                else:
                    shutil.copy2(lat_path, target)
                    log.write(
                        f"""[{timestamp()}] CREATE: русская из латиницы | {lat_path.name} -> {target.name}
"""
                    )
                created += 1
                continue
            skipped_both_absent += 1
            log.write(
                f"""[{timestamp()}] SKIP: нет исходных файлов | name='{name}' (ожидались: '{with_png(name)}' или '{with_png(ru_to_lat(name))}')
"""
            )
        log.write(
            f"""[{timestamp()}] Итог: создано={created}, пропусков(нет базовых)={skipped_both_absent}, уже ок={already_ok}
"""
        )


if __name__ == "__main__":
    main()
