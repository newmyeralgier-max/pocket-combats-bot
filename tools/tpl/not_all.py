# -*- coding: utf-8 -*-
import json
from pathlib import Path

# --- Входные файлы ---
PATH_NOT_FULL = Path(r"C:\bot\tools\cfg\not_full.json")  # нет full
PATH_NOT_ICON = Path(r"C:\bot\tools\cfg\not_icon.json")  # нет иконки
PATH_NOT_NAME = Path(r"C:\bot\tools\cfg\not_name.json")  # нет названия

# --- Выход ---
PATH_NOT_ALL = Path(r"C:\bot\tools\cfg\not_all.json")  # объединённый список (строки)
PATH_NOT_ALL_DETAIL = Path(r"C:\bot\tools\cfg\not_all_detail.json")  # подробности

import json
from pathlib import Path
from typing import Set  # <— нужно для аннотации


def load_names(path: Path) -> Set[str]:
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[WARN] Файл не найден: {path}")
        return set()
    except json.JSONDecodeError as e:
        print(f"[ERR] Некорректный JSON в {path}: {e}")
        return set()
    except Exception as e:
        print(f"[ERR] Неожиданная ошибка при чтении {path}: {type(e).__name__} — {e}")
        return set()

    return set(map(str, data)) if isinstance(data, list) else set(map(str, data.keys()))


def main():
    s_full = load_names(PATH_NOT_FULL)
    s_icon = load_names(PATH_NOT_ICON)
    s_name = load_names(PATH_NOT_NAME)

    # объединяем
    union = sorted(s_full | s_icon | s_name, key=lambda x: (x.lower(), x))

    # статистика
    both_full_icon = s_full & s_icon
    both_full_name = s_full & s_name
    both_icon_name = s_icon & s_name
    all_three = s_full & s_icon & s_name

    print("=== Сводка ===")
    print(f"Без full: {len(s_full)}")
    print(f"Без иконки: {len(s_icon)}")
    print(f"Без названия: {len(s_name)}")
    print(f"Без full+icon: {len(both_full_icon)}")
    print(f"Без full+name: {len(both_full_name)}")
    print(f"Без icon+name: {len(both_icon_name)}")
    print(f"Без всех трёх: {len(all_three)}")
    print(f"Всего уникальных проблемных (not_all): {len(union)}")

    # сохраняем итог
    PATH_NOT_ALL.parent.mkdir(parents=True, exist_ok=True)
    with open(PATH_NOT_ALL, "w", encoding="utf-8") as f:
        json.dump(union, f, ensure_ascii=False, indent=2)

    detail = [
        {"name": name, "missing": {"full": name in s_full, "icon": name in s_icon, "title": name in s_name}}
        for name in union
    ]
    with open(PATH_NOT_ALL_DETAIL, "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False, indent=2)

    print(f"[OK] Список not_all сохранён: {PATH_NOT_ALL}")
    print(f"[OK] Подробности сохранены: {PATH_NOT_ALL_DETAIL}")


if __name__ == "__main__":
    main()
