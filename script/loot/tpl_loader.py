# C:/bot/script/loot/tpl_loader.py
import glob
import json
import os
import re
import unicodedata
from typing import Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from script.loot.utils import BASE_DIR, imread_u8 as _imread_u8


def _tpl(path: str) -> Optional[np.ndarray]:
    if not path:
        return None
    return _imread_u8(path, cv2.IMREAD_COLOR)


def _norm_png_name(name: str) -> str:
    name = name.strip()
    if name.lower().endswith(".png"):
        name = os.path.splitext(name)[0]
    return name


def ensure_png(path):
    root, ext = os.path.splitext(path)
    if ext.lower() != ".png":
        return root + ".png"
    return path


# BASE_DIR imported from utils
CFG_FILE = os.path.join(BASE_DIR, "tools", "cfg", "config.json")
REG_FILE = os.path.join(BASE_DIR, "tools", "cfg", "templates_registry.json")
# Вставить в tpl_loader.py (или в модуль, где ты загружаешь registry)
# Можно прямо после функции, которая читает templates_registry.json


def normalize_registry_entry(key, value):
    """
    Приводит запись из templates_registry к словарю с ожидаемыми ключами.
    Если value — строка (путь к шаблону), оборачивает в dict с полями:
      tpl   — путь к PNG
      group — всё до последней точки в ключе (например, 'skills.name')
      action — None по умолчанию
    """
    if isinstance(value, str):
        # Берём всё до последней точки, чтобы сохранить подгруппу
        if "." in key:
            group_name = key.rsplit(".", 1)[0]
        else:
            group_name = "default"
        return {"tpl": value, "group": group_name, "action": None}
    elif isinstance(value, dict):
        return value
    else:
        raise TypeError(f"Unexpected type for registry entry {key}: {type(value)}")


def load_templates_registry(path):
    """
    Загружает registry и нормализует все записи.
    """

    with open(path, "r", encoding="utf-8") as f:
        raw_registry = json.load(f)

    normalized_registry = {k: normalize_registry_entry(k, v) for k, v in raw_registry.items()}

    return normalized_registry


def _nfkc_lower(s: str) -> str:
    return unicodedata.normalize("NFKC", s).lower()


def _norm_stem(s: str) -> str:
    s = _nfkc_lower(s).replace("\\", "/")
    s = re.sub(r"\.(png|jpg|jpeg)$", "", s)
    s = re.sub(r"_name$", "", s)
    s = re.sub(r"_icon$", "", s)
    s = re.sub(r"_(\d+)$", "", s)
    s = s.replace("yantar", "янтарь")
    s = os.path.basename(s)
    return s


# _imread_u8 imported from utils


def _load_cfg() -> Dict:
    try:
        with open(CFG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


CFG = _load_cfg()
TP = dict(CFG.get("TEMPLATES_PATHS", {}))


def _join(*parts) -> str:
    # всегда возвращаем путь с прямыми слэшами
    path = os.path.join(*parts)
    return path.replace("\\", "/")


def _load_registry() -> Dict[str, dict]:
    try:
        # Загружаем и нормализуем
        reg = load_templates_registry(REG_FILE)
        # нормализуем ключи
        return {_nfkc_lower(k): v for k, v in reg.items()}
    except Exception:
        return {}


REG = _load_registry()

# ---------- key resolvers ----------


def resolve_key(key: str) -> Optional[str]:
    val = REG.get(_nfkc_lower(key))
    if isinstance(val, dict):
        return val.get("tpl")
    return val


def ui_path(name: str) -> str:
    name = _norm_png_name(name)
    k = _nfkc_lower(name)
    if k.endswith("_icon"):
        skill_name = k[:-5]
        p = resolve_key(f"skills_icon.{skill_name}")
        if p:
            return p
    if k.endswith("_name"):
        skill_name = k[:-5]
        p = resolve_key(f"skills_name.{skill_name}")
        if p:
            return p
    # старое поведение для UI
    p = resolve_key(f"ui.{k}")
    if p:
        return p
    return _join(BASE_DIR, "tpl", "служебные", f"{name}.png")


def item_name_path(stem: str) -> str:
    k = f"item.name.{_norm_stem(stem)}"
    p = resolve_key(k)
    if p:
        return p
    d = TP.get("ITEM_NAMES_DIR", _join(BASE_DIR, "tpl", "имя_предметов"))
    return _join(d, f"{_norm_stem(stem)}_name.png")


def item_icon_path(stem: str) -> str:
    k = f"item.icon.{_norm_stem(stem)}"
    p = resolve_key(k)
    if p:
        return p
    d = TP.get("ITEM_ICONS_DIR", _join(BASE_DIR, "tpl", "иконки_предметов"))
    return _join(d, f"{_norm_stem(stem)}_icon.png")


def rune_name_path(stem: str) -> str:
    k = f"rune.name.{_norm_stem(stem)}"
    p = resolve_key(k)
    if p:
        return p
    d = TP.get("RUNE_NAMES_DIR", _join(BASE_DIR, "tpl", "имя_рун"))
    return _join(d, f"{_norm_stem(stem)}_name.png")


def rune_icon_path(stem: str) -> str:
    k = f"rune.icon.{_norm_stem(stem)}"
    p = resolve_key(k)
    if p:
        return p
    d = TP.get("RUNE_ICONS_DIR", _join(BASE_DIR, "tpl", "иконки_рун"))
    return _join(d, f"{_norm_stem(stem)}_icon.png")


# BUGFIX: убираем несуществующую переменную name и нормализуем stem


def item_desc_path(stem: str) -> str:
    k = f"item.desc.{_norm_stem(stem)}"
    p = resolve_key(k)
    if p:
        return p
    d = TP.get("ITEM_DESC_DIR", _join(BASE_DIR, "tpl", "описание_предметов"))
    return _join(d, f"{_norm_stem(stem)}.png")


def rune_desc_path(stem: str) -> str:
    k = f"rune.desc.{_norm_stem(stem)}"
    p = resolve_key(k)
    if p:
        return p
    d = TP.get("RUNE_DESC_DIR", _join(BASE_DIR, "tpl", "описание_рун"))
    return _join(d, f"{_norm_stem(stem)}.png")


def monster_path(stem: str) -> str:
    k = f"monster.{_norm_stem(stem)}"
    p = resolve_key(k)
    if p:
        return p
    d = TP.get("MONSTERS_DIR", _join(BASE_DIR, "tpl", "монстры"))
    return _join(d, f"{_norm_stem(stem)}.png")


def monster_full_path(stem: str) -> str:
    k = f"monster_full.{_norm_stem(stem)}"
    p = resolve_key(k)
    if p:
        return p
    d = TP.get("MONSTERS_FULL_DIR", _join(BASE_DIR, "tpl", "монстры_full"))
    return _join(d, f"{_norm_stem(stem)}.png")


# ---------- image loaders ----------


def load_image(path, flags=cv2.IMREAD_COLOR):
    from script.core.logging import structured_log

    try:
        with open(path, "rb") as f:
            data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, flags)
        if img is None:
            return None
        return img
    except Exception as e:
        try:
            structured_log("tpl_load_fail", file=path, reason=str(e))
        except NameError:
            print(f"[tpl_load_fail] {path} — {e}")
        return None


def load_ui(name: str):
    return load_image(ui_path(name))


def iter_item_name_files() -> List[str]:
    # 1) из реестра (item.name.*)
    reg_list = [v["tpl"] for k, v in REG.items() if k.startswith("item.name.") and isinstance(v, dict) and v.get("tpl")]
    reg_list = [p.replace("\\", "/") for p in reg_list]
    if reg_list:
        return sorted(set(reg_list))
    # 2) fallback — из директории
    d = TP.get("ITEM_NAMES_DIR", _join(BASE_DIR, "tpl", "имя_предметов"))
    return sorted(glob.glob(_join(d, "*.png")))





def load_item_name_templates_unified(
    should_skip: Optional[Callable[[str], bool]] = None,
    detect_whitelist: Optional[List[str]] = None,
    allowed_dirs: Optional[List[str]] = None,
) -> List[Tuple[str, np.ndarray]]:
    out: List[Tuple[str, np.ndarray]] = []

    allow: Set[str] = set()
    victory_targets: List[str] = []
    auto_bot = None  # ← добавлено

    if detect_whitelist:
        allow.update(_norm_stem(n) for n in detect_whitelist)

    try:
        from script.loot import auto_bot as _auto_bot

        auto_bot = _auto_bot
        vt = auto_bot.get_victory_targets()
        if vt:
            victory_targets = list(vt)
            allow.update(_norm_stem(n) for n in victory_targets)
            auto_bot.structured_log(
                "loot_templates_runtime_whitelist", runtime_targets=victory_targets, merged_allow=list(allow)
            )
    except ImportError:
        victory_targets = []

    for p in iter_item_name_files():
        if allowed_dirs and not any(ad in p for ad in allowed_dirs):
            continue
        base = os.path.basename(p)
        if should_skip and should_skip(base):
            continue
        stem = _norm_stem(base)
        if allow and stem not in allow:
            continue
        img = _imread_u8(p, cv2.IMREAD_COLOR)
        if img is not None:
            out.append((base, img))

    if auto_bot:  # ← добавлена проверка
        try:
            auto_bot.structured_log(
                "loot_templates_loaded", total=len(out), whitelist_active=bool(allow), whitelist=list(allow)
            )
        except Exception:
            pass

    return out


def path(group: str, name: str, *rest: str) -> str:
    """
    Возвращает путь к шаблону по ключу "{group}.{name}".
    Остальные части, если есть, добавляются как подкаталоги/файлы.
    """
    key = f"{_nfkc_lower(group)}.{_nfkc_lower(name)}"
    p = resolve_key(key)
    if p:
        return _join(os.path.dirname(p), *rest) if rest else p
    # Фолбэк: прямая сборка пути в tpl
    return _join(BASE_DIR, "tpl", group, name, *rest)





def get_item_score(name: str, width: int, height: int, scale: float, default_name: str) -> float:
    from script.core.logging import structured_log
    from script.core.config import CFG
    ALLOWED_LOOT_DIRS = CFG.get("ALLOWED_LOOT_DIRS", [])

    """
    Вычисляет скор предмета по его шаблону.
    Если шаблон не найден — пробует default_name.
    Возвращает float для сортировки.
    """
    try:
        # Список директорий, где ищем шаблон
        search_dirs = ALLOWED_LOOT_DIRS if "ALLOWED_LOOT_DIRS" in globals() else []
        tpl_path = None

        # Ищем файл по имени
        for base_dir in search_dirs:
            candidate = os.path.join(base_dir, name)
            if os.path.isfile(candidate):
                tpl_path = candidate
                break

        # Если не нашли — пробуем default_name
        if not tpl_path and default_name:
            for base_dir in search_dirs:
                candidate = os.path.join(base_dir, default_name)
                if os.path.isfile(candidate):
                    tpl_path = candidate
                    break

        if not tpl_path:
            structured_log("item_score_tpl_not_found", name=name)
            return 0.0

        # Загружаем изображение
        img = _imread_u8(tpl_path, cv2.IMREAD_COLOR)
        if img is None:
            structured_log("item_score_img_load_fail", name=name)
            return 0.0

        # Ресайз, если заданы размеры
        if width > 0 and height > 0:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

        # Score stub — всегда 1.0 (реальная логика пока не реализована)
        score = 1.0

        structured_log("item_score_computed", name=name, score=score)
        return score

    except Exception as e:
        structured_log("item_score_error", name=name, error=str(e))
        return 0.0
