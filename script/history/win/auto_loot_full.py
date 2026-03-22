"""
auto_loot_v3_39.py
Python 3.8 compatible.

Назначение:
- Приводит экран к стартовому состоянию (монстры: открыты, вещи: закрыты).
- Ищет вкладки по их текстовым названиям (png-шаблоны), клики строго по имени.
- Вертикальная навигация: один длинный свайп к верху/низу.
- Определяет состояние вкладки по шеврону справа (отдельные шаблоны для monsters/items).

Требования:
- ADB в PATH, Android устройство подключено (USB или Wi‑Fi ADB).
- OpenCV + numpy установлены.

Логи:
- Все сообщения пишутся в один TXT-файл (append), опционально дублируются в консоль.
- Каждый запуск помечается заголовком с датой/временем.

Авторский стиль: прозрачная логика, адаптивные ROI, детальные сообщения.
"""

import base64
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

VERSION = "v3.39"
CONFIG = {
    "DRY_RUN": True,
    "TPL_DIR_MY": "C:\\bot\\tpl\\my",
    "TPL_DIR_SHEVRONS": "C:\\bot\\tpl\\chevrons",
    "TAB_TEMPLATES": {"monsters": "monsters_tab.png", "items": "items_tab.png"},
    "SHEVRON_TEMPLATES": {
        "monsters": {"open": "monsters_open.png", "closed": "monsters_close.png"},
        "items": {"open": "items_open.png", "closed": "items_close.png"},
    },
    "THRESHOLDS": {"tab_match": 0.86, "shevron_match": 0.84},
    "START_STATE": {"monsters": "open", "items": "closed"},
    "SWIPE": {"x_rel": 0.85, "y_bottom_rel": 0.82, "y_top_rel": 0.18, "duration_ms": 360},
    "SHEVRON_REL_ROI": {"x_pad_right_rel": 0.02, "width_rel": 0.12, "y_pad_rel": 0.012},
    "SLEEP": {"after_screenshot": 0.08, "after_tap": 0.12, "after_swipe": 0.22},
    "DEVICE_SERIAL": None,
    "LOG": {"FILE_PATH": "C:\\bot\\diag\\loot_run.txt", "TO_CONSOLE": True, "APPEND": True, "SESSION_HEADER": True},
}
_LOG_FILE_PATH: Optional[Path] = None
_LOG_TO_CONSOLE: bool = True
_LOG_SESSION_HEADER: bool = True


def _init_logger(cfg: Dict):
    global _LOG_FILE_PATH, _LOG_TO_CONSOLE, _LOG_SESSION_HEADER
    log_cfg = cfg.get("LOG", {})
    _LOG_FILE_PATH = Path(log_cfg.get("FILE_PATH", "loot_run.txt"))
    _LOG_TO_CONSOLE = bool(log_cfg.get("TO_CONSOLE", True))
    _LOG_SESSION_HEADER = bool(log_cfg.get("SESSION_HEADER", True))
    out_dir = _LOG_FILE_PATH.parent
    if not out_dir.exists():
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[LOGGER] Не удалось создать директорию {out_dir}: {e}")
            _LOG_FILE_PATH = Path("loot_run.txt")
    if _LOG_SESSION_HEADER:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"\n========== AUTOLoot {VERSION} :: {ts} :: DRY_RUN={cfg['DRY_RUN']} ==========\n"
        try:
            with open(_LOG_FILE_PATH, "a", encoding="utf-8") as f:
                f.write(header)
        except Exception as e:
            print(f"[LOGGER] Ошибка записи заголовка лога: {e}")


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}] {msg}"
    if _LOG_TO_CONSOLE:
        print(line)
    if _LOG_FILE_PATH is not None:
        try:
            with open(_LOG_FILE_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            if not _LOG_TO_CONSOLE:
                print(line)
            print(f"[LOGGER] Ошибка записи в файл {_LOG_FILE_PATH}: {e}")


def safe_imread(path: Path) -> Optional[np.ndarray]:
    """
    Надёжная загрузка png/jpg с русскими путями для Windows.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data is None or data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        log(f"Ошибка загрузки изображения: {path} -> {e}")
        return None


def ensure_dir_exists(path: Path, create: bool = True):
    if not path.exists():
        if create:
            try:
                path.mkdir(parents=True, exist_ok=True)
                log(f"Создана директория: {path}")
            except Exception as e:
                log(f"Внимание: не удалось создать директорию {path}: {e}")
        else:
            log(f"Внимание: директория отсутствует: {path}")


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class ADBClient:

    def __init__(self, serial: Optional[str] = None, dry_run: bool = True):
        self.serial = serial
        self.dry_run = dry_run
        self._screen_size = None

    def _adb_cmd(self) -> list:
        base = ["adb"]
        if self.serial:
            base += ["-s", self.serial]
        return base

    def shell(self, cmd: str) -> str:
        full = self._adb_cmd() + ["shell"] + cmd.split(" ")
        try:
            out = subprocess.check_output(full, stderr=subprocess.STDOUT)
            return out.decode("utf-8", errors="ignore")
        except subprocess.CalledProcessError as e:
            log(f"ADB shell error: {e.output.decode('utf-8', errors='ignore')}")
            return ""

    def tap(self, x: int, y: int):
        if self.dry_run:
            log(f"DRY_RUN: TAP at ({x}, {y})")
            return
        self.shell(f"input tap {x} {y}")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300):
        if self.dry_run:
            log(f"DRY_RUN: SWIPE from ({x1},{y1}) to ({x2},{y2}) for {duration_ms}ms")
            return
        self.shell(f"input swipe {x1} {y1} {x2} {y2} {duration_ms}")

    def screenshot(self) -> Optional[np.ndarray]:
        """
        Получает скриншот через adb exec-out screencap -p
        """
        try:
            proc = subprocess.Popen(self._adb_cmd() + ["exec-out", "screencap", "-p"], stdout=subprocess.PIPE)
            data = proc.communicate()[0]
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                log("Не удалось декодировать скриншот.")
            return img
        except Exception as e:
            log(f"Ошибка скриншота: {e}")
            return None

    def get_screen_size(self) -> Tuple[int, int]:
        """
        Пытается получить размер экрана. Сначала через wm size, иначе — по скриншоту.
        """
        if self._screen_size:
            return self._screen_size
        out = self.shell("wm size").strip()
        w, h = None, None
        if "Physical size:" in out:
            try:
                part = out.split("Physical size:")[1].strip().splitlines()[0].strip()
                w, h = map(int, part.split("x"))
            except Exception:
                pass
        if not w or not h:
            img = self.screenshot()
            if img is not None:
                h, w = img.shape[:2]
        if not w or not h:
            raise RuntimeError("Не удалось определить размер экрана (wm size и скриншот недоступны).")
        self._screen_size = w, h
        log(f"Размер экрана: {w}x{h}")
        return w, h


class Vision:

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.tpl_dir_my = Path(cfg["TPL_DIR_MY"])
        self.tpl_dir_shevrons = Path(cfg["TPL_DIR_SHEVRONS"])
        ensure_dir_exists(self.tpl_dir_my, create=False)
        ensure_dir_exists(self.tpl_dir_shevrons, create=False)

    def load_tab_template(self, key: str) -> Optional[np.ndarray]:
        fname = self.cfg["TAB_TEMPLATES"].get(key)
        if not fname:
            log(f"Не указан шаблон вкладки для ключа: {key}")
            return None
        p = self.tpl_dir_my / fname
        img = safe_imread(p)
        if img is None:
            log(f"Не удалось загрузить шаблон вкладки: {p}")
        return img

    def load_shevron(self, key: str, state: str) -> Optional[np.ndarray]:
        """
        key: 'monsters' | 'items'
        state: 'open' | 'closed'
        """
        mapping = self.cfg["SHEVRON_TEMPLATES"].get(key, {})
        fname = mapping.get(state)
        if not fname:
            log(f"Нет файла шеврона для {key}:{state} в конфиге.")
            return None
        p = self.tpl_dir_shevrons / fname
        img = safe_imread(p)
        if img is None:
            log(f"Не удалось загрузить шаблон шеврона {key}:{state} -> {p}")
        return img

    def match_template(
        self, haystack: np.ndarray, needle: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[Dict]:
        """
        Возвращает лучший матч: {"score": float, "pt": (x,y), "rect": (x1,y1,x2,y2)}
        """
        img = haystack
        x1 = y1 = 0
        x2 = img.shape[1]
        y2 = img.shape[0]
        if roi:
            x1, y1, x2, y2 = roi
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)
            x1 = clamp(x1, 0, img.shape[1] - 1)
            y1 = clamp(y1, 0, img.shape[0] - 1)
            x2 = clamp(x2, x1 + 1, img.shape[1])
            y2 = clamp(y2, y1 + 1, img.shape[0])
            img = haystack[y1:y2, x1:x2]
        res = cv2.matchTemplate(img, needle, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        h, w = needle.shape[:2]
        top_left = max_loc[0] + x1, max_loc[1] + y1
        br = top_left[0] + w, top_left[1] + h
        return {"score": float(max_val), "pt": top_left, "rect": (top_left[0], top_left[1], br[0], br[1])}


class Bot:

    def __init__(self, adb: ADBClient, vision: Vision, cfg: Dict):
        self.adb = adb
        self.vision = vision
        self.cfg = cfg

    def swipe_to_top(self):
        w, h = self.adb.get_screen_size()
        sx = int(self.cfg["SWIPE"]["x_rel"] * w)
        yb = int(self.cfg["SWIPE"]["y_bottom_rel"] * h)
        yt = int(self.cfg["SWIPE"]["y_top_rel"] * h)
        dur = int(self.cfg["SWIPE"]["duration_ms"])
        log(f"Свайп к верху: ({sx},{yb}) → ({sx},{yt}), {dur}мс")
        self.adb.swipe(sx, yb, sx, yt, dur)
        time.sleep(self.cfg["SLEEP"]["after_swipe"])

    def swipe_to_bottom(self):
        w, h = self.adb.get_screen_size()
        sx = int(self.cfg["SWIPE"]["x_rel"] * w)
        yt = int(self.cfg["SWIPE"]["y_top_rel"] * h)
        yb = int(self.cfg["SWIPE"]["y_bottom_rel"] * h)
        dur = int(self.cfg["SWIPE"]["duration_ms"])
        log(f"Свайп к низу: ({sx},{yt}) → ({sx},{yb}), {dur}мс")
        self.adb.swipe(sx, yt, sx, yb, dur)
        time.sleep(self.cfg["SLEEP"]["after_swipe"])

    def screenshot(self) -> Optional[np.ndarray]:
        img = self.adb.screenshot()
        if img is None:
            log("Скриншот не получен.")
            return None
        time.sleep(self.cfg["SLEEP"]["after_screenshot"])
        return img

    def find_tab_by_key_current_view(self, key: str, img: np.ndarray) -> Optional[Dict]:
        """
        Ищем текст вкладки в текущем экране. Возвращаем матч с rect.
        """
        tpl = self.vision.load_tab_template(key)
        if tpl is None:
            return None
        match = self.vision.match_template(img, tpl, roi=None)
        if not match:
            return None
        score = match["score"]
        if score < self.cfg["THRESHOLDS"]["tab_match"]:
            log(f"Вкладка '{key}' не найдена (score {score:.3f} < {self.cfg['THRESHOLDS']['tab_match']}).")
            return None
        log(f"Вкладка '{key}' найдена. score={score:.3f}, rect={match['rect']}")
        return match

    def ensure_tab_visible(self, key: str) -> Optional[Dict]:
        """
        Гарантирует, что вкладка видима. Стратегия: текущий экран -> свайп к верху -> свайп к низу.
        Возвращает матч таба (rect) или None.
        """
        log(f"Ищу вкладку '{key}' в текущем экране...")
        img = self.screenshot()
        if img is None:
            return None
        m = self.find_tab_by_key_current_view(key, img)
        if m:
            return m
        log("Не вижу вкладку. Пробую один длинный свайп к ВЕРХУ.")
        self.swipe_to_top()
        img = self.screenshot()
        if img is None:
            return None
        m = self.find_tab_by_key_current_view(key, img)
        if m:
            return m
        log("Вверху тоже нет. Пробую один длинный свайп к НИЗУ.")
        self.swipe_to_bottom()
        img = self.screenshot()
        if img is None:
            return None
        m = self.find_tab_by_key_current_view(key, img)
        if m:
            return m
        log(f"Вкладка '{key}' не найдена ни в текущем, ни при верх/низ.")
        return None

    def shevron_state_for_tab(self, key: str, img: np.ndarray, tab_rect: Tuple[int, int, int, int]) -> Optional[str]:
        """
        Пытаемся определить состояние вкладки по шеврону справа: 'open'/'closed'/None (неопределённо).
        """
        open_tpl = self.vision.load_shevron(key, "open")
        closed_tpl = self.vision.load_shevron(key, "closed")
        if open_tpl is None and closed_tpl is None:
            log(f"Шаблоны шевронов для '{key}' отсутствуют — пропускаю определение состояния по стрелке.")
            return None
        x1, y1, x2, y2 = tab_rect
        w, h = self.adb.get_screen_size()
        pad = self.cfg["SHEVRON_REL_ROI"]["y_pad_rel"]
        xpad_right = self.cfg["SHEVRON_REL_ROI"]["x_pad_right_rel"]
        wrel = self.cfg["SHEVRON_REL_ROI"]["width_rel"]
        shev_x1 = int(min(x2 + xpad_right * w, w - 1))
        shev_x2 = int(min(shev_x1 + wrel * w, w))
        shev_y1 = int(max(y1 - pad * h, 0))
        shev_y2 = int(min(y2 + pad * h, h))
        roi = shev_x1, shev_y1, shev_x2, shev_y2
        log(f"ROI шеврона для '{key}': {roi}")
        best = {"state": None, "score": -1.0}
        if open_tpl is not None:
            mo = self.vision.match_template(img, open_tpl, roi=roi)
            if mo and mo["score"] > best["score"]:
                best = {"state": "open", "score": mo["score"]}
        if closed_tpl is not None:
            mc = self.vision.match_template(img, closed_tpl, roi=roi)
            if mc and mc["score"] > best["score"]:
                best = {"state": "closed", "score": mc["score"]}
        if best["state"] is None or best["score"] < self.cfg["THRESHOLDS"]["shevron_match"]:
            log(f"Не удалось надёжно определить состояние шеврона для '{key}' (score={best['score']:.3f}).")
            return None
        log(f"Состояние по шеврону '{key}': {best['state']} (score={best['score']:.3f})")
        return best["state"]

    def tap_center(self, rect: Tuple[int, int, int, int]):
        x1, y1, x2, y2 = rect
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        log(f"Тап по центру текста вкладки: ({cx},{cy})")
        self.adb.tap(cx, cy)
        time.sleep(self.cfg["SLEEP"]["after_tap"])

    def ensure_tab_state(self, key: str, target_state: str):
        """
        Приводим вкладку к нужному состоянию: 'open' / 'closed'.
        Кликаем по имени (не по иконкам). Состояние проверяем по шеврону, если он доступен.
        """
        assert target_state in ("open", "closed")
        log(f"Привожу вкладку '{key}' к состоянию: {target_state}")
        m = self.ensure_tab_visible(key)
        if not m:
            log(f"Невозможно обеспечить видимость вкладки '{key}'. Пропускаю.")
            return
        img = self.screenshot()
        if img is None:
            return
        current = self.shevron_state_for_tab(key, img, m["rect"])
        if current is None:
            log("Состояние неизвестно (шеврон не распознан). Выполняю один клик по названию для переключения.")
            self.tap_center(m["rect"])
            return
        if current == target_state:
            log("Состояние уже соответствует цели. Ничего делать не нужно.")
            return
        self.tap_center(m["rect"])
        img2 = self.screenshot()
        if img2 is None:
            return
        new_state = self.shevron_state_for_tab(key, img2, m["rect"])
        if new_state is not None:
            if new_state == target_state:
                log(f"Успешно: вкладка '{key}' теперь {target_state}.")
            else:
                log(f"Внимание: вкладка '{key}' после клика стала '{new_state}', ожидалось '{target_state}'.")
        else:
            log("Не удалось подтвердить новое состояние (шеврон не распознан). Продолжаю по допущению.")

    def ensure_start_state(self):
        """
        Константа на старт цикла:
        - monsters: open
        - items:   closed
        """
        target_monsters = self.cfg["START_STATE"].get("monsters", "open")
        target_items = self.cfg["START_STATE"].get("items", "closed")
        log("=== Привожу экран к стартовому состоянию ===")
        self.ensure_tab_state("monsters", target_monsters)
        self.ensure_tab_state("items", target_items)
        log("=== Стартовое состояние установлено ===")

    def loot_cycle_stub(self):
        """
        Заглушка лут-цикла: только демонстрирует возврат к старту.
        Добавляйте сюда поиск предметов, нажатие 'Подобрать' и т.д.
        """
        log("--- Старт лут-цикла (заглушка) ---")
        self.ensure_start_state()
        log("--- Завершение лут-цикла (заглушка) ---")


def main():
    cfg = CONFIG
    _init_logger(cfg)
    log(f"Автолут {VERSION} — запуск")
    adb = ADBClient(serial=cfg["DEVICE_SERIAL"], dry_run=cfg["DRY_RUN"])
    vision = Vision(cfg)
    bot = Bot(adb, vision, cfg)
    _ = bot.screenshot()
    bot.ensure_start_state()
    bot.loot_cycle_stub()
    log("Готово. DRY_RUN=" + str(cfg["DRY_RUN"]))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Остановлено пользователем (Ctrl+C).")
    except Exception as e:
        log(f"Критическая ошибка: {e}")
        sys.exit(1)
