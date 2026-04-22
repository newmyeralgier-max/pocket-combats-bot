import os
import json
from typing import Dict, Any

from script.loot.utils import BASE_DIR, short_time_tag

CFG_DIR = os.path.join(BASE_DIR, "tools", "cfg")
CFG_FILE = os.path.join(CFG_DIR, "config.json")

DEFAULT_CFG: Dict[str, Any] = {
    "use_slot_grouping": True,
    "device_id": None,
    "dry_run": False,
    "DEBUG": True,
    "save_debug_images": True,
    "sleep_after_tap": 0.65,
    "step_delay": 0.35,
    "delay_between_loops": 5.0,
    "max_loot_rounds": 0,
    "max_tab_attempts": 2,
    "item_name_threshold": 0.86,
    "item_scales": [0.95, 1.0, 1.05],
    "max_lines_to_collect": 12,
    "item_group_step": 150,
    "item_group_threshold": 50,
    "pickup_rel": [0.72, 0.83, 0.95, 0.94],
    "color_ratio_threshold": 0.12,
    "safe_x1": 0,
    "safe_x2": 1080,
    "safe_y1": 400,
    "safe_y2": 2100,
    "max_screenshot_attempts": 2,
    "pickup_timeout_sec": 6.0,
    "chevron_threshold": 0.86,
    "label_threshold": 0.86,
    "label_threshold_fallback": 0.83,
    "THRESHOLDS": {
        "TAB_CHEVRON": 0.86,
        "TAB_CHEVRON_VERIFY": 0.82,
        "TAB_LABEL": 0.86,
        "TAB_LABEL_FALLBACK": 0.83,
        "FIGHT_CONTINUE": 0.82,
        "FIND_DEFAULT": 0.85,
        "OVERLAY_TEXT": 0.85,
        "OVERLAY_CLOSE": 0.85,
        "FIGHT_IMAGE": 0.82,
        "SCHARR_FALLBACK_MIN": 0.83,
        "SCHARR_FALLBACK_OFFSET": 0.03,
        "FIND_ITEM_NAME": 0.86,
        "SYSCHAT": 0.85
    },
    # Опциональный HP/death-recovery в STATE_RECOVER
    # (см. script/detection/player_status.py, script/overlays/heal.py).
    # По умолчанию выключен — STATE_RECOVER работает как раньше
    # (только clear_overlays → STATE_FIND).
    "RECOVER": {
        "ENABLED": False,
        "DEAD_TPL": None,              # путь к шаблону экрана смерти/воскрешения
        "DEAD_THRESHOLD": 0.85,
        "DEAD_WAIT_S": 10.0,
        "HP_ROI": None,                # [x1, y1, x2, y2] HP-бара
        "HP_LOW_RATIO": 0.5,           # доля "красноты" >= этой → heal
        "HEAL_TAP_XY": None,           # [x, y] — явные координаты кнопки лечения
        "HEAL_BTN_TPL": None,          # альтернатива: путь к шаблону кнопки
        "HEAL_BTN_THRESHOLD": 0.85,
        "HEAL_POST_SLEEP_S": 0.6,
        "DEBUG": False
    }
}

def load_config() -> Dict[str, Any]:
    cfg = DEFAULT_CFG.copy()
    if os.path.isfile(CFG_FILE):
        try:
            with open(CFG_FILE, "r", encoding="utf-8") as f:
                user = json.load(f)
            for k, v in user.items():
                cfg[k] = v
        except Exception:
            pass
    else:
        try:
            os.makedirs(CFG_DIR, exist_ok=True)
            with open(CFG_FILE, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CFG, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return cfg

CFG = load_config()

# Global config-based constants extracted from auto_bot.py
FIND = CFG.get("FIND", {})
DETECT_ONLY_WHITELIST = bool(FIND.get("DETECT_ONLY_WHITELIST", False))
SKIP_SUBSTR = list(FIND.get("SKIP_SUBSTR", ["tab", "chevron", "hdr", "label"]))
EXCLUDE_ZONES = list(FIND.get("EXCLUDE_ZONES", [[0, 1900, 500, 2100]]))

QT = CFG.get("QUICK_TAP", {})
QT_X1 = int(QT.get("X1", 910))
QT_X2 = int(QT.get("X2", 1000))
QT_Y_OFFSET = int(QT.get("Y_OFFSET", -15))
QT_JITTER_PCT = float(QT.get("JITTER_PCT", 0.15))
QT_DELAY_BETWEEN = float(QT.get("DELAY_BETWEEN_TAPS_SEC", 0.04))
QT_POST_SERIES = float(QT.get("POST_SERIES_DELAY_SEC", 0.12))

DRY_RUN = CFG.get("DRY_RUN", CFG.get("dry_run", False))
DEBUG = CFG.get("DEBUG", True)
SAVE_DEBUG_IMAGES = CFG.get("save_debug_images", CFG.get("DEBUG", True))
DEBUG_DIR = CFG.get("DEBUG_DIR", "debug")
LOG_FILE = CFG.get("LOG_FILE", "bot.log")

SCREEN_W = CFG.get("SCREEN_W", 1080)
SCREEN_H = CFG.get("SCREEN_H", 2460)

safe = CFG.get("SAFE_TAP_AREA", {"x1": 0, "y1": 400, "x2": 1080, "y2": 2100})
SAFE_X1, SAFE_Y1, SAFE_X2, SAFE_Y2 = safe["x1"], safe["y1"], safe["x2"], safe["y2"]

ITEMS_ROI = CFG.get("ITEMS_ROI", [0, 400, 500, 2100])
STEP_DELAY = CFG.get("STEP_DELAY", 0.3)
SLEEP_AFTER_TAP = CFG.get("SLEEP_AFTER_TAP", 0.35)
DELAY_BETWEEN_LOOPS = float(CFG.get("DELAY_BETWEEN_LOOPS", CFG.get("delay_between_loops", 5.0)) or 5.0)
MAX_LOOT_ROUNDS = int(CFG.get("max_loot_rounds", 0) or 0)

MATCH = CFG.get("MATCH", {})
PICKUP_TPL_THRESHOLD = MATCH.get("PICKUP_TPL_THRESHOLD", 0.86)
PICKUP_SCALES = MATCH.get("PICKUP_SCALES", [0.9, 0.95, 1.0, 1.05])
PICKUP_COLOR_SAT_MIN = MATCH.get("PICKUP_COLOR_SAT_MIN", 90)
PICKUP_COLOR_VAL_MIN = MATCH.get("PICKUP_COLOR_VAL_MIN", 160)
PICKUP_COLOR_RATIO_THRESHOLD = MATCH.get("PICKUP_COLOR_RATIO_THRESHOLD", 0.12)
PICKUP_METHOD_WEIGHTS = MATCH.get("PICKUP_METHOD_WEIGHTS", {"templ": 0.6, "edge": 0.25, "color": 0.15})
PICKUP_ACTIVE_THRESHOLD = MATCH.get("PICKUP_ACTIVE_THRESHOLD", 0.92)
PICKUP_INACTIVE_THRESHOLD = MATCH.get("PICKUP_INACTIVE_THRESHOLD", 0.9)
ITEM_NAME_THRESHOLD = MATCH.get("ITEM_NAME_THRESHOLD", CFG.get("item_name_threshold", 0.86))

ROI_CFG = CFG.get("ROI", {})
PICKUP_REL = ROI_CFG.get("PICKUP_REL", CFG.get("pickup_rel", [0.72, 0.83, 0.95, 0.94]))
PICKUP_REL_CARD_FALLBACK = ROI_CFG.get("PICKUP_REL_CARD_FALLBACK", [0.56, 0.35, 0.98, 0.96])

SWP = CFG.get("SWIPE", {})
MAX_SWIPE_ATTEMPTS = int(CFG.get("max_swipe_attempts", SWP.get("MAX_SWIPE_ATTEMPTS", 4)) or 4)
SWIPE_VECTORS = SWP.get("VECTORS", [[0, 700], [0, 600], [0, 500], [0, 400]])
SWIPE_DURATION_MS = SWP.get("DURATION_MS", 300)
SWIPE_PAUSE_MS = SWP.get("PAUSE_MS", 500)
SWIPE_STOP_ON_REPEAT_HASH = SWP.get("STOP_ON_REPEAT_HASH", True)
SWIPE_SAME_HASH_STOP_N = int(SWP.get("SAME_HASH_STOP_N", 2))

TIMINGS = CFG.get("TIMINGS", {})
CARD_OPEN_DELAY_MS = int(TIMINGS.get("CARD_OPEN_DELAY_MS", 250))
POST_SWIPE_DELAY_MS = int(TIMINGS.get("POST_SWIPE_DELAY_MS", 200))
VERIFY1_MS = int(TIMINGS.get("VERIFY_ITEM_REMOVED_1_MS", 1200))
VERIFY2_MS = int(TIMINGS.get("VERIFY_ITEM_REMOVED_2_MS", 800))

ORDER = CFG.get("ORDER", {})
RESCAN_AFTER_PICKUP = ORDER.get("RESCAN_AFTER_PICKUP", True)

LOGIC = CFG.get("LOGIC", {})
ALLOW_PICKUP_OTHER = LOGIC.get("ALLOW_PICKUP_OTHER", False)
ABORT_ON_FIRST_INACTIVE = LOGIC.get("ABORT_ON_FIRST_INACTIVE", True)

TPLS = CFG.get("TEMPLATES", {})
TPL_PICKUP_ACTIVE_LIST = TPLS.get("PICKUP_ACTIVE", [])
TPL_PICKUP_INACTIVE = TPLS.get("PICKUP_INACTIVE", None)

ALLOWED_ITEM_NAMES = set(CFG.get("ALLOWED_ITEM_NAMES", []))
FIND_STABILIZE_FRAMES = int(CFG.get("FIND", {}).get("STABILIZE_FRAMES", 1) or 1)

DEVICE_ID: str = CFG.get("device_id", "")
MAX_SCREENSHOT_ATTEMPTS = int(CFG.get("MAX_SCREENSHOT_ATTEMPTS", CFG.get("max_screenshot_attempts", 2)) or 2)
ITEM_SCALES = list(CFG.get("item_scales", [0.95, 1.0, 1.05]))
