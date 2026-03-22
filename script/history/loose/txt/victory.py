import os
import subprocess

import cv2
import numpy as np
import pytesseract
from inventory_logic import decide_pick_simple, load_config, load_items_db

cfg = load_config("config.json")
db = load_items_db("items_db.json")
DEVICE_ID = "emulator-5554"
DEBUG_PATH = "debug"
SCREENSHOT = os.path.join(DEBUG_PATH, "last.png")


def capture_screen(device_id, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"[📸] Захват экрана: {out_path}")
    with open(out_path, "wb") as f:
        subprocess.run(["adb", "-s", device_id, "exec-out", "screencap", "-p"], stdout=f)


def find_text(img_path):
    img = cv2.imread(img_path)
    text = pytesseract.image_to_string(img, lang="eng")
    print(f"[🔤] Распознанный текст:\n{text}")
    return text


def find_template(img_path, template_path, threshold=0.8):
    screen = cv2.imread(img_path)
    template = cv2.imread(template_path)
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val >= threshold:
        print(f"[🎯] Шаблон найден (точность {max_val:.2f}) в {max_loc}")
        return max_loc
    else:
        print(f"[❌] Шаблон не найден (макс. точность: {max_val:.2f})")
        return None


def handle_loot_cycle():
    weight_curr, weight_max = get_current_weight()
    free_slots = get_free_slots()
    loot_list = detect_loot()
    for loot in loot_list:
        name = loot["name"]
        qty = loot.get("qty", 1)
        bbox = loot["bbox"]
        decision = decide_pick_simple(
            item_name=name,
            qty_found=qty,
            weight_curr=weight_curr,
            weight_max=weight_max,
            free_slots=free_slots,
            cfg=cfg,
            db=db,
        )
        if decision.action == "go_vendor":
            go_to_vendor()
            return
        if decision.action == "take_all":
            pick_item(bbox)
            free_slots -= 1


def main():
    capture_screen(DEVICE_ID, SCREENSHOT)
    find_text(SCREENSHOT)
    find_template(SCREENSHOT, "attack.png")


if __name__ == "__main__":
    main()
