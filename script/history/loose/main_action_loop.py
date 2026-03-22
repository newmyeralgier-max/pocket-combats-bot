import csv
import json
import os
import time
from typing import Dict, List

import cv2

SCREENS_DIR = "C:/bot/screens"
OUT_DIR = "C:/bot/pickup_out"
ANNOT_DIR = os.path.join(OUT_DIR, "annotated")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ANNOT_DIR, exist_ok=True)
from panel_detector_v1 import annotate as annotate_panel
from panel_detector_v1 import detect_panel_state
from pickup_detector_v1 import annotate as annotate_pickup
from pickup_detector_v1 import detect_pickup_on_image
from script.win.actions import click_center_of_box, swipe

ACTIONS_LOG = os.path.join(OUT_DIR, "actions.log")
ACTIONS_CSV = os.path.join(OUT_DIR, "actions.csv")


def log_action(text: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} | {text}"
    print(line)
    with open(ACTIONS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ensure_actions_csv_header():
    if not os.path.isfile(ACTIONS_CSV):
        with open(ACTIONS_CSV, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["ts", "file", "action", "reason", "x", "y", "w", "h", "state_pickup", "state_panel"])


def write_actions_csv(file: str, action: str, reason: str, box, state_pickup: str, state_panel: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    x, y, w, h = box if box else (0, 0, 0, 0)
    with open(ACTIONS_CSV, "a", encoding="utf-8", newline="") as f:
        wcsv = csv.writer(f, delimiter=";")
        wcsv.writerow([ts, file, action, reason, x, y, w, h, state_pickup, state_panel])


def combined_annotation(img, result_pickup, result_panel):
    vis = img.copy()
    vis = annotate_pickup(vis, result_pickup)
    vis = annotate_panel(vis, result_panel)
    return vis


def decide_and_act(fname: str, img, res_pickup: Dict[str, any], res_panel: Dict[str, any]):
    H, W = img.shape[:2]
    state_pickup = res_pickup.get("state", "unknown")
    state_panel = res_panel.get("state", "unknown")
    acted = False
    if state_pickup == "active" and res_pickup["active"]["found"]:
        box = res_pickup["active"]["box"]
        log_action(f"{fname} | click_pickup | box={box} | pickup_score={res_pickup['active']['score']}")
        write_actions_csv(fname, "click_pickup", "pickup_active", box, state_pickup, state_panel)
        x, y, w, h = box
        click_center_of_box((x, y, w, h), W, H, tag="pickup")
        acted = True
    elif state_pickup in ("inactive", "not visible") and state_panel in ("collapsed", "hidden"):
        x1, y1 = int(0.5 * W), int(0.85 * H)
        x2, y2 = int(0.5 * W), int(0.35 * H)
        log_action(f"{fname} | swipe_up | reason=panel_{state_panel}")
        write_actions_csv(
            fname, "swipe_up", f"panel_{state_panel}", (x1, y1, x2 - x1, y2 - y1), state_pickup, state_panel
        )
        swipe(x1, y1, x2, y2, W, H, duration_ms=300, tag="open_panel")
        acted = True
    else:
        log_action(f"{fname} | no_action | pickup={state_pickup} panel={state_panel}")
        write_actions_csv(fname, "none", "no_condition", (0, 0, 0, 0), state_pickup, state_panel)
    return acted


def main():
    ensure_actions_csv_header()
    files = [f for f in os.listdir(SCREENS_DIR) if f.lower().endswith(".png")]
    files.sort()
    combined_results: List[Dict[str, any]] = []
    for i, fname in enumerate(files, 1):
        path = os.path.join(SCREENS_DIR, fname)
        try:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise Exception("Не удалось загрузить изображение")
            H, W = img.shape[:2]
            res_pickup = detect_pickup_on_image(path)
            res_panel = detect_panel_state(path)
            vis = combined_annotation(img, res_pickup, res_panel)
            cv2.imwrite(os.path.join(ANNOT_DIR, fname), vis)
            acted = decide_and_act(fname, img, res_pickup, res_panel)
            combined_results.append({"file": fname, "pickup": res_pickup, "panel": res_panel, "acted": acted})
            print(
                f"[{i}/{len(files)}] {fname} -> pickup={res_pickup['state']}, panel={res_panel.get('state', '—')}, acted={acted}"
            )
        except Exception as e:
            log_action(f"{fname} | ERROR | {e}")
            print(f"[{i}/{len(files)}] {fname} -> ERROR: {e}")
    out_json = os.path.join(OUT_DIR, "results_actions.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Готово. Аннотации, логи и результаты в {OUT_DIR}")


if __name__ == "__main__":
    main()
