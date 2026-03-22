import csv
import json
import os
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


def combined_annotation(img, result_pickup, result_panel):
    vis = img.copy()
    vis = annotate_pickup(vis, result_pickup)
    vis = annotate_panel(vis, result_panel)
    return vis


def main():
    files = [f for f in os.listdir(SCREENS_DIR) if f.lower().endswith(".png")]
    files.sort()
    results: List[Dict[str, any]] = []
    for i, fname in enumerate(files, 1):
        path = os.path.join(SCREENS_DIR, fname)
        try:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise Exception("Не удалось загрузить изображение")
            res_pickup = detect_pickup_on_image(path)
            res_panel = detect_panel_state(path)
            res_combined = {"file": fname, "pickup": res_pickup, "panel": res_panel}
            results.append(res_combined)
            vis = combined_annotation(img, res_pickup, res_panel)
            cv2.imwrite(os.path.join(ANNOT_DIR, fname), vis)
            print(f"[{i}/{len(files)}] {fname} -> pickup={res_pickup['state']}, panel={res_panel.get('state', '—')}")
        except Exception as e:
            print(f"[{i}/{len(files)}] {fname} -> ERROR: {e}")
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_DIR, "results.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["file", "pickup_state", "panel_state"])
        for r in results:
            w.writerow([r["file"], r["pickup"]["state"], r["panel"].get("state", "")])
    print(f"\n✅ Готово! Всё собрано в {OUT_DIR}")


if __name__ == "__main__":
    main()
