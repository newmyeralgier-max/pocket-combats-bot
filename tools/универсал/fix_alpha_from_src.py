# -*- coding: utf-8 -*-
"""
Rebuild alpha PNGs from original source images by copying exact RGB
and setting alpha=0 for white pixels (R,G,B >= WHITE_MIN).

Usage:
  python fix_alpha_from_src.py --dry-run --limit 10 --white-min 255
"""
import argparse
from pathlib import Path
import sys
import importlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0=all, otherwise first N")
    parser.add_argument("--white-min", type=int, default=255, help="threshold for white")
    parser.add_argument("--dry-run", action="store_true", help="don't overwrite, just report")
    parser.add_argument("--outline", type=int, default=0, help="remove white outline by expanding transparent area N pixels")
    parser.add_argument("--outline-threshold", type=int, default=None, help="brightness threshold for outline removal (0-255). If not set, defaults to max(250, white_min-5)")
    args = parser.parse_args()

    mod_name = "преобразователь_шаблонов"
    try:
        prep = importlib.import_module(mod_name)
    except Exception:
        sys.path.insert(0, str(Path(__file__).parent))
        prep = importlib.import_module(mod_name)

    SRC_DIR = getattr(prep, "SRC_DIR")
    OUT_DIRS = getattr(prep, "OUT_DIRS")

    white_min = int(args.white_min)
    files = sorted(SRC_DIR.glob("*.png"))
    if args.limit > 0:
        files = files[: args.limit]

    cnt = 0
    replaced = 0
    for f in files:
        cnt += 1
        src = f
        dst = OUT_DIRS["alpha"] / f.name
        print(f"[{cnt}] {f.name} -> {dst}")
        try:
            import cv2
            import numpy as np

            data = prep.cv2_imread_unicode(src, cv2.IMREAD_UNCHANGED)
            if data is None:
                print("  read fail")
                continue
            if data.ndim == 2:
                b = data
                g = data
                r = data
            else:
                if data.shape[2] >= 3:
                    b, g, r = data[:, :, 0], data[:, :, 1], data[:, :, 2]
                else:
                    print("  unexpected channels")
                    continue

            if data.ndim == 3 and data.shape[2] >= 4:
                a = data[:, :, 3].copy()
            else:
                a = np.full_like(b, 255)

            white_mask = (b >= white_min) & (g >= white_min) & (r >= white_min)
            a[white_mask] = 0

            outline_removed = 0
            if args.outline and args.outline > 0:
                # remove near-white pixels adjacent to transparent area
                mask_trans = (a == 0).astype(np.uint8)
                kernel = np.ones((3, 3), np.uint8)
                dil = cv2.dilate(mask_trans, kernel, iterations=int(args.outline)).astype(bool)
                candidates = dil & (a != 0)
                if args.outline_threshold is None:
                    thr2 = max(250, white_min - 5)
                else:
                    thr2 = int(args.outline_threshold)
                outline_mask = candidates & (b >= thr2) & (g >= thr2) & (r >= thr2)
                outline_removed = int(np.sum(outline_mask))
                a[outline_mask] = 0

            rgba = cv2.merge((b, g, r, a))

            if args.dry_run:
                whites = int(white_mask.sum())
                print(f"  dry: whites={whites} outline_removed={outline_removed}")
            else:
                prep.cv2_imwrite_unicode(dst, rgba)
                replaced += 1
        except Exception as e:
            print(f"  err: {e}")

    print(f"done. processed={cnt} replaced={replaced}")


if __name__ == "__main__":
    main()
