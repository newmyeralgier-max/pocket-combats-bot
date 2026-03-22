# -*- coding: utf-8 -*-
"""
Run preprocessor on a subset of files for quick testing.

Usage:
  python run_preprocess.py --limit 10 --no-debug
"""
import argparse
from pathlib import Path
import importlib
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="How many files to process (0 = all)")
    parser.add_argument("--no-debug", dest="debug", action="store_false", help="Disable debug artifacts")
    parser.add_argument("--autotune-only", action="store_true", help="Run autotune only for files with zalpha")
    parser.add_argument("--strict", action="store_true", help="Use strict white removal (exact 255,255,255)")
    parser.add_argument("--white-min", type=int, default=None, help="If set, treat pixels with R,G,B >= WHITE_MIN as white (e.g. 250)")
    args = parser.parse_args()

    # import the preprocessor module (localized name)
    mod_name = "преобразователь_шаблонов"
    try:
        prep = importlib.import_module(mod_name)
    except Exception as e:
        # try package-relative import
        sys.path.insert(0, str(Path(__file__).parent))
        prep = importlib.import_module(mod_name)

    SRC_DIR = getattr(prep, "SRC_DIR")
    OUT_DIRS = getattr(prep, "OUT_DIRS")

    # Передаём опции в модуль предобработчика
    if args.strict:
        try:
            setattr(prep, "BG_MODE", "pure_white_strict")
        except Exception:
            pass
    if args.white_min is not None:
        try:
            setattr(prep, "WHITE_MIN", int(args.white_min))
        except Exception:
            pass

    files = sorted(SRC_DIR.glob("*.png"))
    if args.limit > 0:
        files = files[: args.limit]

    print(f"Processing {len(files)} files from {SRC_DIR}")
    processed = 0
    skipped = 0
    failed = []

    for f in files:
        print(f"-> {f.name}")
        try:
            if args.autotune_only:
                # autotune is invoked inside process_one if zalpha exists; to force autotune-only, we run process_one but skip edge/bw output
                ok = prep.process_one(f)
            else:
                ok = prep.process_one(f)
            if ok:
                processed += 1
            else:
                skipped += 1
                failed.append(f)
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(f)

    print("\nSummary:")
    print(f" processed: {processed}")
    print(f" skipped/failed: {skipped + len(failed) - skipped}")
    if failed:
        print("Failed files and debug dirs:")
        for f in failed:
            dbg = (OUT_DIRS["alpha"].parent / "_debug_white" / f.stem)
            print(f" - {f.name} -> dbg: {dbg} (alpha: {OUT_DIRS['alpha']/f.name})")


if __name__ == "__main__":
    main()
