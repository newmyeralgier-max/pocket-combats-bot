import os
from typing import Optional

# ===== НАСТРОЙКИ =====
TARGET_FOLDER = r"C:\bot\tpl\имя_рун"
PREFIX = "_name"
ONLY_FORMAT = "all"  # "all", "png" или ".png"
DRY_RUN = False
RECURSIVE = False
REPLACE_SPACES = True
RENAME_FOLDERS = True


def normalize_only_format(fmt: str) -> Optional[str]:
    if not fmt or fmt.lower() == "all":
        return None
    return fmt if fmt.startswith(".") else "." + fmt.lower()


def should_process_file(ext: str, normalized_fmt: Optional[str]) -> bool:
    if normalized_fmt is None:
        return True
    return ext.lower() == normalized_fmt


def build_new_name(base: str, ext: str, prefix: str, replace_spaces: bool) -> str:
    if replace_spaces:
        base = base.replace(" ", "_")
    if prefix.startswith("_"):
        new_base = f"{base}{prefix}"
    else:
        new_base = f"{prefix}{base}"
    return f"{new_base}{ext}"


def ensure_unique_name(root: str, new_name: str) -> str:
    candidate = new_name
    name, ext = os.path.splitext(new_name)
    i = 1
    while os.path.exists(os.path.join(root, candidate)):
        candidate = f"{name}_{i}{ext}"
        i += 1
    return candidate


def process_folder(
    folder: str,
    prefix: str,
    only_format: str,
    dry_run: bool,
    recursive: bool,
    replace_spaces: bool,
    rename_folders: bool,
):
    fmt = normalize_only_format(only_format)
    renamed = 0

    for root, dirs, files in os.walk(folder, topdown=True):
        if rename_folders:
            for i, d in enumerate(list(dirs)):
                old_path = os.path.join(root, d)
                base, ext = os.path.splitext(d)
                new_name = build_new_name(base, ext, prefix, replace_spaces)
                if new_name == d:
                    continue
                new_name = ensure_unique_name(root, new_name)
                new_path = os.path.join(root, new_name)
                if dry_run:
                    print(f"[DRY] DIR  {d} -> {new_name}")
                else:
                    os.rename(old_path, new_path)
                    print(f"[OK]  DIR  {d} -> {new_name}")
                    renamed += 1
                if recursive:
                    try:
                        idx = dirs.index(d)
                        dirs[idx] = new_name
                    except ValueError:
                        pass

        for f in files:
            old_path = os.path.join(root, f)
            base, ext = os.path.splitext(f)
            if not should_process_file(ext, fmt):
                continue
            new_name = build_new_name(base, ext, prefix, replace_spaces)
            if new_name == f:
                continue
            new_name = ensure_unique_name(root, new_name)
            new_path = os.path.join(root, new_name)
            if dry_run:
                print(f"[DRY] FILE {f} -> {new_name}")
            else:
                os.rename(old_path, new_path)
                print(f"[OK]  FILE {f} -> {new_name}")
                renamed += 1

        if not recursive:
            break

    print(f"[DONE] Переименовано: {renamed}")


if __name__ == "__main__":
    process_folder(TARGET_FOLDER, PREFIX, ONLY_FORMAT, DRY_RUN, RECURSIVE, REPLACE_SPACES, RENAME_FOLDERS)
