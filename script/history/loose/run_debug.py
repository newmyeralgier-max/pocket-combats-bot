import hashlib
import os
import time

from actions import _adb_cmd, click, screenshot, short_sleep, swipe

OUT = "C:/bot/_debug_out"
os.makedirs(OUT, exist_ok=True)


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def ensure_awake():
    import subprocess

    from actions import _adb_cmd

    for key in ("224", "82"):
        cmd = _adb_cmd() + ["shell", "input", "keyevent", key]
        print(f"[KEY] -> {' '.join(cmd)}")
        subprocess.run(cmd)
        time.sleep(0.2)


def step_shot(idx, tag):
    path = os.path.join(OUT, f"step_{idx:03d}_{tag}.png")
    ok = screenshot(path)
    return path if ok else None


def run():
    print("[RUN] debug cycle start")
    ensure_awake()
    imgs = []
    p0 = step_shot(0, "start")
    from PIL import Image

    img_w, img_h = Image.open(p0).size
    if not p0:
        print("[ABORT] screenshot failed at start")
        return
    imgs.append(p0)
    click(540, 1200, img_w, img_h, tag="center_tap")
    short_sleep()
    p1 = step_shot(1, "after_center_tap")
    imgs.append(p1)
    swipe(540, 1800, 540, 600, img_w, img_h, tag="scroll_up")
    short_sleep()
    p2 = step_shot(2, "after_swipe")
    imgs.append(p2)
    click(140, 2200, img_w, img_h, tag="bottom_left")
    short_sleep()
    p3 = step_shot(3, "after_bottom_left")
    imgs.append(p3)
    prev_hash = None
    for i, p in enumerate(imgs):
        if not p:
            print(f"[STEP {i}] no image")
            continue
        size = os.path.getsize(p)
        h = md5sum(p)
        changed = "?" if prev_hash is None else "CHANGED" if h != prev_hash else "SAME"
        print(f"[STEP {i}] {os.path.basename(p)} size={size} md5={h} {changed}")
        prev_hash = h
    print(f"[DONE] Saved {len([p for p in imgs if p])} frames to {OUT}")


if __name__ == "__main__":
    run()
