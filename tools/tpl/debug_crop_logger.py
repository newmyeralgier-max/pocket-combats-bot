import json
from pathlib import Path
import cv2
import numpy as np


OUT_ROOT = Path(__file__).resolve().parents[2] / 'debug' / 'out_icons_debug'
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def _safe_imwrite(p: Path, img: np.ndarray):
    p.parent.mkdir(parents=True, exist_ok=True)
    # normalize image dtype and channels
    arr = img
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype('uint8')
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        # drop alpha channel for writing
        arr = arr[:, :, :3]
    # always encode as PNG to avoid OpenCV path issues
    ok, buf = cv2.imencode('.png', arr)
    if ok:
        p.write_bytes(buf.tobytes())
    else:
        raise IOError('imencode failed')


def save_step(screen_name: str, tpl_key: str, step_name: str, img: np.ndarray):
    """Save an image for a particular step of processing.

    Files are saved under debug/out_icons_debug/crops/<screen>/<tpl_key>/<step>.png
    """
    out_dir = OUT_ROOT / 'crops' / screen_name / tpl_key
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = out_dir / f"{step_name}.png"
    _safe_imwrite(fn, img)
    return fn


def save_metadata(screen_name: str, tpl_key: str, meta: dict):
    out_dir = OUT_ROOT / 'crops' / screen_name / tpl_key
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / 'meta.json'
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    return p


def make_overlay(tpl_bgr, crop_bgr, tpl_alpha=None):
    # Create a side-by-side overlay with alpha blended template top-left on crop
    try:
        H = max(tpl_bgr.shape[0], crop_bgr.shape[0])
        W = tpl_bgr.shape[1] + crop_bgr.shape[1]
        vis = np.zeros((H, W, 3), dtype=np.uint8)
        vis[0:tpl_bgr.shape[0], 0:tpl_bgr.shape[1]] = tpl_bgr
        vis[0:crop_bgr.shape[0], tpl_bgr.shape[1] : tpl_bgr.shape[1] + crop_bgr.shape[1]] = crop_bgr

        if tpl_alpha is not None:
            # draw alpha channel as red mask at the top-right corner of the template area
            a = (tpl_alpha > 0).astype('uint8') * 255
            a_col = cv2.merge([a, np.zeros_like(a), np.zeros_like(a)])
            # place it scaled down inside the template rect
            h, w = a.shape
            hh = min(h, 64)
            ww = min(w, 64)
            a_res = cv2.resize(a_col, (ww, hh), interpolation=cv2.INTER_NEAREST)
            vis[0:hh, 0:ww] = a_res

        return vis
    except Exception:
        return None
