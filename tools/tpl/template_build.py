import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pytesseract
from PIL import Image, ImageDraw, ImageFont
from pytesseract import Output

try:
    import yaml
except Exception:
    yaml = None


class Narrator:

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def log(self, msg: str):
        print(msg)

    def dbg(self, msg: str):
        if self.verbose:
            print(msg)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


MULTISPACE = re.compile("\\s+")
NON_WORDS = re.compile("[^0-9A-Za-zА-Яа-яЁё\\s-]")


def normalize_ru(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("ё", "е")
    s = NON_WORDS.sub(" ", s)
    s = MULTISPACE.sub(" ", s).strip()
    return s


def sanitize_name(text: str, limit: int = 64) -> str:
    t = normalize_ru(text)
    t = t.replace(" ", "_").replace("-", "_")
    t = re.sub("[^0-9a-zа-я_]+", "_", t)
    t = t.strip("_")
    if not t:
        t = "tpl"
    return t[:limit]


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        try:
            return font.getsize(text)
        except Exception:
            return max(8, 8 * len(text)), 16


def draw_boxes(
    img: Image.Image, boxes: List[Tuple[int, int, int, int]], labels: List[str], colors: List[Tuple[int, int, int]]
) -> Image.Image:
    diag = img.copy()
    draw = ImageDraw.Draw(diag)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for (x, y, w, h), label, color in zip(boxes, labels, colors):
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        if label:
            tw, th = measure_text(draw, label, font)
            pad = 2
            bx0, by0 = x, max(0, y - th - pad * 2)
            bx1, by1 = x + tw + pad * 2, y
            draw.rectangle([bx0, by0, bx1, by1], fill=(0, 0, 0))
            draw.text((x + pad, by0 + pad), label, fill=(255, 255, 255), font=font)
    return diag


def load_item_names(narrator: Narrator, path: Path) -> Dict[str, str]:
    if not path.exists():
        narrator.log(f"[error] item_names file not found: {path}")
        sys.exit(1)
    text = path.read_text(encoding="utf-8")
    items: List[str] = []
    if yaml is not None:
        try:
            data = yaml.safe_load(text)
            if isinstance(data, dict) and "item_names" in data and isinstance(data["item_names"], list):
                items = [str(x) for x in data["item_names"]]
            elif isinstance(data, list):
                items = [str(x) for x in data]
        except Exception as e:
            narrator.log(f"[warn] yaml parse failed ({e}), fallback to plain list")
    if not items:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("- "):
                items.append(line[2:].strip())
            else:
                items.append(line)
    items = [i.strip() for i in items if i and i.strip()]
    if not items:
        narrator.log("[error] item_names is empty")
        sys.exit(1)
    norm_map = {normalize_ru(i): i for i in items}
    narrator.dbg(f"[names] loaded {len(norm_map)} items")
    return norm_map


class Word:
    __slots__ = ("text", "conf", "left", "top", "width", "height", "page", "block", "par", "line", "word")

    def __init__(
        self,
        text: str,
        conf: int,
        left: int,
        top: int,
        width: int,
        height: int,
        page: int,
        block: int,
        par: int,
        line: int,
        word: int,
    ):
        self.text = text
        self.conf = conf
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.page = page
        self.block = block
        self.par = par
        self.line = line
        self.word = word


def tesseract_words(img: Image.Image, lang: str, config: str) -> List[Word]:
    data = pytesseract.image_to_data(img, lang=lang, output_type=Output.DICT, config=config)
    n = len(data.get("text", []))
    words: List[Word] = []
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if txt == "":
            continue
        conf_str = data.get("conf", ["-1"] * n)[i]
        try:
            conf = int(float(conf_str))
        except Exception:
            conf = -1
        left = int(data.get("left", [0] * n)[i])
        top = int(data.get("top", [0] * n)[i])
        width = int(data.get("width", [0] * n)[i])
        height = int(data.get("height", [0] * n)[i])
        page = int(data.get("page_num", [1] * n)[i])
        block = int(data.get("block_num", [0] * n)[i])
        par = int(data.get("par_num", [0] * n)[i])
        line = int(data.get("line_num", [0] * n)[i])
        word = int(data.get("word_num", [0] * n)[i])
        if width <= 0 or height <= 0:
            continue
        words.append(Word(txt, conf, left, top, width, height, page, block, par, line, word))
    return words


class Line:
    __slots__ = ("text", "norm", "conf", "left", "top", "width", "height", "words")

    def __init__(self, text: str, conf: float, left: int, top: int, width: int, height: int, words: List[Word]):
        self.text = text
        self.norm = normalize_ru(text)
        self.conf = conf
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.words = words


def group_into_lines(words: List[Word]) -> List[Line]:
    groups: Dict[Tuple[int, int, int, int], List[Word]] = {}
    for w in words:
        key = w.page, w.block, w.par, w.line
        groups.setdefault(key, []).append(w)
    lines: List[Line] = []
    for key, ws in groups.items():
        ws_sorted = sorted(ws, key=lambda w: w.left)
        text = " ".join(w.text for w in ws_sorted)
        mean_conf = sum(max(0, w.conf) for w in ws_sorted) / max(1, len(ws_sorted))
        left = min(w.left for w in ws_sorted)
        top = min(w.top for w in ws_sorted)
        right = max(w.left + w.width for w in ws_sorted)
        bottom = max(w.top + w.height for w in ws_sorted)
        lines.append(
            Line(
                text=text, conf=mean_conf, left=left, top=top, width=right - left, height=bottom - top, words=ws_sorted
            )
        )
    lines.sort(key=lambda l: (l.top, l.left))
    return lines


def contains_phrase(hay_norm: str, needle_norm: str) -> bool:
    if not needle_norm or not hay_norm:
        return False
    hay = f" {hay_norm} "
    needle = f" {needle_norm} "
    return needle in hay


def match_line_to_item(line_norm: str, items_norm_to_orig: Dict[str, str]) -> Optional[str]:
    if line_norm in items_norm_to_orig:
        return items_norm_to_orig[line_norm]
    for inorm, orig in items_norm_to_orig.items():
        if contains_phrase(line_norm, inorm):
            return orig
    tokens = line_norm.split()
    for inorm, orig in items_norm_to_orig.items():
        ntoks = inorm.split()
        if len(ntoks) == 0:
            continue
        for i in range(0, max(0, len(tokens) - len(ntoks) + 1)):
            if tokens[i : i + len(ntoks)] == ntoks:
                return orig
    return None


def save_crop(
    img: Image.Image, box: Tuple[int, int, int, int], out_dir: Path, base_name: str, used: Dict[str, int]
) -> Path:
    ensure_dir(out_dir)
    stem = sanitize_name(base_name)
    if stem in used:
        used[stem] += 1
        stem = f"{stem}_{used[stem]}"
    else:
        used[stem] = 1
    path = out_dir / f"{stem}.png"
    x, y, w, h = box
    crop = img.crop((x, y, x + w, y + h))
    crop.save(path)
    return path


def process_image(
    narrator: Narrator,
    img_path: Path,
    out_dir: Path,
    diag_dir: Path,
    rejects_dir: Path,
    items_norm_to_orig: Dict[str, str],
    min_conf: int,
    lang: str,
    tesseract_config: str,
) -> Tuple[int, int]:
    narrator.dbg(f"[scan] {img_path.name}")
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        narrator.log(f"[error] cannot open image {img_path}: {e}")
        return 0, 0
    words = tesseract_words(img, lang=lang, config=tesseract_config)
    lines = group_into_lines(words)
    accepted, rejected = 0, 0
    used_names: Dict[str, int] = {}
    acc_boxes: List[Tuple[int, int, int, int]] = []
    acc_labels: List[str] = []
    acc_colors: List[Tuple[int, int, int]] = []
    matched_items_in_image: Set[str] = set()
    for ln in lines:
        if ln.conf < min_conf:
            continue
        match_name = match_line_to_item(ln.norm, items_norm_to_orig)
        if match_name is not None:
            if match_name in matched_items_in_image:
                continue
            matched_items_in_image.add(match_name)
            save_crop(img, (ln.left, ln.top, ln.width, ln.height), out_dir, match_name, used_names)
            acc_boxes.append((ln.left, ln.top, ln.width, ln.height))
            acc_labels.append(f"{match_name} ({int(ln.conf)})")
            acc_colors.append((0, 200, 0))
            accepted += 1
        elif ln.width > 0 and ln.height > 0 and len(ln.norm) >= 2:
            save_crop(img, (ln.left, ln.top, ln.width, ln.height), rejects_dir, f"rej_{ln.text}", used_names)
            acc_boxes.append((ln.left, ln.top, ln.width, ln.height))
            acc_labels.append(f"rej:{ln.text} ({int(ln.conf)})")
            acc_colors.append((220, 0, 0))
            rejected += 1
    if acc_boxes:
        ensure_dir(diag_dir)
        diag = draw_boxes(img, acc_boxes, acc_labels, acc_colors)
        diag_path = diag_dir / f"{img_path.stem}_diag.png"
        diag.save(diag_path)
        narrator.dbg(f"[diag] {diag_path.name}: {len(acc_boxes)} boxes")
    narrator.log(f"[done] {img_path.name}: accepted={accepted}, rejected={rejected}")
    return accepted, rejected


def main():
    parser = argparse.ArgumentParser(
        description="Template builder: Russian-only, line grouping, strict match to item_names."
    )
    parser.add_argument("--input", required=True, help="Входная папка с изображениями")
    parser.add_argument("--out", required=True, help="Папка для шаблонов (совпадения)")
    parser.add_argument("--diag", required=True, help="Папка для диагностических коллажей")
    parser.add_argument("--rejects", required=True, help="Папка для отклонённых кропов")
    parser.add_argument("--names", required=True, help="YAML/текстовый файл со списком item_names")
    parser.add_argument("--tess", default="", help="Путь к tesseract.exe (Windows)")
    parser.add_argument("--lang", default="rus", help="Язык Tesseract (по умолчанию 'rus')")
    parser.add_argument("--psm", default="6", help="Tesseract --psm (default 6)")
    parser.add_argument("--oem", default="1", help="Tesseract --oem (default 1)")
    parser.add_argument("--min-conf", type=int, default=60, help="Минимальная средняя уверенность строки (0-100)")
    parser.add_argument("--verbose", action="store_true", help="Подробный лог")
    args = parser.parse_args()
    narrator = Narrator(verbose=args.verbose)
    if args.tess:
        pytesseract.pytesseract.tesseract_cmd = args.tess
    in_dir = Path(args.input).resolve()
    out_dir = Path(args.out).resolve()
    diag_dir = Path(args.diag).resolve()
    rejects_dir = Path(args.rejects).resolve()
    names_path = Path(args.names).resolve()
    if not in_dir.exists():
        narrator.log(f"[error] input not found: {in_dir}")
        sys.exit(1)
    for d in (out_dir, diag_dir, rejects_dir):
        ensure_dir(d)
    items_norm_to_orig = load_item_names(narrator, names_path)
    config = f"--psm {args.psm} --oem {args.oem}"
    images = [p for p in sorted(in_dir.iterdir()) if p.is_file() and is_image_file(p)]
    if not images:
        narrator.log("[warn] no images found in input")
        sys.exit(0)
    narrator.log(f"[start] images={len(images)}, out='{out_dir}', diag='{diag_dir}', rejects='{rejects_dir}'")
    narrator.log(f"[cfg] lang='{args.lang}', psm={args.psm}, oem={args.oem}, min_conf={args.min_conf}")
    total_acc, total_rej = 0, 0
    for img_path in images:
        a, r = process_image(
            narrator=narrator,
            img_path=img_path,
            out_dir=out_dir,
            diag_dir=diag_dir,
            rejects_dir=rejects_dir,
            items_norm_to_orig=items_norm_to_orig,
            min_conf=args.min_conf,
            lang=args.lang,
            tesseract_config=config,
        )
        total_acc += a
        total_rej += r
    narrator.log(f"[summary] accepted={total_acc}, rejected={total_rej}, files={len(images)}")
    narrator.log("[ok] templates ready.")


if __name__ == "__main__":
    main()
# python "C:\bot\tools\tpl\template_build.py" `
# --input "C:/путь/к/папке/с_картинками" `
# --out "C:/путь/к/результатам/OK" `
# --diag "C:/путь/к/диагностике" `
# --rejects "C:/путь/к/браку" `
# --names "C:/путь/к/списку/item_names.json" `
# --lang rus `
# --min-conf 60 `
# --verbose
# python "C:\bot\tools\tpl\template_build.py" --input "C:\bot\tpl\иконки_предметов" --out "C:/bot/data" --diag "C:/bot/out_diag" --rejects "C:/bot/out_rejects" --names "C:/bot/tools/cfg/not_name.json" --lang rus --min-conf 60 --verbose
