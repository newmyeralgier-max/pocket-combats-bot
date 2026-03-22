import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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


MULTISPACE = re.compile(r"\s+")
NON_WORDS = re.compile(r"[^0-9A-Za-zА-Яа-яЁё\s-]")


def normalize_ru(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("ё", "е")
    s = NON_WORDS.sub(" ", s)
    s = MULTISPACE.sub(" ", s).strip()
    return s


def sanitize_name(text: str, limit: int = 64) -> str:
    t = normalize_ru(text)
    t = t.replace(" ", "_").replace("-", "_")
    t = re.sub(r"[^0-9a-zа-я_]+", "_", t)
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
        key = (w.page, w.block, w.par, w.line)
        groups.setdefault(key, []).append(w)
    lines: List[Line] = []
    for _, ws in groups.items():
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


def parse_rgb(s: str) -> Tuple[int, int, int]:
    parts = re.split(r"[,\s]+", s.strip())
    vals = [int(p) for p in parts if p != ""]
    if len(vals) != 3:
        raise ValueError(f"invalid RGB triple: {s}")
    return tuple(max(0, min(255, v)) for v in vals)


def is_near_solid_color(
    crop: Image.Image, target_rgb: Tuple[int, int, int], tolerance: int, min_ratio: float, sample_step: int = 3
) -> Tuple[bool, float]:
    """
    Быстрая проверка однотонности: доля пикселей в пределах tolerance от target_rgb.
    Возвращает (is_near, ratio).
    """
    if crop.width <= 0 or crop.height <= 0:
        return False, 0.0
    im = crop.convert("RGB")
    px = im.load()
    tr, tg, tb = target_rgb
    total = 0
    match_count = 0
    step_x = max(1, sample_step)
    step_y = max(1, sample_step)
    for y in range(0, im.height, step_y):
        for x in range(0, im.width, step_x):
            r, g, b = px[x, y]
            if abs(r - tr) <= tolerance and abs(g - tg) <= tolerance and abs(b - tb) <= tolerance:
                match_count += 1
            total += 1
    ratio = (match_count / total) if total else 0.0
    return (ratio >= min_ratio), ratio


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
    enable_bg_filter: bool,
    bg_rgb: Tuple[int, int, int],
    bg_tolerance: int,
    bg_min_ratio: float,
    top_percent: float,
    bg_sample_step: int = 3,
) -> Tuple[int, int]:
    narrator.dbg(f"[scan] {img_path.name}")
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        narrator.log(f"[error] cannot open image {img_path}: {e}")
        return 0, 0

    y_limit = int(img.height * (max(0.0, min(100.0, top_percent)) / 100.0))

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
        if ln.width <= 0 or ln.height <= 0:
            continue

        # Фильтр по верхним N% по Y (используем центр строки для честности)
        y_center = ln.top + ln.height // 2
        if y_center > y_limit:
            # Добавим легкую диагностику (в verbose)
            narrator.dbg(
                f"[skip-top] {img_path.name} line@({ln.left},{ln.top},{ln.width}x{ln.height}) "
                f"center_y={y_center} > limit={y_limit}"
            )
            continue

        crop_box = (ln.left, ln.top, ln.width, ln.height)
        crop_img = img.crop((ln.left, ln.top, ln.left + ln.width, ln.top + ln.height))

        # Фильтр фона RGB≈225
        if enable_bg_filter:
            is_bg, ratio = is_near_solid_color(
                crop=crop_img,
                target_rgb=bg_rgb,
                tolerance=bg_tolerance,
                min_ratio=bg_min_ratio,
                sample_step=bg_sample_step,
            )
            if is_bg:
                save_crop(img, crop_box, rejects_dir, f"rejbg_{ln.text}", used_names)
                acc_boxes.append((ln.left, ln.top, ln.width, ln.height))
                acc_labels.append(f"rejbg:{ln.text} ({int(ln.conf)}) r={ratio:.2f}")
                acc_colors.append((128, 128, 128))
                rejected += 1
                narrator.dbg(
                    f"[bg-skip] {img_path.name} @({ln.left},{ln.top},{ln.width}x{ln.height}) "
                    f"ratio={ratio:.2f} rgb≈{bg_rgb} tol={bg_tolerance}"
                )
                continue

        match_name = match_line_to_item(ln.norm, items_norm_to_orig)

        if match_name is not None:
            if match_name in matched_items_in_image:
                continue
            matched_items_in_image.add(match_name)
            save_crop(img, crop_box, out_dir, match_name, used_names)
            acc_boxes.append((ln.left, ln.top, ln.width, ln.height))
            acc_labels.append(f"{match_name} ({int(ln.conf)})")
            acc_colors.append((0, 200, 0))
            accepted += 1
        else:
            if len(ln.norm) >= 2:
                save_crop(img, crop_box, rejects_dir, f"rej_{ln.text}", used_names)
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


def canon_file_stem(stem: str) -> str:
    """
    Канонизация имени файла для отбора:
    - отрезаем суффикс '_full'
    - нормализуем/санитизируем как названия
    """
    if stem.endswith("_full"):
        stem = stem[:-5]
    return sanitize_name(stem)


def main():
    parser = argparse.ArgumentParser(
        description="Template builder: Russian-only, line grouping, strict match to item_names + RGB(225) bg filter + top-percent + file-mode."
    )
    parser.add_argument("--input", required=True, help="Входная папка с изображениями")
    parser.add_argument("--out", required=True, help="Папка для шаблонов (совпадения)")
    parser.add_argument("--diag", required=True, help="Папка для диагностических коллажей")
    parser.add_argument("--rejects", required=True, help="Папка для отклонённых кропов")
    parser.add_argument(
        "--names",
        required=True,
        help="YAML/текстовый файл со списком item_names (и для отбора файлов при file-mode=none)",
    )
    parser.add_argument("--tess", default="", help="Путь к tesseract.exe (Windows)")
    parser.add_argument("--lang", default="rus", help="Язык Tesseract (по умолчанию 'rus')")
    parser.add_argument("--psm", default="6", help="Tesseract --psm (default 6)")
    parser.add_argument("--oem", default="1", help="Tesseract --oem (default 1)")
    parser.add_argument("--min-conf", type=int, default=60, help="Минимальная средняя уверенность строки (0-100)")

    # Фильтр по фону
    parser.add_argument("--no-bg-filter", action="store_true", help="Отключить фильтр однотонного фона RGB≈225")
    parser.add_argument(
        "--bg-rgb", default="225,225,225", help='Цвет фона для игнора, формат: "R,G,B" (по умолчанию 225,225,225)'
    )
    parser.add_argument("--bg-tol", type=int, default=12, help="Допуск по каждому каналу для совпадения с фоном")
    parser.add_argument(
        "--bg-min-ratio", type=float, default=0.85, help="Минимальная доля пикселей фона для отбраковки"
    )
    parser.add_argument("--bg-sample", type=int, default=3, help="Шаг выборки пикселей при проверке фона (>=1)")

    # Фильтр по верхней части изображения
    parser.add_argument(
        "--top-percent", type=float, default=20.0, help="Обрабатывать только верхние N процентов по Y (0-100)"
    )

    # Режим отбора файлов
    parser.add_argument(
        "--file-mode",
        choices=["all", "none"],
        default="all",
        help="Отбор файлов: 'all' — все изображения; 'none' — только те, что соответствуют списку из --names (игнорируя суффикс _full)",
    )

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

    items_norm_to_orig = load_item_names(narrator, names_path)  # используется и для OCR-матчинга, и для фильтра файлов
    config = f"--psm {args.psm} --oem {args.oem}"

    try:
        bg_rgb = parse_rgb(args.bg_rgb)
    except Exception as e:
        narrator.log(f"[error] bad --bg-rgb value: {e}")
        sys.exit(1)

    enable_bg_filter = not args.no_bg_filter
    if args.bg_sample < 1:
        narrator.log("[warn] --bg-sample < 1, исправляю на 1")
        args.bg_sample = 1

    # Готовим список изображений
    all_images = [p for p in sorted(in_dir.iterdir()) if p.is_file() and is_image_file(p)]

    # Фильтрация по списку (file-mode=none): берём только те файлы, чья каноническая основа совпадает со списком имён
    if args.file_mode == "none":
        # Канонизируем список имён (из норм->ориг)
        wanted_stems: Set[str] = {sanitize_name(orig) for orig in items_norm_to_orig.values()}
        filtered_images: List[Path] = []
        for p in all_images:
            canon_stem = canon_file_stem(p.stem)  # срезает _full и санитизирует
            if canon_stem in wanted_stems:
                filtered_images.append(p)
        images = filtered_images
        narrator.log(
            f"[files] mode=none: selected={len(images)} of {len(all_images)} by names list (with _full tolerance)"
        )
    else:
        images = all_images
        narrator.log(f"[files] mode=all: selected all {len(images)}")

    if not images:
        narrator.log("[warn] no images to process after filtering")
        sys.exit(0)

    narrator.log(f"[start] images={len(images)}, out='{out_dir}', diag='{diag_dir}', rejects='{rejects_dir}'")
    narrator.log(
        f"[cfg] lang='{args.lang}', psm={args.psm}, oem={args.oem}, min_conf={args.min_conf}, top_percent={args.top_percent}"
    )
    narrator.log(
        f"[bg] enabled={enable_bg_filter}, rgb={bg_rgb}, tol={args.bg_tol}, min_ratio={args.bg_min_ratio}, sample={args.bg_sample}"
    )

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
            enable_bg_filter=enable_bg_filter,
            bg_rgb=bg_rgb,
            bg_tolerance=args.bg_tol,
            bg_min_ratio=args.bg_min_ratio,
            top_percent=args.top_percent,
            bg_sample_step=args.bg_sample,
        )
        total_acc += a
        total_rej += r

    narrator.log(f"[summary] accepted={total_acc}, rejected={total_rej}, files={len(images)}")
    narrator.log("[ok] templates ready.")


if __name__ == "__main__":
    main()
# python "C:\bot\вырезка названий.py" --input "C:\bot\tpl\характеристики" --out "C:/bot/data" --diag "C:/bot/out_diag" --rejects "C:/bot/out_rejects" --names "C:/bot/tools/cfg/not_name.json" --lang rus --min-conf 60 --top-percent 20 --file-mode none --verbose
# python "C:\bot\tools\tpl\вырезка названий.py" --input "C:\bot\tpl\описание_рун" --out "C:/bot/data" --diag "C:/bot/out_diag" --rejects "C:/bot/out_rejects" --names "C:/bot/tools/cfg/not_name.json" --lang rus --min-conf 60 --top-percent 99 --file-mode all --verbose
