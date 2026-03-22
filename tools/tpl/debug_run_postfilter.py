import sys, os
HERE = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)
from tools.tpl.run_strict_scanner import load_icon_templates, run_on_image
from tools.tpl.run_strict_scanner import _postfilter_matches
import csv
import cv2

tpls = load_icon_templates(r"C:\bot\tpl", "иконки_предметов", max_templates=263)
# read csv rows
csvp = r"C:\bot\debug\out_icons_debug\strict_matches_yantar.csv"
rows = []
with open(csvp, encoding='utf-8') as f:
	reader = csv.DictReader(f)
	for r in reader:
		rows.append(r)

img = cv2.imread(r"C:\bot\back\screens\yantar.png")
out = _postfilter_matches(rows, tpls, img, min_combined=0.5, candidate_boxes=None, pad_frac=0.35, use_overlay=False, image_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV), image_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
print('postfilter out count', len(out))
print(out)
