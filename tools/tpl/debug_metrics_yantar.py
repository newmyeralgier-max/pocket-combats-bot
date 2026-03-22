import cv2
import numpy as np
import sys, os
HERE = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from tools.tpl import find_icon

IMG = r"c:\bot\back\screens\yantar.png"
TPL_ROOT = r"c:\bot\tpl\иконки_предметов\Янтарь_icon.png"
# bbox from CSV
x,y,w,h = 166,897,60,55

def load_img(p):
    # prefer robust unicode-aware loader if available in find_icon
    try:
        if hasattr(find_icon, 'imread_u'):
            return find_icon.imread_u(p)
    except Exception:
        pass
    return cv2.imread(p, cv2.IMREAD_COLOR)

img = load_img(IMG)
tpl = load_img(TPL_ROOT)
if img is None or tpl is None:
    print('failed to read')
    raise SystemExit

crop = img[y:y+h, x:x+w]
print('crop shape', crop.shape)
# prepare tpl_resized
try:
    tpl_rs = cv2.resize(tpl, (max(1, crop.shape[1]), max(1, crop.shape[0])), interpolation=cv2.INTER_AREA)
except Exception:
    tpl_rs = tpl
# mask from alpha if any else color mask like in loader
# attempt to find alpha by reading with IMREAD_UNCHANGED
ai = cv2.imread(TPL_ROOT, cv2.IMREAD_UNCHANGED)
mask = None
if ai is not None and ai.ndim==3 and ai.shape[2]==4:
    alpha = ai[:,:,3]
    alpha_rs = cv2.resize(alpha, (tpl_rs.shape[1], tpl_rs.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = (alpha_rs>0).astype('uint8')
else:
    # simple color mask: non-white
    hsv = cv2.cvtColor(tpl_rs, cv2.COLOR_BGR2HSV)
    s = hsv[:,:,1]; v = hsv[:,:,2]
    mask = ((s>30)|(v<245)).astype('uint8')

# aggressive white exclusion on crop
crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
sc = crop_hsv[:,:,1]; vc = crop_hsv[:,:,2]
white_px = ((vc>250)&(sc<10)).astype('uint8')
mask = (mask * (1-white_px)).astype('uint8')
print('mask sum', mask.sum())
if mask.sum()==0:
    print('empty mask')

# masked agreement: V channel diff <=30
try:
    tpl_hsv = cv2.cvtColor(tpl_rs, cv2.COLOR_BGR2HSV)
    v_tpl = tpl_hsv[:,:,2].astype(int)
    v_crop = crop_hsv[:,:,2].astype(int)
    diff = (abs(v_tpl - v_crop) <= 30).astype('uint8')
    agree = float((diff * mask).sum()) / float(max(1, mask.sum()))
    print('masked_agreement', agree)
except Exception as e:
    print('err v', e)

# masked_v_corr
try:
    tpl_vf = v_tpl.astype('float32') * (mask.astype('float32'))
    crop_vf = v_crop.astype('float32') * (mask.astype('float32'))
    msum = mask.sum()
    mean_tpl = tpl_vf.sum()/msum
    mean_crop = crop_vf.sum()/msum
    a = (tpl_vf - mean_tpl) * (mask.astype('float32'))
    b = (crop_vf - mean_crop) * (mask.astype('float32'))
    denom = (np.linalg.norm(a)*np.linalg.norm(b))
    mv = float(np.dot(a.ravel(), b.ravel())/denom) if denom>0 else 0.0
    print('masked_v_corr', mv)
except Exception as e:
    print('err mv', e)

# edge score
try:
    tpl_edge = cv2.Canny(cv2.cvtColor(tpl_rs, cv2.COLOR_BGR2GRAY),50,150)
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop_edge = cv2.Canny(crop_gray,50,150)
    tpl_e = (tpl_edge.astype('float32')*(mask.astype('float32')))
    crop_e = (crop_edge.astype('float32')*(mask.astype('float32')))
    denom = (np.linalg.norm(tpl_e)*np.linalg.norm(crop_e))
    edge_score = float(np.dot(tpl_e.ravel(), crop_e.ravel())/denom) if denom>0 else 0.0
    print('edge_score', edge_score)
except Exception as e:
    print('err edge', e)

# hist corr
try:
    def masked_hist_corr(tpl_bgr, crop_bgr, mask_u8):
        m=(mask_u8>0).astype('uint8')
        th = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2HSV)
        ch = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        ht = cv2.calcHist([th],[0,1],m,[16,16],[0,180,0,256])
        hc = cv2.calcHist([ch],[0,1],m,[16,16],[0,180,0,256])
        cv2.normalize(ht, ht)
        cv2.normalize(hc, hc)
        return float(cv2.compareHist(ht,hc,cv2.HISTCMP_CORREL))
    hc = masked_hist_corr(tpl_rs, crop, mask.astype('uint8'))
    print('hist_corr', hc)
except Exception as e:
    print('err hist', e)

print('done')

# check candidate boxes intersection
try:
    from tools.tpl.run_strict_scanner import extract_icon_candidates
    img_full = load_img(IMG)
    boxes = extract_icon_candidates(img_full)
    print('candidate boxes count', len(boxes))
    for b in boxes:
        bx,by,bw,bh = b
        ix0 = max(bx, x)
        iy0 = max(by, y)
        ix1 = min(bx+bw, x+w)
        iy1 = min(by+bh, y+h)
        inter = max(0, ix1-ix0) * max(0, iy1-iy0)
        print('box', b, 'inter', inter)
except Exception as e:
    print('candidate check failed', e)

# ORB verify
try:
    from tools.tpl.run_strict_scanner import _orb_verify
    ok = _orb_verify(tpl_rs, crop, min_matches=10)
    print('orb_verify (min_matches=10):', ok)
    ok2 = _orb_verify(tpl_rs, crop, min_matches=4)
    print('orb_verify (min_matches=4):', ok2)
except Exception as e:
    print('orb check failed', e)
