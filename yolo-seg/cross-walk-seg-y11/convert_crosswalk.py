from pathlib import Path
import argparse, random, math
import cv2, numpy as np
from tqdm import tqdm

def contours_to_lines(mask, class_id=0, min_area=20):
    '''
        return list of YOLO‑Seg label lines for a binary mask
    '''
    h, w = mask.shape
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        seg = c.reshape(-1, 2).astype(float)
        seg[:, 0] /= w
        seg[:, 1] /= h
        seg = seg.clip(0, 1).flatten()
        if len(seg) < 6:             # need at least 3 points
            continue
        seg_txt = " ".join(f"{p:.6f}" for p in seg)
        lines.append(f"{class_id} {seg_txt}")
    return lines

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/crosswalk", help="dataset root")
ap.add_argument("--val",  type=float, default=0.2,   help="val split ratio")
args = ap.parse_args()

ROOT = Path(args.root)
VAL_RATIO = args.val
random.seed(42)

train_list, val_list = [], []

# walk every images/ dir once
for img_dir in sorted(ROOT.rglob("images")):
    mask_dir  = img_dir.parent / "masks"
    label_dir = img_dir.parent / "labels"
    label_dir.mkdir(exist_ok=True)

    imgs = sorted(img_dir.glob("*.*"))
    random.shuffle(imgs)
    k = math.ceil(len(imgs) * VAL_RATIO)
    val_list   += imgs[:k]
    train_list += imgs[k:]

    for img_path in tqdm(imgs, desc=f"{img_dir.relative_to(ROOT)}", leave=False):
        stem = img_path.stem
        m_jpg = mask_dir / f"{stem}_mask.jpg"
        m_png = mask_dir / f"{stem}_mask.png"
        m_path = m_png if m_png.exists() else m_jpg
        if not m_path.exists():
            continue

        # read mask, threshold to 0/1
        m = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        _, m_bin = cv2.threshold(m, 128, 1, cv2.THRESH_BINARY)

        # save back as loss‑less PNG
        bin_png = mask_dir / f"{stem}_mask.png"
        cv2.imwrite(str(bin_png), (m_bin * 255).astype(np.uint8))
        if m_jpg.exists():
            m_jpg.unlink()           # remove the jpeg to avoid confusion

        # write label(s)
        lines = contours_to_lines(m_bin, class_id=0)
        if lines:
            (label_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")

# write split files
(ROOT / "train.txt").write_text("\n".join(str(p) for p in train_list))
(ROOT / "val.txt").write_text("\n".join(str(p) for p in val_list))

print(f"{len(train_list)} train   |   {len(val_list)} val images")
print("Masks binarised, labels written (class‑id 0)")
