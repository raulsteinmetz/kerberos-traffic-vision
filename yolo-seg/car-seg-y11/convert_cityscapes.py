#!/usr/bin/env python3
import json, shutil
from pathlib import Path

SRC = Path("data/cityscapes")
DST = Path("data/cityscapes_yolo")
VEHICLES = {"car","truck","bus","train","motorcycle","bicycle","caravan","trailer"}
ALIASES  = {"on rails": "train"}
SPLITS   = ("train","val","test")

def canon(n):
    if n.endswith("group"):
        n = n[:-5]
    n = n.lower()
    return ALIASES.get(n, n)

for split in SPLITS:
    imgs  = SRC/"leftImg8bit"/split
    anns  = SRC/"gtFine"/split
    imgs_out = DST/"images"/split
    lbls_out = DST/"labels"/split
    imgs_out.mkdir(parents=True,exist_ok=True)
    lbls_out.mkdir(parents=True,exist_ok=True)

    for city in imgs.iterdir():
        for img_path in city.glob("*_leftImg8bit.png"):
            img_stem = img_path.stem
            ann_stem = img_stem.replace("_leftImg8bit","")
            ann_path = anns/city.name/f"{ann_stem}_gtFine_polygons.json"
            shutil.copy2(img_path, imgs_out/img_path.name)

            with open(ann_path) as f: js = json.load(f)
            h,w   = js["imgHeight"], js["imgWidth"]
            lines = []
            for obj in js["objects"]:
                n = canon(obj["label"])
                if n not in VEHICLES or len(obj["polygon"])<3:
                    continue
                coords=[]
                for x,y in obj["polygon"]:
                    coords += [x/w, y/h]
                lines.append("0 "+" ".join(f"{c:.6f}" for c in coords))

            if lines:
                (lbls_out/f"{img_stem}.txt").write_text("\n".join(lines))

# dataset yaml
(DST/"cityscapes_vehicle.yaml").write_text(f"""\
path: {DST}
train: images/train
val: images/val
test: images/test
names:
  0: vehicle
""")
print("✅ labels rebuilt with correct names")
