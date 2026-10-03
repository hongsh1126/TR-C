from pathlib import Path
import os

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "kitti_yolo"
SOURCES = {"train": ROOT / "kitti_train", "val": ROOT / "kitti_val"}
CLASS_IDS = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}


def convert(split, source):
    image_out = OUTPUT / "images" / split
    label_out = OUTPUT / "labels" / split
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    files = sorted((source / "image_2").glob("*.png"))
    for index, image_path in enumerate(files, 1):
        linked = image_out / image_path.name
        if not linked.exists():
            os.link(image_path, linked)
        width, height = Image.open(image_path).size
        labels = []
        for line in (source / "label_2" / f"{image_path.stem}.txt").read_text().splitlines():
            p = line.split()
            if p[0] not in CLASS_IDS:
                continue
            x1, y1, x2, y2 = map(float, p[4:8])
            x1, x2 = max(0.0, x1), min(float(width), x2)
            y1, y2 = max(0.0, y1), min(float(height), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            cx = (x1 + x2) / (2 * width)
            cy = (y1 + y2) / (2 * height)
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            labels.append(f"{CLASS_IDS[p[0]]} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
        (label_out / f"{image_path.stem}.txt").write_text("\n".join(labels))
        if index % 1000 == 0:
            print(split, index, "/", len(files), flush=True)


for name, path in SOURCES.items():
    convert(name, path)

(OUTPUT / "kitti.yaml").write_text(
    f"path: {OUTPUT.as_posix()}\n"
    "train: images/train\n"
    "val: images/val\n"
    "names:\n"
    "  0: Car\n"
    "  1: Pedestrian\n"
    "  2: Cyclist\n"
)
print(OUTPUT / "kitti.yaml")
