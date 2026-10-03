import hashlib
import os
from pathlib import Path


ROOT = Path(os.environ.get("TRC_EXPERIMENT_ROOT", Path(__file__).resolve().parents[1]))
YOLO_ROOT = ROOT / "kitti_yolo"
VAL_IMAGES = YOLO_ROOT / "images" / "val"


def split_name(frame_id):
    digest = hashlib.sha256(f"TRC-2026-{frame_id}".encode("ascii")).digest()
    return "calibration" if digest[0] % 2 == 0 else "locked_test"


images = sorted(VAL_IMAGES.glob("*.png"))
groups = {"calibration": [], "locked_test": []}
for image in images:
    groups[split_name(image.stem)].append(image.resolve())

for name, paths in groups.items():
    (YOLO_ROOT / f"{name}_images.txt").write_text(
        "\n".join(path.as_posix() for path in paths) + "\n", encoding="utf-8"
    )
    (ROOT / "trc_kitti" / f"{name}_ids.txt").write_text(
        "\n".join(path.stem for path in paths) + "\n", encoding="ascii"
    )

protocol = YOLO_ROOT / "kitti_locked.yaml"
protocol.write_text(
    f"path: {YOLO_ROOT.as_posix()}\n"
    "train: images/train\n"
    "val: calibration_images.txt\n"
    "test: locked_test_images.txt\n"
    "names:\n"
    "  0: Car\n"
    "  1: Pedestrian\n"
    "  2: Cyclist\n",
    encoding="utf-8",
)

print(f"calibration={len(groups['calibration'])}")
print(f"locked_test={len(groups['locked_test'])}")
print(protocol)
