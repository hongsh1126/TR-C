"""Convert the saved 3D PointPillars outputs (FOV-reduced input) into the per-frame prediction files used by run_benchmark.py.

Output: <out>/predictions/<id>.npz with the same keys as `run_benchmark.py infer --skip-camera`
(lid_boxes2d, lid_scores, lid_labels, lid_locations, lidar_seconds; empty camera arrays).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "PointPillars"))
from pointpillars.utils import read_calib  # noqa: E402
from pointpillars.utils.process import bbox3d2corners_camera, bbox_lidar2camera, points_camera2image  # noqa: E402


def project_lidar_boxes(boxes, calib, image_size):
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    cam = bbox_lidar2camera(boxes, calib["Tr_velo_to_cam"], calib["R0_rect"])
    corners = bbox3d2corners_camera(cam)
    pixels = points_camera2image(corners, calib["P2"])
    width, height = image_size
    xy1 = np.maximum(np.min(pixels, axis=1), [0, 0])
    xy2 = np.minimum(np.max(pixels, axis=1), [width, height])
    return np.concatenate([xy1, xy2], axis=1).astype(np.float32), cam[:, :3].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "kitti_val"))
    ap.add_argument("--pred3d", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    data, src = Path(args.data), Path(args.pred3d)
    dst = Path(args.output) / "predictions"
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.npz")):
        stem = f.stem
        raw = np.load(f)
        calib = read_calib(str(data / "calib" / f"{stem}.txt"))
        size = Image.open(data / "image_2" / f"{stem}.png").size
        b2d, loc = project_lidar_boxes(raw["lidar_bboxes"], calib, size)
        valid = (b2d[:, 2] > b2d[:, 0]) & (b2d[:, 3] > b2d[:, 1]) & (loc[:, 2] > 0)
        np.savez(dst / f"{stem}.npz",
                 cam_boxes=np.empty((0, 4), dtype=np.float32), cam_scores=np.empty((0,), dtype=np.float32),
                 cam_labels=np.empty((0,), dtype=np.int64),
                 lid_boxes2d=b2d[valid], lid_scores=raw["scores"][valid].astype(np.float32),
                 lid_labels=raw["labels"][valid].astype(np.int64), lid_locations=loc[valid],
                 camera_seconds=np.float64(0.0), lidar_seconds=raw["lidar_seconds"])
    print("done", len(list(dst.glob("*.npz"))))


if __name__ == "__main__":
    main()
