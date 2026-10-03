"""Run the CPU PointPillars path and save the full 3D boxes (needed for KITTI 3D/BEV AP).

Same preprocessing and inference call as run_benchmark.py (`infer`), but the raw 7-DoF boxes in LiDAR
coordinates are stored instead of their image-plane projections.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "PointPillars"))
from pointpillars.model import PointPillars  # noqa: E402
from pointpillars.utils import read_calib  # noqa: E402
from pointpillars.utils.process import remove_outside_points  # noqa: E402
from PIL import Image  # noqa: E402


def filter_points(points):
    lo = np.asarray([0.0, -39.68, -3.0])
    hi = np.asarray([69.12, 39.68, 1.0])
    keep = np.all(points[:, :3] > lo, axis=1) & np.all(points[:, :3] < hi, axis=1)
    return points[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "kitti_val"))
    ap.add_argument("--output", default=str(ROOT / "trc_kitti" / "output_3d"))
    ap.add_argument("--checkpoint", default=str(ROOT / "PointPillars" / "pretrained" / "epoch_160.pth"))
    ap.add_argument("--id-list", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reduced", action="store_true",
                    help="keep only points inside the camera frustum, as in the reference evaluation (velodyne_reduced)")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ids = sorted(p.stem for p in (data / "label_2").glob("*.txt"))
    if args.id_list:
        keep = set(Path(args.id_list).read_text().split())
        ids = [i for i in ids if i in keep]
    if args.limit:
        ids = ids[:args.limit]
    model = PointPillars(nclasses=3)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    torch.set_num_threads(args.threads)
    t_start = time.time()
    for n, stem in enumerate(ids, 1):
        target = out / f"{stem}.npz"
        if target.exists():
            continue
        raw_pts = np.fromfile(data / "velodyne" / f"{stem}.bin", dtype=np.float32).reshape(-1, 4)
        if args.reduced:
            calib = read_calib(str(data / "calib" / f"{stem}.txt"))
            w, h = Image.open(data / "image_2" / f"{stem}.png").size
            raw_pts = remove_outside_points(raw_pts, calib["R0_rect"], calib["Tr_velo_to_cam"], calib["P2"], (h, w))
        pts = filter_points(raw_pts)
        t0 = time.perf_counter()
        with torch.inference_mode():
            res = model([torch.from_numpy(pts)], mode="test")[0]
        sec = time.perf_counter() - t0
        np.savez(target, lidar_bboxes=np.asarray(res["lidar_bboxes"], dtype=np.float32),
                 scores=np.asarray(res["scores"], dtype=np.float32), labels=np.asarray(res["labels"], dtype=np.int64),
                 lidar_seconds=np.float64(sec))
        if n % 50 == 0 or n == len(ids):
            print(f"{n}/{len(ids)} ({time.time() - t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
