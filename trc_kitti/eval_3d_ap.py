"""KITTI 2D / BEV / 3D average precision of the CPU PointPillars path (R40 as in the vendored evaluate.py).

Uses the vendored `do_eval` (zhulf0804/PointPillars) unchanged; only `Tensor.cuda()` is turned into a no-op so
that it runs on a CPU-only machine. Evaluates the full 3,769-frame validation split and the 1,883 locked-test frames.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "PointPillars"))
torch.Tensor.cuda = lambda self, *a, **k: self  # CPU-only machine; do_eval calls .cuda() on tensors

import evaluate as ev  # noqa: E402  (vendored PointPillars/evaluate.py)
from pointpillars.utils import read_calib, read_label  # noqa: E402
from pointpillars.utils.process import keep_bbox_from_image_range, keep_bbox_from_lidar_range  # noqa: E402

CLASSES = {"Pedestrian": 0, "Cyclist": 1, "Car": 2}
LABEL2CLASSES = {v: k for k, v in CLASSES.items()}
LIMIT = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)


def judge_difficulty(a):
    h = a["bbox"][:, 3] - a["bbox"][:, 1]
    out = []
    for hh, o, t in zip(h, a["occluded"], a["truncated"]):
        d = -1
        for i in range(2, -1, -1):
            if hh > [40, 25, 25][i] and o <= [0, 1, 2][i] and t <= [0.15, 0.30, 0.50][i]:
                d = i
        out.append(d)
    return np.array(out, dtype=np.int32)


def build(ids, data, pred_dir):
    det, gt = {}, {}
    for stem in ids:
        calib = read_calib(str(data / "calib" / f"{stem}.txt"))
        w, h = Image.open(data / "image_2" / f"{stem}.png").size
        anno = read_label(str(data / "label_2" / f"{stem}.txt"))
        anno["difficulty"] = judge_difficulty(anno)
        gt[int(stem)] = {"annos": anno}
        raw = np.load(pred_dir / f"{stem}.npz")
        res = {"lidar_bboxes": raw["lidar_bboxes"], "labels": raw["labels"], "scores": raw["scores"]}
        res = keep_bbox_from_image_range(res, calib["Tr_velo_to_cam"].astype(np.float32), calib["R0_rect"].astype(np.float32),
                                         calib["P2"].astype(np.float32), (h, w))
        res = keep_bbox_from_lidar_range(res, LIMIT)
        fr = {k: [] for k in ("name", "alpha", "bbox", "dimensions", "location", "rotation_y", "score")}
        for lb, label, score, b2, cb in zip(res["lidar_bboxes"], res["labels"], res["scores"], res["bboxes2d"], res["camera_bboxes"]):
            fr["name"].append(LABEL2CLASSES[int(label)])
            fr["alpha"].append(cb[6] - np.arctan2(cb[0], cb[2]))
            fr["bbox"].append(b2)
            fr["dimensions"].append(cb[3:6])
            fr["location"].append(cb[:3])
            fr["rotation_y"].append(cb[6])
            fr["score"].append(score)
        det[int(stem)] = {k: np.array(v) for k, v in fr.items()}
        if len(fr["name"]) == 0:
            det[int(stem)]["bbox"] = np.zeros((0, 4), dtype=np.float32)
            for k in ("dimensions", "location"):
                det[int(stem)][k] = np.zeros((0, 3), dtype=np.float32)
    return det, gt


def parse(path):
    txt = Path(path).read_text()
    out, sec = {}, None
    for line in txt.splitlines():
        m = re.match(r"=+(\w+)=+$", line.strip())
        if m:
            sec = m.group(1)
            continue
        m = re.match(r"(\w+) (?:AP|AOS)@([\d.]+): ([\d.]+) ([\d.]+) ([\d.]+)", line.strip())
        if m and sec:
            out[f"{sec}/{m.group(1)}"] = [float(m.group(3)), float(m.group(4)), float(m.group(5))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "kitti_val"))
    ap.add_argument("--pred", default=str(ROOT / "trc_kitti" / "output_3d"))
    ap.add_argument("--locked-ids", default=str(ROOT / "trc_kitti" / "locked_test_ids.txt"))
    ap.add_argument("--out", default=str(ROOT / "trc_kitti" / "output_3d_eval"))
    ap.add_argument("--subset", choices=("locked", "all"), default="locked")
    args = ap.parse_args()
    data, pred = Path(args.data), Path(args.pred)
    ids = sorted(p.stem for p in pred.glob("*.npz"))
    if args.subset == "locked":
        keep = set(Path(args.locked_ids).read_text().split())
        ids = [i for i in ids if i in keep]
    out = Path(args.out) / args.subset
    out.mkdir(parents=True, exist_ok=True)
    det, gt = build(ids, data, pred)
    ev.do_eval(det, gt, CLASSES, str(out))
    res = parse(out / "eval_results.txt")
    res["n_frames"] = len(ids)
    json.dump(res, open(out / "ap_summary.json", "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
