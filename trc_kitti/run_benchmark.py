import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)


ROOT = Path(__file__).resolve().parents[1]
PP_ROOT = ROOT / "PointPillars"
sys.path.insert(0, str(PP_ROOT))

from pointpillars.model import PointPillars  # noqa: E402
from pointpillars.utils import read_calib  # noqa: E402
from pointpillars.utils.process import (  # noqa: E402
    bbox3d2corners_camera,
    bbox_lidar2camera,
    points_camera2image,
)


CLASSES = {"Pedestrian": 0, "Cyclist": 1, "Car": 2}
COCO_TO_KITTI = {1: 0, 2: 1, 3: 2}
CLASS_HEIGHT = np.array([1.70, 1.70, 1.50], dtype=np.float32)
CLASS_LENGTH = np.array([0.80, 1.76, 3.90], dtype=np.float32)
RANGE_BINS = ((0.0, 20.0, "near"), (20.0, 40.0, "mid"), (40.0, 70.0, "far"))


def parse_label(path):
    records = []
    for line in path.read_text().splitlines():
        p = line.split()
        bbox = np.asarray(p[4:8], dtype=np.float32)
        if p[0] not in CLASSES:
            if p[0] == "DontCare":
                records.append({"label": -1, "bbox": bbox, "ignore": True})
            continue
        h, w, length = map(float, p[8:11])
        loc = np.asarray(p[11:14], dtype=np.float32)
        records.append({
            "label": CLASSES[p[0]],
            "truncation": float(p[1]),
            "occlusion": int(p[2]),
            "bbox": bbox,
            "dimensions": np.asarray([length, h, w], dtype=np.float32),
            "location": loc,
            "rotation_y": float(p[14]),
            "ignore": False,
        })
    return records


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


def filter_points(points):
    lo = np.asarray([0.0, -39.68, -3.0])
    hi = np.asarray([69.12, 39.68, 1.0])
    keep = np.all(points[:, :3] > lo, axis=1) & np.all(points[:, :3] < hi, axis=1)
    return points[keep]


def run_inference(args):
    data = Path(args.data)
    output = Path(args.output)
    pred_dir = output / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted(p.stem for p in (data / "label_2").glob("*.txt"))
    if args.limit:
        ids = ids[:args.limit]

    weights = None if args.skip_camera else SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    camera = None if args.skip_camera else ssdlite320_mobilenet_v3_large(weights=weights).eval()
    lidar = PointPillars(nclasses=3)
    state = torch.load(args.checkpoint, map_location="cpu")
    lidar.load_state_dict(state)
    lidar.eval()
    torch.set_num_threads(args.threads)

    completed = 0
    for index, stem in enumerate(ids, 1):
        target = pred_dir / f"{stem}.npz"
        if target.exists() and not args.overwrite:
            completed += 1
            continue
        image = Image.open(data / "image_2" / f"{stem}.png").convert("RGB")
        points = np.fromfile(data / "velodyne" / f"{stem}.bin", dtype=np.float32).reshape(-1, 4)

        if camera is None:
            camera_seconds = 0.0
            cam_boxes = np.empty((0, 4), dtype=np.float32)
            cam_scores = np.empty((0,), dtype=np.float32)
            cam_labels = np.empty((0,), dtype=np.int64)
        else:
            t0 = time.perf_counter()
            tensor = weights.transforms()(image)
            with torch.inference_mode():
                cam_raw = camera([tensor])[0]
            camera_seconds = time.perf_counter() - t0
            scores = cam_raw["scores"].cpu().numpy()
            labels_coco = cam_raw["labels"].cpu().numpy()
            keep = (scores >= args.camera_floor) & np.isin(labels_coco, list(COCO_TO_KITTI))
            cam_boxes = cam_raw["boxes"].cpu().numpy()[keep].astype(np.float32)
            cam_scores = scores[keep].astype(np.float32)
            cam_labels = np.asarray([COCO_TO_KITTI[int(x)] for x in labels_coco[keep]], dtype=np.int64)

        points_filtered = filter_points(points)
        t0 = time.perf_counter()
        with torch.inference_mode():
            lid_raw = lidar([torch.from_numpy(points_filtered)], mode="test")[0]
        lidar_seconds = time.perf_counter() - t0
        calib = read_calib(data / "calib" / f"{stem}.txt")
        lid_boxes2d, lid_locations = project_lidar_boxes(lid_raw["lidar_bboxes"], calib, image.size)
        valid = ((lid_boxes2d[:, 2] > lid_boxes2d[:, 0]) &
                 (lid_boxes2d[:, 3] > lid_boxes2d[:, 1]) &
                 (lid_locations[:, 2] > 0))

        np.savez(
            target,
            cam_boxes=cam_boxes,
            cam_scores=cam_scores,
            cam_labels=cam_labels,
            lid_boxes2d=lid_boxes2d[valid],
            lid_scores=lid_raw["scores"][valid].astype(np.float32),
            lid_labels=lid_raw["labels"][valid].astype(np.int64),
            lid_locations=lid_locations[valid],
            camera_seconds=np.float64(camera_seconds),
            lidar_seconds=np.float64(lidar_seconds),
        )
        completed += 1
        if index % 25 == 0 or index == len(ids):
            print(f"inference {index}/{len(ids)}", flush=True)
    print(f"completed predictions: {completed}")


def run_yolo_inference(args):
    from ultralytics import YOLO

    data = Path(args.data)
    output = Path(args.output) / "camera_yolo"
    output.mkdir(parents=True, exist_ok=True)
    ids = sorted(p.stem for p in (data / "label_2").glob("*.txt"))
    if args.limit:
        ids = ids[:args.limit]
    model = YOLO(args.yolo_checkpoint)
    label_map = {0: 2, 1: 0, 2: 1}
    fallback = Path(args.image_fallback) if args.image_fallback else None

    def image_path(stem):
        primary = data / "image_2" / f"{stem}.png"
        if primary.exists():
            return primary
        if fallback is not None:
            candidate = fallback / f"{stem}.png"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No image found for KITTI frame {stem}")

    pending = [stem for stem in ids if args.overwrite or not (output / f"{stem}.npz").exists()]
    for start in range(0, len(pending), args.yolo_batch):
        stems = pending[start:start + args.yolo_batch]
        sources = [str(image_path(stem)) for stem in stems]
        results = model.predict(source=sources, imgsz=640, conf=args.camera_floor,
                                device="cpu", batch=args.yolo_batch, verbose=False)
        for stem, result in zip(stems, results):
            classes = result.boxes.cls.cpu().numpy().astype(np.int64)
            np.savez(
                output / f"{stem}.npz",
                cam_boxes=result.boxes.xyxy.cpu().numpy().astype(np.float32),
                cam_scores=result.boxes.conf.cpu().numpy().astype(np.float32),
                cam_labels=np.asarray([label_map[int(x)] for x in classes], dtype=np.int64),
                camera_seconds=np.float64(
                    (result.speed["preprocess"] + result.speed["inference"] + result.speed["postprocess"]) / 1000
                ),
            )
        finished = min(start + len(stems), len(pending))
        if finished % 160 == 0 or finished == len(pending):
            print(f"YOLO pending {finished}/{len(pending)}", flush=True)


def iou2d(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(rb - lt, 0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    aa = np.maximum(a[:, 2] - a[:, 0], 0) * np.maximum(a[:, 3] - a[:, 1], 0)
    ab = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-7)


def monocular_locations(boxes, labels, calib):
    if len(boxes) == 0:
        return np.empty((0, 3), dtype=np.float32)
    p2 = calib["P2"]
    heights = np.maximum(boxes[:, 3] - boxes[:, 1], 2.0)
    z = p2[1, 1] * CLASS_HEIGHT[labels] / heights
    u = (boxes[:, 0] + boxes[:, 2]) * 0.5
    v = boxes[:, 3]
    x = (u - p2[0, 2]) * z / p2[0, 0]
    y = (v - p2[1, 2]) * z / p2[1, 1]
    return np.stack([x, y, z], axis=1).astype(np.float32)


def project_points(points, calib):
    xyz1 = np.concatenate([points[:, :3], np.ones((len(points), 1), dtype=np.float32)], axis=1)
    camera = xyz1 @ (calib["R0_rect"] @ calib["Tr_velo_to_cam"]).T
    positive = camera[:, 2] > 0.1
    image = camera @ calib["P2"].T
    uv = image[:, :2] / np.maximum(image[:, 2:3], 1e-7)
    return uv[positive], camera[positive, :3]


def laser_proxy(box, label, uv, camera_points):
    width = box[2] - box[0]
    height = box[3] - box[1]
    roi = np.asarray([
        box[0] + 0.15 * width,
        box[1] + 0.20 * height,
        box[2] - 0.15 * width,
        box[3] - 0.05 * height,
    ])
    keep = ((uv[:, 0] >= roi[0]) & (uv[:, 0] <= roi[2]) &
            (uv[:, 1] >= roi[1]) & (uv[:, 1] <= roi[3]))
    pts = camera_points[keep]
    minimum = 5 if label == CLASSES["Car"] else 2
    if len(pts) < minimum:
        return None
    bins = np.floor(pts[:, 2]).astype(np.int32)
    values, counts = np.unique(bins, return_counts=True)
    peak = values[np.argmax(counts)]
    cluster = pts[np.abs(pts[:, 2] - (peak + 0.5)) <= 1.0]
    if len(cluster) < minimum or np.std(cluster[:, 2]) > 1.25:
        return None
    location = np.median(cluster, axis=0)
    location[2] += CLASS_LENGTH[label] * 0.5
    return location.astype(np.float32)


def make_predictions(raw, calib, points, args):
    cam_keep = raw["cam_scores"] >= args.camera_score
    cam_boxes = raw["cam_boxes"][cam_keep]
    cam_scores = raw["cam_scores"][cam_keep]
    cam_labels = raw["cam_labels"][cam_keep]
    cam_locations = monocular_locations(cam_boxes, cam_labels, calib)
    valid_camera = (cam_locations[:, 2] > 0) & (cam_locations[:, 2] < 70.0)
    camera = dict(boxes=cam_boxes[valid_camera], scores=cam_scores[valid_camera],
                  labels=cam_labels[valid_camera], locations=cam_locations[valid_camera])

    lid_keep = raw["lid_scores"] >= args.lidar_score
    lid_keep &= (raw["lid_locations"][:, 2] > 0) & (raw["lid_locations"][:, 2] < 70.0)
    lidar = dict(boxes=raw["lid_boxes2d"][lid_keep], scores=raw["lid_scores"][lid_keep],
                 labels=raw["lid_labels"][lid_keep], locations=raw["lid_locations"][lid_keep])
    pair_iou = iou2d(camera["boxes"], lidar["boxes"])
    candidates = []
    for ci in range(len(camera["boxes"])):
        for li in range(len(lidar["boxes"])):
            if camera["labels"][ci] == lidar["labels"][li] and pair_iou[ci, li] >= args.fusion_iou:
                candidates.append((pair_iou[ci, li], ci, li))
    pairs, used_c, used_l = [], set(), set()
    for _, ci, li in sorted(candidates, reverse=True):
        if ci not in used_c and li not in used_l:
            pairs.append((ci, li)); used_c.add(ci); used_l.add(li)

    fusion = dict(
        boxes=np.asarray([lidar["boxes"][li] for _, li in pairs], dtype=np.float32).reshape(-1, 4),
        scores=np.asarray([(camera["scores"][ci] + lidar["scores"][li]) / 2 for ci, li in pairs]),
        labels=np.asarray([lidar["labels"][li] for _, li in pairs], dtype=np.int64),
        locations=np.asarray([lidar["locations"][li] for _, li in pairs], dtype=np.float32).reshape(-1, 3),
    )

    uv, camera_points = project_points(points, calib)
    proposed_items = []
    for ci in range(len(camera["boxes"])):
        z = camera["locations"][ci, 2]
        if z > args.gate_distance:
            proposed_items.append((camera["boxes"][ci], camera["scores"][ci],
                                   camera["labels"][ci], camera["locations"][ci]))
            continue
        matched = next((li for c, li in pairs if c == ci), None)
        if matched is not None:
            proposed_items.append((lidar["boxes"][matched],
                                   max(camera["scores"][ci], lidar["scores"][matched]),
                                   camera["labels"][ci], lidar["locations"][matched]))
            continue
        proxy = None
        if camera["scores"][ci] >= args.laser_camera_score:
            proxy = laser_proxy(camera["boxes"][ci], int(camera["labels"][ci]), uv, camera_points)
        if proxy is not None:
            proposed_items.append((camera["boxes"][ci], camera["scores"][ci], camera["labels"][ci], proxy))
    for li in range(len(lidar["boxes"])):
        if (li not in used_l and lidar["locations"][li, 2] <= args.gate_distance
                and lidar["scores"][li] >= args.fallback_lidar_score):
            proposed_items.append((lidar["boxes"][li], lidar["scores"][li],
                                   lidar["labels"][li], lidar["locations"][li]))
    proposed = dict(
        boxes=np.asarray([x[0] for x in proposed_items], dtype=np.float32).reshape(-1, 4),
        scores=np.asarray([x[1] for x in proposed_items], dtype=np.float32),
        labels=np.asarray([x[2] for x in proposed_items], dtype=np.int64),
        locations=np.asarray([x[3] for x in proposed_items], dtype=np.float32).reshape(-1, 3),
    )
    return {"camera": camera, "lidar": lidar, "fusion": fusion, "proposed": proposed}


def moderate_ground_truth(records):
    selected, ignored = [], []
    for item in records:
        if item["ignore"]:
            ignored.append(item["bbox"])
            continue
        height = item["bbox"][3] - item["bbox"][1]
        distance = float(np.linalg.norm(item["location"][[0, 2]]))
        if (height >= 25 and item["occlusion"] <= 1 and item["truncation"] <= 0.3
                and distance < 70.0):
            selected.append(item)
        else:
            ignored.append(item["bbox"])
    return selected, np.asarray(ignored, dtype=np.float32).reshape(-1, 4)


def match_frame(pred, gt, ignored, threshold):
    if not gt:
        false_pos = list(range(len(pred["boxes"])))
        if len(ignored):
            overlaps = iou2d(pred["boxes"], ignored)
            false_pos = [i for i in false_pos if np.max(overlaps[i]) < 0.5]
        return [], false_pos
    gt_boxes = np.asarray([x["bbox"] for x in gt])
    gt_labels = np.asarray([x["label"] for x in gt])
    overlaps = iou2d(pred["boxes"], gt_boxes)
    candidates = []
    for pi in range(len(pred["boxes"])):
        for gi in range(len(gt)):
            if pred["labels"][pi] == gt_labels[gi] and overlaps[pi, gi] >= threshold:
                candidates.append((overlaps[pi, gi], pi, gi))
    matches, used_p, used_g = [], set(), set()
    for _, pi, gi in sorted(candidates, reverse=True):
        if pi not in used_p and gi not in used_g:
            matches.append((pi, gi)); used_p.add(pi); used_g.add(gi)
    false_pos = [i for i in range(len(pred["boxes"])) if i not in used_p]
    if len(ignored) and false_pos:
        ignored_iou = iou2d(pred["boxes"][false_pos], ignored)
        false_pos = [pi for row, pi in enumerate(false_pos) if np.max(ignored_iou[row]) < 0.5]
    return matches, false_pos


def run_evaluation(args):
    data = Path(args.data)
    velodyne_fallback = Path(args.velodyne_fallback) if args.velodyne_fallback else None
    label_fallback = Path(args.label_fallback) if args.label_fallback else None

    def data_file(folder, stem, suffix, fallback=None):
        primary = data / folder / f"{stem}{suffix}"
        if primary.exists():
            return primary
        if fallback is not None:
            candidate = fallback / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Missing {folder} file for KITTI frame {stem}")

    output = Path(args.output)
    pred_dir = output / "predictions"
    ids = sorted(p.stem for p in pred_dir.glob("*.npz"))
    if args.id_list:
        allowed = {
            line.strip() for line in Path(args.id_list).read_text().splitlines()
            if line.strip()
        }
        ids = [stem for stem in ids if stem in allowed]
        missing = allowed.difference(ids)
        if missing:
            raise FileNotFoundError(f"Missing predictions for {len(missing)} requested frames")
    totals = {m: {"gt": {b[2]: 0 for b in RANGE_BINS}, "tp": {b[2]: 0 for b in RANGE_BINS},
                  "class_gt": {(c, b[2]): 0 for c in CLASSES.values() for b in RANGE_BINS},
                  "class_tp": {(c, b[2]): 0 for c in CLASSES.values() for b in RANGE_BINS},
                  "fp": 0, "errors": [], "margins": []}
              for m in ("camera", "lidar", "fusion", "proposed")}
    times = {m: [] for m in totals}

    for index, stem in enumerate(ids, 1):
        base_raw = np.load(pred_dir / f"{stem}.npz")
        yolo_path = Path(args.output) / "camera_yolo" / f"{stem}.npz"
        if yolo_path.exists():
            yolo_raw = np.load(yolo_path)
            raw = {key: base_raw[key] for key in base_raw.files}
            for key in ("cam_boxes", "cam_scores", "cam_labels", "camera_seconds"):
                raw[key] = yolo_raw[key]
        else:
            raw = base_raw
        calib = read_calib(data / "calib" / f"{stem}.txt")
        points = np.fromfile(data_file("velodyne", stem, ".bin", velodyne_fallback),
                             dtype=np.float32).reshape(-1, 4)
        t0 = time.perf_counter()
        methods = make_predictions(raw, calib, points, args)
        post_seconds = time.perf_counter() - t0
        gt, ignored = moderate_ground_truth(parse_label(
            data_file("label_2", stem, ".txt", label_fallback)))
        camera_seconds = float(raw["camera_seconds"])
        lidar_seconds = float(raw["lidar_seconds"])
        times["camera"].append(camera_seconds)
        times["lidar"].append(lidar_seconds)
        times["fusion"].append(camera_seconds + lidar_seconds)
        times["proposed"].append(camera_seconds + lidar_seconds + post_seconds)

        for method, pred in methods.items():
            matches, false_pos = match_frame(pred, gt, ignored, args.match_iou)
            totals[method]["fp"] += len(false_pos)
            matched_gt = {gi: pi for pi, gi in matches}
            for gi, item in enumerate(gt):
                distance = float(np.linalg.norm(item["location"][[0, 2]]))
                bin_name = next((name for lo, hi, name in RANGE_BINS if lo <= distance < hi), None)
                if bin_name is None:
                    continue
                totals[method]["gt"][bin_name] += 1
                totals[method]["class_gt"][(item["label"], bin_name)] += 1
                if gi in matched_gt:
                    pi = matched_gt[gi]
                    totals[method]["tp"][bin_name] += 1
                    totals[method]["class_tp"][(item["label"], bin_name)] += 1
                    error = np.linalg.norm(pred["locations"][pi] - item["location"])
                    if bin_name == "near":
                        totals[method]["errors"].append(float(error))
                    margin = max(0.0, distance / args.reference_speed - times[method][-1])
                else:
                    margin = 0.0
                totals[method]["margins"].append(margin)
        if index % 250 == 0:
            print(f"evaluation {index}/{len(ids)}", flush=True)

    def wilson(successes, count, z=1.96):
        if count == 0:
            return [0.0, 0.0]
        p = successes / count
        den = 1 + z * z / count
        center = (p + z * z / (2 * count)) / den
        half = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / den
        return [float(center - half), float(center + half)]

    rows, class_rows = [], []
    for method, item in totals.items():
        recalls = {name: item["tp"][name] / item["gt"][name] if item["gt"][name] else 0
                   for _, _, name in RANGE_BINS}
        rows.append({
            "method": method,
            "frames": len(ids),
            "near_recall": recalls["near"],
            "near_recall_ci95": wilson(item["tp"]["near"], item["gt"]["near"]),
            "mid_recall": recalls["mid"],
            "mid_recall_ci95": wilson(item["tp"]["mid"], item["gt"]["mid"]),
            "far_recall": recalls["far"],
            "far_recall_ci95": wilson(item["tp"]["far"], item["gt"]["far"]),
            "near_localization_error_m": float(np.mean(item["errors"])) if item["errors"] else None,
            "near_localization_median_m": float(np.median(item["errors"])) if item["errors"] else None,
            "false_positives": item["fp"],
            "false_positives_per_frame": item["fp"] / len(ids),
            "mean_latency_ms": float(np.mean(times[method]) * 1000),
            "p95_latency_ms": float(np.percentile(times[method], 95) * 1000),
            "mean_available_margin_s": float(np.mean(item["margins"])) if item["margins"] else 0,
        })
        for class_name, class_id in CLASSES.items():
            class_row = {"method": method, "class": class_name}
            for _, _, bin_name in RANGE_BINS:
                count = item["class_gt"][(class_id, bin_name)]
                success = item["class_tp"][(class_id, bin_name)]
                class_row[f"{bin_name}_n"] = count
                class_row[f"{bin_name}_recall"] = success / count if count else None
            class_rows.append(class_row)
    camera_fp = next(x["false_positives"] for x in rows if x["method"] == "camera")
    lidar_margin = next(x["mean_available_margin_s"] for x in rows if x["method"] == "lidar")
    for row in rows:
        row["fp_reduction_vs_camera"] = ((camera_fp - row["false_positives"]) / camera_fp
                                          if camera_fp else 0)
        row["margin_gain_vs_lidar_s"] = row["mean_available_margin_s"] - lidar_margin

    output.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.metrics_tag}" if args.metrics_tag else ""
    (output / f"metrics{suffix}.json").write_text(json.dumps(
        {"settings": vars(args), "results": rows, "class_results": class_rows}, indent=2))
    with (output / f"metrics{suffix}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    with (output / f"class_metrics{suffix}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=class_rows[0].keys())
        writer.writeheader(); writer.writerows(class_rows)
    print(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("infer", "infer-yolo", "evaluate"))
    parser.add_argument("--data", default=str(ROOT / "kitti_val"))
    parser.add_argument("--output", default=str(ROOT / "trc_kitti" / "output"))
    parser.add_argument("--checkpoint", default=str(PP_ROOT / "pretrained" / "epoch_160.pth"))
    parser.add_argument("--yolo-checkpoint", default=str(ROOT / "trc_kitti" / "training" / "pilot" / "weights" / "best.pt"))
    parser.add_argument("--image-fallback", default="")
    parser.add_argument("--velodyne-fallback", default="")
    parser.add_argument("--label-fallback", default="")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-camera", action="store_true")
    parser.add_argument("--camera-floor", type=float, default=0.10)
    parser.add_argument("--yolo-batch", type=int, default=16)
    parser.add_argument("--camera-score", type=float, default=0.25)
    parser.add_argument("--lidar-score", type=float, default=0.10)
    parser.add_argument("--fusion-iou", type=float, default=0.30)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--gate-distance", type=float, default=40.0)
    parser.add_argument("--fallback-lidar-score", type=float, default=0.75)
    parser.add_argument("--laser-camera-score", type=float, default=0.35)
    parser.add_argument("--reference-speed", type=float, default=13.9)
    parser.add_argument("--metrics-tag", default="")
    parser.add_argument("--id-list", default="")
    args = parser.parse_args()
    if args.mode == "infer":
        run_inference(args)
    elif args.mode == "infer-yolo":
        run_yolo_inference(args)
    else:
        run_evaluation(args)


if __name__ == "__main__":
    main()
