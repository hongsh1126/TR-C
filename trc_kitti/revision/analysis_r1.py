"""Additional analyses for the revision, computed from the saved per-frame detections.
Pure numpy; re-implements run_benchmark.py (evaluate mode) with per-object logging."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import os

EXP = Path(os.environ.get("TRC_EXPERIMENT_ROOT", Path(__file__).resolve().parents[2]))  # repository root
DATA = EXP / "kitti_val"
OUT = EXP / "trc_kitti" / "output_locked"  # needs predictions/ and camera_yolo/ (see README)
RES = EXP / "trc_kitti" / "revision" / "results"
RES.mkdir(parents=True, exist_ok=True)
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "locked"
PRED = Path(os.environ.get("TRC_PRED_DIR", str(OUT / "predictions")))
TAG = os.environ.get("TRC_TAG", "")
IDS = [s.strip() for s in (EXP / "trc_kitti" / ("locked_test_ids.txt" if SPLIT == "locked" else "calibration_ids.txt")).read_text().split() if s.strip()]

CLASSES = {"Pedestrian": 0, "Cyclist": 1, "Car": 2}
CLASS_HEIGHT = np.array([1.70, 1.70, 1.50], dtype=np.float32)
CLASS_LENGTH = np.array([0.80, 1.76, 3.90], dtype=np.float32)
BINS = ((0.0, 20.0, "near"), (20.0, 40.0, "mid"), (40.0, 70.0, "far"))
REF_SPEED = 13.9


def read_calib(path):
    lines = [l.strip() for l in open(path).readlines()]
    g = lambda i, shape: np.array(lines[i].split(" ")[1:], dtype=np.float32).reshape(shape)
    P2 = np.concatenate([g(2, (3, 4)), np.array([[0, 0, 0, 1]], dtype=np.float32)], 0)
    R0 = np.eye(4, dtype=np.float32); R0[:3, :3] = g(4, (3, 3))
    Tr = np.concatenate([g(5, (3, 4)), np.array([[0, 0, 0, 1]], dtype=np.float32)], 0)
    return {"P2": P2, "R0_rect": R0, "Tr_velo_to_cam": Tr}


def parse_label(path):
    out = []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        bbox = np.asarray(p[4:8], dtype=np.float32)
        if p[0] not in CLASSES:
            if p[0] == "DontCare":
                out.append({"label": -1, "bbox": bbox, "ignore": True})
            continue
        out.append({"label": CLASSES[p[0]], "truncation": float(p[1]), "occlusion": int(p[2]), "bbox": bbox,
                    "location": np.asarray(p[11:14], dtype=np.float32), "ignore": False})
    return out


def iou2d(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(rb - lt, 0); inter = wh[:, :, 0] * wh[:, :, 1]
    aa = np.maximum(a[:, 2] - a[:, 0], 0) * np.maximum(a[:, 3] - a[:, 1], 0)
    ab = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-7)


def monocular_locations(boxes, labels, calib):
    if len(boxes) == 0:
        return np.empty((0, 3), dtype=np.float32)
    p2 = calib["P2"]
    heights = np.maximum(boxes[:, 3] - boxes[:, 1], 2.0)
    z = p2[1, 1] * CLASS_HEIGHT[labels] / heights
    u = (boxes[:, 0] + boxes[:, 2]) * 0.5; v = boxes[:, 3]
    return np.stack([(u - p2[0, 2]) * z / p2[0, 0], (v - p2[1, 2]) * z / p2[1, 1], z], axis=1).astype(np.float32)


def project_points(points, calib):
    xyz1 = np.concatenate([points[:, :3], np.ones((len(points), 1), dtype=np.float32)], axis=1)
    camera = xyz1 @ (calib["R0_rect"] @ calib["Tr_velo_to_cam"]).T
    positive = camera[:, 2] > 0.1
    image = camera @ calib["P2"].T
    uv = image[:, :2] / np.maximum(image[:, 2:3], 1e-7)
    return uv[positive], camera[positive, :3]


def laser_proxy(box, label, uv, cp):
    w = box[2] - box[0]; h = box[3] - box[1]
    roi = np.asarray([box[0] + 0.15 * w, box[1] + 0.20 * h, box[2] - 0.15 * w, box[3] - 0.05 * h])
    keep = (uv[:, 0] >= roi[0]) & (uv[:, 0] <= roi[2]) & (uv[:, 1] >= roi[1]) & (uv[:, 1] <= roi[3])
    pts = cp[keep]
    minimum = 5 if label == CLASSES["Car"] else 2
    if len(pts) < minimum:
        return None
    bins = np.floor(pts[:, 2]).astype(np.int32)
    values, counts = np.unique(bins, return_counts=True)
    peak = values[np.argmax(counts)]
    cluster = pts[np.abs(pts[:, 2] - (peak + 0.5)) <= 1.0]
    if len(cluster) < minimum or np.std(cluster[:, 2]) > 1.25:
        return None
    loc = np.median(cluster, axis=0); loc[2] += CLASS_LENGTH[label] * 0.5
    return loc.astype(np.float32)


DEFAULT = dict(camera_score=0.25, lidar_score=0.50, fusion_iou=0.30, gate=40.0, fallback=0.75, laser_cam=0.35)


def make_predictions(raw, calib, uvcp, P):
    cam_keep = raw["cam_scores"] >= P["camera_score"]
    cb, cs, cl = raw["cam_boxes"][cam_keep], raw["cam_scores"][cam_keep], raw["cam_labels"][cam_keep]
    cloc = monocular_locations(cb, cl, calib)
    ok = (cloc[:, 2] > 0) & (cloc[:, 2] < 70.0)
    cam = dict(boxes=cb[ok], scores=cs[ok], labels=cl[ok], locations=cloc[ok], source=["cam"] * int(ok.sum()))
    lk = raw["lid_scores"] >= P["lidar_score"]
    lk &= (raw["lid_locations"][:, 2] > 0) & (raw["lid_locations"][:, 2] < 70.0)
    lid = dict(boxes=raw["lid_boxes2d"][lk], scores=raw["lid_scores"][lk], labels=raw["lid_labels"][lk],
               locations=raw["lid_locations"][lk], source=["lid"] * int(lk.sum()))
    piou = iou2d(cam["boxes"], lid["boxes"])
    cands = [(piou[ci, li], ci, li) for ci in range(len(cam["boxes"])) for li in range(len(lid["boxes"]))
             if cam["labels"][ci] == lid["labels"][li] and piou[ci, li] >= P["fusion_iou"]]
    pairs, uc, ul = [], set(), set()
    for _, ci, li in sorted(cands, reverse=True):
        if ci not in uc and li not in ul:
            pairs.append((ci, li)); uc.add(ci); ul.add(li)
    fus = dict(boxes=np.asarray([lid["boxes"][li] for _, li in pairs], dtype=np.float32).reshape(-1, 4),
               scores=np.asarray([(cam["scores"][ci] + lid["scores"][li]) / 2 for ci, li in pairs]),
               labels=np.asarray([lid["labels"][li] for _, li in pairs], dtype=np.int64),
               locations=np.asarray([lid["locations"][li] for _, li in pairs], dtype=np.float32).reshape(-1, 3),
               source=["fus"] * len(pairs))
    uv, cp = uvcp
    items = []
    pmap = {c: l for c, l in pairs}
    for ci in range(len(cam["boxes"])):
        z = cam["locations"][ci, 2]
        if z > P["gate"]:
            items.append((cam["boxes"][ci], cam["scores"][ci], cam["labels"][ci], cam["locations"][ci], "far_cam"))
            continue
        if ci in pmap:
            li = pmap[ci]
            items.append((lid["boxes"][li], max(cam["scores"][ci], lid["scores"][li]), cam["labels"][ci], lid["locations"][li], "lidar_match"))
            continue
        proxy = None
        if cam["scores"][ci] >= P["laser_cam"]:
            proxy = laser_proxy(cam["boxes"][ci], int(cam["labels"][ci]), uv, cp)
        if proxy is not None:
            items.append((cam["boxes"][ci], cam["scores"][ci], cam["labels"][ci], proxy, "roi_proxy"))
    for li in range(len(lid["boxes"])):
        if li not in ul and lid["locations"][li, 2] <= P["gate"] and lid["scores"][li] >= P["fallback"]:
            items.append((lid["boxes"][li], lid["scores"][li], lid["labels"][li], lid["locations"][li], "lidar_fallback"))
    prop = dict(boxes=np.asarray([x[0] for x in items], dtype=np.float32).reshape(-1, 4),
                scores=np.asarray([x[1] for x in items], dtype=np.float32),
                labels=np.asarray([x[2] for x in items], dtype=np.int64),
                locations=np.asarray([x[3] for x in items], dtype=np.float32).reshape(-1, 3),
                source=[x[4] for x in items])
    return {"camera": cam, "lidar": lid, "fusion": fus, "proposed": prop}


def moderate_gt(records):
    sel, ign = [], []
    for it in records:
        if it["ignore"]:
            ign.append(it["bbox"]); continue
        h = it["bbox"][3] - it["bbox"][1]
        d = float(np.linalg.norm(it["location"][[0, 2]]))
        if h >= 25 and it["occlusion"] <= 1 and it["truncation"] <= 0.3 and d < 70.0:
            sel.append(it)
        else:
            ign.append(it["bbox"])
    return sel, np.asarray(ign, dtype=np.float32).reshape(-1, 4)


def match_frame(pred, gt, ign, thr=0.5):
    n = len(pred["boxes"])
    if not gt:
        fp = list(range(n))
        if len(ign) and n:
            ov = iou2d(pred["boxes"], ign)
            fp = [i for i in fp if np.max(ov[i]) < 0.5]
        return [], fp
    gb = np.asarray([x["bbox"] for x in gt]); gl = np.asarray([x["label"] for x in gt])
    ov = iou2d(pred["boxes"], gb)
    cands = [(ov[p, g], p, g) for p in range(n) for g in range(len(gt)) if pred["labels"][p] == gl[g] and ov[p, g] >= thr]
    m, up, ug = [], set(), set()
    for _, p, g in sorted(cands, reverse=True):
        if p not in up and g not in ug:
            m.append((p, g)); up.add(p); ug.add(g)
    fp = [i for i in range(n) if i not in up]
    if len(ign) and fp:
        io = iou2d(pred["boxes"][fp], ign)
        fp = [p for r, p in enumerate(fp) if np.max(io[r]) < 0.5]
    return m, fp


def build_variants():
    V = {}
    for m in ("camera", "lidar", "fusion", "proposed"):
        V[m] = (m, dict(DEFAULT))
    for g in (30.0, 50.0, 60.0, 70.0):
        V[f"gate{int(g)}"] = ("proposed", dict(DEFAULT, gate=g))
    V["no_roi"] = ("proposed", dict(DEFAULT, laser_cam=2.0))
    V["no_fallback"] = ("proposed", dict(DEFAULT, fallback=2.0))
    # threshold sweeps (camera_score=lidar_score=t) for matched-false-positive comparison
    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        for m in ("camera", "lidar", "fusion", "proposed"):
            V[f"sw{t:.1f}_{m}"] = (m, dict(DEFAULT, camera_score=t, lidar_score=t, laser_cam=max(0.35, t), fallback=max(0.75, t)))
    return V


def main():
    t0 = time.time()
    V = build_variants()
    rows = {k: [] for k in V}       # per-GT rows
    fps = {k: [] for k in V}        # per-frame FP counts
    srcfp = {k: {} for k in ("proposed", "no_roi")}
    extra = []                      # per-GT diagnostics independent of the method
    timing = []
    for fi, stem in enumerate(IDS):
        base = np.load(PRED / f"{stem}.npz")
        yolo = np.load(OUT / "camera_yolo" / f"{stem}.npz")
        raw = {k: base[k] for k in base.files}
        for k in ("cam_boxes", "cam_scores", "cam_labels", "camera_seconds"):
            raw[k] = yolo[k]
        calib = read_calib(DATA / "calib" / f"{stem}.txt")
        pts = np.fromfile(DATA / "velodyne" / f"{stem}.bin", dtype=np.float32).reshape(-1, 4)
        uvcp = project_points(pts, calib)
        gt, ign = moderate_gt(parse_label(DATA / "label_2" / f"{stem}.txt"))
        tp0 = time.perf_counter()
        make_predictions(raw, calib, uvcp, DEFAULT)
        post = time.perf_counter() - tp0
        timing.append((float(raw["camera_seconds"]), float(raw["lidar_seconds"]), post))
        # diagnostics per GT: gate assignment and raw detections
        cb_all = raw["cam_boxes"]; cl_all = raw["cam_labels"]; cs_all = raw["cam_scores"]
        cloc_all = monocular_locations(cb_all, cl_all, calib) if len(cb_all) else np.empty((0, 3))
        for gi, g in enumerate(gt):
            d = float(np.linalg.norm(g["location"][[0, 2]]))
            b = next((n for lo, hi, n in BINS if lo <= d < hi), None)
            if b is None:
                continue
            best_c = (0.0, 0.0, np.nan); best_l = (0.0, 0.0)
            if len(cb_all):
                io = iou2d(cb_all, g["bbox"][None])[:, 0]
                io = np.where(cl_all == g["label"], io, 0)
                if io.max() >= 0.5:
                    j = int(np.argmax(io * (cs_all >= 0)))
                    best_c = (float(io[j]), float(cs_all[j]), float(cloc_all[j, 2]))
            if len(raw["lid_boxes2d"]):
                io = iou2d(raw["lid_boxes2d"], g["bbox"][None])[:, 0]
                io = np.where(raw["lid_labels"] == g["label"], io, 0)
                if io.max() >= 0.5:
                    j = int(np.argmax(io)); best_l = (float(io[j]), float(raw["lid_scores"][j]))
            extra.append((fi, gi, g["label"], b, d, g["location"][2], best_c[0], best_c[1], best_c[2], best_l[0], best_l[1]))
        cache = {}
        for name, (method, P) in V.items():
            key = json.dumps(P, sort_keys=True)
            if key not in cache:
                cache[key] = make_predictions(raw, calib, uvcp, P)
            pred = cache[key][method]
            m, fp = match_frame(pred, gt, ign)
            fps[name].append(len(fp))
            if name in srcfp:
                for i in fp:
                    srcfp[name][pred["source"][i]] = srcfp[name].get(pred["source"][i], 0) + 1
            mg = {g: p for p, g in m}
            for gi, g in enumerate(gt):
                d = float(np.linalg.norm(g["location"][[0, 2]]))
                b = next((n for lo, hi, n in BINS if lo <= d < hi), None)
                if b is None:
                    continue
                if gi in mg:
                    p = mg[gi]
                    err = float(np.linalg.norm(pred["locations"][p] - g["location"]))
                    src = pred["source"][p]
                    rows[name].append((fi, gi, g["label"], b, d, 1, err, src))
                else:
                    rows[name].append((fi, gi, g["label"], b, d, 0, np.nan, ""))
        if (fi + 1) % 200 == 0:
            print(f"{SPLIT} {fi + 1}/{len(IDS)} ({time.time() - t0:.0f}s)", flush=True)
    import pickle
    pickle.dump(dict(rows=rows, fps=fps, srcfp=srcfp, extra=extra, timing=timing, ids=IDS), open(RES / f"analysis_raw_{SPLIT}{TAG}.pkl", "wb"))
    print("saved", time.time() - t0)


if __name__ == "__main__":
    main()
