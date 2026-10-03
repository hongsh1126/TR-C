import json
import os
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

import os

ROOT = Path(os.environ.get("TRC_EXPERIMENT_ROOT", Path(__file__).resolve().parents[2]))
W = ROOT / "trc_kitti" / "revision" / "results"
TAG = os.environ.get("TRC_TAG", "")
LOCK = pickle.load(open(W / f"analysis_raw_locked{TAG}.pkl", "rb"))
try:
    CAL = pickle.load(open(W / "analysis_raw_calib.pkl", "rb")) if not TAG else None
except FileNotFoundError:
    CAL = None
BINS = ["near", "mid", "far"]
B = 2000
rng = np.random.default_rng(2026)
NF = len(LOCK["ids"])
OUT = {}


def table(D, name):
    r = D["rows"][name]
    return {
        "fi": np.array([x[0] for x in r]), "gi": np.array([x[1] for x in r]), "cls": np.array([x[2] for x in r]),
        "bin": np.array([x[3] for x in r]), "dist": np.array([x[4] for x in r]), "m": np.array([x[5] for x in r]),
        "err": np.array([x[6] for x in r], dtype=float), "src": np.array([x[7] for x in r]),
    }


def per_frame(D, name, nf, bins=BINS, what="tp"):
    t = table(D, name)
    out = np.zeros(nf)
    sel = np.isin(t["bin"], bins)
    if what == "tp":
        np.add.at(out, t["fi"][sel], t["m"][sel])
    else:
        np.add.at(out, t["fi"][sel], 1)
    return out


Wts = rng.multinomial(NF, np.full(NF, 1.0 / NF), size=B).astype(float)


def recall_ci(tp, gt):
    r = tp.sum() / gt.sum()
    bs = (Wts @ tp) / (Wts @ gt)
    return r, np.percentile(bs, [2.5, 97.5])


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj


# ---------- S0 replication of the reported numbers
ref = json.load(open(ROOT / "trc_kitti" / "output_locked" / "metrics_locked.json"))["results"]
rep = {}
for r in ref:
    n = r["method"]
    T = table(LOCK, n)
    mine = {b: float(T["m"][T["bin"] == b].mean()) for b in BINS}
    mine["fp"] = int(sum(LOCK["fps"][n]))
    mine["near_err"] = float(np.nanmean(T["err"][(T["bin"] == "near") & (T["m"] == 1)]))
    rep[n] = dict(mine=mine, paper=dict(near=r["near_recall"], mid=r["mid_recall"], far=r["far_recall"], fp=r["false_positives"], near_err=r["near_localization_error_m"]))
OUT["replication"] = rep
print("REPLICATION")
for n, v in rep.items():
    print(n, {k: round(x, 4) for k, x in v["mine"].items()}, "| reported", {k: round(x, 4) for k, x in v["paper"].items()})

# ---------- S1 paired comparisons on recall
gt_b = {b: per_frame(LOCK, "camera", NF, [b], "gt") for b in BINS}
gt_b["all"] = sum(gt_b.values())
tp = {n: {b: per_frame(LOCK, n, NF, [b]) for b in BINS} for n in ("camera", "lidar", "fusion", "proposed", "gate30", "gate50", "gate60", "gate70", "no_roi", "no_fallback")}
for n in tp:
    tp[n]["all"] = sum(tp[n][b] for b in BINS)
comps = [("proposed", "camera"), ("proposed", "lidar"), ("lidar", "camera"), ("proposed", "fusion"), ("proposed", "gate70"), ("gate50", "proposed")]
res1 = []
for a, b in comps:
    for bn in BINS + ["all"]:
        d = tp[a][bn].sum() / gt_b[bn].sum() - tp[b][bn].sum() / gt_b[bn].sum()
        bs = (Wts @ tp[a][bn] - Wts @ tp[b][bn]) / (Wts @ gt_b[bn])
        p = max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1.0 / B)
        Ta, Tb = table(LOCK, a), table(LOCK, b)
        sel = (Ta["bin"] == bn) if bn != "all" else np.ones(len(Ta["m"]), bool)
        n01 = int(((Ta["m"][sel] == 0) & (Tb["m"][sel] == 1)).sum()); n10 = int(((Ta["m"][sel] == 1) & (Tb["m"][sel] == 0)).sum())
        pm = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        res1.append(dict(a=a, b=b, bin=bn, diff=float(d), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), p_boot=float(p),
                         a_only=n10, b_only=n01, p_mcnemar=float(pm)))
primary = [i for i, r in enumerate(res1) if (r["a"], r["b"]) in (("proposed", "camera"), ("proposed", "lidar")) and r["bin"] in BINS]
adj = holm(np.array([res1[i]["p_boot"] for i in primary]))
for i, a_ in zip(primary, adj):
    res1[i]["p_holm_boot"] = float(a_)
adj2 = holm(np.array([res1[i]["p_mcnemar"] for i in primary]))
for i, a_ in zip(primary, adj2):
    res1[i]["p_holm_mcnemar"] = float(a_)
OUT["paired_recall"] = res1
print("\nPAIRED RECALL (diff in percentage points, 95% cluster-bootstrap CI)")
for r in res1:
    print(f"{r['a']:>9} - {r['b']:<9} {r['bin']:>4}: {100*r['diff']:+6.2f} [{100*r['lo']:+6.2f},{100*r['hi']:+6.2f}] pboot={r['p_boot']:.3f} holm={r.get('p_holm_boot', float('nan')):.3f} | mcnemar a-only={r['a_only']} b-only={r['b_only']} p={r['p_mcnemar']:.4f} holm={r.get('p_holm_mcnemar', float('nan')):.3f}")

# false positives (paired by frame)
fpm = {n: np.array(LOCK["fps"][n], dtype=float) for n in ("camera", "lidar", "fusion", "proposed", "gate50", "gate70")}
fp_res = []
for a, b in (("proposed", "camera"), ("proposed", "lidar"), ("lidar", "camera"), ("fusion", "camera")):
    bs = (Wts @ fpm[a] - Wts @ fpm[b]) / NF
    fp_res.append(dict(a=a, b=b, diff_per_frame=float((fpm[a].sum() - fpm[b].sum()) / NF), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                       total_a=int(fpm[a].sum()), total_b=int(fpm[b].sum())))
OUT["paired_fp"] = fp_res
print("\nFALSE POSITIVES per frame difference")
for r in fp_res:
    print(r)

# ---------- S2 localization on a common matched subset (near field)
Tn = {n: table(LOCK, n) for n in ("camera", "lidar", "fusion", "proposed")}
key = lambda T: {(f, g): i for i, (f, g) in enumerate(zip(T["fi"], T["gi"]))}
idx = {n: key(Tn[n]) for n in Tn}
common, own = [], {}
base = Tn["camera"]
for i, (f, g) in enumerate(zip(base["fi"], base["gi"])):
    if base["bin"][i] != "near":
        continue
    ok = all(Tn[n]["m"][idx[n][(f, g)]] == 1 for n in ("camera", "lidar", "proposed"))
    if ok:
        common.append((f, g))
loc = {}
for n in Tn:
    T = Tn[n]
    sel = (T["bin"] == "near") & (T["m"] == 1)
    own[n] = dict(n=int(sel.sum()), mean=float(T["err"][sel].mean()), median=float(np.median(T["err"][sel])))
    ce = np.array([T["err"][idx[n][c]] for c in common])
    cf = np.array([c[0] for c in common])
    loc[n] = (ce, cf)
    own[n].update(common_n=len(common), common_mean=float(ce.mean()), common_median=float(np.median(ce)))
loc_pairs = []
for a, b in (("proposed", "camera"), ("proposed", "lidar")):
    ea, cf = loc[a]; eb, _ = loc[b]
    d = ea - eb
    sf = np.zeros(NF); cn = np.zeros(NF)
    np.add.at(sf, cf, d); np.add.at(cn, cf, 1)
    bs = (Wts @ sf) / (Wts @ cn)
    loc_pairs.append(dict(a=a, b=b, mean_diff=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5))))
OUT["localization"] = dict(own=own, pairs=loc_pairs)
print("\nLOCALIZATION near field: own-matched vs common subset (n common =", len(common), ")")
for n, v in own.items():
    print(n, {k: round(x, 3) if isinstance(x, float) else x for k, x in v.items()})
print(loc_pairs)
# source composition of proposed near-field matches
Tp = Tn["proposed"]
srcs = {}
for s in sorted(set(Tp["src"][(Tp["m"] == 1)])):
    sel = (Tp["m"] == 1) & (Tp["src"] == s)
    nb = (sel & (Tp["bin"] == "near"))
    srcs[s] = dict(n_all=int(sel.sum()), n_near=int(nb.sum()), mean_err_near=float(Tp["err"][nb].mean()) if nb.any() else None,
                   median_err_near=float(np.median(Tp["err"][nb])) if nb.any() else None)
OUT["proposed_sources"] = srcs
print("\nPROPOSED matched detections by source", srcs)
# ROI-proxy accuracy vs monocular and LiDAR for the same objects
Tc, Tl = Tn["camera"], Tn["lidar"]
px = [(i, f, g) for i, (f, g, s, m) in enumerate(zip(Tp["fi"], Tp["gi"], Tp["src"], Tp["m"])) if m == 1 and s == "roi_proxy"]
pe = np.array([Tp["err"][i] for i, _, _ in px]); ce = np.array([Tc["err"][idx["camera"][(f, g)]] for _, f, g in px])
lmatch = [(Tl["m"][idx["lidar"][(f, g)]] == 1) for _, f, g in px]
le = np.array([Tl["err"][idx["lidar"][(f, g)]] if mm else np.nan for (_, f, g), mm in zip(px, lmatch)])
cm = np.array([Tc["m"][idx["camera"][(f, g)]] == 1 for _, f, g in px])
proxy = dict(n=len(px), proxy_mean=float(pe.mean()), proxy_median=float(np.median(pe)), proxy_p90=float(np.percentile(pe, 90)),
             camera_mean_same_objects=float(np.nanmean(np.where(cm, ce, np.nan))), camera_n=int(cm.sum()),
             lidar_mean_same_objects=float(np.nanmean(le)), lidar_n=int(np.sum(lmatch)))
OUT["roi_proxy_accuracy"] = proxy
print("ROI proxy accuracy", proxy)
OUT["fp_by_source"] = {k: LOCK["srcfp"][k] for k in LOCK["srcfp"]}
print("FP by source", OUT["fp_by_source"])

# ---------- S3 gate misassignment
ex = np.array(LOCK["extra"], dtype=object)
cls_, binn, dist_, ciou, cscore, cz = ex[:, 2].astype(int), ex[:, 3], ex[:, 4].astype(float), ex[:, 6].astype(float), ex[:, 7].astype(float), ex[:, 8].astype(float)
kept = (ciou >= 0.5) & (cscore >= 0.25) & ~np.isnan(cz)
mis = {}
for b in BINS:
    s = kept & (binn == b)
    mis[b] = dict(n=int(s.sum()), z_le_40=float((cz[s] <= 40).mean()), z_gt_40=float((cz[s] > 40).mean()),
                  median_abs_depth_err=float(np.median(np.abs(cz[s] - ex[:, 5].astype(float)[s]))),
                  median_rel_depth_err=float(np.median((cz[s] - ex[:, 5].astype(float)[s]) / ex[:, 5].astype(float)[s])))
OUT["gate_assignment"] = mis
print("\nGATE ASSIGNMENT (share of camera-detected objects whose monocular z is <=40 or >40 m)")
for b, v in mis.items():
    print(b, {k: round(x, 3) for k, x in v.items()})

# ---------- S4 far-field vulnerable road users
far = (binn == "far")
vr = {}
for name, cset in (("pedestrian", [0]), ("cyclist", [1]), ("car", [2])):
    s = far & np.isin(cls_, cset)
    lidar_iou, lidar_sc = ex[:, 9].astype(float), ex[:, 10].astype(float)
    vr[name] = dict(n=int(s.sum()), cam_any=int((ciou[s] >= 0.5).sum()), cam_ge025=int(((ciou[s] >= 0.5) & (cscore[s] >= 0.25)).sum()),
                    lid_any=int((lidar_iou[s] >= 0.5).sum()), lid_ge05=int(((lidar_iou[s] >= 0.5) & (lidar_sc[s] >= 0.5)).sum()),
                    proposed_tp=0)
Tfar = table(LOCK, "proposed")
for name, c in (("pedestrian", 0), ("cyclist", 1), ("car", 2)):
    sel = (Tfar["bin"] == "far") & (Tfar["cls"] == c)
    vr[name]["proposed_tp"] = int(Tfar["m"][sel].sum())
    for m in ("camera", "lidar"):
        T = Tn[m]; s2 = (T["bin"] == "far") & (T["cls"] == c); vr[name][m + "_tp"] = int(T["m"][s2].sum())
OUT["far_field_classes"] = vr
print("\nFAR FIELD per class", json.dumps(vr))

# ---------- S5 timing dispersion
tm = np.array(LOCK["timing"]) * 1000
t_all = {"camera": tm[:, 0], "lidar": tm[:, 1], "fusion (camera+LiDAR)": tm[:, 0] + tm[:, 1], "range gated": tm.sum(1)}
tim = {}
for k, v in t_all.items():
    tim[k] = dict(n=len(v), mean=float(v.mean()), sd=float(v.std(ddof=1)), median=float(np.median(v)), q1=float(np.percentile(v, 25)), q3=float(np.percentile(v, 75)),
                  p95=float(np.percentile(v, 95)), p99=float(np.percentile(v, 99)), min=float(v.min()), max=float(v.max()))
OUT["timing_ms"] = tim
print("\nTIMING (ms)")
for k, v in tim.items():
    print(k, {a: round(b, 1) for a, b in v.items()})
# delay-vs-lidar paired difference
dd = (tm.sum(1) - tm[:, 1])
OUT["gated_minus_lidar_ms"] = dict(mean=float(dd.mean()), sd=float(dd.std(ddof=1)))
print("gated minus LiDAR-only latency per frame (ms): mean %.1f sd %.1f" % (dd.mean(), dd.std(ddof=1)))

# ---------- S6 matched false-positive comparison
sweep = {}
for m in ("camera", "lidar", "fusion", "proposed"):
    pts = []
    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        n = f"sw{t:.1f}_{m}"
        T = table(LOCK, n)
        pts.append((t, float(sum(LOCK["fps"][n])) / NF, float(T["m"].mean()), {b: float(T["m"][T["bin"] == b].mean()) for b in BINS}))
    sweep[m] = pts
OUT["sweep"] = sweep


def at_fp(points, target):
    pts = sorted((p[1], p[2]) for p in points)
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    if target < xs.min() or target > xs.max():
        return None
    return float(np.interp(target, xs, ys))


print("\nSWEEP (threshold, FP/frame, overall recall)")
for m, pts in sweep.items():
    print(m, [(p[0], round(p[1], 2), round(p[2], 3)) for p in pts])
OUT["recall_at_fp"] = {str(f): {m: at_fp(sweep[m], f) for m in sweep} for f in (0.5, 1.0, 2.0)}
print("recall at matched FP/frame:", OUT["recall_at_fp"])

# ---------- S7 gate variants and range-independent comparator
gv = {}
for n in ("camera", "lidar", "proposed", "gate30", "gate50", "gate60", "gate70", "no_roi", "no_fallback"):
    T = table(LOCK, n)
    gv[n] = {b: float(T["m"][T["bin"] == b].mean()) for b in BINS}
    gv[n]["all"] = float(T["m"].mean())
    gv[n]["fp"] = int(sum(LOCK["fps"][n]))
    gv[n]["near_err"] = float(np.nanmean(T["err"][(T["bin"] == "near") & (T["m"] == 1)]))
OUT["variants"] = gv
print("\nVARIANTS")
for n, v in gv.items():
    print(n, {k: round(x, 4) for k, x in v.items()})

# ---------- S8 calibration-selected gate and per-modality thresholds
if CAL is not None:
    NFc = len(CAL["ids"])
    def overall(D, n, nf):
        T = table(D, n); return float(T["m"].mean()), float(sum(D["fps"][n]))
    gcal = {}
    for n in ("proposed", "gate30", "gate50", "gate60", "gate70"):
        r, fpv = overall(CAL, n, NFc)
        Tl = table(LOCK, n)
        gcal[n] = dict(cal_recall=r, cal_fp=fpv, locked_recall=float(Tl["m"].mean()), locked_fp=int(sum(LOCK["fps"][n])))
    OUT["gate_selection"] = gcal
    print("\nGATE SELECTION on calibration subset"); [print(k, {a: round(b, 4) for a, b in v.items()}) for k, v in gcal.items()]
    f1 = {}
    for m in ("camera", "lidar", "proposed"):
        row = []
        for t in (0.1, 0.3, 0.5, 0.7, 0.9):
            Tc_ = table(CAL, f"sw{t:.1f}_{m}"); tpv = Tc_["m"].sum(); fnv = (Tc_["m"] == 0).sum(); fpv = sum(CAL["fps"][f"sw{t:.1f}_{m}"])
            row.append((t, float(2 * tpv / (2 * tpv + fpv + fnv))))
        f1[m] = row
    best = {m: max(f1[m], key=lambda x: x[1])[0] for m in f1}
    OUT["f1_calibration"] = dict(f1=f1, best=best)
    print("F1 on calibration", f1, "best", best)
    cl = {}
    for m in ("camera", "lidar"):
        n = f"sw{best[m]:.1f}_{m}"; T = table(LOCK, n)
        cl[m] = dict(threshold=best[m], **{b: float(T["m"][T["bin"] == b].mean()) for b in BINS}, fp=int(sum(LOCK["fps"][n])))
    OUT["calibrated_thresholds_locked"] = cl
    print("locked results at calibration-selected thresholds", cl)

json.dump(OUT, open(W / f"r1_results{TAG}.json", "w"), indent=1, default=float)
