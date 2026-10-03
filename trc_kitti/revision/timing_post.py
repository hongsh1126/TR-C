import sys, time, pickle
sys.argv = ["x", "locked"]
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import analysis_r1 as A
post = []
for i, stem in enumerate(A.IDS):
    base = np.load(A.OUT / "predictions" / f"{stem}.npz"); yolo = np.load(A.OUT / "camera_yolo" / f"{stem}.npz")
    raw = {k: base[k] for k in base.files}
    for k in ("cam_boxes", "cam_scores", "cam_labels", "camera_seconds"):
        raw[k] = yolo[k]
    calib = A.read_calib(A.DATA / "calib" / f"{stem}.txt")
    pts = np.fromfile(A.DATA / "velodyne" / f"{stem}.bin", dtype=np.float32).reshape(-1, 4)
    t0 = time.perf_counter()
    uvcp = A.project_points(pts, calib)
    A.make_predictions(raw, calib, uvcp, A.DEFAULT)
    post.append(time.perf_counter() - t0)
pickle.dump(post, open(A.RES / "post_seconds.pkl", "wb"))
print("done", np.mean(post) * 1000, np.std(post) * 1000)
