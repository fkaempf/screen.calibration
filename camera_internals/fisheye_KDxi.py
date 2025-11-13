import cv2
import numpy as np
import glob
import sys
sys.path.append(r"D:\screen.calibration")
import os, json, hashlib, numpy as np, cv2

# --- config ---
IMG_GLOB = 'calibration.pictures/1800 U 501m NIR-07XC0/automatic.chessboard/*.png'
SQUARE_SIZE = 1.0
PATTERNS = [(6, 9), (9, 6)]
TOP_N_BAD = 10

# --- criteria ---
SUBPIX_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-5)
CALIB_CRIT  = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-10)

def build_obj_grid(cols, rows, sq):
    obj = np.zeros((rows * cols, 3), np.float64)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq
    return obj.reshape(-1, 1, 3)

def collect_points(images, shape):
    cols, rows = shape
    obj_grid = build_obj_grid(cols, rows, SQUARE_SIZE)
    objpoints, imgpoints, keep = [], [], []
    _img_shape = None

    for fname in images:
        img = cv2.imread(fname, cv2.IMREAD_COLOR)
        if img is None:
            continue
        if _img_shape is None:
            _img_shape = img.shape[:2]  # (h,w)
        else:
            if _img_shape != img.shape[:2]:
                continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cb_flags = (cv2.CALIB_CB_NORMALIZE_IMAGE |
                    cv2.CALIB_CB_EXHAUSTIVE |
                    cv2.CALIB_CB_ACCURACY)
        ok, corners = cv2.findChessboardCornersSB(gray, (cols, rows), flags=cb_flags)
        if not ok:
            continue

        corners = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), SUBPIX_CRIT)

        objpoints.append(obj_grid.copy())
        imgpoints.append(corners.reshape(-1, 1, 2).astype(np.float64))
        keep.append(fname)

    return objpoints, imgpoints, keep, _img_shape

def omnidir_calibrate(objpoints, imgpoints, img_shape):
    h, w = img_shape
    size = (int(w), int(h))
    flags = cv2.omnidir.CALIB_FIX_SKEW

    K  = np.eye(3, dtype=np.float64)
    D  = np.zeros((1, 4), dtype=np.float64)
    xi = np.array([1.2], dtype=np.float64)

    rms, K, xi, D, rvecs, tvecs, used_idx = cv2.omnidir.calibrate(
        objectPoints=objpoints,
        imagePoints=imgpoints,
        size=size,
        K=K, xi=xi, D=D,
        rvecs=None, tvecs=None,
        flags=flags,
        criteria=CALIB_CRIT
    )
    return rms, K, xi, D, rvecs, tvecs, used_idx

# OpenCV projector wrapper (flag positional in 4.12)
def project_points_cv(op, rv, tv, K, xi, D, perspective=True):
    op = np.asarray(op, np.float64).reshape(-1, 1, 3)
    rv = np.asarray(rv, np.float64).reshape(3, 1)
    tv = np.asarray(tv, np.float64).reshape(3, 1)
    K  = np.asarray(K,  np.float64)
    D  = np.asarray(D,  np.float64)
    xi_val = float(np.asarray(xi, np.float64).ravel()[0])
    flag = 1 if perspective else 0
    out = cv2.omnidir.projectPoints(op, rv, tv, K, xi_val, D, flag)
    return out[0] if isinstance(out, tuple) else out  # (N,1,2)

# Errors when arrays are ALREADY ALIGNED (same ordering/length across lists)
def per_view_errors_aligned(obj_list, img_list, rvecs, tvecs, K, xi, D, names_list):
    errs, per_file = [], []
    for op, ip, rv, tv, name in zip(obj_list, img_list, rvecs, tvecs, names_list):
        proj = project_points_cv(op, rv, tv, K, xi, D, perspective=True)
        e = np.linalg.norm(proj.reshape(-1, 2) - ip.reshape(-1, 2), axis=1).mean()
        errs.append(e)
        per_file.append((name, e))
    return np.asarray(errs, dtype=np.float64), per_file

def robust_filter_aligned(obj_list, img_list, rvecs, tvecs, names_list, errs, z=2.5):
    med = np.median(errs)
    mad = np.median(np.abs(errs - med)) + 1e-9
    thr = med + z * 1.4826 * mad
    mask = errs <= thr
    select = lambda xs: [x for x, m in zip(xs, mask) if m]
    return select(obj_list), select(img_list), select(rvecs), select(tvecs), select(names_list), mask, thr

def run_for_shape(images, shape):
    objpoints, imgpoints, keep, img_shape = collect_points(images, shape)
    if len(objpoints) < 3 or img_shape is None:
        return None

    # First calibration
    rms, K, xi, D, rvecs, tvecs, used_idx = omnidir_calibrate(objpoints, imgpoints, img_shape)

    # Align lists to used_idx
    used_idx = np.asarray(used_idx, dtype=int).ravel()
    obj_used = [objpoints[i] for i in used_idx]
    img_used = [imgpoints[i] for i in used_idx]
    names_used = [keep[i] for i in used_idx]

    # Errors on aligned lists
    errs, per_file = per_view_errors_aligned(obj_used, img_used, rvecs, tvecs, K, xi, D, names_used)

    # Robust prune and refit on ALIGNED lists
    op_f, ip_f, rv_f, tv_f, names_f, mask, thr = robust_filter_aligned(
        obj_used, img_used, rvecs, tvecs, names_used, errs, z=2.5
    )

    if len(op_f) >= 3 and len(op_f) < len(obj_used):
        # Refit
        h, w = img_shape
        size = (int(w), int(h))
        flags = cv2.omnidir.CALIB_FIX_SKEW
        K2  = np.eye(3, dtype=np.float64)
        D2  = np.zeros((1, 4), dtype=np.float64)
        xi2 = np.array([1.2], dtype=np.float64)

        rms2, K2, xi2, D2, rvecs2, tvecs2, used_idx2 = cv2.omnidir.calibrate(
            objectPoints=op_f,
            imagePoints=ip_f,
            size=size,
            K=K2, xi=xi2, D=D2,
            rvecs=None, tvecs=None,
            flags=flags,
            criteria=CALIB_CRIT
        )

        # Align second pass (indices relative to filtered lists)
        used_idx2 = np.asarray(used_idx2, dtype=int).ravel()
        op2 = [op_f[i] for i in used_idx2]
        ip2 = [ip_f[i] for i in used_idx2]
        names2 = [names_f[i] for i in used_idx2]

        errs2, per_file2 = per_view_errors_aligned(op2, ip2, rvecs2, tvecs2, K2, xi2, D2, names2)

        return {
            "shape": shape, "rms": rms2, "K": K2, "xi": xi2, "D": D2,
            "rvecs": rvecs2, "tvecs": tvecs2, "errs": errs2,
            "kept": len(errs2), "total": len(errs),
            "img_shape": img_shape, "per_file": per_file2
        }
    else:
        # No refit; return first pass
        return {
            "shape": shape, "rms": rms, "K": K, "xi": xi, "D": D,
            "rvecs": rvecs, "tvecs": tvecs, "errs": errs,
            "kept": len(errs), "total": len(errs),
            "img_shape": img_shape, "per_file": per_file
        }

# --- run both orientations; pick best by RMS ---
images = sorted(glob.glob(IMG_GLOB))
results = [run_for_shape(images, shp) for shp in PATTERNS]
results = [r for r in results if r is not None]
assert results, "No valid detections."
best = min(results, key=lambda r: r["rms"])

errs = best["errs"]
print(f"Chosen shape: {best['shape']}")
print(f"Views kept: {best['kept']} / {best['total']}")
print(f"RMS: {best['rms']:.3f} px")
print(f"Per-view mean error: mean={errs.mean():.2f} px, median={np.median(errs):.2f} px, max={errs.max():.2f} px")
print("K:\n", best["K"])
print("xi:", best["xi"].ravel())
print("D:",  best["D"].ravel())

# --- list worst N views (from the pass that 'best' represents) ---
per_file_sorted = sorted(best["per_file"], key=lambda x: x[1], reverse=True)
print("\nWorst views:")
for i, (name, e) in enumerate(per_file_sorted[:TOP_N_BAD], 1):
    print(f" {i}. {name:>20}  error={e:.2f} px")


# %%
def draw_reprojection(img_path, shape, K, xi, D, rvec, tvec, square_size=1.0, out_path=None):
    cols, rows = shape
    obj = np.zeros((rows*cols,3), np.float64)
    obj[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2)*square_size
    obj = obj.reshape(-1,1,3)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    proj, _ = cv2.omnidir.projectPoints(
        obj, np.asarray(rvec, np.float64).reshape(3,1),
        np.asarray(tvec, np.float64).reshape(3,1),
        np.asarray(K, np.float64),
        float(np.asarray(xi).ravel()[0]),
        np.asarray(D, np.float64),
        1  # 1=perspective
    )
    for p in proj.reshape(-1,2).astype(int):
        cv2.drawMarker(img, tuple(p), (0,255,0), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)

    if out_path: cv2.imwrite(out_path, img)
    else:
        cv2.imshow("reprojection", img); cv2.waitKey(0); cv2.destroyAllWindows()

# example: overlay the current worst view
import os
os.makedirs("debug/fisheye.KDxi/reproj", exist_ok=True)
for it in per_file_sorted:
    name, _ = it
    idx = [n for n,_ in best["per_file"]].index(name)
    rv, tv = best["rvecs"][idx], best["tvecs"][idx]
    draw_reprojection(
    name, best["shape"], best["K"], best["xi"], best["D"], rv, tv,
    out_path=f"debug/fisheye.KDxi/reproj/{os.path.basename(name)}_reproj.png"
)


name = 'calibration.pictures/1800 U 501m NIR-07XC0/automatic.chessboard/0061.png'
img = cv2.imread(name, cv2.IMREAD_COLOR)
h, w = img.shape[:2]


K  = np.ascontiguousarray(np.asarray(best["K"],  np.float64).reshape(3,3))
D  = np.ascontiguousarray(np.asarray(best["D"],  np.float64).reshape(1,4))
xi = np.ascontiguousarray(np.array([float(np.asarray(best["xi"]).ravel()[0])], dtype=np.float64))

zoom = 0.3  # < 1 zooms out, > 1 zooms in
Knew = K.copy()
Knew[0, 0] *= zoom   # fx
Knew[1, 1] *= zoom   # fy
Knew[0, 2] = w / 2.0 # keep principal point centered
Knew[1, 2] = h / 2.0


new_size=None
if new_size is None:
        new_size = (w, h)
K  = np.asarray(K,  np.float64).reshape(3, 3)
D  = np.asarray(D,  np.float64).reshape(1, 4)   # (1,4)
xi = float(np.asarray(xi).ravel()[0])
xi = np.array([xi], dtype=np.float64)    



os.makedirs("debug/fisheye.KDxi/rectified", exist_ok=True)
os.makedirs("debug/fisheye.KDxi/reproj", exist_ok=True)

for it in per_file_sorted:
    name, _ = it
    idx = [n for n,_ in best["per_file"]].index(name)
    
    img = cv2.imread(name, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
        
    rectified = cv2.omnidir.undistortImage(
    img, 
    K = K, 
    xi = xi,
    D = D, 
    flags  = 1, 
    Knew = Knew, 
    new_size  = new_size
)
    
    scale = 1/3
    orig_small = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    rect_small = cv2.resize(rectified, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # make sure sizes match
    h = min(orig_small.shape[0], rect_small.shape[0])
    w = min(orig_small.shape[1], rect_small.shape[1])
    orig_small = orig_small[:h, :w]
    rect_small = rect_small[:h, :w]

    combined = cv2.hconcat([orig_small, rect_small])
    out_path=f"debug/fisheye.KDxi/rectified/{os.path.basename(name)}_rectified.png"
    cv2.imwrite(out_path, combined)
    

    import glob, cv2, os

def make_video(folder, fps=15):
    files = sorted(glob.glob(f"{folder}/*.png"))
    if not files: return
    h, w = cv2.imread(files[0]).shape[:2]
    out = cv2.VideoWriter(f"{folder}/output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    [out.write(cv2.imread(f)) for f in files if cv2.imread(f) is not None]
    out.release()

make_video("debug/fisheye.KDxi/rectified")
make_video("debug/fisheye.KDxi/reproj")


    
    
import os, glob, cv2, numpy as np

def build_rectify_map_for_size_alt(w, h, K, D, xi, zoom=0.3):
    Knew = K.copy()
    Knew[0,0] = K[0,0] * zoom
    Knew[1,1] = K[1,1] * zoom
    Knew[0,2] = w / 2.0
    Knew[1,2] = h / 2.0
    size = (int(w), int(h))
    # OpenCV 4.12 order: (K, D, xi, R, Knew, size, m1type, flags)
    map1, map2 = cv2.omnidir.initUndistortRectifyMap(K, D, xi, np.eye(3, dtype=np.float64), Knew, size, 5, 1)
    return map1, map2

# choose files; reuse your per_file_sorted if present
files_alt = [n for (n, _) in per_file_sorted] if 'per_file_sorted' in globals() else sorted(glob.glob("checkerboard_samples/*.png"))


os.makedirs("debug/fisheye.KDxi/rectified.map", exist_ok=True)
for path in files_alt:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        continue
    h_, w_ = img.shape[:2]
    m1, m2 = build_rectify_map_for_size_alt(w_, h_, K, D, xi, zoom=0.30)
    rect = cv2.remap(img, m1, m2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # quick side-by-side preview at 1/3 size
    s = 1/3
    a = cv2.resize(img,  (0,0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
    b = cv2.resize(rect, (0,0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
    h2 = min(a.shape[0], b.shape[0]); w2 = min(a.shape[1], b.shape[1])
    combo = cv2.hconcat([a[:h2,:w2], b[:h2,:w2]])
    cv2.imwrite(f"debug/fisheye.KDxi/rectified.map/{os.path.basename(path)}_rect.png", combo)

# optional: make a video from the previews
def make_video_alt(folder, fps=15):
    fr = sorted(glob.glob(f"{folder}/*.png"))
    if not fr:
        return
    frame0 = cv2.imread(fr[0]); H, W = frame0.shape[:2]
    vw = cv2.VideoWriter(f"{folder}/output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for f in fr:
        im = cv2.imread(f)
        if im is not None and im.shape[:2] == (H, W):
            vw.write(im)
    vw.release()

make_video_alt("debug/fisheye.KDxi/rectified.map")





# --- persist raw calibration (already computed as 'best') ---
os.makedirs("configs/fisheye.config", exist_ok=True)
np.save("configs/fisheye.config/fisheye.K.npy",  best["K"])
np.save("configs/fisheye.config/fisheye.D.npy",  best["D"])
np.save("configs/fisheye.config/fisheye.xi.npy", best["xi"].ravel())  # (1,)
with open("configs/fisheye.config/fisheye.meta.json","w") as f:
    json.dump({
        "model":"omnidir-mei",
        "shape": list(best["shape"]),
        "rms_px": float(best["rms"]),
        "errs_mean_px": float(best["errs"].mean()),
        "errs_median_px": float(np.median(best["errs"])),
        "errs_max_px": float(best["errs"].max())
    }, f, indent=2)


 