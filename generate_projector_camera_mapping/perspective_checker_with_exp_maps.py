# pipeline.py — basic with heat maps

import os, time
import sys
import numpy as np
import cv2
import pygame

sys.path.append(r"D:\screen.calibration")

from cameras.CamAlvium import CamAlvium
from generate_projector_camera_mapping.mapping_utils import (
    capture_and_decode_sine_hybrid,
    capture_and_decode,
    CamRotPy,
    
)
from stimulus.warp_stimulus import (
    make_camera_grid,
    make_uv_map,
)

from generate_projector_camera_mapping.mapping_pipeline import save_heat01, make_equirect_checker, project_and_capture_single




CAMTYPE = "alvium"        # "rotpy" or "alvium"
MODE = "sine_hybrid"           # "gray" or "sine_hybrid"
PERIODS_X = 128          # ~12.5px period for 800px width
PERIODS_Y = 96          # ~12.5px period for 600px height
NPHASE    = 15
AVG_PER   = 5           # you can increase to 3–5 if needed
GAMMA_INV = None        # set 2.2 if you want inverse-gamma on the projected sines
PROJ_W = 1280
PROJ_H = 800
ts = time.strftime("%Y%m%d_%H%M%S")
OUT = f"D:/screen.calibration/debug/perspective.checker.with.exp.maps/{ts}"
os.makedirs(OUT, exist_ok=True)

mapx = np.load('D:/screen.calibration/configs/camera.projector.mapping/mapx.experimental.npy').astype(np.float32)
mapy = np.load('D:/screen.calibration/configs/camera.projector.mapping/mapy.experimental.npy').astype(np.float32)

# 3) build a simple camera grid and warp it to projector pixels

cam = CamAlvium(exposure_ms=2, gain_db=0)
cam.start()
frame = cam.grab()
cam.stop()
cam_h, cam_w = frame.shape[:2]

grid = make_camera_grid(cam_h, cam_w, step=40, thick=5)
proj_grid = cv2.remap(grid, mapx, mapy, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
cv2.imwrite(os.path.join(OUT, "projector_grid.png"), proj_grid)

# 4) show the warped grid and capture one camera photo
project_and_capture_single(
    proj_grid,
    save_path=os.path.join(OUT, "camera_view_of_warp.png"),
    exposure_ms=3,
    gain_db=0.0,
    hold_seconds=3.0,
    settle_seconds=0.2,
    camtype=CAMTYPE
)



# 6) uv_map: compute, save, visualize, render a VR checker, project, capture
if CAMTYPE.lower() == "rotpy":
    K_path, D_path = "D:/screen.calibration/configs/pinhole.config/pinhole.K.npy", "D:/screen.calibration/configs/pinhole.config/pinhole.D.npy"
elif CAMTYPE.lower() == "alvium":
    K_path, D_path, xi_path = "D:/screen.calibration/configs/fisheye.config/fisheye.K.npy", "D:/screen.calibration/configs/fisheye.config/fisheye.D.npy","D:/screen.calibration/configs/fisheye.config/fisheye.xi.npy"
    if not (os.path.exists(K_path) and os.path.exists(D_path) and os.path.exists(xi_path)):
        raise FileNotFoundError("omnidir intrinsics missing: K/D/xi")
    
if os.path.exists(K_path) and os.path.exists(D_path):
    if CAMTYPE.lower() == "rotpy":
        K = np.load(K_path).astype(np.float64)
        D = np.load(D_path).astype(np.float64)
        MODEL = "pinhole"
        uv = make_uv_map(mapx, mapy, K, D, model=MODEL)
    elif CAMTYPE.lower() == "alvium":
        K = np.load(K_path).astype(np.float64)
        D = np.load(D_path).astype(np.float64)
        xi = np.load(xi_path).astype(np.float64)
        MODEL = "fisheye_xi"
        uv = make_uv_map(mapx, mapy, K, D, model=MODEL, xi=xi, w=cam_w, h=cam_h)
    

    


    # UV heat maps
    save_heat01(os.path.join(OUT, "uv_U_heat.png"), uv[..., 0])
    save_heat01(os.path.join(OUT, "uv_V_heat.png"), uv[..., 1])

    # render via uv_map
    stim = make_equirect_checker(1024, 2048, step_deg=30, line_px=2)
    Ht, Wt = stim.shape[:2]
    mapx_uv = (uv[..., 0] * (Wt - 1)).astype(np.float32)
    mapy_uv = (uv[..., 1] * (Ht - 1)).astype(np.float32)
    proj_vr = cv2.remap(stim, mapx_uv, mapy_uv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    cv2.imwrite(os.path.join(OUT, "projector_vr_checker.png"), proj_vr)

    # project and grab one photo
    project_and_capture_single(
        proj_vr,
        save_path=os.path.join(OUT, "camera_view_of_vr_checker.png"),
        exposure_ms=2,
        gain_db=0.0,
        hold_seconds=3.0,
        settle_seconds=0.2,
        camtype=CAMTYPE
    )
else:
    print("K_cam.npy or D_cam.npy not found, uv_map step skipped")