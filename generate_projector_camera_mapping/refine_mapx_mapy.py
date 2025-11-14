# pipeline.py  — basic with heat maps
import os, time
import numpy as np
import cv2
import pygame
import sys
import matplotlib.pyplot as plt
sys.path.append(r"D:\screen.calibration")
from cameras.CamAlvium import CamAlvium
from generate_projector_camera_mapping.mapping_utils import (
    capture_and_decode_sine_hybrid,
    capture_and_decode,
    CamRotPy,
)
from stimulus.warp_stimulus import build_proj_to_cam_map, make_camera_grid, make_uv_map
from generate_projector_camera_mapping.mapping_pipeline import (
    _pick_monitor_rightmost,
    _frame_to_surface,
)


def refine_mapx_mapy(
    STEP,
    DOT_R,   # currently unused, kept for completeness
    EXP_MS,
    GAIN_DB,
    SETTLE,
    SIZE,
    CAMTYPE,
):
# %%
    mapx = np.load(
        "D:/screen.calibration/configs/camera.projector.mapping/mapx.npy"
    ).astype(np.float32)
    mapy = np.load(
        "D:/screen.calibration/configs/camera.projector.mapping/mapy.npy"
    ).astype(np.float32)
    valid = np.load(
        "D:/screen.calibration/configs/camera.projector.mapping/valid.mask.npy"
    ).astype(bool)

    PROJ_H, PROJ_W = mapx.shape[0], mapx.shape[1]

    cam_h, cam_w = valid.shape

    # init projector window once
    m = _pick_monitor_rightmost(PROJ_W, PROJ_H)
    os.environ.setdefault("SDL_VIDEODRIVER", "windows")
    os.environ.setdefault("SDL_RENDER_DRIVER", "software")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("SDL_HINT_VIDEO_HIGHDPI_DISABLED", "1")
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{m.x},{m.y}"

    pygame.init()
    screen = pygame.display.set_mode(
        (m.width, m.height), pygame.SWSURFACE | pygame.NOFRAME
    )
    pygame.display.set_caption("Probe projector mapping")

    # init camera once
    if CAMTYPE.lower() == "alvium":
        cam = CamAlvium(exposure_ms=EXP_MS, gain_db=GAIN_DB)
    else:
        cam = CamRotPy(exposure_ms=EXP_MS, gain_db=GAIN_DB)
    cam.start()

    null_frame = cam.grab()
    # one reusable stimulus buffer
    stim = np.zeros((PROJ_H, PROJ_W), np.uint8)
    vis_map = np.zeros((PROJ_H, PROJ_W), np.uint8)

    try:
        for y in range(0, stim.shape[0], STEP):
            for x in range(0, stim.shape[1], STEP):
                stim_temp = stim.copy()
                stim_temp[y : y + SIZE, x : x + SIZE] = 255

                # minimal event handling so ESC quits
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        raise KeyboardInterrupt
                    if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt

                surf = _frame_to_surface(stim_temp, (m.width, m.height))
                screen.blit(surf, (0, 0))
                pygame.display.flip()

                time.sleep(SETTLE)

                frame = cam.grab()
                
                if frame.dtype != np.uint8:
                    frame = cv2.convertScaleAbs(frame)

                # ensure grayscale for threshold/OTSU
                if frame.ndim == 3:
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    frame_gray = frame

                # threshold camera image
                _, bw = cv2.threshold(
                    frame_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

                # mask with valid (camera-space)
                bw_bool = bw.astype(bool)
                combined = bw_bool & valid

                # if there is *no* overlap between bright dot and valid mask → skip
                if combined.any():
                    
                    vis_map[y : y + SIZE, x : x + SIZE] += 1
         

    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        pygame.display.quit()
        pygame.quit()

    vis_map = vis_map != 0
    mapx[~vis_map] = np.nan
    mapy[~vis_map] = np.nan

    np.save(
        "D:/screen.calibration/configs/camera.projector.mapping/mapx.experimental.npy",
        mapx,
    )
    np.save(
        "D:/screen.calibration/configs/camera.projector.mapping/mapy.experimental.npy",
        mapy,
    )
    mx = mapx.astype(np.float32)
    my = mapy.astype(np.float32)
    vm = vis_map.astype(np.float32)

    mx = np.nan_to_num(mx, nan=0.0)
    my = np.nan_to_num(my, nan=0.0)
    vm = np.nan_to_num(vm, nan=0.0)

    mx_show = cv2.normalize(mx, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    my_show = cv2.normalize(my, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vm_show = cv2.normalize(vm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    mx_show = cv2.resize(mx_show, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    my_show = cv2.resize(my_show, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    vm_show = cv2.resize(vm_show, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

    cv2.imshow("mapx", mx_show)
    cv2.imshow("mapy", my_show)
    cv2.imshow("vis_map", vm_show)
    cv2.waitKey(4000)
    cv2.destroyAllWindows()







if __name__ == "__main__":
    STEP = 50       # projector grid spacing in px
    DOT_R = 5      # radius of projected dot (currently unused)
    EXP_MS = 3.0
    GAIN_DB = 0.0
    SETTLE = 0.05  # seconds to wait after updating projector
    SIZE = 50
    CAMTYPE = "alvium"

    refine_mapx_mapy(
        STEP=STEP,
        DOT_R=DOT_R,
        EXP_MS=EXP_MS,
        GAIN_DB=GAIN_DB,
        SETTLE=SETTLE,
        SIZE=SIZE,
        CAMTYPE=CAMTYPE,
    )
