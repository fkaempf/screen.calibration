# pipeline.py  — basic with heat maps
import os, time
import numpy as np
import cv2
import pygame
import sys
sys.path.append(r"D:\screen.calibration")
from cameras.CamAlvium import CamAlvium
from generate_projector_camera_mapping.mapping_utils import capture_and_decode_sine_hybrid, capture_and_decode, CamRotPy
from stimulus.warp_stimulus import build_proj_to_cam_map, make_camera_grid, make_uv_map
from generate_projector_camera_mapping.mapping_pipeline import _pick_monitor_rightmost, _frame_to_surface



STEP    = 7      # projector grid spacing in px
DOT_R   = 5       # radius of projected dot
EXP_MS  = 3.0
GAIN_DB = 0.0
SETTLE  = 0.05    # seconds to wait after updating projector
SIZE = 10
CAMTYPE = "alvium"


mapx = np.load('D:/screen.calibration/configs/camera.projector.mapping/mapx.npy').astype(np.float32)
mapy = np.load('D:/screen.calibration/configs/camera.projector.mapping/mapy.npy').astype(np.float32)
valid = np.load('D:/screen.calibration/configs/camera.projector.mapping/valid.mask.npy').astype(bool)

PROJ_H, PROJ_W = mapx.shape[0], mapx.shape[1]

cam_h, cam_w = valid.shape
mapx_exp = np.full((PROJ_H, PROJ_W), np.nan, np.float32)
mapy_exp = np.full((PROJ_H, PROJ_W), np.nan, np.float32)

# init projector window once
m = _pick_monitor_rightmost()
os.environ.setdefault("SDL_VIDEODRIVER", "windows")
os.environ.setdefault("SDL_RENDER_DRIVER", "software")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_HINT_VIDEO_HIGHDPI_DISABLED", "1")
os.environ["SDL_VIDEO_WINDOW_POS"] = f"{m.x},{m.y}"

pygame.init()
screen = pygame.display.set_mode((m.width, m.height),
                                    pygame.SWSURFACE | pygame.NOFRAME)
pygame.display.set_caption("Probe projector mapping")

# init camera once
if CAMTYPE.lower() == "alvium":
    cam = CamAlvium(exposure_ms=EXP_MS, gain_db=GAIN_DB)
else:
    cam = CamRotPy(exposure_ms=EXP_MS, gain_db=GAIN_DB)
cam.start()

# one reusable stimulus buffer
stim = np.zeros((PROJ_H, PROJ_W), np.uint8)
vis_map = np.zeros((PROJ_H, PROJ_W), np.uint8)

try:
    
    for y in range(0,stim.shape[0],STEP):
        for x in range(0,stim.shape[1],STEP):
            stim_temp = stim.copy()
            stim_temp[y:y+SIZE,x:x+SIZE]=255




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
                frame_gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # mask with valid (camera-space)
            bw_bool = bw.astype(bool)
            combined = bw_bool & valid

            # if there is *no* overlap between bright dot and valid mask → skip
            if combined.any():
                vis_map[y:y+SIZE,x:x+SIZE]+=1



except KeyboardInterrupt:
    pass
finally:
    cam.stop()
    pygame.display.quit()
    pygame.quit()
vis_map = vis_map!=0
mapx[~vis_map] = np.nan
mapy[~vis_map] = np.nan

np.save("D:/screen.calibration/configs/camera.projector.mapping/mapx.experimental.npy", mapx)
np.save("D:/screen.calibration/configs/camera.projector.mapping/mapy.experimental.npy", mapy)





