# warp_circle_and_capture.py — keep stimulus visible WHILE capturing

import os, time
import numpy as np
import cv2
import pygame

# ---------- config ----------
CAMTYPE   = "alvium"   # "alvium" or "rotpy"
PROJ_W    = 1280
PROJ_H    = 800
EXPOSURE  = 10.0       # ms
GAIN_DB   = 0.0
HOLD_S    = 3.0        # total on-screen time
SETTLE_S  = 0.2        # wait after showing before capture
DOT_RADIUS = 20

OUTDIR = "out"
os.makedirs(OUTDIR, exist_ok=True)

# ---------- load projector<-camera maps (DEST=projector; VALUES=camera coords) ----------
mapx = np.load("mapx.npy").astype(np.float32)
mapy = np.load("mapy.npy").astype(np.float32)
assert mapx.shape == (PROJ_H, PROJ_W) and mapy.shape == (PROJ_H, PROJ_W), "map size must equal projector size"

# ---------- infer required camera frame size from map ranges ----------
mx_max = float(np.nanmax(mapx))
my_max = float(np.nanmax(mapy))
cam_w  = int(np.ceil(mx_max)) + 1
cam_h  = int(np.ceil(my_max)) + 1
assert cam_w > 0 and cam_h > 0

# ---------- sanitize maps ----------
mx = mapx.copy(); my = mapy.copy()
bad = ~np.isfinite(mx) | ~np.isfinite(my)
mx[bad] = -1.0; my[bad] = -1.0
mx = np.clip(mx, 0, cam_w - 1)
my = np.clip(my, 0, cam_h - 1)

# ---------- build a CAMERA-space stimulus (circle) ----------
cam_img = np.zeros((cam_h, cam_w, 3), np.uint8)
cx, cy  = cam_w // 2, cam_h // 2
radius  = min(cam_w, cam_h) // DOT_RADIUS
cv2.circle(cam_img, (cx, cy), radius, (255, 255, 255), -1, lineType=cv2.LINE_8)
cv2.imwrite(os.path.join(OUTDIR, "circle_camera_space.png"), cam_img)

# ---------- warp to PROJECTOR space ----------
proj_frame = cv2.remap(
    cam_img, mx, my,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
)
cv2.imwrite(os.path.join(OUTDIR, f"circle_projector_warped_{PROJ_W}x{PROJ_H}.png"), proj_frame)

# ---------- monitor pick ----------
try:
    from screeninfo import get_monitors
except Exception:
    get_monitors = None

def _pick_rightmost_monitor():
    if get_monitors:
        mons = get_monitors()
        if mons:
            return max(mons, key=lambda m: m.x)
    class M: pass
    m = M(); m.x = 0; m.y = 0; m.width = PROJ_W; m.height = PROJ_H
    return m

def _img_to_surface(img_u8, size_wh):
    W, H = size_wh
    if img_u8.dtype != np.uint8:
        img_u8 = cv2.convertScaleAbs(img_u8)
    if img_u8.ndim == 2:
        img_rgb = np.dstack([img_u8]*3)
    else:
        img_rgb = img_u8[..., ::-1] if img_u8.shape[2] == 3 else img_u8
    if (img_rgb.shape[1], img_rgb.shape[0]) != (W, H):
        img_rgb = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_NEAREST)
    return pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))

# ---------- camera capture ----------
def capture_one_camera_frame(save_path, exposure_ms=EXPOSURE, gain_db=GAIN_DB, settle_s=0.0, camtype=CAMTYPE):
    if camtype.lower() == "alvium":
from screen_calibration.camera.CamAlvium import CamAlvium
        cam = CamAlvium(exposure_ms=exposure_ms, gain_db=gain_db)
        cam.start(); 
        if settle_s > 0: time.sleep(settle_s)
        try: cam.grab(timeout_s=0.5)
        except Exception: pass
        frame = cam.grab()
        cam.stop()
    else:
        RotCam = None
        for mod in ("screen_calibration.capture.gray_capture_rotpy", "gray_capture_rotpy_basic", "gray_capture_rotpy"):
            try:
                RotCam = __import__(mod, fromlist=["CamRotPy"]).CamRotPy
                break
            except Exception:
                pass
        if RotCam is not None:
            cam = RotCam(exposure_ms=exposure_ms, gain_db=gain_db)
            cam.start(); 
            if settle_s > 0: time.sleep(settle_s)
            try: cam.grab(timeout_s=0.5)
            except Exception: pass
            frame = cam.grab(); cam.stop()
        else:
            cap = cv2.VideoCapture(0, cv2.CAP_ANY)
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure_ms/1000.0)
            if settle_s > 0: time.sleep(settle_s)
            for _ in range(2): cap.read()
            ok, frame = cap.read(); cap.release()
            if not ok: raise RuntimeError("OpenCV could not read a frame")
            if frame.ndim == 3: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if frame.dtype != np.uint8: frame = cv2.convertScaleAbs(frame)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, frame)
    return frame

# ---------- show WHILE capturing ----------
if __name__ == "__main__":
    m = _pick_rightmost_monitor()
    os.environ["SDL_VIDEODRIVER"] = "windows"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ["SDL_HINT_VIDEO_HIGHDPI_DISABLED"] = "1"
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{m.x},{m.y}"

    pygame.display.init()
    flags = pygame.FULLSCREEN | pygame.SCALED
    screen = pygame.display.set_mode((m.width, m.height), flags)
    pygame.display.set_caption("Projector")

    surf = _img_to_surface(proj_frame, (m.width, m.height))
    screen.blit(surf, (0, 0)); pygame.display.flip()

    # keep window visible; capture after SETTLE_S without closing the window
    captured = False
    t0 = time.time()
    clock = pygame.time.Clock()
    while time.time() - t0 < HOLD_S:
        now = time.time() - t0
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.display.quit(); raise SystemExit
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                pygame.display.quit(); raise SystemExit

        if (not captured) and (now >= SETTLE_S):
            capture_one_camera_frame(os.path.join(OUTDIR, "circle_camera_view.png"),
                                     exposure_ms=EXPOSURE, gain_db=GAIN_DB,
                                     settle_s=0.0, camtype=CAMTYPE)
            captured = True

        clock.tick(60)

    pygame.display.quit()

    print("Saved:")
    print(" -", os.path.join(OUTDIR, "circle_camera_space.png"))
    print(" -", os.path.join(OUTDIR, f"circle_projector_warped_{PROJ_W}x{PROJ_H}.png"))
    print(" -", os.path.join(OUTDIR, "circle_camera_view.png"))
