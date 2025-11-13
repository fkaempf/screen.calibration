# warp_circle_oscillate_and_record_omni_alvium.py
# Projector-sized maps (projector <- camera). Camera = Alvium + omnidir (Mei) with K, D, xi.
import sys
sys.path.append(r"D:\screen.calibration")

import os, time, math
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import cv2
import pygame
from cameras.CamAlvium import CamAlvium  # must be available in PYTHONPATH

# ---------- CONFIG ----------
PROJ_W, PROJ_H = 1280, 800
EXPOSURE_MS    = 7.0
GAIN_DB        = 0.0
OSC_FREQ_HZ    = 0.25
SAVE_UNDISTORT = True
OMNI_CFG_DIR   = "D:\screen.calibration/configs/fisheye.config"
OUTDIR         = "D:\screen.calibration/debug/moving.circle.fish.alvium"
ZOOM_UNDISTORT = 0.1

# === Circle control parameters ===
DOT_RADIUS_FRAC = 0.05       # fraction of min(cam_w, cam_h)
DOT_Y_FRAC      = 0.65   # vertical placement as fraction of height (1=top, 0=bottom)
OSC_AMPL_FRAC   = 0.8       # horizontal motion amplitude fraction (0=no motion, 1=max full width)

# === Paths ===
RAW_PATH       = os.path.join(OUTDIR, "osc_circle_alvium_raw.mp4")
UND_PATH       = os.path.join(OUTDIR, "osc_circle_alvium_undist.mp4")
LAST_PROJ_PNG  = os.path.join(OUTDIR, "osc_circle_last_projector_frame.png")
os.makedirs(OUTDIR, exist_ok=True)

# ---------- LOAD MAPS ----------
mapx = np.load("mapx.npy").astype(np.float32)
mapy = np.load("mapy.npy").astype(np.float32)
assert mapx.shape == (PROJ_H, PROJ_W)
assert mapy.shape == (PROJ_H, PROJ_W)

valid = np.isfinite(mapx) & np.isfinite(mapy) & (mapx >= 0) & (mapy >= 0)
if not np.any(valid):
    raise RuntimeError("mapx/mapy contain no valid entries")
cam_w = int(np.ceil(mapx[valid].max())) + 1
cam_h = int(np.ceil(mapy[valid].max())) + 1
mx = np.clip(np.where(valid, mapx, -1.0), 0, cam_w - 1).astype(np.float32)
my = np.clip(np.where(valid, mapy, -1.0), 0, cam_h - 1).astype(np.float32)

# ---------- LOAD OMNIDIR INTRINSICS ----------
def load_omni(cfg_dir):
    K  = np.load(os.path.join(cfg_dir, "fisheye.K.npy")).astype(np.float64).reshape(3,3)
    D  = np.load(os.path.join(cfg_dir, "fisheye.D.npy")).astype(np.float64).reshape(1,4)
    xi = np.load(os.path.join(cfg_dir, "fisheye.xi.npy")).astype(np.float64).ravel()
    xi = np.array([float(xi[0])], dtype=np.float64)
    return K, D, xi

K_omni, D_omni, xi_omni = load_omni(OMNI_CFG_DIR)

def undistort_omnidir_frame(gray_u8, K, D, xi, zoom=0.3):
    h, w = gray_u8.shape[:2]
    Knew = K.copy()
    Knew[0,0] *= zoom
    Knew[1,1] *= zoom
    Knew[0,2] = w / 2.0
    Knew[1,2] = h / 2.0
    return cv2.omnidir.undistortImage(
    gray_u8, 
    K = K, 
    xi = xi,
    D = D, 
    flags  = 1, 
    Knew = Knew, 
    new_size  = (w, h)
)

# ---------- PYGAME FULLSCREEN PROJECTOR ----------
try:
    from screeninfo import get_monitors
except Exception:
    get_monitors = None

def pick_rightmost_monitor():
    if get_monitors:
        mons = get_monitors()
        if mons:
            return max(mons, key=lambda m: m.x)
    class M: pass
    m = M(); m.x = 0; m.y = 0; m.width = PROJ_W; m.height = PROJ_H
    return m

def to_surface(img_u8, size_wh):
    W, H = size_wh
    if img_u8.dtype != np.uint8: img_u8 = cv2.convertScaleAbs(img_u8)
    if img_u8.ndim == 2:
        rgb = np.dstack([img_u8]*3)
    else:
        rgb = img_u8[..., ::-1] if img_u8.shape[2] == 3 else img_u8
    if (rgb.shape[1], rgb.shape[0]) != (W, H):
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_NEAREST)
    return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

m = pick_rightmost_monitor()
os.environ["SDL_VIDEODRIVER"] = "windows"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["SDL_HINT_VIDEO_HIGHDPI_DISABLED"] = "1"
os.environ["SDL_VIDEO_WINDOW_POS"] = f"{m.x},{m.y}"
pygame.display.init()
flags = pygame.FULLSCREEN | pygame.SCALED
screen = pygame.display.set_mode((m.width, m.height), flags)
pygame.display.set_caption("Projector")

# ---------- ALVIUM CAMERA ----------
cam = CamAlvium(exposure_ms=EXPOSURE_MS, gain_db=GAIN_DB)
cam.start()
try: cam.grab(timeout_s=0.5)
except Exception: pass

samples = []
prev = time.perf_counter()
for _ in range(100):
    _ = cam.grab()
    now = time.perf_counter()
    samples.append(now - prev)
    prev = now
dt = np.median(samples[5:]) if len(samples) > 5 else (np.mean(samples) if samples else 1/30)
fps_meas = max(1.0, min(240.0, 1.0 / max(1e-6, dt)))
print(f"Measured Alvium FPS ~ {fps_meas:.2f}")

probe = cam.grab()
if probe.dtype != np.uint8: probe = cv2.convertScaleAbs(probe)
fh, fw = probe.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
raw_writer = cv2.VideoWriter(RAW_PATH, fourcc, fps_meas, (fw, fh), True)
und_writer = cv2.VideoWriter(UND_PATH, fourcc, fps_meas, (fw, fh), True) if SAVE_UNDISTORT else None

# ---------- STIMULUS ----------
radius   = int(max(4, min(cam_w, cam_h) * DOT_RADIUS_FRAC))
y_center = int(cam_h * DOT_Y_FRAC)
A        = (cam_w * OSC_AMPL_FRAC) / 2.0
x0       = cam_w / 2.0

cam_img = np.zeros((cam_h, cam_w, 3), np.uint8)
last_proj = None

# ---------- MAIN LOOP ----------
loop_start = time.perf_counter()
frames_written = 0
paused = False

try:
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: raise KeyboardInterrupt
            if e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q): raise KeyboardInterrupt
                if e.key in (pygame.K_SPACE, pygame.K_p): paused = not paused

        if not paused:
            t = time.perf_counter() - loop_start
            x_center = int(round(x0 + A * math.sin(2 * math.pi * OSC_FREQ_HZ * t)))

            cam_img.fill(0)
            cv2.circle(cam_img, (x_center, y_center), radius, (255, 255, 255), -1, lineType=cv2.LINE_8)

            proj_frame = cv2.remap(cam_img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            last_proj = proj_frame

            surf = to_surface(proj_frame, (m.width, m.height))
            screen.blit(surf, (0, 0)); pygame.display.flip()

        frame = cam.grab()
        if frame.dtype != np.uint8: frame = cv2.convertScaleAbs(frame)
        raw_writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        if SAVE_UNDISTORT:
            und = undistort_omnidir_frame(frame, K_omni, D_omni, xi_omni, zoom=ZOOM_UNDISTORT)
            und_writer.write(cv2.cvtColor(und, cv2.COLOR_GRAY2BGR))
        frames_written += 1

except KeyboardInterrupt:
    pass
finally:
    loop_end = time.perf_counter()
    loop_fps = frames_written / max(1e-6, (loop_end - loop_start))
    print(f"Loop FPS ~ {loop_fps:.3f}")
    if last_proj is not None:
        cv2.imwrite(LAST_PROJ_PNG, last_proj)
    raw_writer.release()
    if SAVE_UNDISTORT and und_writer is not None:
        und_writer.release()
    cam.stop()
    pygame.display.quit(); pygame.quit()

print("Saved:")
print(f" - {RAW_PATH}")
if SAVE_UNDISTORT: print(f" - {UND_PATH}")
print(f" - {LAST_PROJ_PNG}")

