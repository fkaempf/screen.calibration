# warp_circle_and_capture.py
import os
import time
import numpy as np
import cv2
import pygame

CAMTYPE = "alvium"        # "rotpy" or "alvium"
MODE = "sine_hybrid"           # "gray" or "sine_hybrid"
PERIODS_X = 64          # ~12.5px period for 800px width
PERIODS_Y = 48          # ~12.5px period for 600px height
NPHASE    = 4
AVG_PER   = 1           # you can increase to 3–5 if needed
GAMMA_INV = None        # set 2.2 if you want inverse-gamma on the projected sines
PROJ_W = 1280
PROJ_H = 800



# ------------- load projector->camera maps -------------
mapx = np.load("D:/screen.calibration/configs/camera.projector.mapping/mapx.npy").astype(np.float32)
mapy = np.load("D:/screen.calibration/configs/camera.projector.mapping/mapy.npy").astype(np.float32)
proj_h, proj_w = mapx.shape  # expect 600, 800

# ------------- infer camera image size -------------
valid = (mapx >= 0) & (mapy >= 0)
cam_w = int(np.max(mapx[valid])) + 1
cam_h = int(np.max(mapy[valid])) + 1
if cam_w <= 0 or cam_h <= 0:
    raise RuntimeError("mapx/mapy look empty")

# ------------- make a circle in camera space -------------
cam_img = np.zeros((cam_h, cam_w, 3), np.uint8)
cx, cy = cam_w // 2, cam_h // 2
radius = min(cam_w, cam_h) // 6
cv2.circle(cam_img, (cx, cy), radius, (255, 255, 255), -1, lineType=cv2.LINE_8)

# ------------- warp to projector space -------------
proj_frame = cv2.remap(
    cam_img, mapx, mapy,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
)

# ------------- save images -------------
os.makedirs("out", exist_ok=True)
cv2.imwrite("out/circle_camera_space.png", cam_img)
cv2.imwrite("out/circle_projector_warped_800x600.png", proj_frame)

# ------------- show on projector with pygame -------------
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
    m = M(); m.x = 0; m.y = 0; m.width = proj_w; m.height = proj_h
    return m

def _to_surface(img_u8, size_wh):
    W, H = size_wh
    if img_u8.dtype != np.uint8:
        img_u8 = cv2.convertScaleAbs(img_u8)
    if img_u8.ndim == 2:
        rgb = np.dstack([img_u8]*3)
    else:
        rgb = img_u8[..., ::-1] if img_u8.shape[2] == 3 else img_u8
    if (rgb.shape[1], rgb.shape[0]) != (W, H):
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_NEAREST)
    return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

def show_on_projector(img_u8, hold_seconds=3.0):
    m = _pick_rightmost_monitor()
    os.environ.setdefault("SDL_VIDEODRIVER", "windows")
    os.environ.setdefault("SDL_RENDER_DRIVER", "software")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("SDL_HINT_VIDEO_HIGHDPI_DISABLED", "1")
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{m.x},{m.y}"

    pygame.init()
    screen = pygame.display.set_mode((m.width, m.height), pygame.SWSURFACE | pygame.NOFRAME)
    pygame.display.set_caption("Projector")
    surf = _to_surface(img_u8, (m.width, m.height))
    screen.blit(surf, (0, 0)); pygame.display.flip()
    t0 = time.time()
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: pygame.display.quit(); pygame.quit(); return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE: pygame.display.quit(); pygame.quit(); return
        if (time.time() - t0) >= hold_seconds:
            break
        time.sleep(0.01)
    pygame.display.quit(); pygame.quit()

# ------------- camera capture helper -------------
def capture_one_camera_frame(save_path, exposure_ms=16.7, gain_db=0.0, settle_s=0.2):
    """
    Tries to use CamRotPy from your gray_capture module.
    Falls back to OpenCV VideoCapture(0) if not available.
    """
    RotCam = None
    cam = None
    # prefer packaged path, fall back to old names
    for mod in ("screen_calibration.capture.gray_capture_rotpy", "gray_capture_rotpy_basic", "gray_capture_rotpy"):
        try:
            RotCam = __import__(mod, fromlist=["CamRotPy"]).CamRotPy
            break
        except Exception:
            pass

    if RotCam is not None:
        cam = RotCam(exposure_ms=exposure_ms, gain_db=gain_db)
        cam.start()
        time.sleep(settle_s)
        # discard one to avoid stale frame, then grab
        try: cam.grab(timeout_s=0.5)
        except Exception: pass
        frame = cam.grab()
        cam.stop()
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure_ms/1000.0)
        time.sleep(settle_s)
        for _ in range(2): cap.read()
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("OpenCV could not read a frame")
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if frame.dtype != np.uint8:
        frame = cv2.convertScaleAbs(frame)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, frame)
    return frame

# ------------- run: show warped and capture one photo -------------
if __name__ == "__main__":
    # show the warped stimulus for a few seconds while we capture
    # start capture slightly after it is on screen
    # show for 3 s, capture during that window
    # show first, then capture immediately
    m = _pick_rightmost_monitor()
    os.environ.setdefault("SDL_VIDEODRIVER", "windows")
    os.environ.setdefault("SDL_RENDER_DRIVER", "software")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("SDL_HINT_VIDEO_HIGHDPI_DISABLED", "1")
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{m.x},{m.y}"

    pygame.init()
    screen = pygame.display.set_mode((m.width, m.height), pygame.SWSURFACE | pygame.NOFRAME)
    pygame.display.set_caption("Projector")
    surf = _to_surface(proj_frame, (m.width, m.height))
    screen.blit(surf, (0, 0)); pygame.display.flip()

    # give the projector a moment to refresh, then capture
    time.sleep(0.2)
    cap_path = "out/circle_camera_view.png"
    frame = capture_one_camera_frame(cap_path, exposure_ms=16.7, gain_db=0.0, settle_s=0.0)

    # keep it visible a bit more, or ESC to close
    t0 = time.time()
    hold_seconds = 2.8
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: pygame.display.quit(); pygame.quit(); exit(0)
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE: pygame.display.quit(); pygame.quit(); exit(0)
        if (time.time() - t0) >= hold_seconds:
            break
        time.sleep(0.01)
    pygame.display.quit(); pygame.quit()

    print("Saved:")
    print(" - out/circle_camera_space.png")
    print(" - out/circle_projector_warped_800x600.png")
    print(" - out/circle_camera_view.png")
