# warp_circle_oscillate_and_record.py  — uncapped FPS
import os, time, math
import numpy as np
import cv2
import pygame

# ---------- load projector->camera maps ----------
mapx = np.load("mapx.npy").astype(np.float32)
mapy = np.load("mapy.npy").astype(np.float32)
proj_h, proj_w = mapx.shape  # e.g. 600, 800

# ---------- infer camera image size ----------
valid = (mapx >= 0) & (mapy >= 0)
cam_w = int(np.max(mapx[valid])) + 1
cam_h = int(np.max(mapy[valid])) + 1
if cam_w <= 0 or cam_h <= 0:
    raise RuntimeError("mapx/mapy look empty")

# ---------- pygame helpers ----------
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

# ---------- camera helpers ----------
def _get_rotpy_cam():
    for mod in ("screen_calibration.capture.gray_capture_rotpy", "gray_capture_rotpy_basic", "gray_capture_rotpy"):
        try:
            return __import__(mod, fromlist=["CamRotPy"]).CamRotPy
        except Exception:
            pass
    return None

class CamFallback:
    def __init__(self, exposure_ms=16.7, gain_db=0.0, index=0):
        self.cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        # do not force FPS, only exposure
        self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure_ms/1000.0)
    def start(self): pass
    def grab(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("OpenCV could not read a frame")
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
    def stop(self):
        self.cap.release()

def open_camera(exposure_ms=16.7, gain_db=0.0):
    RotCam = _get_rotpy_cam()
    if RotCam is not None:
        cam = RotCam(exposure_ms=exposure_ms, gain_db=gain_db)
        # IMPORTANT: do not set AcquisitionFrameRate to a fixed number (no cap)
        cam.start()
        return cam, "rotpy"
    cam = CamFallback(exposure_ms=exposure_ms, gain_db=0.0, index=0)
    cam.start()
    return cam, "cv"

# ---------- params ----------
radius =  max(4, min(cam_w, cam_h) // 10)
y_center = (cam_h // 2)+ ((cam_h // 4))
x_min = radius/2
x_max = cam_w - radius/2
A = (x_max - x_min) / 2.0
x0 = x_min + A
freq_hz = 0.25                # oscillation frequency (motion timing only)
exposure_ms = 7           # camera exposure target

# ---------- setup projector ----------
m = _pick_rightmost_monitor()
os.environ.setdefault("SDL_VIDEODRIVER", "windows")
os.environ.setdefault("SDL_RENDER_DRIVER", "software")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_HINT_VIDEO_HIGHDPI_DISABLED", "1")
os.environ["SDL_VIDEO_WINDOW_POS"] = f"{m.x},{m.y}"

pygame.init()
screen = pygame.display.set_mode((m.width, m.height), pygame.SWSURFACE | pygame.NOFRAME)
pygame.display.set_caption("Projector")

# ---------- camera and measured fps ----------
os.makedirs("out", exist_ok=True)
cam, backend = open_camera(exposure_ms=exposure_ms, gain_db=0.0)

# discard one stale
try: cam.grab()
except Exception: pass

# measure camera grab interval briefly (uncapped)
samples = []
prev = time.perf_counter()
for _ in range(30):
    _ = cam.grab()
    now = time.perf_counter()
    samples.append(now - prev)
    prev = now

if len(samples) > 5:
    dt = np.median(samples[5:])
else:
    dt = np.mean(samples) if samples else 1.0/30.0
fps_meas = max(1.0, min(240.0, 1.0 / max(1e-6, dt)))
print(f"Measured camera fps ~ {fps_meas:.2f}")

# open writer with initial guess; we retime later anyway
probe = cam.grab()
if probe.dtype != np.uint8:
    probe = cv2.convertScaleAbs(probe)
fh, fw = probe.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
raw_path = "out/osc_circle_camera.mp4"
writer = cv2.VideoWriter(raw_path, fourcc, fps_meas, (fw, fh), True)

# ---------- preallocate ----------
cam_img = np.zeros((cam_h, cam_w, 3), np.uint8)

# ---------- run until quit (NO FPS CAPS) ----------
loop_start = time.perf_counter()
frames_written = 0
paused = False
last_proj = None

try:
    while True:
        # events
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                raise KeyboardInterrupt
            if e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    raise KeyboardInterrupt
                if e.key in (pygame.K_SPACE, pygame.K_p):
                    paused = not paused

        if not paused:
            t = time.perf_counter() - loop_start
            x_center = int(round(x0 + A * math.sin(2 * math.pi * freq_hz * t)))

            # draw in camera space
            cam_img.fill(0)
            cv2.circle(cam_img, (x_center, y_center), radius, (255, 255, 255), -1, lineType=cv2.LINE_8)

            # warp and show
            proj_frame = cv2.remap(
                cam_img, mapx, mapy,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
            )
            last_proj = proj_frame
            surf = _to_surface(proj_frame, (m.width, m.height))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        # grab & write as fast as possible
        frame = cam.grab()
        if frame.dtype != np.uint8:
            frame = cv2.convertScaleAbs(frame)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        frames_written += 1

        # NOTE: no sleep, no clock.tick -> fully uncapped
except KeyboardInterrupt:
    pass
finally:
    loop_end = time.perf_counter()
    loop_seconds = max(1e-6, loop_end - loop_start)
    loop_fps = frames_written / loop_seconds
    print(f"Loop ran for {loop_seconds:.3f}s, frames={frames_written}, loop FPS={loop_fps:.3f}")

    if last_proj is not None:
        cv2.imwrite("out/osc_circle_last_projector_frame.png", last_proj)

    writer.release()
    cam.stop()
    pygame.display.quit(); pygame.quit()

# ---------- re-encode with measured loop FPS ----------
input_path  = raw_path
output_path = "out/osc_circle_camera_retimed.mp4"

cap = cv2.VideoCapture(input_path)
if cap.isOpened():
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(output_path, fourcc, loop_fps, (w, h), True)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        vw.write(fr)
    vw.release()
    cap.release()
    print(f"Re-encoded with loop FPS ({loop_fps:.3f}): {output_path}")
else:
    print("Warning: could not reopen recorded video to re-encode; kept original timing.")

print("Saved:")
print(f" - {raw_path}")
print(" - out/osc_circle_camera_retimed.mp4")
print(" - out/osc_circle_last_projector_frame.png")
