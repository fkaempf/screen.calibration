# warp_stimulus.py  — basic version (no quality enhancements)
import os, time
import numpy as np
import cv2
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "windows")
os.environ.setdefault("SDL_RENDER_DRIVER", "software")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_HINT_VIDEO_HIGHDPI_DISABLED", "1")

try:
    from screeninfo import get_monitors
except Exception:
    get_monitors = None

# ---------- projector display ----------
def _pick_monitor():
    if get_monitors:
        mons = get_monitors()
        if mons:
            return max(mons, key=lambda m: m.x)
    class M: pass
    m = M(); m.x = 0; m.y = 0; m.width = 800; m.height = 600
    return m

def _frame_to_surface(img_u8, size_wh):
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

def project_image(img_u8, seconds=None):
    m = _pick_monitor()
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{m.x},{m.y}"
    pygame.init()
    screen = pygame.display.set_mode((m.width, m.height), pygame.SWSURFACE | pygame.NOFRAME)
    pygame.display.set_caption("Projector")
    surf = _frame_to_surface(img_u8, (m.width, m.height))
    screen.blit(surf, (0, 0)); pygame.display.flip()
    t0 = time.time()
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.display.quit(); pygame.quit(); return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                pygame.display.quit(); pygame.quit(); return
        if seconds is not None and (time.time() - t0) >= seconds:
            break
        time.sleep(0.01)
    pygame.display.quit(); pygame.quit()

# ---------- mapping and helpers ----------
def build_proj_to_cam_map(proj_x, proj_y, proj_w, proj_h, valid_mask=None):
    """
    Minimal inversion: fill only where we have measurements, leave others at -1.
    No hole filling, no smoothing.
    """
    cam_h, cam_w = proj_x.shape
    mapx = np.full((proj_h, proj_w), -1.0, np.float32)
    mapy = np.full((proj_h, proj_w), -1.0, np.float32)

    yy, xx = np.indices((cam_h, cam_w))
    ok = (proj_x >= 0) & (proj_y >= 0) & (proj_x < proj_w) & (proj_y < proj_h)
    if valid_mask is not None:
        ok &= valid_mask.astype(bool)

    u = proj_x[ok]; v = proj_y[ok]
    mapx[v, u] = xx[ok].astype(np.float32)
    mapy[v, u] = yy[ok].astype(np.float32)
    return mapx, mapy

def build_gain_map(mapx, mapy, white_cap_u8, eps=1e-3):
    """
    Minimal photometric map: sample white image and invert.
    No blurring or post-processing.
    """
    samp = cv2.remap(white_cap_u8.astype(np.float32)/255.0, mapx, mapy,
                     interpolation=cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    gain = 1.0 / np.maximum(samp, eps)
    return gain.astype(np.float32)

def make_camera_grid(h, w, step=40, thick=1):
    img = np.full((h, w), 255, np.uint8)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w-1, y), 0, thick)
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h-1), 0, thick)
    cv2.drawMarker(img, (w//2, h//2), 0, cv2.MARKER_CROSS, step, max(1, thick))
    return img

def make_uv_map(mapx, mapy, K, D, model="pinhole"):
    """
    Minimal UV map: undistort to normalized rays, convert to equirect UV.
    No flips or adjustments.
    """
    proj_h, proj_w = mapx.shape
    pts = np.stack([mapx.ravel(), mapy.ravel()], axis=1).astype(np.float32).reshape(-1, 1, 2)

    if model == "fisheye":
        norm = cv2.fisheye.undistortPoints(pts, K, D)
    else:
        norm = cv2.undistortPoints(pts, K, D)

    x = norm[:, 0, 0]
    y = norm[:, 0, 1]
    z = np.ones_like(x)

    v = np.stack([x, y, z], axis=1)
    v /= np.linalg.norm(v, axis=1, keepdims=True)

    theta = np.arctan2(v[:, 1], v[:, 0])                         # azimuth
    phi   = np.arctan2(v[:, 2], np.hypot(v[:, 0], v[:, 1]))      # elevation

    U = (theta + np.pi) / (2*np.pi)
    V = (phi + 0.5*np.pi) / np.pi

    return np.stack([U, V], axis=1).reshape(proj_h, proj_w, 2).astype(np.float32)
