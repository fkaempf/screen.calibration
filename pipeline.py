# pipeline.py  — basic with heat maps
import os, time
import numpy as np
import cv2
import pygame

from gray_capture_rotpy import capture_and_decode, CamRotPy
from warp_stimulus import build_proj_to_cam_map, make_camera_grid, make_uv_map

PROJ_W = 800
PROJ_H = 600

try:
    from screeninfo import get_monitors
except Exception:
    get_monitors = None

def _pick_monitor_rightmost():
    if get_monitors:
        mons = get_monitors()
        if mons:
            return max(mons, key=lambda m: m.x)
    class M: pass
    m = M(); m.x = 0; m.y = 0; m.width = PROJ_W; m.height = PROJ_H
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
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

def project_and_capture_single(img_u8, save_path, exposure_ms=10.0, gain_db=0.0, hold_seconds=3.0, settle_seconds=0.2):
    m = _pick_monitor_rightmost()
    os.environ.setdefault("SDL_VIDEODRIVER", "windows")
    os.environ.setdefault("SDL_RENDER_DRIVER", "software")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("SDL_HINT_VIDEO_HIGHDPI_DISABLED", "1")
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{m.x},{m.y}"

    pygame.init()
    screen = pygame.display.set_mode((m.width, m.height), pygame.SWSURFACE | pygame.NOFRAME)
    pygame.display.set_caption("Projector")
    surf = _frame_to_surface(img_u8, (m.width, m.height))
    screen.blit(surf, (0, 0)); pygame.display.flip()

    cam = CamRotPy(exposure_ms=exposure_ms, gain_db=gain_db)
    cam.start()
    time.sleep(settle_seconds)
    frame = cam.grab()
    cam.stop()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if frame.dtype != np.uint8:
        frame = cv2.convertScaleAbs(frame)
    cv2.imwrite(save_path, frame)

    t0 = time.time()
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: pygame.display.quit(); pygame.quit(); return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE: pygame.display.quit(); pygame.quit(); return
        if (time.time() - t0) >= hold_seconds: break
        time.sleep(0.01)
    pygame.display.quit(); pygame.quit()

def save_heat(path, A, vmax):
    # generic colored heat map saver for arrays with known max
    A = A.astype(np.float32).copy()
    A[A < 0] = 0  # hide invalids
    if vmax <= 0: vmax = 1.0
    u8 = np.clip(A / float(vmax), 0.0, 1.0)
    u8 = (u8 * 255.0).astype(np.uint8)
    vis = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    cv2.imwrite(path, vis)

def save_heat01(path, A01):
    # colored heat map for values already in [0,1]
    u8 = (np.clip(A01, 0.0, 1.0) * 255.0).astype(np.uint8)
    vis = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    cv2.imwrite(path, vis)

def make_equirect_checker(h=1024, w=2048, step_deg=30, line_px=2):
    img = np.full((h, w, 3), 255, np.uint8)
    step_x = max(1, int(round(w * step_deg / 360.0)))
    step_y = max(1, int(round(h * step_deg / 180.0)))
    for x in range(0, w, step_x):
        cv2.line(img, (x, 0), (x, h-1), (0, 0, 0), line_px)
    for y in range(0, h, step_y):
        cv2.line(img, (0, y), (w-1, y), (0, 0, 0), line_px)
    cv2.line(img, (w // 2, 0), (w // 2, h - 1), (0, 0, 255), line_px + 1)
    cv2.line(img, (0, h // 2), (w - 1, h // 2), (0, 0, 255), line_px + 1)
    return img

if __name__ == "__main__":
    ts = time.strftime("%Y%m%d_%H%M%S")
    OUT = os.path.join("out", ts)
    os.makedirs(OUT, exist_ok=True)
    print("Saving to", OUT)

    # 1) capture and decode
    proj_x, proj_y, black_cap, white_cap, valid = capture_and_decode(
        proj_w=PROJ_W, proj_h=PROJ_H,
        exposure_ms=7, gain_db=0.0,wait_s=0.1,
        proj_monitor_mode="index", proj_monitor_index=1,avg_per_pattern=100, avg_mode="mean"
    )

    # save valid mask and camera side heat maps
    cam_h, cam_w = proj_x.shape
    cv2.imwrite(os.path.join(OUT, "valid_mask.png"), (valid.astype(np.uint8) * 255))
    save_heat(os.path.join(OUT, "proj_x_heat.png"), proj_x, PROJ_W - 1)
    save_heat(os.path.join(OUT, "proj_y_heat.png"), proj_y, PROJ_H - 1)

    # 2) invert to projector space
    mapx, mapy = build_proj_to_cam_map(proj_x, proj_y, PROJ_W, PROJ_H, valid_mask=valid)
    mask = (mapx < 0) | (mapy < 0)
    if np.any(mask):
        mx = cv2.inpaint(mapx, (mask*255).astype(np.uint8), 5, cv2.INPAINT_NS)
        my = cv2.inpaint(mapy, (mask*255).astype(np.uint8), 5, cv2.INPAINT_NS)
        mapx, mapy = mx, my

    # save projector side heat maps
    save_heat(os.path.join(OUT, "mapx_heat.png"), mapx, cam_w - 1)
    save_heat(os.path.join(OUT, "mapy_heat.png"), mapy, cam_h - 1)

    # 3) build a simple camera grid and warp it to projector pixels
    grid = make_camera_grid(cam_h, cam_w, step=40, thick=5)
    proj_grid = cv2.remap(grid, mapx, mapy, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cv2.imwrite(os.path.join(OUT, "projector_grid.png"), proj_grid)

    # 4) show the warped grid and capture one camera photo
    project_and_capture_single(
        proj_grid,
        save_path=os.path.join(OUT, "camera_view_of_warp.png"),
        exposure_ms=10.0,
        gain_db=0.0,
        hold_seconds=3.0,
        settle_seconds=0.2
    )

    # 5) save the LUTs
    np.save(os.path.join(OUT, "mapx.npy"), mapx)
    np.save(os.path.join(OUT, "mapy.npy"), mapy)

    # 6) uv_map: compute, save, visualize, render a VR checker, project, capture
    K_path, D_path = "K_cam.npy", "D_cam.npy"
    if os.path.exists(K_path) and os.path.exists(D_path):
        K = np.load(K_path).astype(np.float64)
        D = np.load(D_path).astype(np.float64)
        MODEL = "pinhole"  # set to "fisheye" if you calibrated with cv2.fisheye

        uv = make_uv_map(mapx, mapy, K, D, model=MODEL)
        np.save(os.path.join(OUT, "uv_map.npy"), uv)

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
            exposure_ms=10.0,
            gain_db=0.0,
            hold_seconds=3.0,
            settle_seconds=0.2
        )
    else:
        print("K_cam.npy or D_cam.npy not found, uv_map step skipped")
