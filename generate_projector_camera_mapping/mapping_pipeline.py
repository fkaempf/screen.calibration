import os, time
import numpy as np
import cv2
import pygame
import sys

sys.path.append(r"D:\screen.calibration")

from cameras.CamAlvium import CamAlvium
from cameras.CamRotPy import CamRotPy
from generate_projector_camera_mapping.mapping_utils import (
    capture_and_decode_sine_hybrid,
    capture_and_decode,
)
from stimulus.warp_stimulus import build_proj_to_cam_map, make_camera_grid, make_uv_map


try:
    from screeninfo import get_monitors
except Exception:
    get_monitors = None


def _pick_monitor_rightmost(proj_w, proj_h):
    """
    Pick rightmost monitor; fall back to a virtual monitor of size proj_w x proj_h.
    """
    if get_monitors:
        mons = get_monitors()
        if mons:
            return max(mons, key=lambda m: m.x)

    class M:
        pass

    m = M()
    m.x = 0
    m.y = 0
    m.width = proj_w
    m.height = proj_h
    return m


def _frame_to_surface(img_u8, size_wh):
    W, H = size_wh
    if img_u8.dtype != np.uint8:
        img_u8 = cv2.convertScaleAbs(img_u8)
    if img_u8.ndim == 2:
        rgb = np.dstack([img_u8] * 3)
    else:
        rgb = img_u8[..., ::-1] if img_u8.shape[2] == 3 else img_u8
    if (rgb.shape[1], rgb.shape[0]) != (W, H):
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))


def project_and_capture_single(
    img_u8,
    save_path,
    exposure_ms=10.0,
    gain_db=0.0,
    hold_seconds=3.0,
    settle_seconds=0.2,
    camtype="rotpy",
):
    h_img, w_img = img_u8.shape[:2]
    m = _pick_monitor_rightmost(w_img, h_img)

    os.environ.setdefault("SDL_VIDEODRIVER", "windows")
    os.environ.setdefault("SDL_RENDER_DRIVER", "software")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("SDL_HINT_VIDEO_HIGHDPI_DISABLED", "1")
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{m.x},{m.y}"

    pygame.init()
    screen = pygame.display.set_mode((m.width, m.height), pygame.SWSURFACE | pygame.NOFRAME)
    pygame.display.set_caption("Projector")

    surf = _frame_to_surface(img_u8, (m.width, m.height))
    screen.blit(surf, (0, 0))
    pygame.display.flip()

    if camtype.lower() == "alvium":
        cam = CamAlvium(exposure_ms=exposure_ms, gain_db=gain_db)
        cam.start()
    elif camtype.lower() == "rotpy":
        cam = CamRotPy(exposure_ms=exposure_ms, gain_db=gain_db)
        cam.start()
    else:
        raise ValueError(f"Unknown camtype {camtype}")

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
            if e.type == pygame.QUIT:
                pygame.display.quit()
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                pygame.display.quit()
                pygame.quit()
                return
        if (time.time() - t0) >= hold_seconds:
            break
        time.sleep(0.01)

    pygame.display.quit()
    pygame.quit()


def save_heat(path, A, vmax):
    A = A.astype(np.float32).copy()
    A[A < 0] = 0
    if vmax <= 0:
        vmax = 1.0
    u8 = np.clip(A / float(vmax), 0.0, 1.0)
    u8 = (u8 * 255.0).astype(np.uint8)
    vis = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    cv2.imwrite(path, vis)


def save_heat01(path, A01):
    u8 = (np.clip(A01, 0.0, 1.0) * 255.0).astype(np.uint8)
    vis = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    cv2.imwrite(path, vis)


def make_equirect_checker(h=1024, w=2048, step_deg=30, line_px=2):
    img = np.full((h, w, 3), 255, np.uint8)
    step_x = max(1, int(round(w * step_deg / 360.0)))
    step_y = max(1, int(round(h * step_deg / 180.0)))
    for x in range(0, w, step_x):
        cv2.line(img, (x, 0), (x, h - 1), (0, 0, 0), line_px)
    for y in range(0, h, step_y):
        cv2.line(img, (0, y), (w - 1, y), (0, 0, 0), line_px)
    cv2.line(img, (w // 2, 0), (w // 2, h - 1), (0, 0, 255), line_px + 1)
    cv2.line(img, (0, h // 2), (w - 1, h // 2), (0, 0, 255), line_px + 1)
    return img


def despeckle_maps(mapx, mapy, valid_mask=None, k_med=3, k_avg=5, tol=2.0):
    mx = mapx.astype(np.float32, copy=True)
    my = mapy.astype(np.float32, copy=True)

    if valid_mask is not None and valid_mask.shape == mx.shape:
        vm = valid_mask.astype(bool)
    else:
        vm = (mx >= 0) & (my >= 0)

    mx_med = cv2.medianBlur(mx, k_med)
    my_med = cv2.medianBlur(my, k_med)

    out_x = (np.abs(mx - mx_med) > tol) & vm
    out_y = (np.abs(my - my_med) > tol) & vm

    mx[out_x] = mx_med[out_x]
    my[out_y] = my_med[out_y]

    vm_f = vm.astype(np.float32)
    ksize = (k_avg, k_avg)
    sum_w = cv2.boxFilter(vm_f, -1, ksize, normalize=False)
    sum_x = cv2.boxFilter(mx * vm_f, -1, ksize, normalize=False)
    sum_y = cv2.boxFilter(my * vm_f, -1, ksize, normalize=False)

    good = sum_w > 0
    mx[good] = sum_x[good] / sum_w[good]
    my[good] = sum_y[good] / sum_w[good]

    return mx, my


def execute_mapping(
    # camera / projector
    CAMTYPE="alvium",
    PROJ_W=1280,
    PROJ_H=800,
    # pattern / mode
    MODE="sine_hybrid",
    PERIODS_X=128,
    PERIODS_Y=96,
    NPHASE=15,
    AVG_PER=5,
    GAMMA_INV=None,
):
    ts = time.strftime("%Y%m%d_%H%M%S")
    OUT = f"D:/screen.calibration/debug/pipeline/{ts}"
    os.makedirs(OUT, exist_ok=True)
    print("Saving to", OUT)

    if MODE == "sine_hybrid":
        proj_x_f, proj_y_f, black_cap, white_cap, valid = capture_and_decode_sine_hybrid(
            proj_w=PROJ_W,
            proj_h=PROJ_H,
            exposure_ms=2,
            gain_db=0.0,
            proj_monitor_mode="index",
            proj_monitor_index=1,
            periods_x=PERIODS_X,
            periods_y=PERIODS_Y,
            nphase=NPHASE,
            wait_s=None,
            avg_per=AVG_PER,
            gamma=GAMMA_INV,
            mod_thresh=0.2,
            camtype=CAMTYPE,
        )
        proj_x = np.where(valid, np.rint(proj_x_f).astype(np.int32), -1)
        proj_y = np.where(valid, np.rint(proj_y_f).astype(np.int32), -1)

    else:
        proj_x, proj_y, black_cap, white_cap, valid = capture_and_decode(
            proj_w=PROJ_W,
            proj_h=PROJ_H,
            exposure_ms=7,
            gain_db=0.0,
            proj_monitor_mode="index",
            proj_monitor_index=1,
        )

    cam_h, cam_w = proj_x.shape
    cv2.imwrite(os.path.join(OUT, "valid_mask.png"), (valid.astype(np.uint8) * 255))
    save_heat(os.path.join(OUT, "proj_x_heat.png"), proj_x, PROJ_W - 1)
    save_heat(os.path.join(OUT, "proj_y_heat.png"), proj_y, PROJ_H - 1)

    mapx, mapy = build_proj_to_cam_map(proj_x, proj_y, PROJ_W, PROJ_H, valid_mask=valid)
    mask_invalid = (mapx < 0) | (mapy < 0)
    if np.any(mask_invalid):
        mx = cv2.inpaint(mapx, (mask_invalid * 255).astype(np.uint8), 5, cv2.INPAINT_NS)
        my = cv2.inpaint(mapy, (mask_invalid * 255).astype(np.uint8), 5, cv2.INPAINT_NS)
        mapx, mapy = mx, my

    mapx, mapy = despeckle_maps(mapx, mapy, valid_mask=valid, k_med=3, k_avg=5, tol=2.0)

    mapx = cv2.medianBlur(mapx, 3)
    mapy = cv2.medianBlur(mapy, 3)

    save_heat(os.path.join(OUT, "mapx_heat.png"), mapx, cam_w - 1)
    save_heat(os.path.join(OUT, "mapy_heat.png"), mapy, cam_h - 1)

    grid = make_camera_grid(cam_h, cam_w, step=40, thick=5)
    proj_grid = cv2.remap(
        grid,
        mapx,
        mapy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    cv2.imwrite(os.path.join(OUT, "projector_grid.png"), proj_grid)

    project_and_capture_single(
        proj_grid,
        save_path=os.path.join(OUT, "camera_view_of_warp.png"),
        exposure_ms=3,
        gain_db=0.0,
        hold_seconds=3.0,
        settle_seconds=0.2,
        camtype=CAMTYPE,
    )

    np.save(os.path.join(OUT, "mapx.npy"), mapx)
    np.save(os.path.join(OUT, "mapy.npy"), mapy)

    if CAMTYPE.lower() == "rotpy":
        K_path = "D:/screen.calibration/configs/pinhole.config/pinhole.K.npy"
        D_path = "D:/screen.calibration/configs/pinhole.config/pinhole.D.npy"
        xi_path = None
    elif CAMTYPE.lower() == "alvium":
        K_path = "D:/screen.calibration/configs/fisheye.config/fisheye.K.npy"
        D_path = "D:/screen.calibration/configs/fisheye.config/fisheye.D.npy"
        xi_path = "D:/screen.calibration/configs/fisheye.config/fisheye.xi.npy"
        if not (os.path.exists(K_path) and os.path.exists(D_path) and os.path.exists(xi_path)):
            raise FileNotFoundError("omnidir intrinsics missing: K/D/xi")
    else:
        raise ValueError(f"Unknown CAMTYPE {CAMTYPE}")

    if os.path.exists(K_path) and os.path.exists(D_path):
        if CAMTYPE.lower() == "rotpy":
            K = np.load(K_path).astype(np.float64)
            D = np.load(D_path).astype(np.float64)
            uv = make_uv_map(mapx, mapy, K, D, model="pinhole")

        elif CAMTYPE.lower() == "alvium":
            K = np.load(K_path).astype(np.float64)
            D = np.load(D_path).astype(np.float64)
            xi = np.load(xi_path).astype(np.float64)
            uv = make_uv_map(
                mapx,
                mapy,
                K,
                D,
                model="fisheye_xi",
                xi=xi,
                w=cam_w,
                h=cam_h,
            )

        np.save(os.path.join(OUT, "uv_map.npy"), uv)
        save_heat01(os.path.join(OUT, "uv_U_heat.png"), uv[..., 0])
        save_heat01(os.path.join(OUT, "uv_V_heat.png"), uv[..., 1])

        stim = make_equirect_checker(1024, 2048, step_deg=30, line_px=2)
        Ht, Wt = stim.shape[:2]
        mapx_uv = (uv[..., 0] * (Wt - 1)).astype(np.float32)
        mapy_uv = (uv[..., 1] * (Ht - 1)).astype(np.float32)
        proj_vr = cv2.remap(
            stim,
            mapx_uv,
            mapy_uv,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        cv2.imwrite(os.path.join(OUT, "projector_vr_checker.png"), proj_vr)

        project_and_capture_single(
            proj_vr,
            save_path=os.path.join(OUT, "camera_view_of_vr_checker.png"),
            exposure_ms=2,
            gain_db=0.0,
            hold_seconds=3.0,
            settle_seconds=0.2,
            camtype=CAMTYPE,
        )

    np.save("D:/screen.calibration/configs/camera.projector.mapping/mapx.npy", mapx)
    np.save("D:/screen.calibration/configs/camera.projector.mapping/mapy.npy", mapy)

    _, bw = cv2.threshold(
        valid.astype(np.uint8) * 255,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest_label).astype("uint8").astype(bool)
    np.save("D:/screen.calibration/configs/camera.projector.mapping/valid.mask.npy", mask)


if __name__ == "__main__":
    # camera / projector
    CAMTYPE = "alvium"   # "rotpy" or "alvium"
    PROJ_W = 1280
    PROJ_H = 800

    # pattern / mode
    MODE = "sine_hybrid"  # "gray" or "sine_hybrid"
    PERIODS_X = 128       # ~12.5px period for 800px width
    PERIODS_Y = 96        # ~12.5px period for 600px height
    NPHASE = 15
    AVG_PER = 5           # increase to 3–5 if needed
    GAMMA_INV = None      # set 2.2 to apply inverse gamma to sines

    execute_mapping(
        CAMTYPE=CAMTYPE,
        PROJ_W=PROJ_W,
        PROJ_H=PROJ_H,
        MODE=MODE,
        PERIODS_X=PERIODS_X,
        PERIODS_Y=PERIODS_Y,
        NPHASE=NPHASE,
        AVG_PER=AVG_PER,
        GAMMA_INV=GAMMA_INV,
    )
