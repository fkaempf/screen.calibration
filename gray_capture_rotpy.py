# gray_capture_rotpy_basic.py  with averaging
import os, time, math
import numpy as np
import cv2
import pygame

cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "windows")
os.environ.setdefault("SDL_RENDER_DRIVER", "software")
os.environ.setdefault("SDL_HINT_VIDEO_HIGHDPI_DISABLED", "1")

try:
    from screeninfo import get_monitors
except Exception:
    get_monitors = None

# ---------------- projector helpers ----------------
def _pick_monitor(mode="index", index=1, fallback=(800, 600)):
    if get_monitors:
        mons = get_monitors()
        if mons:
            if mode == "index":
                i = max(0, min(index, len(mons) - 1))
                return mons[i]
            return max(mons, key=lambda m: m.x)
    class M: pass
    m = M(); m.x = 0; m.y = 0; m.width, m.height = fallback
    return m

def _setup_window_at_monitor(m):
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{m.x},{m.y}"
    pygame.display.init()
    surf = pygame.display.set_mode((m.width, m.height), pygame.NOFRAME | pygame.SWSURFACE)
    pygame.display.set_caption("Projector")
    return surf

def projector_show(surface, img_u8):
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            raise KeyboardInterrupt
    if img_u8.dtype != np.uint8:
        img_u8 = cv2.convertScaleAbs(img_u8)
    if img_u8.ndim == 2:
        rgb = np.dstack([img_u8]*3)
    else:
        rgb = img_u8[..., ::-1] if img_u8.shape[2] == 3 else img_u8
    pygame.surfarray.blit_array(surface, rgb.swapaxes(0, 1))
    pygame.display.flip()

# ---------------- RotPy camera backend ----------------
class CamRotPy:
    def __init__(self, exposure_ms=10.0, gain_db=0.0):
        from rotpy.system import SpinSystem
        from rotpy.camera import CameraList
        self.system = SpinSystem()
        cam_list = CameraList.create_from_system(self.system, True, True)
        if cam_list.get_size() < 1:
            raise RuntimeError("No FLIR camera found")
        self.cam = cam_list.create_camera_by_index(0)
        self.exposure_ms = exposure_ms
        self.gain_db = gain_db

    def start(self):
        c = self.cam
        c.init_cam()
        try: c.camera_nodes.PixelFormat.set_node_value_from_str("Mono8")
        except: pass
        try: c.camera_nodes.ExposureAuto.set_node_value_from_str("Off")
        except: pass
        try: c.camera_nodes.ExposureTime.set_node_value(max(500.0, min(self.exposure_ms*1000.0, 3e7)))
        except: pass
        try: c.camera_nodes.GainAuto.set_node_value_from_str("Off")
        except: pass
        try: c.camera_nodes.Gain.set_node_value(self.gain_db)
        except: pass
        # force software trigger
        try:
            c.camera_nodes.TriggerMode.set_node_value_from_str("Off")
            c.camera_nodes.TriggerSelector.set_node_value_from_str("FrameStart")
            c.camera_nodes.TriggerSource.set_node_value_from_str("Software")
            c.camera_nodes.TriggerMode.set_node_value_from_str("On")
        except: pass
        c.begin_acquisition()
        for _ in range(3):
            try:
                im = c.get_next_image(timeout=0.2)
                im.release()
            except:
                break

    def grab(self, timeout_s=1.0):
        c = self.cam
        try: c.camera_nodes.TriggerSoftware.execute_node()
        except: pass
        im = c.get_next_image(timeout=timeout_s)
        try:
            try: im = im.convert_fmt("Mono8")
            except: pass
            h, w, stride = im.get_height(), im.get_width(), im.get_stride()
            try: b = im.get_image_data_bytes()
            except: b = bytes(im.get_image_data_memoryview())
            a = np.frombuffer(b, np.uint8)[:h*stride].reshape(h, stride)[:, :w]
            return np.ascontiguousarray(a)
        finally:
            im.release()

    def stop(self):
        try: self.cam.end_acquisition()
        except: pass
        try: self.cam.deinit_cam()
        except: pass
        try: self.cam.release()
        except: pass

# ---------------- Gray code generator and minimal decode ----------------
def generate_graycode_patterns(w, h):
    nx = int(math.ceil(math.log2(w)))
    ny = int(math.ceil(math.log2(h)))
    xs = np.arange(w, dtype=np.uint32)
    ys = np.arange(h, dtype=np.uint32)
    gx = xs ^ (xs >> 1)
    gy = ys ^ (ys >> 1)
    pats = []
    for k in range(nx - 1, -1, -1):
        col = ((gx >> k) & 1).astype(np.uint8) * 255
        img = np.repeat(col[np.newaxis, :], h, axis=0)
        pats += [img, 255 - img]
    for k in range(ny - 1, -1, -1):
        row = ((gy >> k) & 1).astype(np.uint8) * 255
        img = np.repeat(row[:, np.newaxis], w, axis=1)
        pats += [img, 255 - img]
    black = np.zeros((h, w), np.uint8)
    white = np.full((h, w), 255, np.uint8)
    return pats, black, white

def gray_to_binary(bits):
    out = bits.copy()
    for i in range(1, out.shape[-1]):
        out[..., i] ^= out[..., i-1]
    return out

def decode_gray_minimal(captured, black_cap, white_cap, proj_w, proj_h):
    C = np.stack(captured, 0).astype(np.int16, copy=False)
    B = black_cap.astype(np.int16, copy=False)
    W = white_cap.astype(np.int16, copy=False)

    nx = int(math.ceil(math.log2(proj_w)))
    ny = int(math.ceil(math.log2(proj_h)))

    x_pos = C[0:nx*2:2]; x_neg = C[1:nx*2:2]
    y_pos = C[nx*2:nx*2+ny*2:2]; y_neg = C[nx*2+1:nx*2+ny*2:2]

    gx_bits = (x_pos > x_neg).astype(np.uint8)
    gy_bits = (y_pos > y_neg).astype(np.uint8)

    bx_bits = gray_to_binary(np.moveaxis(gx_bits, 0, -1))
    by_bits = gray_to_binary(np.moveaxis(gy_bits, 0, -1))

    def bits_to_int(bits):
        v = np.zeros(bits.shape[:2], np.int32)
        for i in range(bits.shape[-1]):
            v = (v << 1) | bits[..., i].astype(np.int32)
        return v

    proj_x = bits_to_int(bx_bits)
    proj_y = bits_to_int(by_bits)

    valid = (W - B) > 10
    proj_x[(proj_x < 0) | (proj_x >= proj_w) | (~valid)] = -1
    proj_y[(proj_y < 0) | (proj_y >= proj_h) | (~valid)] = -1
    return proj_x, proj_y, valid.astype(np.uint8)

# ---------------- averaging helper ----------------
def _grab_avg(cam, n=1, mode="median", discard_first=True):
    if n <= 1:
        f = cam.grab()
        return f if f.dtype == np.uint8 else cv2.convertScaleAbs(f)
    if discard_first:
        try: cam.grab(timeout_s=0.5)
        except: pass
    stack = []
    for _ in range(int(n)):
        frm = cam.grab()
        if frm.dtype != np.uint8:
            frm = cv2.convertScaleAbs(frm)
        stack.append(frm)
    S = np.stack(stack, 0).astype(np.float32)
    if mode == "mean":
        out = S.mean(0)
    else:
        out = np.median(S, 0)
    return np.clip(out, 0, 255).astype(np.uint8)

# ---------------- capture API ----------------
def capture_and_decode(proj_w, proj_h,
                       exposure_ms=10.0, gain_db=0.0,
                       proj_monitor_mode="index", proj_monitor_index=1,
                       wait_s=None,
                       avg_per_pattern=3, avg_mode="mean"):
    """
    Timing safe with averaging:
      - show each pattern
      - optional discard of one stale frame
      - trigger and average N frames for that pattern
    """
    patterns, black, white = generate_graycode_patterns(proj_w, proj_h)
    m = _pick_monitor(proj_monitor_mode, proj_monitor_index, (proj_w, proj_h))
    surf = _setup_window_at_monitor(m)

    cam = CamRotPy(exposure_ms, gain_db)
    cam.start()

    if wait_s is None:
        wait_s = max(exposure_ms/1000.0 + 0.03, 0.07)

    captured = []
    try:
        projector_show(surf, black); time.sleep(0.2)

        for pat in patterns:
            projector_show(surf, pat)
            time.sleep(wait_s)
            frame = _grab_avg(cam, n=avg_per_pattern, mode=avg_mode, discard_first=True)
            captured.append(np.ascontiguousarray(frame))

        projector_show(surf, black); time.sleep(wait_s)
        black_cap = _grab_avg(cam, n=avg_per_pattern, mode=avg_mode, discard_first=True)

        projector_show(surf, white); time.sleep(wait_s)
        white_cap = _grab_avg(cam, n=avg_per_pattern, mode=avg_mode, discard_first=True)

    finally:
        cam.stop()
        pygame.display.quit()
        pygame.quit()

    proj_x, proj_y, valid = decode_gray_minimal(captured, black_cap, white_cap, proj_w, proj_h)
    return proj_x, proj_y, black_cap, white_cap, valid

if __name__ == "__main__":
    px, py, b, w, val = capture_and_decode(800, 600, exposure_ms=16.7, avg_per_pattern=3, avg_mode="median")
    print("Got maps:", px.shape, py.shape, "valid ratio:", float(val.mean()))
