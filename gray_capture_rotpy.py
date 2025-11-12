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


# ---------------- Alvium camera backend ----------------

from CamAlvium import CamAlvium
           
# ---------------- RotPy camera backend ----------------


from CamRotPy import CamRotPy

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
                       avg_per_pattern=3, avg_mode="mean",
                       camtype = "rotpy"):
    """
    Timing safe with averaging:
      - show each pattern
      - optional discard of one stale frame
      - trigger and average N frames for that pattern
    """
    patterns, black, white = generate_graycode_patterns(proj_w, proj_h)
    m = _pick_monitor(proj_monitor_mode, proj_monitor_index, (proj_w, proj_h))
    surf = _setup_window_at_monitor(m)
    if camtype.lower() == "alvium":
        cam = CamAlvium(exposure_ms, gain_db)
    elif camtype.lower() == "rotpy":
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

# ==== SINE / FRINGE PATTERNS (4-step) ========================================

def gen_sine_patterns(W, H, periods, nphase=4, axis='x', gamma=None):
    """
    Generate nphase phase-shifted cos fringes along x or y.
    periods: number of periods across the axis (e.g., 64 for W=800 -> ~12.5px period)
    gamma: if not None, apply inverse-gamma LUT (e.g., 2.2) to linearize projector
    """
    import numpy as np, cv2
    if axis == 'x':
        u = np.linspace(0, 2*np.pi*periods, W, endpoint=False)[None, :]
        u = np.repeat(u, H, axis=0)
    else:
        u = np.linspace(0, 2*np.pi*periods, H, endpoint=False)[:, None]
        u = np.repeat(u, W, axis=1)

    pats = []
    for k in range(nphase):
        phi = 2*np.pi * (k / nphase)   # 0, pi/2, pi, 3pi/2
        img = 0.5 + 0.5*np.cos(u + phi)             # [0..1]
        u8  = (img * 255.0 + 0.5).astype(np.uint8)
        if gamma is not None:
            # inverse gamma so the *camera* sees closer to linear sinusoid
            x = np.arange(256, dtype=np.float32)/255.0
            lut = np.clip((x ** (1.0/gamma))*255.0 + 0.5, 0, 255).astype(np.uint8)
            u8 = cv2.LUT(u8, lut)
        pats.append(u8)
    return pats

def decode_phase_nstep(frames):
    """
    General N-step phase-shifting decoder (arbitrary number of evenly spaced phase shifts).

    frames : list/tuple of N equal-sized uint8/float32 images
             each image corresponds to one projected sine with phase offset φ_k = 2π*k/N
    returns:
        phase ∈ [-π, π)
        modulation ∈ [0,1]
    """
    import numpy as np

    # Convert to float32 stack [H, W, N]
    I = np.stack([f.astype(np.float32) for f in frames], axis=-1)
    N = I.shape[-1]
    if N < 3:
        raise ValueError("Need at least 3 phase-shifted images")

    # Precompute equally spaced phase offsets (in radians)
    phi = np.linspace(0, 2*np.pi, N, endpoint=False).astype(np.float32)

    # Compute the first-harmonic complex Fourier coefficient
    # a1 = Σ I_k * exp(-j*φ_k)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    a_real = np.tensordot(I, cos_phi, axes=([-1],[0]))
    a_imag = np.tensordot(I, sin_phi, axes=([-1],[0]))

    # Wrapped phase [-π, π)
    phase = np.arctan2(-a_imag, a_real)   # negative imag for conventional orientation

    # Modulation and mean intensity
    A = np.sqrt(a_real**2 + a_imag**2) * (2.0 / N)
    Iavg = np.mean(I, axis=-1)
    mod = np.clip(A / (Iavg + 1e-6), 0, 1)

    return phase, mod


def _show_and_grab_sequence(surface, cam, patterns, wait_s, avg_per=1, sleep_after=0.0,camtype = "rotpy"):
    """
    Projects each pattern and grabs one (or averaged) frame for each.
    """
    import time, numpy as np, cv2, pygame
    captured = []
    for pat in patterns:
        # show
        for e in pygame.event.get():
            if e.type == pygame.QUIT: raise KeyboardInterrupt
        if pat.dtype != np.uint8: pat = cv2.convertScaleAbs(pat)
        rgb = np.dstack([pat]*3)
        pygame.surfarray.blit_array(surface, rgb.swapaxes(0,1)); pygame.display.flip()

        time.sleep(wait_s)
        if avg_per <= 1:
            frm = cam.grab()
        else:
            stack = [cam.grab() for _ in range(avg_per)]
            frm = np.median(np.stack(stack,0),0).astype(np.uint8)
        if frm.dtype != np.uint8:
            frm = cv2.convertScaleAbs(frm)
        captured.append(np.ascontiguousarray(frm))
        if sleep_after > 0:
            time.sleep(sleep_after)
    return captured

def capture_sine_sets(proj_w, proj_h,
                      exposure_ms=10.0, gain_db=0.0,
                      proj_monitor_mode="index", proj_monitor_index=1,
                      periods_x=5000, periods_y=5000, nphase=4,
                      wait_s=None, avg_per=1, gamma=None,camtype = "rotpy"):
    """
    Projects 4-step sine sets for X and Y, captures frames, and returns:
      (frames_x[4], frames_y[4], black_cap, white_cap)
    """
    import pygame, time, numpy as np, cv2
    # pick monitor and window
    m = _pick_monitor(proj_monitor_mode, proj_monitor_index, (proj_w, proj_h))
    surface = _setup_window_at_monitor(m)

    # camera
    if camtype.lower() == "alvium":
        cam = CamAlvium(exposure_ms, gain_db); cam.start()
    elif camtype.lower() == "rotpy":
        cam = CamRotPy(exposure_ms, gain_db); cam.start()

    if wait_s is None:
        wait_s = max(exposure_ms/1000.0 + 0.02, 0.06)

    # patterns
    patt_x = gen_sine_patterns(proj_w, proj_h, periods_x, nphase=nphase, axis='x', gamma=gamma)
    patt_y = gen_sine_patterns(proj_w, proj_h, periods_y, nphase=nphase, axis='y', gamma=gamma)
    black = np.zeros((proj_h, proj_w), np.uint8)
    white = np.full((proj_h, proj_w), 255, np.uint8)

    try:
        # settle on black
        surface.fill((0,0,0)); pygame.display.flip(); time.sleep(0.2)

        frames_x = _show_and_grab_sequence(surface, cam, patt_x, wait_s, avg_per=avg_per)
        frames_y = _show_and_grab_sequence(surface, cam, patt_y, wait_s, avg_per=avg_per)

        # capture black/white for validity
        # black
        for e in pygame.event.get():
            if e.type == pygame.QUIT: raise KeyboardInterrupt
        pygame.surfarray.blit_array(surface, np.dstack([black]*3).swapaxes(0,1)); pygame.display.flip()
        time.sleep(wait_s); black_cap = cam.grab()
        # white
        for e in pygame.event.get():
            if e.type == pygame.QUIT: raise KeyboardInterrupt
        pygame.surfarray.blit_array(surface, np.dstack([white]*3).swapaxes(0,1)); pygame.display.flip()
        time.sleep(wait_s); white_cap = cam.grab()

    finally:
        cam.stop(); pygame.display.quit(); pygame.quit()

    if black_cap.dtype != np.uint8: black_cap = cv2.convertScaleAbs(black_cap)
    if white_cap.dtype != np.uint8: white_cap = cv2.convertScaleAbs(white_cap)
    return frames_x, frames_y, black_cap, white_cap

def unwrap_with_gray(phase_wrapped, coarse_int):
    """
    Simple hybrid: use coarse Gray integer as pixel index; phase gives subpixel offset.
    Returns float coordinates: coarse + (phase/2pi - 0.5).
    """
    frac = (phase_wrapped + np.pi) / (2*np.pi)   # [-pi,pi) -> [0,1)
    return coarse_int.astype(np.float32) + (frac - 0.5)

def capture_and_decode_sine_hybrid(proj_w, proj_h,
                                   exposure_ms=10.0, gain_db=0.0,
                                   proj_monitor_mode="index", proj_monitor_index=1,
                                   periods_x=64, periods_y=48, nphase=4,
                                   wait_s=None, avg_per=1, gamma=None,
                                   gray_minimal=True, mod_thresh=0.2,camtype = "rotpy"):
    """
    Full hybrid:
      1) run your existing GRAY decode to get coarse proj_x, proj_y (ints)
      2) capture 4-step sine X and Y; decode wrapped phases + modulation
      3) combine to get float projector coords with subpixel precision
    Returns: proj_x_float, proj_y_float, black_cap, white_cap, valid_mask
    """
    # 1) gray
    px_i, py_i, black_cap, white_cap, valid_gray = capture_and_decode(
        proj_w, proj_h, exposure_ms, gain_db,
        proj_monitor_mode, proj_monitor_index, wait_s,camtype=camtype,
    )

    # 2) sine sets
    frames_x, frames_y, black_cap2, white_cap2 = capture_sine_sets(
        proj_w, proj_h, exposure_ms, gain_db,
        proj_monitor_mode, proj_monitor_index,
        periods_x, periods_y, nphase, wait_s, avg_per, gamma,camtype=camtype
    )
    # prefer the later black/white (same pose)
    black_cap = black_cap2; white_cap = white_cap2

    # decode 4-step phases
    phase_x, mod_x = decode_phase_nstep(frames_x)
    phase_y, mod_y = decode_phase_nstep(frames_y)

    # 3) hybrid unwrap
    px_f = unwrap_with_gray(phase_x, px_i)
    py_f = unwrap_with_gray(phase_y, py_i)

    # validity: gray valid + not saturated + modulation ok
    B = black_cap.astype(np.int16); W = white_cap.astype(np.int16)
    valid = (W - B) > 10
    valid = valid & valid_gray.astype(bool) & (mod_x > mod_thresh) & (mod_y > mod_thresh)

    # clamp out of range to -1
    px_f[~valid] = -1
    py_f[~valid] = -1
    return px_f.astype(np.float32), py_f.astype(np.float32), black_cap, white_cap, valid.astype(np.uint8)

if __name__ == "__main__":
    px, py, b, w, val = capture_and_decode(800, 600, exposure_ms=16.7, avg_per_pattern=3, avg_mode="median")
    print("Got maps:", px.shape, py.shape, "valid ratio:", float(val.mean()))
