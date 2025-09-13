import os, sys, time, gc
import numpy as np
import cv2
import pygame
import matplotlib.pyplot as plt

# ================== user settings ==================
W, H = 800, 600            # fallback size; will be replaced by projector monitor size
EXPOSURE_MS = 10           # camera exposure in ms
GAIN_DB = 0.0
NUM_DUMMY_FRAMES = 5
PROJ_MONITOR_MODE = "rightmost"  # "rightmost" or "index"
PROJ_MONITOR_INDEX = 1           # only used if PROJ_MONITOR_MODE == "index"
# ===================================================

# try to get monitor geometry
try:
    from screeninfo import get_monitors
except Exception:
    get_monitors = None

# ---------- pygame projector helpers ----------
SURF = None
CLOCK = None
_proj_xywh = None  # (x, y, w, h)

def init_projector_geometry():
    global _proj_xywh, W, H
    if get_monitors is None:
        _proj_xywh = (0, 0, W, H)
        return
    mons = get_monitors()
    if not mons:
        _proj_xywh = (0, 0, W, H)
        return
    if PROJ_MONITOR_MODE == "index":
        i = max(0, min(PROJ_MONITOR_INDEX, len(mons) - 1))
        m = mons[i]
    else:
        m = max(mons, key=lambda M: M.x)  # pick rightmost
    _proj_xywh = (m.x, m.y, m.width, m.height)
    W, H = m.width, m.height  # ensure patterns match projector

def setup_projector_window():
    global SURF, CLOCK
    if _proj_xywh is None:
        raise RuntimeError("Call init_projector_geometry() first")
    x, y, w, h = _proj_xywh
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{x},{y}"
    pygame.init()
    SURF = pygame.display.set_mode((w, h), pygame.NOFRAME)
    pygame.display.set_caption("Projector")
    CLOCK = pygame.time.Clock()

def _blit_gray(img_u8):
    # img_u8 is HxW uint8
    if img_u8.dtype != np.uint8:
        img_u8 = cv2.convertScaleAbs(img_u8)
    rgb = np.dstack([img_u8] * 3)  # HxWx3
    pygame.surfarray.blit_array(SURF, rgb.swapaxes(0, 1))
    pygame.display.flip()

def projector_show(img):
    # process events so the OS keeps the window responsive
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            raise KeyboardInterrupt
    _blit_gray(img)

# ---------- camera backends ----------
class CamBase:
    def start(self): ...
    def stop(self): ...
    def grab(self): ...

class CamPySpin(CamBase):
    def __init__(self, exposure_ms, gain_db):
        import PySpin
        self.PySpin = PySpin
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        if self.cam_list.GetSize() < 1:
            self.system.ReleaseInstance()
            raise RuntimeError("No FLIR camera found")
        self.cam = self.cam_list[0]
        self.exposure_ms = exposure_ms
        self.gain_db = gain_db

    def start(self):
        PS = self.PySpin
        self.cam.Init()

        # Stream: newest only, small buffer
        s_map = self.cam.GetTLStreamNodeMap()
        try:
            sbhm = PS.CEnumerationPtr(s_map.GetNode("StreamBufferHandlingMode"))
            newest = sbhm.GetEntryByName("NewestOnly")
            sbhm.SetIntValue(newest.GetValue())
            bc_mode = PS.CEnumerationPtr(s_map.GetNode("StreamBufferCountMode"))
            manu = bc_mode.GetEntryByName("Manual")
            bc_mode.SetIntValue(manu.GetValue())
            bc = PS.CIntegerPtr(s_map.GetNode("StreamBufferCountManual"))
            bc.SetValue(4)
        except Exception:
            pass

        n_map = self.cam.GetNodeMap()

        # Pixel format
        try:
            pf = PS.CEnumerationPtr(n_map.GetNode("PixelFormat"))
            mono8 = pf.GetEntryByName("Mono8")
            pf.SetIntValue(mono8.GetValue())
        except Exception:
            pass

        # Manual exposure and gain
        self._set_enum(n_map, "ExposureAuto", "Off")
        self._set_float(n_map, "ExposureTime", max(500.0, min(self.exposure_ms*1000.0, 30000000.0)))  # us
        self._set_enum(n_map, "GainAuto", "Off")
        self._set_float(n_map, "Gain", self.gain_db)

        # Trigger: software, FrameStart
        self._set_enum(n_map, "TriggerMode", "Off")
        self._set_enum(n_map, "TriggerSelector", "FrameStart")
        self._set_enum(n_map, "TriggerSource", "Software")
        self._set_enum(n_map, "TriggerMode", "On")

        # Start stream
        self.cam.BeginAcquisition()

        # Flush any stale images
        for _ in range(NUM_DUMMY_FRAMES):
            try:
                img = self.cam.GetNextImage(100)
                img.Release()
            except Exception:
                break

    def trigger_and_grab(self, timeout_ms=100):
        PS = self.PySpin
        trig = PS.CCommandPtr(self.cam.GetNodeMap().GetNode("TriggerSoftware"))
        trig.Execute()
        img = self.cam.GetNextImage(timeout_ms)
        if img.IsIncomplete():
            img.Release()
            raise RuntimeError("Incomplete image")
        arr = img.GetNDArray()
        img.Release()
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        return arr

    def grab(self):
        return self.trigger_and_grab()

    def _set_enum(self, nodemap, name, entry):
        PS = self.PySpin
        node = PS.CEnumerationPtr(nodemap.GetNode(name))
        val = node.GetEntryByName(entry)
        node.SetIntValue(val.GetValue())

    def _set_float(self, nodemap, name, value):
        PS = self.PySpin
        node = PS.CFloatPtr(nodemap.GetNode(name))
        node.SetValue(value)

    def stop(self):
        try:
            self.cam.EndAcquisition()
        except Exception:
            pass
        try:
            self.cam.DeInit()
        except Exception:
            pass
        try:
            self.cam_list.Clear()
        except Exception:
            pass
        try:
            self.system.ReleaseInstance()
        except Exception:
            pass
        gc.collect()

class CamCV(CamBase):
    def __init__(self, index, exposure_ms, gain_db):
        self.cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        self.exposure_ms = exposure_ms
        self.gain_db = gain_db

    def start(self):
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'Y800'))
        if not self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure_ms/1000.0):
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -7)
        self.cap.set(cv2.CAP_PROP_GAIN, self.gain_db)
        for _ in range(NUM_DUMMY_FRAMES):
            self.cap.read()

    def grab(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read from camera")
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def stop(self):
        self.cap.release()

# ---------- main capture ----------
def capture_graycode():
    init_projector_geometry()
    setup_projector_window()

    pat = cv2.structured_light_GrayCodePattern.create(W, H)
    ok, patterns = pat.generate()
    if not ok:
        raise RuntimeError("GrayCodePattern.generate failed")

    # black and white masks
    black = np.zeros((H, W), np.uint8)
    white = np.full((H, W), 255, np.uint8)
    try:
        b, w = pat.getImagesForShadowMasks()
        black, white = b, w
    except Exception:
        pass

    cam = CamPySpin(EXPOSURE_MS, GAIN_DB)

    captured = []
    black_cap = None
    white_cap = None

    try:
        cam.start()
        projector_show(black)
        time.sleep(0.2)

        for pat_img in patterns:
            p8 = pat_img if pat_img.dtype == np.uint8 else cv2.convertScaleAbs(pat_img)
            projector_show(p8)
            time.sleep(0.1)
            frame = cam.grab()
            if frame.dtype != np.uint8:
                frame = cv2.convertScaleAbs(frame)
            captured.append(np.ascontiguousarray(frame).copy())

        projector_show(black); time.sleep(0.1); black_cap = np.ascontiguousarray(cam.grab()).copy()
        projector_show(white); time.sleep(0.1); white_cap = np.ascontiguousarray(cam.grab()).copy()

    finally:
        try:
            cam.stop()
        except Exception:
            pass
        try:
            pygame.display.quit()
            pygame.quit()
        except Exception:
            pass
        gc.collect()

    patterns_u8 = [p if p.dtype == np.uint8 else cv2.convertScaleAbs(p) for p in patterns]
    return patterns_u8, captured, black_cap, white_cap

# ---------- example usage ----------
if __name__ == "__main__":
    try:
        patterns, captured, black_cap, white_cap = capture_graycode()
        pat = cv2.structured_light_GrayCodePattern.create(W, H)
        ok, proj_xy = pat.decode(captured, black_cap, white_cap)
        if not ok:
            raise RuntimeError("GrayCode decode failed")
        proj_x, proj_y = proj_xy
        print("Decoded maps shape:", proj_x.shape, proj_y.shape)
    except Exception as e:
        print("Error:", e)
        sys.exit(1)
