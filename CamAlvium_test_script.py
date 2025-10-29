# test_cam_alvium.py
# Purpose: Verify Alvium acquisition using the exact API calls that work in your environment (Harvesters 1.4.x + VimbaUSBTL).
# Uses: add_cti_file • update_device_info_list • create_image_acquirer(list_index=0) • start_acquisition • fetch_buffer.
# Prints PASS/FAIL per step and exits non-zero on failure.

import os, sys, time, traceback
import numpy as np
import matplotlib.pyplot as plt

# --- User parameters -----------------------------------------------------------
CTI_PATH   = os.environ.get("CAM_CTI_PATH", r"C:\Program Files\Allied Vision\Vimba X\cti\VimbaUSBTL.cti")
SERIAL     = os.environ.get("ALVIUM_SERIAL", None)  # not used; we open list_index=0 to match your working script
EXPOSURE_MS = 10.0   # 10 ms
GAIN_DB     = 6.0    # 6 dB
TIMEOUT_S   = 2.0
# ------------------------------------------------------------------------------


def _fail(msg, exc=None):
    print(f"[FAIL] {msg}")
    if exc:
        print("       " + "".join(traceback.format_exception_only(type(exc), exc)).strip())
    sys.exit(1)

def _pass(msg):
    print(f"[PASS] {msg}")

def main():
    # Import here so missing deps fail cleanly
    try:
        from harvesters.core import Harvester
    except Exception as e:
        _fail("Import harvesters failed", e)

    # 0) CTI
    if not os.path.isfile(CTI_PATH):
        _fail(f"CTI not found at: {CTI_PATH}")
    _pass(f"CTI located: {CTI_PATH}")

    # 1) Initialize Harvester
    h = Harvester()
    try:
        h.add_file(CTI_PATH)                # legacy call that works in your env
        h.update()             # legacy call that works in your env
        if not h.device_info_list:
            _fail("No devices found via GenTL")
        _pass("Harvester initialized and device list populated")

        # 2) Create ImageAcquirer exactly like your working snippet
        ia = h.create()
        nm = ia.remote_device.node_map

        # 3) Node configuration – exact sequence
        nm.PixelFormat.value     = "Mono8"
        nm.ExposureAuto.value    = "Off"
        nm.GainAuto.value        = "Off"
        nm.TriggerMode.value     = "Off"
        nm.TriggerSelector.value = "FrameStart"
        nm.TriggerSource.value   = "Software"
        nm.TriggerMode.value     = "On"

        nm.ExposureTime.value    = float(EXPOSURE_MS) * 1000.0   # ms → µs
        nm.Gain.value            = float(GAIN_DB)

        # 4) Start acquisition
        ia.start()
        _pass("Acquisition started")


        # 6) Single grab (software trigger → fetch_buffer)
        nm.TriggerSoftware.execute()
        buf = ia.fetch(timeout=int(max(1, TIMEOUT_S*1000)))
        try:
            comp = buf.payload.components[0]
            h_, w_ = int(comp.height), int(comp.width)
            mv = comp.data
            stride = len(mv) // h_
            a = np.frombuffer(mv, np.uint8, count=h_ * stride).reshape(h_, stride)[:, :w_].copy()
            plt.imshow(a)
            plt.show()
        finally:
            buf.queue()

        if not isinstance(a, np.ndarray): _fail("grab() did not return a numpy array")
        if a.dtype != np.uint8: _fail(f"dtype is {a.dtype}, expected uint8")
        if a.ndim != 2 or min(a.shape) <= 0: _fail(f"frame shape invalid: {a.shape}")
        _pass(f"First frame OK: shape={a.shape}, dtype={a.dtype}")

        # 7) Basic stats
        mn, mx, mean, std = int(a.min()), int(a.max()), float(a.mean()), float(a.std())
        print(f"[INFO] stats: min={mn} max={mx} mean={mean:.2f} std={std:.2f}")

        # 8) Multiple grabs to ensure stability
        n = 3
        for i in range(n):
            nm.TriggerSoftware.execute()
            buf = ia.fetch(timeout=int(max(1, TIMEOUT_S*1000)))
            try:
                comp = buf.payload.components[0]
                mv = comp.data
                stride = len(mv) // h_
                f = np.frombuffer(mv, np.uint8, count=h_ * stride).reshape(h_, stride)[:, :w_]
            finally:
                buf.queue()
            if f.shape != a.shape or f.dtype != np.uint8:
                _fail(f"Frame {i} mismatch: shape={f.shape}, dtype={f.dtype}")
        _pass(f"{n} additional grabs stable")

    except Exception as e:
        _fail("Test raised", e)
    finally:
        # 9) Shutdown
        try:
            ia.stop()
        except Exception:
            pass
        try:
            ia.destroy()
        except Exception:
            pass
        try:
            h.reset()
        except Exception:
            pass

    print("\nALL CHECKS PASSED")

if __name__ == "__main__":
    main()
