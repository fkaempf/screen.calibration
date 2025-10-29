# Alvium camera backend via Harvesters/GenTL — mirrors CamRotPy API
# Requirements: pip install harvesters numpy ; install Allied Vision Vimba X (GenTL .cti)
import os
import numpy as np
import matplotlib.pyplot as plt

class CamAlvium:
    def __init__(self, exposure_ms=10.0, gain_db=6, cti_path=None):
        """
        exposure_ms: desired exposure in milliseconds (mapped to ExposureTime [µs])
        gain_db: analog gain in dB
        cti_path: explicit path to a GenTL .cti (e.g., VimbaUSBTL.cti). If None, tries common defaults/env.
        """
        
        self.CTI_PATH   = os.environ.get("CAM_CTI_PATH", r"C:\Program Files\Allied Vision\Vimba X\cti\VimbaUSBTL.cti")
        self.SERIAL     = os.environ.get("ALVIUM_SERIAL", None)  # not used; we open list_index=0 to match your working script
        self.EXPOSURE_MS = exposure_ms
        self.GAIN_DB     = gain_db
        self.TIMEOUT_S   = 2.0
        
    def start(self):
        # 1) Initialize Harvester
        from harvesters.core import Harvester
        self.h = Harvester()
        self.h.add_file(self.CTI_PATH)   
        self.h.update()
        
    # 2) Create ImageAcquirer
        self.ia = self.h.create()
        self.nm = self.ia.remote_device.node_map
        
    # 3) Node configuration – exact sequence
        self.nm.PixelFormat.value     = "Mono8"
        self.nm.ExposureAuto.value    = "Off"
        self.nm.GainAuto.value        = "Off"
        self.nm.TriggerMode.value     = "Off"
        self.nm.TriggerSelector.value = "FrameStart"
        self.nm.TriggerSource.value   = "Software"
        self.nm.TriggerMode.value     = "On"

        self.nm.ExposureTime.value    = float(self.EXPOSURE_MS) * 1000.0   # ms → µs
        self.nm.Gain.value            = float(self.GAIN_DB)
        
    # 4) Start acquisition
        self.ia.start()

    def grab_frame_and_show(self):
        self.nm.TriggerSoftware.execute()
        buf = self.ia.fetch(timeout=int(max(1, self.TIMEOUT_S*1000)))
        comp = buf.payload.components[0]
        h_, w_ = int(comp.height), int(comp.width)
        mv = comp.data
        stride = len(mv) // h_
        a = np.frombuffer(mv, np.uint8, count=h_ * stride).reshape(h_, stride)[:, :w_].copy()
        buf.queue()
        plt.imshow(a, cmap='gray')
        plt.show()
        
    def grab(self):
        self.nm.TriggerSoftware.execute()
        buf = self.ia.fetch(timeout=int(max(1, self.TIMEOUT_S*1000)))
        comp = buf.payload.components[0]
        h_, w_ = int(comp.height), int(comp.width)
        mv = comp.data
        stride = len(mv) // h_
        a = np.frombuffer(mv, np.uint8, count=h_ * stride).reshape(h_, stride)[:, :w_].copy()
        buf.queue()
        return np.ascontiguousarray(a)
    def stop(self):
        try:
            self.ia.stop()
        except Exception:
            pass
        try:
            self.ia.destroy()
        except Exception:
            pass
        try:
            self.h.reset()
        except Exception:
            pass
if __name__ == "__main__":   
    a = CamAlvium()
    a.start()
    lol = list()
    for i in range(100):
        print(i)
        lol.append(a.grab())
        
    a.stop()
    plt.imshow(np.mean(np.array(lol),axis=0), cmap='viridis')
    plt.show()