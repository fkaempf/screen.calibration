from harvesters.core import Harvester
import numpy as np
import matplotlib.pyplot as plt

h = Harvester()
h.add_cti_file(r"C:\Program Files\Allied Vision\Vimba X\cti\VimbaUSBTL.cti")
h.update_device_info_list()
ia = h.create_image_acquirer(list_index=0)
nm = ia.remote_device.node_map

# basic setup
nm.PixelFormat.value   = "Mono8"
nm.ExposureAuto.value  = "Off"
nm.ExposureTime.value  = 10000.0
nm.GainAuto.value      = "Off"
nm.Gain.value          = 0.0
nm.TriggerMode.value   = "Off"
nm.TriggerSelector.value = "FrameStart"
nm.TriggerSource.value = "Software"
nm.TriggerMode.value   = "On"

ia.start_acquisition()
nm.TriggerSoftware.execute()

# Harvesters 1.4.x: fetch_buffer() returns a buffer; no 'with_payload' and no 'num_bytes_per_line'
buf = ia.fetch_buffer(timeout=1000)
try:
    comp = buf.payload.components[0]
    h_, w_ = int(comp.height), int(comp.width)
    # stride = bytes per row = total bytes / height
    mv = comp.data  # memoryview
    stride = len(mv) // h_
    a = np.frombuffer(mv, np.uint8, count=h_ * stride).reshape(h_, stride)[:, :w_].copy()
finally:
    buf.queue()  # return buffer
    ia.stop_acquisition()
    ia.destroy()
    h.reset()

print(a.shape, a.dtype)
plt.imshow(a)
plt.show()