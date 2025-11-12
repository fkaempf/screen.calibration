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

