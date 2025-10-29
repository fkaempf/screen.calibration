from vimba import *
with Vimba.get_instance () as vimba:
    cams = vimba.get_all_cameras ()
    with cams [0] as cam:
        # Aquire single frame synchronously
        frame = cam.get_frame ()

        # Aquire 10 frames synchronously
        for frame in cam.get_frame_generator(limit =10):
            pass