from screen_calibration.camera.CamAlvium import CamAlvium
import os
import cv2
import numpy as np
from datetime import datetime
import keyboard  # pip install keyboard
import time

def sample_checkerboard_images(cam, save_dir="checkerboard_samples"):
    """
    Start capturing checkerboard images on first SPACE press and stop on next SPACE press.
    Saves images as timestamped PNGs in the given folder.
    """
    os.makedirs(save_dir, exist_ok=True)
    print("Press SPACE to start recording checkerboard samples...")
    keyboard.wait("space")
    print("Recording started. Press SPACE again to stop.")

    counter = 0
    time.sleep(2)
    while not keyboard.is_pressed("space"):
        frame = cam.grab()
        fname = f"{counter}.png"
        path = os.path.join(save_dir, fname)
        cv2.imwrite(path, frame)    
        print(f" {counter} Saved {path}")
        time.sleep(0.1) 
        counter += 1

    print(f"Recording stopped. {counter} images saved in '{save_dir}'.")


if __name__ == "__main__":
     cam = CamAlvium(exposure_ms=10.0, gain_db=6)
     cam.start()
     sample_checkerboard_images(cam)
     cam.stop()

