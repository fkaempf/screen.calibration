import cv2
import numpy as np
import glob
import scipy.io as sio

# --- load MATLAB params ---
data = sio.loadmat('fisheye_to_opencv.mat')
K = np.asarray(data['K'], dtype=np.float64)
D = np.asarray(data['D'], dtype=np.float64).reshape(4, 1)  # (4,1) float64




def undistort(path, balance=1.0, alpha=0.0):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]

    Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=balance, fov_scale=1.0
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), Knew, (w, h), cv2.CV_16SC2
    )
    und = cv2.remap(img, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    scale = 0.25  # 1/4 size
    img_small = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    und_small = cv2.resize(und, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    side = cv2.hconcat([img_small, und_small])
    cv2.imshow('original | undistorted', side)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# balance↑ → farther zoomed out (more FOV, more black borders)


if __name__ == '__main__':
    images = glob.glob('checkerboard_samples/*.png')
    undistort(images[69])
