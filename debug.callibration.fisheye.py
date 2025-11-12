import os
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from detect_pose import detect_pose
charuco_img = cv2.imread("ChArUco_Marker.png")


ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.03
MARKER_LENGTH = 0.015
PATH_TO_YOUR_IMAGES = Path(os.getcwd()).joinpath('1800 U 501m NIR-07XC0')

USE_FISHEYE = True

dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard(
    (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY),
    SQUARE_LENGTH, MARKER_LENGTH, dictionary
)
params = cv2.aruco.DetectorParameters()

image_files = sorted([os.path.join(PATH_TO_YOUR_IMAGES, f)
                        for f in os.listdir(PATH_TO_YOUR_IMAGES)
                        if f.lower().endswith((".jpg", ".png"))])

all_charuco_corners = []
all_charuco_ids = []
img_size = None

for image_file in image_files:
    image = cv2.imread(image_file)
    plt.imshow(image)
    if image is None:
        continue
    if img_size is None:
        h, w = image.shape[:2]
        img_size = (w, h)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    img_markers = image.copy()
    cv2.aruco.drawDetectedMarkers(img_markers, marker_corners, marker_ids)
    plt.imshow(cv2.cvtColor(img_markers, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Detected Aruco Markers')
    plt.show()

    
    if marker_ids is None or len(marker_ids) == 0:
        continue

    _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray, board
    )
    
    img_vis = image.copy()
    if marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(img_vis, marker_corners, marker_ids)
    if charuco_ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(img_vis, charuco_corners, charuco_ids)
    
    plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    plt.title('Interpolated Charuco Corners')
    plt.axis('off')
    plt.show()
    
    
    if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) >= 6    :
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)

subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC  + cv2.fisheye.CALIB_FIX_SKEW


objpoints, imgpoints = [], []
for corners, ids in zip(all_charuco_corners, all_charuco_ids):
    if ids is None or corners is None or len(ids) == 0:
        continue
    ids = ids.flatten().astype(int)

    # 3D points (board coords, Z=0) -> shape (N,1,3), CV_64FC3
    obj = board.getChessboardCorners()[ids, :].reshape(-1, 1, 3)
    obj = np.ascontiguousarray(obj, dtype=np.float64)

    # 2D points -> shape (N,1,2), CV_64FC2
    cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
    img = corners.reshape(-1, 1, 2)
    img = np.ascontiguousarray(img, dtype=np.float64)

    objpoints.append(obj)
    imgpoints.append(img)
    
    



K = np.eye(3, dtype=np.float64)
D = np.zeros((4, 1), dtype=np.float64) 

Knew = K.copy()
# Optionally shrink to reduce peripheral stretching:
# Knew[0,0] = Knew[1,1] = 0.8 * K[0,0]
und = cv2.fisheye.undistortImage(gray, K, D, Knew=Knew)
plt.imshow(und)



criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6)


rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
    objectPoints=objpoints,     # list of (N,1,3) CV_64FC3
    imagePoints=imgpoints,      # list of (N,1,2) CV_64FC2
    image_size=img_size,        # (w,h)
    K=K,
    D=D,
    criteria=criteria,
    flags=calibration_flags
)
Knew = K.copy()
# Optionally shrink to reduce peripheral stretching:
# Knew[0,0] = Knew[1,1] = 0.8 * K[0,0]
und = cv2.fisheye.undistortImage(gray, K, D, Knew=Knew)
plt.imshow(und)
plt.show()
plt.imshow(gray)
plt.show()