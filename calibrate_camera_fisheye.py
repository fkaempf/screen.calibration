import os
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from screen_calibration.calibration.detect_pose import detect_pose  # kept if you still want your custom overlay

# ------------------------------
# Read existing ChArUco
img = cv2.imread("ChArUco_Marker.png")
if img is None:
    print("Error: could not read image")
else:
    print("Image shape:", img.shape)
#cv2.imshow("Loaded Image", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
# ------------------------------

# ------------------------------
# REQUIREMENTS
ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 5
SQUARES_HORIZONTALLY = 7
SQUARE_LENGTH = 0.03
MARKER_LENGTH = 0.015
PATH_TO_YOUR_IMAGES = Path(os.getcwd()).joinpath('1800 U 501m NIR-07XC0')
# If you calibrated with the fisheye model, keep this True and use K,D from that calibration.
USE_FISHEYE = True
# ------------------------------

def calibrate_and_save_parameters():
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
        plt.show()

        
        if marker_ids is None or len(marker_ids) == 0:
            continue

        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board
        )
        if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) >= 4:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)

    # Calibrate with standard Charuco (pinhole) if you already have fisheye K,D saved skip this
    # Replace with your saved fisheye K,D if available.
    retval, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners, all_charuco_ids, board, img_size, None, None
    )

    np.save('K_cam_fish.npy', K)
    np.save('D_cam_fish.npy', D)

    # Iterate through displaying all the images
    for image_file in image_files:
        image = cv2.imread(image_file)
        if image is None:
            continue
        h, w = image.shape[:2]

        if USE_FISHEYE:
            # Rectify fisheye image to pinhole using K,D
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, np.eye(3), K, (w, h), cv2.CV_16SC2
            )
            undistorted = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
        else:
            undistorted = cv2.undistort(image, K, D)

        # Orientation arrow in the undistorted image
        uh, uw = undistorted.shape[:2]
        center = (uw // 2, uh // 2)
        arrow_len = int(min(uw, uh) * 0.2)
        end_point = (center[0], center[1] - arrow_len)  # upward
        cv2.arrowedLine(undistorted, center, end_point, (255, 0, 0), 3, tipLength=0.2)

        # Pose and 3D axes on undistorted view (detect on undistorted with zero distortion)
        gray_u = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        m_c, m_ids, _ = cv2.aruco.detectMarkers(gray_u, dictionary, parameters=params)
        if m_ids is not None and len(m_ids) > 0:
            _, ch_c, ch_ids = cv2.aruco.interpolateCornersCharuco(m_c, m_ids, gray_u, board)
            if ch_c is not None and ch_ids is not None and len(ch_ids) >= 4:
                ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    ch_c, ch_ids, board, K, None  # undistorted => set distortion to None/zeros
                )
                if ok:
                    # Draw axes with a 3 cm axis length
                    cv2.drawFrameAxes(undistorted, K, None, rvec, tvec, SQUARE_LENGTH)

        # Optional: your custom pose overlay on the original image
        pose_image = detect_pose(image, K, D)

        # Show scaled windows
        show = lambda im, title: (cv2.imshow(title, cv2.resize(im, None, fx=0.3, fy=0.3)), cv2.waitKey(0))
        show(image, 'Original Image')
        show(undistorted, 'Undistorted + Orientation + Axes')
        show(pose_image, 'Pose Image')

    cv2.destroyAllWindows()

calibrate_and_save_parameters()
