import os
import numpy as np
import cv2
from pathlib import Path

# ------------------------------
# Read existing ChArUco
img = cv2.imread("ChArUco_Marker.png")   # Replace with your path

if img is None:
    print("Error: could not read image")
else:
    print("Image shape:", img.shape)  # (height, width, channels)


cv2.imshow("Loaded Image", img)
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()


# ------------------------------
# ENTER YOUR REQUIREMENTS HERE:
ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.03
MARKER_LENGTH = 0.015
# ...
PATH_TO_YOUR_IMAGES = Path(os.getcwd()).joinpath('GSR-3U3-4IC-6NIR-C')
# ------------------------------

def calibrate_and_save_parameters():
    # Define the aruco dictionary and charuco board
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()

    # Load PNG images from folder
    image_files = [os.path.join(PATH_TO_YOUR_IMAGES, f) for f in os.listdir(PATH_TO_YOUR_IMAGES) if f.endswith(".jpg") or f.endswith(".png")]
    image_files.sort()  # Ensure files are in order

    all_charuco_corners = []
    all_charuco_ids = []

    for image_file in image_files:
        image = cv2.imread(image_file)
        image_copy = image.copy()
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=params)
        
        # If at least one marker is detected
        if len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, board)
            if charuco_retval:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)

    # Calibrate camera
    all_obj = board.getChessboardCorners()
    objpoints = [all_obj[ids.flatten()].reshape(-1,1,3).astype(np.float32) for ids in all_charuco_ids]
    imgpoints = [corners.reshape(-1,1,2).astype(np.float32) for corners in all_charuco_corners]

    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board, image.shape[:2], None, None)
    #rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.fisheye.calibrate(objpoints, imgpoints, image.shape[1::-1], np.eye(3), np.zeros((4,1)), None, None, cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC|cv2.fisheye.CALIB_CHECK_COND|cv2.fisheye.CALIB_FIX_SKEW, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))


    # Save calibration data
    np.save('K_cam.npy', camera_matrix)
    np.save('D_cam.npy', dist_coeffs)

    # Iterate through displaying all the images
    for image_file in image_files:
        image = cv2.imread(image_file)
        undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
        #map1,map2=cv2.fisheye.initUndistortRectifyMap(camera_matrix,dist_coeffs,np.eye(3),camera_matrix,(image.shape[1],image.shape[0]),cv2.CV_16SC2); undistorted_image=cv2.remap(image,map1,map2,cv2.INTER_LINEAR)

        cv2.imshow('Undistorted Image', undistorted_image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

calibrate_and_save_parameters()