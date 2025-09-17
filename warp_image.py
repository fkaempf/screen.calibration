import cv2, numpy as np

# load your projector->camera map
mapx = np.load("mapx.npy").astype(np.float32)
mapy = np.load("mapy.npy").astype(np.float32)
proj_h, proj_w = mapx.shape

# load the image
img = cv2.imread("fly.jpg")               # BGR
cam_h, cam_w = int(mapy.max()+1), int(mapx.max()+1)  # or proj_x.shape
img_cam = cv2.resize(img, (cam_w, cam_h), interpolation=cv2.INTER_AREA)

# warp to projector space
proj_frame = cv2.remap(img_cam, mapx, mapy, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)

cv2.imwrite("out/warped_fly_mapxy.png", proj_frame)