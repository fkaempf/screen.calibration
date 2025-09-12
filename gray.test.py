import cv2
W = 1280
H = 720

pat = cv2.structured_light_GrayCodePattern.create(W, H)
ok, patterns = pat.generate()
# Create a window on the projector screen
cv2.namedWindow("proj", cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow("proj", -1920, 0)  # adjust so it goes to projector display
cv2.setWindowProperty("proj", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

for i, p in enumerate(patterns):
    cv2.imshow("proj", p)
    cv2.waitKey(500)  # wait a bit so the camera can capture

cv2.destroyAllWindows()