import cv2
import numpy as np
from glob import glob


img = cv2.imread(*glob("./gallery/face0*"))
height, width, channels = img.shape
b, g, r = cv2.split(img)    # splits each color channel matrix
# empty array of triple width of img source
bgr_split = np.empty((height, 3*width, 3), dtype="uint8")
# cloning blue channel three times and pasted in the first third
bgr_split[:, 0:width] = cv2.merge([b, b, b])  # pastes a shape of (1994, 2900, 3)
# cloning green channel three times and pasted in the first third
bgr_split[:, width: width*2] = cv2.merge([g, g, g])   # pastes a shape of (1994, 2900, 3)
# cloning red channel three times and pasted in the first third
bgr_split[:, width*2:] = cv2.merge([r, r, r])   # pastes a shape of (1994, 2900, 3)

hsv_split = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)    # converting BGR to HSV color
# concatenate splitted channels side by side
hsv_split = np.concatenate([*cv2.split(hsv_split)], axis=1)

# creates image container window to control later
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("BGR", cv2.WINDOW_NORMAL)
cv2.namedWindow("HSV", cv2.WINDOW_NORMAL)
# moves the window contour to certain locations
cv2.moveWindow("Image", 1000, 100)
cv2.moveWindow("BGR", 0, 100)
cv2.moveWindow("HSV", 500, 500)
# display the image
cv2.imshow("Image", img)
cv2.imshow("BGR", bgr_split)
cv2.imshow("HSV", hsv_split)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("testImg.png", img)

