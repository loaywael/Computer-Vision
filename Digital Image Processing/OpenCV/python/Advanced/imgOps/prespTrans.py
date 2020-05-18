import cv2
import numpy as np


hourseImg = cv2.imread("../../gallery/grid.png", cv2.IMREAD_COLOR)
height, width = hourseImg.shape[:2]
src = np.float32([[50, 50], [50, 380], [400, 380]])
dst = np.float32([[50, 50], [50, 380], [300, 380]])

cv2.circle(hourseImg, (50, 50), 5, (255, 0, 0), -1)
cv2.circle(hourseImg, (50, 380), 5, (255, 0, 0), -1)
cv2.circle(hourseImg, (400, 380), 5, (255, 0, 0), -1)


transMatrix = cv2.getAffineTransform(src, dst)
transImg = cv2.warpAffine(hourseImg, transMatrix, (width, height))

cv2.imshow("img", hourseImg)
cv2.imshow("transformed-img", transImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
