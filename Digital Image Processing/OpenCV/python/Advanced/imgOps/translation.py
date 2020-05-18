import cv2
import numpy as np


hourseImg = cv2.imread("../../gallery/hourse.jpeg", cv2.IMREAD_COLOR)
height, width = hourseImg.shape[:2]
transMatrix = np.float32([[1, 0, 100], [0, 1, 50]])
transImg = cv2.warpAffine(hourseImg, transMatrix, (width, height))

cv2.imshow("img", hourseImg)
cv2.imshow("transformed-img", transImg)

cv2.waitKey(0)
cv2.destroyAllWindows()
