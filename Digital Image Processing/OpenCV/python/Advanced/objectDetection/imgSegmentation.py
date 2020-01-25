import cv2
import numpy as np
import matplotlib.pyplot as plt


pennyImg = cv2.imread("../../../gallery/penny.jpg", cv2.IMREAD_COLOR)
pennyImg = cv2.resize(pennyImg, None, fx=0.2, fy=0.2)
grayPenny = cv2.cvtColor(pennyImg, cv2.COLOR_BGR2GRAY)

grayPenny = cv2.medianBlur(grayPenny, 15)
_, mask = cv2.threshold(grayPenny, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# mask = cv2.morphologyEx(grayPenny, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
distTrans = cv2.distanceTransform(mask, cv2.DIST_L2, maskSize=5)

# contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(pennyImg, contours, -1, (255, 0, 0), 3)
# cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
cv2.imshow("mask", distTrans)
# cv2.imshow("outImg", pennyImg)
# cv2.imshow("ranged-frame", grayPenny)
cv2.waitKey(0)
cv2.destroyAllWindows()
