import cv2
import numpy as np
import matplotlib.pyplot as plt


pennyImg = cv2.imread("../../gallery/penny.jpg", cv2.IMREAD_COLOR)
pennyImg = cv2.resize(pennyImg, None, fx=0.2, fy=0.2)
grayPenny = cv2.cvtColor(pennyImg, cv2.COLOR_BGR2GRAY)

kernel = np.ones((3, 3), np.uint8)
# reduce background noise unnecessary details
grayPenny = cv2.medianBlur(grayPenny, 15)
# binary segmentation identify regions
_, mask = cv2.threshold(grayPenny, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# remove any white noise due to thresholding
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
# close the gaps in the objects shape "missing pixels"
closing = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
# increasing space between objects to nearly partion them
erosion = cv2.erode(closing, kernel, iterations=2)
#
distTrans = np.uint8(cv2.distanceTransform(erosion, cv2.DIST_L2, 5))
_, distTrans = cv2.threshold(distTrans, distTrans.max()*0.5, 255, cv2.THRESH_BINARY)

pennyFG = distTrans
pennyBKG = cv2.dilate(opening, kernel=kernel, iterations=3)
unknownPenny = cv2.subtract(pennyBKG, pennyFG)
ret, markers = cv2.connectedComponents(pennyFG)
markers += 1        # let background be 1 not zero and shift other objects
markers[unknownPenny == 255] = 0
markers = cv2.watershed(pennyImg, markers)
pennyImg[markers == -1] = [255, 0, 0]
# contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(pennyImg, contours, -1, (255, 0, 0), 3)
# cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
fullOps = np.hstack([mask, closing, erosion, distTrans])
cv2.imshow("operations", fullOps)
cv2.imshow("img", pennyImg)
cv2.imshow("img0", erosion)

cv2.waitKey(0)
cv2.destroyAllWindows()
