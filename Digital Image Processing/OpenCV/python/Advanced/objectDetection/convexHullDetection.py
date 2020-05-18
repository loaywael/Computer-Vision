import cv2
import numpy as np
from decimal import Decimal


img = cv2.imread("../../gallery/rhand.jpg", cv2.IMREAD_COLOR)
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grayImg = cv2.GaussianBlur(grayImg, (7, 7), 0)
# mask = cv2.adaptiveThreshold(grayImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                              cv2.THRESH_BINARY, 3, 0)
kernel = np.ones((5, 5), "uint8")

mask = cv2.morphologyEx(grayImg, cv2.MORPH_GRADIENT, kernel)
# mask = cv2.erode(mask, kernel)
# mask = cv2.Canny(grayImg, 55, 115, L2gradient=True)
contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
convexContours = cv2.convexHull(contours[0], returnPoints=True)
cv2.polylines(img, [convexContours], True, (0, 0, 255), 9)
cv2.drawContours(img, convexContours, -1, (255, 0, 0), 3)

# returns indixes of the convex points
# concaveContours = cv2.convexHull(contours[0], returnPoints=False
# concaveContours = [i[0] for i in concaveContours]
# concaveContours = np.array([contours[0][i] for i in concaveContours])
x, y, w, h = cv2.boundingRect(contours[0])
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

rect = cv2.minAreaRect(contours[0])
box = np.int0(cv2.boxPoints(rect))
cv2.drawContours(img, [box], -1, (255, 0, 0), 3)

(cx, cy), r = cv2.minEnclosingCircle(contours[0])
cv2.circle(img, (int(cx), int(cy)), int(r), (255, 255, 0), 5)

ellips = cv2.fitEllipse(contours[0])
cv2.ellipse(img, ellips, (0, 255, 255), 5)

[vx, vy, x, y] = cv2.fitLine(contours[0], cv2.DIST_L2, 0, 0.01, 0.01)
leftY = int((-x*vy/vx) + y)
rightY = int((img.shape[1] - x) * vy/vx + y)
cv2.line(img, (img.shape[1]-1, rightY), (0, leftY), (128, 0, 128), 5)

cnt = contours[0]
leftMost = cnt[cnt[:, :, 0].argmin()].ravel()
rightMost = cnt[cnt[:, :, 0].argmax()].ravel()
topMost = cnt[cnt[:, :, 1].argmin()].ravel()
bottomMost = cnt[cnt[:, :, 1].argmax()].ravel()

cv2.circle(img, tuple(leftMost), 9, (0, 0, 0), -1)
cv2.circle(img, tuple(rightMost), 9, (0, 0, 0), -1)
cv2.circle(img, tuple(topMost), 9, (0, 0, 0), -1)
cv2.circle(img, tuple(bottomMost), 9, (0, 0, 0), -1)


print(cv2.isContourConvex(contours[0]))
cv2.namedWindow("img")
cv2.moveWindow("img", 0, 0)
cv2.imshow("img", img)
cv2.imshow("mask", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
