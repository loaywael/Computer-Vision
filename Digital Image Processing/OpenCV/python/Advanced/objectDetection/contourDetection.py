import cv2
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal


pennyImg = cv2.imread("../../gallery/square.png", cv2.IMREAD_COLOR)
squareImg = cv2.imread("../../gallery/square.png", cv2.IMREAD_COLOR)

img = squareImg
# img = cv2.resize(img, None, fx=3, fy=3)
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grayImg = cv2.medianBlur(grayImg, 5)
# _, mask = cv2.threshold(grayPenny, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
edges = cv2.Canny(grayImg, 11, 30)
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

# distTrans = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
# _, mask = cv2.threshold(distTrans, 0.7*distTrans.max(), 255, 0)

# chain simple fetches the main 2 points to describe each line that belongs to a contour shape
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
fontType = cv2.FONT_HERSHEY_SIMPLEX

print(len(contours))
for cnt in contours:
    M = cv2.moments(cnt)
    print(M)
    # center: cx = M01/M00, cy = M10/M00
    area = cv2.contourArea(cnt)
    perimeter = Decimal(cv2.arcLength(cnt, closed=True), 3)
    cv2.putText(img, "area: "+str(area), (75, 200), fontType, 0.3, 255, 1)
    cv2.putText(img, "perimeter: "+str(perimeter), (75, 215), fontType, 0.3, 255, 1)
    center = (int(M["m01"] / M["m00"]), int(M["m10"] / M["m00"]))
    cv2.circle(img, center, 11, (0, 255, 0), -1)
    for c in cnt:
        print(c)
        cv2.circle(img, tuple(c.ravel()), 11, (0, 0, 255), -1)
cv2.drawContours(img, contours, -1, (255, 0, 0), 3)
cv2.namedWindow("outImg")
cv2.moveWindow("outImg", 0, 0)
cv2.imshow("mask", edges)
cv2.imshow("outImg", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
