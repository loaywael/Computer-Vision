import cv2
import numpy as np


img = cv2.imread("../../gallery/penny.jpg", cv2.IMREAD_COLOR)
img = cv2.resize(img, None, fx=0.25, fy=0.25)
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grayImg = cv2.medianBlur(grayImg, 5)
houghParams = {
    "image": grayImg,
    "method": cv2.HOUGH_GRADIENT,
    "dp": 1,
    "minDist": 75,
    "param1": 75,
    "param2": 50
}

circles = cv2.HoughCircles(**houghParams).squeeze()
for x, y, r in circles:
    cv2.circle(img, (x, y), r, (255, 0, 0), 3)
    cv2.circle(img, (x, y), 3, (255, 0, 0), 3)


cv2.imshow("im", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
