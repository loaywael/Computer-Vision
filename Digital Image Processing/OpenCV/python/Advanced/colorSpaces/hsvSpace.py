import cv2
import numpy as np


hourseImg = cv2.imread("../../../gallery/hourse.jpeg", cv2.IMREAD_COLOR)  # bluish img
hsvHourseImg = cv2.cvtColor(hourseImg, cv2.COLOR_BGR2HSV)
hslHourseImg = cv2.cvtColor(hourseImg, cv2.COLOR_BGR2HLS)

cv2.imshow("hsv-space", hsvHourseImg)
cv2.waitKey(0)
cv2.imshow("hsl-space", hslHourseImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
