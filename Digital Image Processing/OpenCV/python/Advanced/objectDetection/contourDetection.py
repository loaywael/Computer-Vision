import cv2
import numpy as np
import matplotlib.pyplot as plt

absPath = "/home/ezio/Rebos/AI-Machine-Learning-Curriculum/Computer Vision/OpenCV/python/Image Processing Tutorials/"
pennyImg = cv2.imread(absPath+"gallery/penny.jpg", cv2.IMREAD_COLOR)
pennyImg = cv2.resize(pennyImg, None, fx=0.5, fy=0.5)
grayPenny = cv2.cvtColor(pennyImg, cv2.COLOR_BGR2GRAY)
grayPenny = cv2.medianBlur(grayPenny, 9)
_, mask = cv2.threshold(grayPenny, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

# distTrans = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
# _, mask = cv2.threshold(distTrans, 0.7*distTrans.max(), 255, 0)

contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(pennyImg, contours, -1, (255, 0, 0), 3)


cv2.imshow("mask", pennyImg)
cv2.imshow("ranged-frame", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
