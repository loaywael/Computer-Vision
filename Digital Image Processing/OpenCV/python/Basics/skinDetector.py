import cv2
import numpy as np
from glob import glob
import time

threshold = 115
img = cv2.imread(*glob("./gallery/face1*"))
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsvImg)


adaBinImg = cv2.adaptiveThreshold(grayImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
_, h = cv2.threshold(h, 115, 255, cv2.THRESH_BINARY_INV)
_, s = cv2.threshold(s, 15, 255, cv2.THRESH_BINARY)
maskedImg = cv2.bitwise_and(cv2.bitwise_and(h, s), adaBinImg)
contours, hierarchy = cv2.findContours(maskedImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contImg = img.copy()
cv2.drawContours(contImg, contours, -1, (0, 255, 0), 3)
cv2.imshow("Masked Image", contImg)
cv2.imshow("Gray Image", grayImg)
cv2.imshow("HSV Image", hsvImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
