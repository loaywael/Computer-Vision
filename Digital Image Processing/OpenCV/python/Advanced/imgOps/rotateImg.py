import cv2
import numpy as np


hourseImg = cv2.imread("../../gallery/hourse.jpeg", cv2.IMREAD_COLOR)
height, width, chn = hourseImg.shape
transMatrix = cv2.getRotationMatrix2D(center=(width/2, height/2), angle=0, scale=2)
transImg = cv2.warpAffine(hourseImg, transMatrix, (width, height))


cv2.imshow("img", hourseImg)
cv2.imshow("transformed-img", transImg)

cv2.waitKey(0)
cv2.destroyAllWindows()
