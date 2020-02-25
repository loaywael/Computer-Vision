import cv2
import numpy as np


hourseImg = cv2.imread("../../gallery/hourse.jpeg", cv2.IMREAD_COLOR)
hourseImg = cv2.resize(hourseImg, None, fx=0.5, fy=0.5)

downImg = hourseImg.copy()
gaussianPyr = [downImg]
for i in range(1, 5):
    downImg = cv2.pyrDown(gaussianPyr[i-1])
    gaussianPyr.append(downImg)


laplacianPyr = [gaussianPyr[4]]
for i in range(4, 0, -1):
    size = (gaussianPyr[i-1].shape[1], gaussianPyr[i-1].shape[0])
    upImg = cv2.pyrUp(gaussianPyr[i], dstsize=size)
    lapImg = cv2.subtract(gaussianPyr[i-1], upImg)
    laplacianPyr.append(lapImg)


for i in range(5):
    cv2.imshow("laplacian"+str(i), laplacianPyr[i])
    cv2.imshow("gaussian-"+str(i), gaussianPyr[i])


cv2.waitKey(0)
cv2.destroyAllWindows()
