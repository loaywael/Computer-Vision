import cv2
import numpy as np
from glob import glob


pth = "/home/ezio/PycharmProjects/python/playground/CV/OpenCV/bTut"
img = cv2.imread(*glob(pth+"/gallery/tomato1*"))
cimg = img.copy()
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bluredImg = cv2.GaussianBlur(grayImg, (3, 3), 0)
threshImg = cv2.adaptiveThreshold(
    bluredImg, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 205, 1
)
contours = cv2.findContours(threshImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
cv2.rectangle(img, pt1=(40, 40), pt2=(100, 100), color=(255, 0, 0), thickness=-1)
img = cv2.addWeighted(cimg, 0.5, img, 1-0.7, 0)
print(len(contours))
# kernel = np.ones((7, 7), "uint8")
# hsvImg = cv2.cvtColor(bluredImg, cv2.COLOR_BGR2HSV)
# imgEdges = cv2.Canny(img, 15, 115)
# imgEdges = cv2.erode(imgEdges, kernel=None, iterations=1)
# mixGril = cv2.bitwise_and(hsvImg[:, :, 1], imgEdges)

cv2.imshow("Tomato Image", img)
# cv2.imshow("Blured Image", bluredImg)
# cv2.imshow("Mix Tomato Image", mixGril)
# cv2.imshow("Tomato Edges", imgEdges)
# cv2.imshow("HSV Tomato Image", hsvImg)

cv2.waitKey(0)
cv2.destroyAllWindows()
