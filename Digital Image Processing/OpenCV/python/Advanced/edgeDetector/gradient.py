import cv2
import numpy as np
import matplotlib.pyplot as plt


hourseImg = cv2.imread("../../../gallery/logo.png", cv2.IMREAD_GRAYSCALE)

# xGrad = cv2.Sobel(hourseImg, cv2.CV_64F, 1, 0, ksize=3)
# yGrad = cv2.Sobel(hourseImg, cv2.CV_64F, 0, 1, ksize=3)

Grad = cv2.Sobel(hourseImg, cv2.CV_64F, 1, 1, ksize=3)
lapGrads = cv2.Laplacian(hourseImg, cv2.CV_64F, ksize=3)
edges = cv2.bitwise_or(Grad, lapGrads)
canny = cv2.Canny(hourseImg, 100, 200)
gradMorph = cv2.morphologyEx(hourseImg, cv2.MORPH_GRADIENT, np.ones((5, 5), dtype=np.uint8))

cv2.imshow("sobel & laplacian", edges)
cv2.waitKey(0)

cv2.imshow("canny", canny)
cv2.waitKey(0)

cv2.imshow("morph", gradMorph)
cv2.waitKey(0)

cv2.destroyAllWindows()
