import cv2
import numpy as np

# -----------------------
# image color-compression
# -----------------------

pennyImg = cv2.imread("../../gallery/penny.jpg", cv2.IMREAD_COLOR)
pennyImg = cv2.resize(pennyImg, None, fx=0.2, fy=0.2)
Z = np.float32(pennyImg.reshape((-1, 3)))   # flatten into (m, colored-pixel)

# accuracy and iterations as a stopping indicator
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# random centroids initialization for K-mean
randCent = cv2.KMEANS_RANDOM_CENTERS
ret, labeledPixels, centroids = cv2.kmeans(Z, K=16, criteria=criteria,
                                           attempts=10, flags=randCent, bestLabels=None)

centroids = np.uint8(centroids)
quantizedPixels = centroids[labeledPixels.flatten()]
quantizedImg = quantizedPixels.reshape(pennyImg.shape)

cv2.imshow("segmented-img", quantizedImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
