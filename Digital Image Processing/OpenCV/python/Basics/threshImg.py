import cv2
import numpy as np
from glob import glob


threshold = 85
img = cv2.imread(*glob("./gallery/face1*"))
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgHeight, imgWidth = grayImg.shape[:2]
binImg = np.zeros((imgHeight, imgWidth), dtype="uint8")

# for row in range(0, imgHeight):
#     for col in range(0, imgWidth):
#         if grayImg[row, col] > threshold:
#             binImg[row, col] = 255

ret, binImg = cv2.threshold(grayImg, threshold, 255, cv2.THRESH_BINARY)
adaBinImg = cv2.adaptiveThreshold(grayImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
cv2.namedWindow("Gray Image")
cv2.namedWindow("Adaptive Threshold Image")
cv2.namedWindow("Binary Image")

cv2.imshow("Gray Image", grayImg)
cv2.imshow("Adaptive Threshold Image", adaBinImg)
cv2.imshow("Binary Image", binImg)
ch = cv2.waitKey(0)

cv2.destroyAllWindows()
