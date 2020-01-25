import cv2
import numpy as np


logoImg = cv2.imread("../../../gallery/tomato.jpg", cv2.IMREAD_GRAYSCALE)
# logoImg = cv2.imread("../../../gallery/logo.png", cv2.IMREAD_GRAYSCALE)

# cv2.addWeighted(hourseImg, 0.4, tomatoImg, 0.7, 0, outImg)
# cv2.imshow("logo", outImg)
# cv2.waitKey(0)
# cv2.add(hourseImg, tomatoImg, outImg)
# cv2.imshow("logo", outImg)
# cv2.waitKey(0)
# outImg = hourseImg + tomatoImg
# cv2.imshow("logo", outImg)
# cv2.waitKey(0)


_, binary = cv2.threshold(logoImg, 125, 255, cv2.THRESH_BINARY)
_, binaryInv = cv2.threshold(logoImg, 125, 255, cv2.THRESH_BINARY_INV)
_, trunc = cv2.threshold(logoImg, 125, 255, cv2.THRESH_TRUNC)
_, tozero = cv2.threshold(logoImg, 125, 255, cv2.THRESH_TOZERO)
_, tozeroInv = cv2.threshold(logoImg, 125, 255, cv2.THRESH_TOZERO_INV)
_, otsu = cv2.threshold(logoImg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


cv2.imshow("grayScale", logoImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("BINARY", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("BINARY-INV", binaryInv)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("TRUNC", trunc)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("TOZERO", tozero)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("TOZERO-INV", tozeroInv)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("OTSU", otsu)
cv2.waitKey(0)
cv2.destroyAllWindows()
