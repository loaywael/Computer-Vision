import cv2
import numpy as np

logoImg = cv2.imread("../../../gallery/tomato.jpg", cv2.IMREAD_GRAYSCALE)
# logoImg = cv2.imread("../../../gallery/logo.png", cv2.IMREAD_GRAYSCALE)

bluredLogo = cv2.medianBlur(logoImg, 5)
meanThresh = cv2.adaptiveThreshold(
    bluredLogo, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY, 75, 3
)

gausThresh = cv2.adaptiveThreshold(
    bluredLogo, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 3
)


cv2.imshow("grayScale", logoImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("BINARY", meanThresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("BINARY-INV", gausThresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
