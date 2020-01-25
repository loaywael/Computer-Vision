import cv2
import numpy as np


bookImg = cv2.imread("../../../gallery/bookpage.jpg", cv2.IMREAD_GRAYSCALE)
# bluredImg = cv2.GaussianBlur(bookImg, (5, 5), 0)
threshTxt = cv2.adaptiveThreshold(
    bookImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 75, 2)


cv2.imshow("txt", bookImg)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imshow("txt", threshTxt)
cv2.waitKey(0)
cv2.destroyAllWindows()
