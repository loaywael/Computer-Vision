import cv2
import numpy as np
import matplotlib.pyplot as plt


hourseImg = cv2.imread("../../../gallery/hourse.jpeg", cv2.IMREAD_COLOR)


def plotImgHisto(img):
    for i, ch in enumerate(("B", "G", "R")):
        histo = cv2.calcHist([img], [i], mask=None, histSize=[255], ranges=[0, 256])
        plt.plot(histo, color=ch)
    plt.xlim([0, 256])
    plt.title("R-G-B pixels-histograms")
    plt.xlabel("pixels")
    plt.ylabel("frequency")
    plt.show()


def equalizeColoredImg(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


plotImgHisto(hourseImg)
contrastedHourse = equalizeColoredImg(hourseImg)
plotImgHisto(contrastedHourse)

cv2.imshow("original", hourseImg)
cv2.waitKey(0)
cv2.imshow("original", contrastedHourse)
cv2.waitKey(0)
cv2.destroyAllWindows()
