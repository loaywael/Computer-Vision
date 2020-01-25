import cv2
import numpy as np
import matplotlib.pyplot as plt


hourseImg = cv2.imread("../../../gallery/hourse.jpeg", cv2.IMREAD_COLOR)

for i, chn in enumerate(["r", "g", "b"]):
    hist = cv2.calcHist([hourseImg], channels=[i], mask=None, histSize=[255], ranges=[0, 256])
    plt.plot(hist, color=chn)

plt.xlim([0, 256])
plt.title("R-G-B pixels-histograms")
plt.xlabel("pixels")
plt.ylabel("frequency")
plt.show()
