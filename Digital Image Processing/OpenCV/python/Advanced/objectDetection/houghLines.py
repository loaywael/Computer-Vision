import cv2
import numpy as np


gridImg = cv2.imread("../../gallery/grid.png", cv2.IMREAD_COLOR)
testImg = cv2.imread("../../gallery/grid1.jpg", cv2.IMREAD_COLOR)


def getLines(r, theta, length):
    xOrg = r * np.cos(theta)
    yOrg = r * np.sin(theta)
    x1 = int(xOrg - (length * np.sin(theta)))
    x2 = int(xOrg + (length * np.sin(theta)))
    y1 = int(yOrg + (length * np.cos(theta)))
    y2 = int(yOrg - (length * np.cos(theta)))
    return (x1, y1), (x2, y2)


def API(img):
    if img.shape[0] > 1000:
        img = cv2.resize(img, None, fx=0.2, fy=0.2)
    kSize, minThresh, maxThresh = 5, 50, 150
    bluredImg = cv2.GaussianBlur(img, (kSize, kSize), 0)
    gridEdges = cv2.Canny(bluredImg, minThresh, maxThresh, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLines(gridEdges, rho=1, theta=np.pi/180, threshold=200)
    if isinstance(lines, (tuple, list, np.ndarray)):
        lines = lines.squeeze()
        for r, theta in lines:
            pt1, pt2 = getLines(r, theta, 750)
            cv2.line(gridImg, pt1, pt2, (0, 255, 0), 1)

    cv2.imshow("im", img)
    cv2.imshow("edges", gridEdges)
    cv2.waitKey(0)
    print("exit clear")
    cv2.destroyAllWindows()


API(gridImg)
