import cv2
import numpy as np
import pickle
import os


def updatePoints(event, x, y, flags, params):
    global points
    points, maxPoints = params
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if len(points) == maxPoints:
            points = []
        else:
            points.append([x, y])


def drawRoIPoly(img, points):
    for pt in points:
        cv2.circle(img, pt, 7, (0, 0, 255), -1)
    cv2.polylines(img, [np.array(points)], True, (255, 0, 0), 2)


def getSatThreshMask(hlsImg, minThresh, maxThresh=255):
    sChannel = hlsImg[:, :, 2]
    satMask = np.zeros_like(sChannel)
    satMask[(sChannel >= minThresh) & (sChannel <= maxThresh)] = 255
    return satMask


def getGRThreshMask(bgrImg, minThresh, maxThresh=255):
    g, r = bgrImg[:, :, 1], bgrImg[:, :, 2]
    grMask = np.zeros_like(r)
    grMask[((r >= minThresh) & (r <= maxThresh)) & (
        (g >= 1.125*minThresh) & (g <= maxThresh))] = 255
    return grMask


def getHueThreshMask(hlsImg, minThresh, maxThresh=255):
    hChannel = hlsImg[:, :, 0]
    hueMask = np.zeros_like(hChannel)
    hueMask[(hChannel <= minThresh) & (hChannel >= 0)] = 255
    return hueMask


def getSobelGrads(grayImg, axis="xy", K=9):
    if axis == "x":
        xGrads = cv2.Sobel(grayImg, cv2.CV_64F, 1, 0, ksize=K)
        absXSobel = np.absolute(xGrads)
        return absXSobel
    elif axis == "y":
        yGrads = cv2.Sobel(grayImg, cv2.CV_64F, 0, 1, ksize=K)
        absYSobel = np.absolute(yGrads)
        return absYSobel
    elif axis == "xy":
        xGrads = cv2.Sobel(grayImg, cv2.CV_64F, 1, 0, ksize=K)
        yGrads = cv2.Sobel(grayImg, cv2.CV_64F, 0, 1, ksize=K)
        absXSobel = np.absolute(xGrads)
        absYSobel = np.absolute(yGrads)
        return absXSobel, absYSobel


def getSobelMag(absGrads, minThresh, maxThresh):
    if isinstance(absGrads, (list, tuple, np.ndarray)) and len(absGrads) == 2:
        absXSobel, absYSobel = absGrads
        sobelMag = np.sqrt(np.square(absXSobel) + np.square(absYSobel))
        sobelMag = np.uint8(sobelMag * 255 / np.max(sobelMag))
        edgesMask = np.zeros_like(sobelMag)
        edgesMask[(sobelMag > minThresh) & (sobelMag < maxThresh)] = 255
        return edgesMask
    else:
        sobelMag = np.sqrt(np.square(absGrads[-1]))
        sobelMag = np.uint8(sobelMag * 255 / np.max(sobelMag))
        edgesMask = np.zeros_like(sobelMag)
        edgesMask[(sobelMag > minThresh) & (sobelMag < maxThresh)] = 255
        return edgesMask


def getSobelDir(absXSobel, absYSobel, minThresh, maxThresh):
    sobelMag = np.arctan2(absYSobel, absXSobel)
    edgesMask = np.zeros_like(sobelMag)
    edgesMask[(sobelMag >= minThresh) & (sobelMag <= maxThresh)] = 255
    return edgesMask


def getLaneMask(bgrFrame, minLineThresh, maxLineThresh,
                satThresh, hueThresh, redThresh, filterSize=5):
    bgrFrame = cv2.medianBlur(bgrFrame, filterSize)
    grayImg = cv2.cvtColor(bgrFrame, cv2.COLOR_BGR2GRAY)
    grayImg = cv2.GaussianBlur(grayImg, (filterSize, filterSize), 0)
    hlsImg = cv2.cvtColor(bgrFrame, cv2.COLOR_BGR2HLS)
    # edgesMask = cv2.Canny(grayImg, minLineThresh, maxLineThresh)
    xGrads = getSobelGrads(grayImg, "x", K=5)
    edgesMask = getSobelMag([xGrads], minLineThresh, maxLineThresh)
    satMask = getSatThreshMask(hlsImg, satThresh)
    hueMask = getHueThreshMask(hlsImg, hueThresh)
    grMask = getGRThreshMask(bgrFrame, redThresh)
    hsMask = cv2.bitwise_and(hueMask, satMask)
    lanesMask = np.zeros_like(grMask)
    lanesMask[(hsMask > 0) | (grMask > 0)] = 255
    outMask = cv2.bitwise_or(lanesMask, edgesMask)
    # cv2.imshow("0", outMask)
    return outMask


def warped2BirdPoly(bgrFrame, points, width, height):
    warpedPoints = np.float32(points)
    birdPoints = np.float32([
        [points[-1][0], points[0][1]],
        [points[2][0], points[0][1]],
        points[2], points[-1]
    ])
    Matrix = cv2.getPerspectiveTransform(warpedPoints, birdPoints)
    return cv2.warpPerspective(bgrFrame, Matrix, (width, height), flags=cv2.INTER_LINEAR)


def save2File(path, object):
    with open(path, "wb") as wf:
        pickle.dump(object, wf)
        print("roi is saved!")


def loadFile(path):
    with open(path, "rb") as rf:
        return pickle.load(rf)
