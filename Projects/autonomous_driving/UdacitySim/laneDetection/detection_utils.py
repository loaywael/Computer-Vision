import cv2
import numpy as np
import pickle
import os


def updatePoints(event, x, y, flags, params):
    """
    Receives mouse clicking locations on screen to append them to a list to draw a polygon
    
    @param event: mouse event object from OpenCV
    @param x: x-location of the current point
    @param y: y-location of the current point
    @param flags: any given flag to the function
    @param params: additional parameters to be passed to the function
    """
    global points
    points, maxPoints = params
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if len(points) == maxPoints:
            points = []
        else:
            points.append([x, y])


def drawRoIPoly(img, points):
    """
    Draws the Lane ROI polygon that visualize the points to be warped
    
    @param img: (np.ndarray) BGR image
    @points: list of points of the Lane ROI
    """
    for pt in points:
        cv2.circle(img, pt, 7, (0, 0, 255), -1)
    cv2.polylines(img, [np.array(points)], True, (255, 0, 0), 2)


def getSatThreshMask(hlsImg, minThresh, maxThresh=255):
    """
    Creates a binary mask of the lane for the saturation channel (color thresholding)
    
    @param hlsImg: HLS image formate np.ndarray
    @param minThresh: the minimum theshold anything below is black
    @param maxThresh: the maximum theshold anything above is white
    """
    sChannel = hlsImg[:, :, 2]
    satMask = np.zeros_like(sChannel)
    satMask[(sChannel >= minThresh) & (sChannel <= maxThresh)] = 255
    return satMask


def getGRThreshMask(bgrImg, minThresh, maxThresh=255):
    """
    Creates a binary mask of the lane for the green and the red channel
    
    @param bgrImg: (np.ndarray) BGR image
    @param minThresh: the minimum theshold anything below is black
    @param maxThresh: the maximum theshold anything above is white
    """
    g, r = bgrImg[:, :, 1], bgrImg[:, :, 2]
    grMask = np.zeros_like(r)
    grMask[((r >= minThresh) & (r <= maxThresh)) & (
        (g >= 1.125*minThresh) & (g <= maxThresh))] = 255
    return grMask


def getHueThreshMask(hlsImg, minThresh, maxThresh=255):
    """
    Creates a binary mask of the lane for the hue channel (color thresholding)
    
    @param hlsImg: HLS image formate np.ndarray
    @param minThresh: the minimum theshold anything below is black
    @param maxThresh: the maximum theshold anything above is white
    """
    hChannel = hlsImg[:, :, 0]
    hueMask = np.zeros_like(hChannel)
    hueMask[(hChannel <= minThresh) & (hChannel >= 0)] = 255
    return hueMask


def getSobelGrads(grayImg, axis="xy", K=9):
    """
    Creates a binary mask of the absolute edges Sobel gradients from a grayscale image
    
    @param grayImg: Gray image formate np.ndarray
    @param axis: the axis to extract gradients over
    @param K: Sobel kernel size
    """
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
    """"
    Applies gradient magnitude for the absGrads and returns binary edges mask
    
    @param absGrads: np.ndarray of the absolute gradients binary mask
    @param minThresh: the minimum theshold anything below is black
    @param maxThresh: the maximum theshold anything above is white
    """
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
    """
    Finds gradients direction of a given absolute gradients of an image
    
    @param absXGrads: np.ndarray of the absolute gradients binary mask of X-axis
    @param absYGrads: np.ndarray of the absolute gradients binary mask of Y-axis
    @param minThresh: the minimum theshold anything below is black
    @param maxThresh: the maximum theshold anything above is white
    """
    sobelMag = np.arctan2(absYSobel, absXSobel)
    edgesMask = np.zeros_like(sobelMag)
    edgesMask[(sobelMag >= minThresh) & (sobelMag <= maxThresh)] = 255
    return edgesMask


def getLaneMask(bgrFrame, minLineThresh, maxLineThresh,
                satThresh, hueThresh, redThresh, filterSize=5):
    """
    Applies color bluring, thresholding and edge detection on a bgr image
    Returns lanes detected in a binary mask
    
    @param bgrImg: (np.ndarray) BGR image
    @param minLineThresh: minimum theshold for edge detector
    @param maxLineThresh: maximum theshold for edge detector
    @param satThresh: saturation channel threshold any value above is 255 else is 0
    @param hueThresh: hue channel threshold any value above is 255 else is 0
    @param redThresh: red, and green channel threshold any value above is 255 else is 0
    """
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


def getinitialCenters(warpedFrame):
    xPixsHisto = np.sum(warpedFrame[warpedFrame.shape[0]//2:], axis=0)
    midPoint = xPixsHisto.shape[0]//2
    leftXcPoint = np.argmax(xPixsHisto[:midPoint])
    rightXcPoint = np.argmax(xPixsHisto[midPoint:]) + midPoint
    return leftXcPoint, rightXcPoint, xPixsHisto


def fitLaneLines(leftLinePoints, rightLinePoints, lineLength, order=2):
    leftX, leftY = leftLinePoints
    rightX, rightY = rightLinePoints
    leftLineParams = np.polyfit(leftY, leftX, order)
    rightLineParams = np.polyfit(rightY, rightX, order)
    save2File("linesFit", dict(leftParams=leftLineParams, rightParams=rightLineParams))
    lineYVals = np.linspace(0, lineLength, birdFrame.shape[0])
    leftLineXVals = leftLineParams[0]*lineYVals**2 + lineYVals*leftLineParams[1] + leftLineParams[2]
    rightLineXVals = rightLineParams[0]*lineYVals**2 + lineYVals*rightLineParams[1] + rightLineParams[2]
    return (leftLineXVals, lineYVals), (rightLineXVals, lineYVals)
    

def getLanePoints(warpedFrame, nWindows, windowWidth, pixelsThresh):
    leftLanePixelsIds = []
    rightLanePixelsIds = []
    noneZeroIds = warpedFrame.nonzero()
    noneZeroXIds = np.array(noneZeroIds[1])
    noneZeroYIds = np.array(noneZeroIds[0])
    leftXcPoint, rightXcPoint, xPixsHisto = getinitialCenters(warpedFrame)
    windowHeight = np.int(warpedFrame.shape[0] // nWindows)
    rightCenter = (rightXcPoint, warpedFrame.shape[0] - windowHeight//2)
    leftCenter = (leftXcPoint, warpedFrame.shape[0] - windowHeight//2)
    outImg = np.dstack([warpedFrame, warpedFrame, warpedFrame])
    
    for i in range(1, nWindows+1):
        leftLinePt1 = (leftCenter[0]-windowWidth//2, leftCenter[1]-windowHeight//2)
        leftLinePt2 = (leftCenter[0]+windowWidth//2, leftCenter[1]+windowHeight//2)
        rightLinePt1 = (rightCenter[0]-windowWidth//2, rightCenter[1]-windowHeight//2)
        rightLinePt2 = (rightCenter[0]+windowWidth//2, rightCenter[1]+windowHeight//2)
        cv2.rectangle(outImg, leftLinePt1, leftLinePt2, (0, 255, 0), 3)
        cv2.rectangle(outImg, rightLinePt1, rightLinePt2, (0, 255, 0), 3)

        leftWindowXIds = (noneZeroXIds > leftLinePt1[0]) & (noneZeroXIds < leftLinePt2[0])
        leftWindowYIds = (noneZeroYIds > leftLinePt1[1]) & (noneZeroYIds < leftLinePt2[1])
        leftWindowPoints = leftWindowXIds & leftWindowYIds
        rightWindowXIds = (noneZeroXIds > rightLinePt1[0]) & (noneZeroXIds < rightLinePt2[0])
        rightWindowYIds = (noneZeroYIds > rightLinePt1[1]) & (noneZeroYIds < rightLinePt2[1])
        rightWindowPoints = rightWindowXIds & rightWindowYIds

        leftLanePixelsIds.append(leftWindowPoints)
        rightLanePixelsIds.append(rightWindowPoints)

        leftCenter = leftCenter[0], leftCenter[1] - windowHeight
        if leftWindowPoints.sum() > pixelsThresh:
    #         xC = np.int(np.median(noneZeroXIds[leftWindowPoints]))
            lXc = np.int(noneZeroXIds[leftWindowPoints].mean())
            leftCenter = (lXc, leftCenter[1])

        rightCenter = rightCenter[0], rightCenter[1] - windowHeight
        if rightWindowPoints.sum() > pixelsThresh:
    #         xC = np.int(np.median(noneZeroXIds[leftWindowPoints]))
            rXc = np.int(noneZeroXIds[rightWindowPoints].mean())
            rightCenter = (rXc, rightCenter[1])

    leftLanePixelsIds = np.sum(np.array(leftLanePixelsIds), axis=0).astype("bool")
    rightLanePixelsIds = np.sum(np.array(rightLanePixelsIds), axis=0).astype("bool")
    leftX = noneZeroXIds[leftLanePixelsIds]
    leftY = noneZeroYIds[leftLanePixelsIds]
    rightX = noneZeroXIds[rightLanePixelsIds]
    rightY = noneZeroYIds[rightLanePixelsIds]
    outImg[leftY, leftX] = [255, 0, 0]
    outImg[rightY, rightX] = [0, 0, 255]
    return outImg, (leftX, leftY), (rightX, rightY)


def predictLaneLines(warpedFrame, linesParams, margin):
    leftLineParams = linesParams["leftParams"]
    rightLineParams = linesParams["rightParams"]
    noneZeroIds = warpedFrame.nonzero()
    noneZeroXIds = np.array(noneZeroIds[1])
    noneZeroYIds = np.array(noneZeroIds[0])
    outImg = np.dstack([warpedFrame, warpedFrame, warpedFrame])

    l_a, l_b, l_c = leftLineParams
    r_a, r_b, r_c = rightLineParams
    leftLineLBoundry = l_a*noneZeroYIds**2 + l_b*noneZeroYIds + l_c - margin
    leftLineRBoundry = l_a*noneZeroYIds**2 + l_b*noneZeroYIds + l_c + margin
    rightLineLBoundry = r_a*noneZeroYIds**2 + r_b*noneZeroYIds + r_c - margin
    rightLineRBoundry = r_a*noneZeroYIds**2 + r_b*noneZeroYIds + r_c + margin

    leftLineBoundryIds = (noneZeroXIds > leftLineLBoundry) & (noneZeroXIds < leftLineRBoundry)
    rightLineBoundryIds = (noneZeroXIds > rightLineLBoundry) & (noneZeroXIds < rightLineRBoundry)

    leftLineBoundryX = noneZeroXIds[leftLineBoundryIds]
    leftLineBoundryY = noneZeroYIds[leftLineBoundryIds]
    rightLineBoundryX = noneZeroXIds[rightLineBoundryIds]
    rightLineBoundryY = noneZeroYIds[rightLineBoundryIds]

    outImg[leftLineBoundryY, leftLineBoundryX] = [255, 0, 0]
    outImg[rightLineBoundryY, rightLineBoundryX] = [0, 0, 255]
    leftLine, rightLine = fitLaneLines(
        (leftLineBoundryX, leftLineBoundryY), 
        (rightLineBoundryX, rightLineBoundryY), 
        birdFrame.shape[0]-1
    )
    return outImg, leftLine, rightLine


def plotPredictionBoundry(warpedImg, leftLine, rightLine, margin):
    boundryMask = np.zeros_like(warpedImg)
    leftLineLeftMargin = leftLine[0] - margin, leftLine[1]
    leftLineRightMargin = leftLine[0] + margin, leftLine[1]
    rightLineLeftMargin = rightLine[0] - margin, rightLine[1]
    rightLineRightMargin = rightLine[0] + margin, rightLine[1]

    leftLineLeftMargin = np.array(list(zip(*leftLineLeftMargin)), "int")
    leftLineRightMargin = np.flipud(np.array(list(zip(*leftLineRightMargin)), "int"))
    leftBoundry = list(np.vstack([leftLineLeftMargin, leftLineRightMargin]).reshape(1, -1, 2))
    cv2.fillPoly(boundryMask, leftBoundry, (255, 255, 0))

    rightLineLeftMargin = np.array(list(zip(*rightLineLeftMargin)), "int")
    rightLineRightMargin = np.flipud(np.array(list(zip(*rightLineRightMargin)), "int"))
    rightBoundry = list(np.vstack([rightLineLeftMargin, rightLineRightMargin]).reshape(1, -1, 2))
    cv2.fillPoly(boundryMask, rightBoundry, (255, 255, 0))
    outImg = cv2.addWeighted(warpedImg, 1, boundryMask, 0.2, 0)
    return outImg


def predictXVal(y, params):
    """
    Predicts the x-axis value of a given y-axis value
    that belongs to a curve given its parameters
    
    @param y: y-axis value of the points to be estimated
    @param params: (tuple) of the lane line fitted parameters
    """
    a, b , c = params
    return a*y**2 + b*y + c


def measureCurveRadius(y, params):
    """
    Computes the radius of curvature for a given (x, y) point 
    that belogns to a lane curve.
    
    @param y: y-axis value of the points to find radius of curvature at
    @param params: (tuple) of a lane line fitted parameters
    """
    a, b , c = params
    x = predictXVal(y, params)
    dydx = 2*a*y + b
    d2ydx2 = 2*a
    r = ((1 + (dydx**2))**1.5)/d2ydx2
    return r


def warped2BirdPoly(bgrFrame, points, width, height):
    """
    Warps a given image to a prespective bird view (top-plane)
    
    @param bgrFrame: (np.ndarray) BGR image
    @param points: ROI lane points to be warped from polygon to rectangle
    @param width: width of the target warped image
    @param height: height of the target warped image
    """
    warpedPoints = np.float32(points)
    birdPoints = np.float32([
        [points[-1][0], points[0][1]],
        [points[2][0], points[0][1]],
        points[2], points[-1]
    ])
    Matrix = cv2.getPerspectiveTransform(warpedPoints, birdPoints)
    return cv2.warpPerspective(bgrFrame, Matrix, (width, height), flags=cv2.INTER_LINEAR)


def save2File(path, obj):
    """
    Saves a data object to a local binary file
    
    @param path: local path to store the object in
    @param obj: data object to be saved.
    """
    with open(path, "wb") as wf:
        pickle.dump(obj, wf)
        print("roi is saved!")


def loadFile(path):
    """
    Loads and Returns a data object from a local binary file
    
    @param path: local path to store the object in
    """
    with open(path, "rb") as rf:
        return pickle.load(rf)
