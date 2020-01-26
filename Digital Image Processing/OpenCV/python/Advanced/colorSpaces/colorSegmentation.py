import cv2
import numpy as np


def creatTrackBar(name, cam="mi"):
    web = 170, 60, 115, 179, 92, 206
    mi = 147, 176, 85, 179, 255, 193
    mode = mi if cam == "mi" else web
    cv2.namedWindow(name)
    cv2.createTrackbar("lowerHue", name, mode[0], 179, lambda x: None)
    cv2.createTrackbar("lowerSat", name, mode[1], 255, lambda x: None)
    cv2.createTrackbar("lowerVal", name, mode[2], 255, lambda x: None)
    cv2.createTrackbar("upperHue", name, mode[3], 179, lambda x: None)
    cv2.createTrackbar("upperSat", name, mode[4], 255, lambda x: None)
    cv2.createTrackbar("upperVal", name, mode[5], 255, lambda x: None)


def getTrackBarVals(name):
    lH = cv2.getTrackbarPos("lowerHue", name)
    lS = cv2.getTrackbarPos("lowerSat", name)
    lV = cv2.getTrackbarPos("lowerVal", name)
    uH = cv2.getTrackbarPos("upperHue", name)
    uS = cv2.getTrackbarPos("upperSat", name)
    uV = cv2.getTrackbarPos("upperVal", name)
    return np.array((lH, lS, lV)), np.array((uH, uS, uV))


name = "frame"
cap = cv2.VideoCapture(1)
width, height = int(cap.get(3)), int(cap.get(4))
creatTrackBar(name, "web")
codex = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
writer = cv2.VideoWriter("detectedColor.avi", codex, 20, (width, height))

while True:
    global recording
    _, frame = cap.read()
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowObjColor, highObjColor = getTrackBarVals(name)
    colorRange = cv2.inRange(hsvFrame, lowObjColor, highObjColor)
    # ------------------------------------------------------------
    gauBlr = cv2.GaussianBlur(colorRange, (9, 9), 1)
    medBlr = cv2.medianBlur(colorRange, 15)
    detectedObj = cv2.bitwise_or(gauBlr, medBlr)
    # ------------------------------------------------------------
    colorRange = cv2.morphologyEx(detectedObj, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8))
    colorRange = cv2.bitwise_and(frame, frame,  mask=detectedObj)
    # ------------------------------------------------------------
    writer.write(colorRange)
    cv2.imshow("frame", colorRange)
    cv2.imshow("ranged-frame", detectedObj)
    if cv2.waitKey(1) & 0xFF == ord('q') or not cap.isOpened():
        break


cap.release()
writer.release()
cv2.destroyAllWindows()
