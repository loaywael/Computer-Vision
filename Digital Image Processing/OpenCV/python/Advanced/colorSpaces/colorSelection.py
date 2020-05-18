import cv2
import numpy as np


def getColor(event, x, y, flags, params):
    global clicks, colorRange, frame
    if event == cv2.EVENT_LBUTTONDBLCLK:
        clicks = not clicks
        cv2.circle(frame, (x, y), 11, 255, -1)
        color = np.array(frame[x, y], "uint8", ndmin=3)
        colorRange = cv2.cvtColor(color, cv2.COLOR_BGR2HSV).ravel()
        print(hsvColor)


cap = cv2.VideoCapture(1)
camWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
camHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
vidCodex = cv2.VideoWriter_fourcc(*"XVID")
vidWriter = cv2.VideoWriter("objColor.avi", vidCodex, fps, (camWidth, camHeight))

colorRange = []
clicks = 0
cv2.namedWindow("cam-stream")
cv2.setMouseCallback("cam-stream", getColor)
while cap.isOpened():
    ret, frame = cap.read()
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or not ret:
        break
    print(colorRange, f"num clicks: {clicks}")
    if clicks:
        h, s, v = colorRange[0], colorRange[1], colorRange[2]
        lowerRange = np.uint8([h, s, v])
        upperRange = np.uint8([h + 10, s, v])
        hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # lowerRange = np.flip(colorRange)    # reverse array oreder
        # upperRange=np.array([colorRange[2] + 10, 175, 185], dtype="uint8")
        mask = cv2.inRange(hsvFrame, lowerRange, upperRange)
        objFrame = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow("color-stream", objFrame)
        vidWriter.write(objFrame)
    cv2.imshow("cam-stream", frame)


cap.release()
vidWriter.release()
cv2.destroyAllWindows()
