import os
import cv2
import numpy as np
from detection_utils import updatePoints, drawRoIPoly
from detection_utils import save2File, loadFile, getLaneMask
from detection_utils import warped2BirdPoly


path = "./driving_datasets/"
videoPath = path + "project_video.mp4"

cap = cv2.VideoCapture(videoPath)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(width, height)
i = 0
key = None
points = []
while cap.isOpened() and key != ord('q'):
    key = cv2.waitKey(1)
    ret, frame = cap.read()    # reads bgr frame
    if ret:
        camModel = loadFile("camCalibMatCoeffs")
        camMtx = camModel["camMtx"]
        dstCoeffs = camModel["dstCoeffs"]
        frame = cv2.undistort(frame, camMtx, dstCoeffs, None, camMtx)
        displayedFrame = frame.copy()
        if os.path.exists("./roiPoly"):
            points = loadFile("./roiPoly")
        if points is not None:
            drawRoIPoly(displayedFrame, points)
        binaryLanes = getLaneMask(frame, 33, 200, 110, 30, 200)
        birdFrame = warped2BirdPoly(binaryLanes, points, width, height)
        if key & 0xFF == ord('s'):
            cv2.imwrite(f"frame{str(i)}.jpg", frame)
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # cv2.imshow("lanesMask", binaryLanes)
        cv2.imshow("frame", displayedFrame)
        cv2.imshow("bird", birdFrame)
        i += 1


cap.release()
cv2.destroyAllWindows()
