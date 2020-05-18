import cv2
import numpy as np

from LaneUtils.LaneUtils import makeJPGS
from LaneUtils.LaneUtils import getSceneCanny
from LaneUtils.LaneUtils import getRegOfInterest
from LaneUtils.LaneUtils import getPointsFromLine
from LaneUtils.LaneUtils import drawLanesLines


# imgsPath = sorted(makeJPGS("./data"))
cap = cv2.VideoCapture("data/test2.mp4")
width, height = int(cap.get(3)), int(cap.get(4))
outVid = cv2.VideoWriter(
    "data/outImgs/lanePath.avi",
    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (width, height)
)

while cap.isOpened():
    reading, laneImg = cap.read()
    if reading:
        sceneEdges = getSceneCanny(laneImg)
        regOfInterest = np.array([[(30, height-75), (1250, height-75), (650, 290)]])
        laneEdges = getRegOfInterest(sceneEdges, regOfInterest)
        sceneLanes = drawLanesLines(laneImg, laneEdges, minLineLength=50, maxLineGap=5)
        outVid.write(sceneLanes)
        cv2.imshow("window", sceneLanes)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break
# cv2.imwrite("data/outImgs/sceneEdges.jpg", sceneEdges)
# cv2.imwrite("data/outImgs/laneEdges.jpg", laneEdges)
# cv2.imwrite("data/outImgs/detectedLane.jpg", sceneLanes)

cap.release()
cv2.destroyAllWindows()
