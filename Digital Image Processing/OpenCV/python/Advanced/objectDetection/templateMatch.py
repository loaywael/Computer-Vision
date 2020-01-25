import cv2
import numpy as np
import matplotlib.pyplot as plt


hourseImg = cv2.imread("../../../gallery/hourse.jpeg", cv2.IMREAD_COLOR)
# hourseImg = cv2.cvtColor(hourseImg, cv2.COLOR_RGB2BGR)
# grayHourse = cv2.cvtColor(hourseImg, cv2.COLOR_BGR2GRAY)

eyeTemp = hourseImg[205:217, 639:657]
faceTemp = hourseImg[175:280, 575:700]
faceRes = cv2.matchTemplate(hourseImg, faceTemp, cv2.TM_CCOEFF)


tempHeight, tempWidth = faceTemp.shape[:2]
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(faceRes)
recPt1 = maxLoc
recPt2 = recPt1[0] + tempWidth, recPt1[1] + tempHeight
cv2.rectangle(hourseImg, recPt1, recPt2, color=(0, 255, 0), thickness=2)


# threshold = 0.9
# correctLocs = np.where(locations > threshold)
# h, w = eyeTemp.shape
# for xLoc, yLoc in zip(*correctLocs[::-1]):
#     pt1, pt2 = (xLoc, yLoc), (xLoc + w, yLoc + h)
#     cv2.rectangle(hourseImg, pt1, pt2, (0, 255, 0), 2)

# cv2.imshow("eye-result", eyeRes)
# cv2.imshow("face-result", faceRes)
cv2.imshow("matched-temp", hourseImg)
cv2.imshow("temp-heatmap", faceRes)
cv2.waitKey(0)

cv2.destroyAllWindows()
