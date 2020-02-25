import cv2
import numpy as np
import matplotlib.pyplot as plt


hourseImg = cv2.imread("../../gallery/hourse.jpeg", cv2.IMREAD_COLOR)
pennImg = cv2.imread("../../gallery/penny.jpg", cv2.IMREAD_COLOR)
pennImg = cv2.resize(pennImg, None, fx=0.5, fy=0.5)

# hourseImg = cv2.cvtColor(hourseImg, cv2.COLOR_RGB2BGR)
# grayHourse = cv2.cvtColor(hourseImg, cv2.COLOR_BGR2GRAY)

eyeTemp = hourseImg[205:217, 639:657]
faceTemp = hourseImg[175:280, 575:700]
# -------------------------------------
pennyTemp = pennImg[35:287, 115:380]

faceRes = cv2.matchTemplate(hourseImg, faceTemp, cv2.TM_CCOEFF)


tempHeight, tempWidth = faceTemp.shape[:2]
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(faceRes)
recPt1 = maxLoc
recPt2 = recPt1[0] + tempWidth, recPt1[1] + tempHeight
cv2.rectangle(hourseImg, recPt1, recPt2, color=(0, 255, 0), thickness=2)


threshold = 0.7
matchedMap = cv2.matchTemplate(pennImg, pennyTemp, cv2.TM_CCOEFF_NORMED)
locs = np.where(matchedMap > threshold)

for pt1 in zip(*locs[::-1]):
    pt2 = pt1[0] + pennyTemp.shape[1], pt1[1] + pennyTemp.shape[0]
    cv2.rectangle(pennImg, pt1, pt2, (0, 255, 0), 2)

# cv2.imshow("eye-result", eyeRes)
# cv2.imshow("face-result", faceRes)
cv2.imshow("matched-temp", pennImg)
# cv2.imshow("temp-heatmap", matchedMap)
cv2.waitKey(0)

cv2.destroyAllWindows()
