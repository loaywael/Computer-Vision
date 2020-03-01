import cv2
import numpy as np
from decimal import Decimal


img = cv2.imread("../../gallery/rhand.jpg", cv2.IMREAD_COLOR)
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.ones((5, 5), "uint8")
mask = cv2.morphologyEx(grayImg, cv2.MORPH_GRADIENT, kernel)
contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
hull = cv2.convexHull(contours[0], returnPoints=False)
defects = cv2.convexityDefects(contours[0], hull).squeeze()
cnt = contours[0]
for s, e, f, d in defects:
    start = tuple(*cnt[s])
    end = tuple(*cnt[e])
    far = tuple(*cnt[f])

    cv2.line(img, start, end, (0, 255, 0), 3)
    cv2.circle(img, far, 9, (0, 0, 255), -1)

leftMost = cnt[cnt[:, :, 0].argmin()].ravel()
rightMost = cnt[cnt[:, :, 0].argmax()].ravel()
topMost = cnt[cnt[:, :, 1].argmin()].ravel()
bottomMost = cnt[cnt[:, :, 1].argmax()].ravel()

cv2.circle(img, tuple(leftMost), 9, (0, 0, 0), -1)
cv2.circle(img, tuple(rightMost), 9, (0, 0, 0), -1)
cv2.circle(img, tuple(topMost), 9, (0, 0, 0), -1)
cv2.circle(img, tuple(bottomMost), 9, (0, 0, 0), -1)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
