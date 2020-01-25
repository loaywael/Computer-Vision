import cv2
import numpy as np
import matplotlib.pyplot as plt


objImg = cv2.imread("../../../gallery/object.jpg", cv2.IMREAD_GRAYSCALE)
sceneImg = cv2.imread("../../../gallery/scene.jpg", cv2.IMREAD_GRAYSCALE)

bfr = cv2.ORB_create()
kp1, dst1 = bfr.detectAndCompute(objImg, None)
kp2, dst2 = bfr.detectAndCompute(sceneImg, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(dst1, dst2)
matches = sorted(matches, key=lambda x: x.distance)

# matchMask = [[0, 0] for i in range((len(matches)))]
drawParams = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0))
matchedScene = cv2.drawMatches(objImg, kp2, sceneImg, kp1,
                               matches[:11], None, flags=0, **drawParams)
matchedScene = cv2.resize(matchedScene, (1200, 800))

cv2.imshow("detected-features", matchedScene)
cv2.waitKey(0)
cv2.destroyAllWindows()
