import cv2
import numpy as np
import matplotlib.pyplot as plt


objImg = cv2.imread("../../../gallery/object.jpg", cv2.IMREAD_GRAYSCALE)
sceneImg = cv2.imread("../../../gallery/scene.jpg", cv2.IMREAD_GRAYSCALE)

sift = cv2.xfeatures2d_SIFT()
kpObj, distObj = sift.detectAndCompute(objImg, None)
kpScene, distScene = sift.detectAndCompute(sceneImg, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.knnMatch(distObj, distScene, k=2)
bestFeatures = []
for match1, match2 in matches:
    if match1.distance < 0.75 * match2.distance:
        bestFeatures.append(match1)
print(bestFeatures)
