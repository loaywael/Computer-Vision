import cv2
import numpy as np


srcImg = cv2.imread("../../../gallery/hourse.jpeg", cv2.IMREAD_GRAYSCALE)
detParams = cv2.SimpleBlobDetector_Params()
detParams.minThreshold = 0
detParams.maxThreshold = 255
# -----------------------
detParams.filterByArea = True
detParams.minArea = 1000
detParams.minArea = 10000
detector = cv2.SimpleBlobDetector(detParams)


keyPoints = detector.detect(srcImg)
blobedImg = cv2.drawKeypoints(srcImg, keyPoints,
                              np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


cv2.imshow("blobed-img", blobedImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
