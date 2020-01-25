import cv2
import numpy as np
import matplotlib.pyplot as plt


chessImg = cv2.imread("../../../gallery/chessBoard.jpg", cv2.IMREAD_COLOR)
grayChess = cv2.cvtColor(chessImg, cv2.COLOR_BGR2GRAY)
floatGrayChess = np.float32(grayChess)

# haris corner-detector
# harisRes = cv2.cornerHarris(floatGrayChess, blockSize=2, ksize=3, k=0.05)
# harisRes = cv2.dilate(harisRes, None)
# chessImg[harisRes > 0.01*harisRes.max()] = [255, 0, 0]

# shi-tomasi corner-detector
corners = cv2.goodFeaturesToTrack(floatGrayChess, maxCorners=64, qualityLevel=0.01, minDistance=10)
corners = np.int0(corners)
for c in corners:
    x, y = c.ravel()
    print(c, (x, y))
    cv2.circle(chessImg, (x, y), 3, (255, 0, 0), -1)

cv2.imshow("detected-corners", chessImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
