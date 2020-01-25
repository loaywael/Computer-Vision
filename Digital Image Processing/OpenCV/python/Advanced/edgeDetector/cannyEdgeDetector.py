import cv2
import numpy as np
import matplotlib.pyplot as plt


chessImg = cv2.imread("../../../gallery/chessBoard.jpg", cv2.IMREAD_COLOR)
grayChess = cv2.cvtColor(chessImg, cv2.COLOR_BGR2GRAY)

median = np.median(grayChess)
minVal = int(max(0, 0.7*median))
maxVal = int(min(255, 1.3*median))
cannyEdges = cv2.Canny(grayChess, minVal, maxVal)


cv2.imshow("detected-edges", cannyEdges)
cv2.waitKey(0)
cv2.destroyAllWindows()
