import cv2
import numpy as np
import matplotlib.pyplot as plt


chessImg = cv2.imread("../../../gallery/chessBoard.jpg", cv2.IMREAD_COLOR)
# grayChess = cv2.cvtColor(chessImg, cv2.COLOR_BGR2GRAY)

found, corners = cv2.findChessboardCornersSB(chessImg, (8, 8))
found, corners = cv2.findCirclesGrid(chessImg, (8, 8), cv2.CALIB_CB_SYMMETRIC_GRID)

print(found)
cv2.drawChessboardCorners(chessImg, (8, 8), corners, found)
cv2.imshow("detected-corners", chessImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
