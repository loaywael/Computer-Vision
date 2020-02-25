import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 0)
    cv2.circle(frame, (200, 107), 5, (255, 0, 0), -1)
    cv2.circle(frame, (300, 107), 5, (255, 0, 0), -1)
    cv2.circle(frame, (100, 350), 5, (255, 0, 0), -1)
    cv2.circle(frame, (400, 350), 5, (255, 0, 0), -1)
    planePts = np.float32([[200, 107], [300, 107], [100, 350], [400, 350]])
    outPts = np.float32([[0, 0], [600, 0], [0, 800], [600, 800]])
    prepecMatrix = cv2.getPerspectiveTransform(planePts, outPts)
    board = cv2.warpPerspective(frame, prepecMatrix, (600, 800))
    if cv2.waitKey(10) & 0xFF == ord('q') or not ret:
        break
    cv2.imshow("plane", frame)
    cv2.imshow("board", board)
