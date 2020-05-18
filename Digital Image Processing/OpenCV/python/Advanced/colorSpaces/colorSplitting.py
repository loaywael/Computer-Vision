import cv2
import numpy as np


# bgrHourse = cv2.imread("../../gallery/hourse.jpeg", cv2.IMREAD_COLOR)
# cv2.imshow("frame", bgrHourse)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

recording = False
cap = cv2.VideoCapture(1)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
vidCodex = cv2.VideoWriter_fourcc(*"MPEG")
vidWriter = cv2.VideoWriter("camCap.avi", vidCodex, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    b, g, r = frame[:, :, 1], frame[:, :, 0], frame[:, :, 0]
    borderEffect = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    # different img size make demux problems, or bgr in compatible
    borderEffect = cv2.resize(borderEffect, (width, height))
    print(frame.shape, borderEffect.shape)
    # b, g, r = cv2.split(frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or not ret:
        break
    if key == ord('r') or recording:
        recording = True
        vidWriter.write(borderEffect)
    if key == ord('s'):
        recording = False
    # cv2.imshow("blue", b)
    # cv2.imshow("green", g)
    # cv2.imshow("red", r)
    cv2.imshow("red", borderEffect)


cap.release()
vidWriter.release()
cv2.destroyAllWindows()
