import cv2
import numpy as np
import datetime
import time


mins, seconds = 0, 0
vidCap = cv2.VideoCapture("../../../gallery/lanePath.avi")
font = cv2.FONT_HERSHEY_SIMPLEX
width, height, fps = int(vidCap.get(3)), int(vidCap.get(3)), int(vidCap.get(cv2.CAP_PROP_FPS))
codex = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter("test.avi", codex, fps, (width, height))

while vidCap.isOpened():
    ret, frame = vidCap.read()
    if cv2.waitKey(1) & 0xFF == ord("q") or not ret:
        break
    if seconds % 60 == 0 and mins != 0:
        seconds = 0
        mins += 1
    seconds += 1/30

    time.sleep(1/30)
    cv2.rectangle(frame, (0, 0), (175, 30), (0, 255, 0), 1)
    cv2.putText(frame, f"time elapsed: {mins}:{int(seconds)}", (5, 20), font, 0.50, (0, 255, 0), 1)
    writer.write(frame)
    cv2.imshow("frame", frame)

vidCap.release()
writer.release()
cv2.destroyAllWindows()
