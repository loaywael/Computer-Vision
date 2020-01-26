import cv2
import numpy as np
import datetime

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

font = cv2.FONT_HERSHEY_SIMPLEX
fontColor = (255, 0, 0)

vidCodex = cv2.VideoWriter_fourcc(*"XVID")
vidWriter = cv2.VideoWriter("camCap.avi", vidCodex, fps, (width, height))

recording = False
while cap.isOpened():
    ret, frame = cap.read()
    dateTime = datetime.datetime.now()
    timeNow = dateTime.strftime("%H:%M:%S")
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or not ret:
        break
    cv2.putText(frame, f"width: {width}, height: {height}, fps: {fps}",
                (15, 30), font, 0.75, fontColor, 2)
    cv2.putText(frame, f"recording-time: {timeNow}", (15, 60), font, 0.75, fontColor, 2)
    if key == ord('r') or recording:
        recording = True
        cv2.putText(frame, f"recording: {recording}", (width - 250, 30), font, 0.75, fontColor, 2)
        vidWriter.write(frame)
    if key == ord('s'):
        recording = False
    cv2.imshow("frame", frame)


cap.release()
vidWriter.release()
cv2.destroyAllWindows()
