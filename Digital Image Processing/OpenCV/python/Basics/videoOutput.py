#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 17:51:04 2020

@author: ezio
"""

import cv2
import numpy as np


cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter("video.avi", fourcc, 20, (800, 600))

while True:
    _, frame = cap.read()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edgeFrames = cv2.Canny(grayFrame, 75, 135)
    cv2.rectangle(frame, (0, 0), (300, 300), (0, 255, 0), 3)
    cv2.circle(frame, (150, 150), 100, (0, 0, 255), -1)
    cv2.putText(frame, "Hello, World!", (150, 400), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("edges", frame)
    writer.write(grayFrame)
    if not cap.isOpened() or cv2.waitKey(1) == ord('q'):
        break
    

cap.release()
writer.release()
cv2.destroyAllWindows()