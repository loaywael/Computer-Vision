import cv2
import numpy as np


class TrackBars:
    def __init__(self, **kwargs):
        self.window = kwargs["windowName"]
        self.names = kwargs["barNames"]
        self.values = kwargs["values"]
        cv2.namedWindow(self.window)
        for i in range(len(self.names)):
            cv2.createTrackbar(self.names[i], self.window, *self.values[i], lambda x: x)

    def getTrackPos(self):
        positions = []
        for i in range(len(self.names)):
            positions.append(cv2.getTrackbarPos(self.names[i], self.window))
        return positions

    @staticmethod
    def oddSizer(x):
        return x + 1 if x % 2 == 0 else x
