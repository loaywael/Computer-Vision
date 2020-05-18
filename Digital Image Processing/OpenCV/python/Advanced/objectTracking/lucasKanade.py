import cv2
import numpy as np


lucKan = dict(winSize=(200, 200), maxLevel=2,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
