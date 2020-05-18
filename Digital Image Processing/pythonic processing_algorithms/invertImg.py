#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 23:06:11 2020

@author: ezio
"""

import cv2
import math
import numpy as np


CONTRAST = 1.3
img = cv2.imread("./gallery/tomato1.jpeg", cv2.IMREAD_GRAYSCALE)
invImg = np.copy(img)

height, width = img.shape[:2]

for i in range(height):
    for j in range(width):
        pix = img.item(i, j)
        modPix = 255 - pix
        invImg.itemset((i, j), modPix)

cv2.imshow("frame", img)
cv2.waitKey(0)

cv2.imshow("contrasted-frame", invImg)
cv2.waitKey(0)

cv2.destroyAllWindows()
