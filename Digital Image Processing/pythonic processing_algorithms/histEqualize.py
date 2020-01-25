#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 23:46:30 2020

@author: ezio
"""

import cv2
import math
import numpy as np


_max, _min = 0, 255
img = cv2.imread("./gallery/tomato1.jpeg", cv2.IMREAD_GRAYSCALE)
equaImg = np.copy(img)


def histogram(img):
    """
    Computes the histogram of pixels for a given Gray-Scale image
    """
    height, width = img.shape[:2]
    histo = np.arange((256))
    for i in range(height):
        for j in range(width):
            pix = img.item((i, j))
            histo[pix] += 1
    return histo


def cummHisto(hist):
    """
    Computes the cummulative histogram of pixels for a given Gray-Scale image
    """
    cummHist = hist.copy()
    for i in range(1, len(hist)):
        cummHist[i] += cummHist[i-1]
    return cummHist


height, width = img.shape[:2]
pixs = height * width

hist = histogram(img)
cumHist = cummHisto(hist)

for i in range(height):
    for j in range(width):
        pix = img.item(i, j)
        modPix = math.floor(cumHist[pix] * 255. / pixs)
        equaImg.itemset((i, j), modPix)
        
cv2.imshow("img", img)
cv2.waitKey(0)

cv2.imshow("contrasted-img", equaImg)
cv2.waitKey(0)

cv2.destroyAllWindows()