#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:49:35 2020

@author: ezio
"""
import cv2
import numpy as np



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

    
p = 0.005
low = 0
high = 255
img = cv2.imread("./gallery/tomato1.jpeg", cv2.IMREAD_GRAYSCALE)
contImg = img.copy()
height, width = img.shape[:2]
pixs = height * width

hist = histogram(img)
cumHist = cummHisto(hist)

for i in range(256):    # get the %p of the lowest thresthold
    if cumHist[i] >= p * pixs:
        low = i
        break
    
for i in range(255, -1, -1):    # get the %p of the highest thresthold
    if cumHist[i] <= pixs * (1 - p):
        high = i
        break


for i in range(height):
    for j in range(width):
        pix = img.item((i, j))
        if pix <= low:  # lower region
            modPix = 0
        elif pix >= high:   # higher region
            modPix = 255
        else:   # min-max scaler
            modPix = (pix - low) / (high - low) * 255
        contImg.itemset((i, j), modPix)

cv2.imshow("img", img)
cv2.waitKey(0)

cv2.imshow("contrasted-img", contImg)
cv2.waitKey(0)

cv2.destroyAllWindows()
    
