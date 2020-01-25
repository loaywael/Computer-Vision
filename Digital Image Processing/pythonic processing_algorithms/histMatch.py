#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 11:03:14 2020

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

    
img = cv2.imread("./gallery/face1.jpg", cv2.IMREAD_GRAYSCALE)
refImg = cv2.imread("./gallery/tomato1.jpeg", cv2.IMREAD_GRAYSCALE)
outImg = img.copy()

imgHeight, imgWidth = img.shape[:2]
refImgHeight, refImgWidth = refImg.shape[:2]
imgPixs = imgHeight * imgWidth
refImgPixs = refImgHeight * refImgWidth

imgHist = histogram(img)
refImgHist = histogram(refImg)
imgCummHist = cummHisto(imgHist)
refImgCummHist = cummHisto(refImgHist)

k = 256
normImgCummHist = imgCummHist / imgPixs
normRefImgCummHist = refImgCummHist / refImgPixs
newPixVals = np.zeros((k))

for a in range(k):
    j = k - 1
    while True: 
        newPixVals[a] = j
        j -= 1
        if j < 0 or normImgCummHist[a] > normRefImgCummHist[j]:
            break
        
for i in range(imgHeight):
    for j in range(imgWidth):
        pix = img.item(i, j)
        modPix = newPixVals[pix]
        outImg.itemset((i, j), modPix)
  
      
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.imshow("matched-img", outImg)
cv2.waitKey(0)
cv2.destroyAllWindows()