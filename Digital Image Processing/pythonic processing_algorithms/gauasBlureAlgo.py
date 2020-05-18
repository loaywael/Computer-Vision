#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:02:32 2020

@author: ezio
"""

import cv2
import numpy as np


img = cv2.imread("gallery/tomato.jpg", cv2.IMREAD_GRAYSCALE)
outImg = img.copy()

gaussKernel = (1.0 / 57) * np.array(
        [[0, 1, 2, 1, 0],
         [1, 3, 5, 3, 1],
         [2, 5, 9, 5, 2],
         [1, 3, 5, 3, 1],
         [0, 1, 2, 1, 0]]
        )

imgHeight, imgWidth = img.shape
imgSize = imgHeight * imgWidth
kHeight, kWidth = gaussKernel.shape
kSize = kHeight * kWidth

for i in range(2, imgHeight - 2):
    for j in range(2, imgWidth - 2):
        kPSum = 0
        for kii in range(-2, 3):
            for kjj in range(-2, 3):
                kPix = img.item(i + kii, j + kjj)
                gPix = gaussKernel.item(2 + kii, 2 + kjj)
                kPSum += kPix * gPix
        modPix = kPSum
        outImg.itemset((i, j), modPix)
        
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.imshow("blured-img", outImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
