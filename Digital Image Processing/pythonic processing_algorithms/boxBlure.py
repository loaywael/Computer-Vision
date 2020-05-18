#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 14:40:50 2020

@author: ezio
"""

import cv2


img = cv2.imread("gallery/tomato.jpg", cv2.IMREAD_GRAYSCALE)
outImg = img.copy()

height, width = img.shape[:2]
for i in range(3, height-3):
    for j in range(3, width-3):
        sum = 0
        for kii in range(-3, 4):
            for kjj in range(-3, 4):
                kPix = img.item(i + kii, j + kjj)
                sum += kPix
        sum /= 7 * 7
        modPix = int(sum)
        outImg.itemset((i, j), modPix)
 
       
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.imshow("blured-img", outImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
