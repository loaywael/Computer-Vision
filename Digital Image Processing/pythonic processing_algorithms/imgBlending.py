#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 11:41:53 2020

@author: ezio
"""

import cv2


img1 = cv2.imread("gallery/face1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("gallery/tomato.jpg", cv2.IMREAD_GRAYSCALE)
outImg = img1.copy()

alpha = 0.75
height, width = img1.shape[:2]

for i in range(height):
    for j in range(width):
        pix1 = img1.item(i, j)
        pix2 = img2.item(i, j)
        modPix = alpha * pix1 + (1-alpha) * pix2
        outImg.itemset((i, j), modPix)
   
     
cv2.imshow("img", img1)
cv2.waitKey(0)
cv2.imshow("blended-img", outImg)
cv2.waitKey(0)
cv2.destroyAllWindows()