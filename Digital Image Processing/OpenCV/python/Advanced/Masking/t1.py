import cv2
import numpy as np


hourseImg = cv2.imread("../../../gallery/hourse.jpeg", cv2.IMREAD_COLOR)
logoImg = cv2.imread("../../../gallery/logo.png", cv2.IMREAD_COLOR)


# select the starting offset
x_offset, y_offset = 175, 150
height, width = logoImg.shape[:2]

# extract the region of image
# roi = hourseImg[y_offset:y_offset + height, x_offset:x_offset + width]
roi = hourseImg[y_offset:y_offset+height, 0:width]

# create the mask of the logo
grayLogo = cv2.cvtColor(logoImg, cv2.COLOR_BGR2GRAY)
grayLogo = cv2.blur(grayLogo, (5, 5))
_, mask = cv2.threshold(grayLogo, 220, 255, cv2.THRESH_BINARY_INV)
invMask = cv2.bitwise_not(mask)

# black out the region of the logo in the ROI
roiBKG = cv2.bitwise_and(roi, roi, mask=invMask)

# cut the colored-logo without background
roiFG = cv2.bitwise_and(logoImg, logoImg, mask=mask)

# patch the roi over the hourse image
finalROI = cv2.add(roiBKG, roiFG)
hourseImg[y_offset:y_offset+height, 0:width] = finalROI

cv2.imshow("frame", hourseImg)
cv2.waitKey(0)

cv2.destroyAllWindows()
