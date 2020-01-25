import cv2
import numpy as np

# =====================
# CODE OBJECTIVE
# =====================
# substituting white background with black one for the mask logo
# then overwrite logo mask on the region of interest of the target image


hourseImg = cv2.imread("../../../gallery/hourse.jpeg", cv2.IMREAD_COLOR)
logoImg = cv2.imread("../../../gallery/logo.png", cv2.IMREAD_COLOR)

rows, cols = logoImg.shape[:2]                      # size of the logo-img
hourseROI = hourseImg[0:rows, 0:cols]               # target location to overwrite

grayLogo = cv2.cvtColor(logoImg, cv2.COLOR_BGR2GRAY)
_, logoMask = cv2.threshold(grayLogo, 220, 255, cv2.THRESH_BINARY_INV)  # white logo over black BKG
invLogoMask = cv2.bitwise_not(logoMask)                                 # black logo over white BKG

hourseBKG = cv2.bitwise_and(hourseROI, hourseROI, mask=invLogoMask)
hourseFG = cv2.bitwise_and(logoImg, logoImg, mask=logoMask)

logoPatch = cv2.add(hourseBKG, hourseFG)
hourseImg[:rows, :cols] = logoPatch

cv2.imshow("over-layed hourse", hourseImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.destroyAllWindows()
