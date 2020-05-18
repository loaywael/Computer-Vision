import cv2
import numpy as np
from glob import glob


img = cv2.imread(*glob("./gallery/face1*"))

fatImg = cv2.resize(img, (0, 0), fx=2, fy=1)
doubImg = cv2.resize(img, (1024, 1024))		# interploted pixels
# no interpolation just selects the next pixel for each pixel
pixDoubImg = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)

rotMatrix = cv2.getRotationMatrix2D((0, 0), -30, 1)		# clockwise rotation
rotatedImg = cv2.warpAffine(img, rotMatrix, (img.shape[1], img.shape[0]))

cv2.imshow("Image", fatImg)
cv2.imshow("Pixeled Big Image", pixDoubImg)
cv2.imshow("Smooth Big Image", doubImg)
cv2.imshow("Rotated Image", rotatedImg)

cv2.waitKey(0)
cv2.destroyAllWindows()