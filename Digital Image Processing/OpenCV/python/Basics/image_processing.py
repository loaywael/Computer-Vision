import cv2
import numpy as np
from glob import glob


img = cv2.imread(*glob("./gallery/face0*"))
kernel = np.ones((5, 5), dtype="uint8")

b = img[:, :, 0]
g = img[:, :, 1]
r = img[:, :, 2]

bgra = cv2.merge((b, g, r, g))
cv2.imwrite("bgra.png", bgra)

# averages nearby pixels in range
blured = cv2.GaussianBlur(img, (55, 5), 0)
# removes black noise from image by a sliding kernel
dilated = cv2.dilate(img, kernel, iterations=3)
# removes white noise from image by a sliding kernel
eroded = cv2.erode(img, kernel, iterations=3)

cv2.imshow("Blured", blured)
cv2.imshow("Dilated", dilated)
cv2.imshow("Eroded", eroded)

cv2.waitKey(0)
cv2.destroyAllWindows()
