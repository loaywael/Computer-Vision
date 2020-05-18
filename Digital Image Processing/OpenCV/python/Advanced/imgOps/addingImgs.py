import cv2
import numpy as np


img1 = np.array([10], "uint8")
img2 = np.array([250], "uint8")

out1 = cv2.add(img1, img2)
print(out1)

out2 = img1 + img2
print(out2)

cv2.imshow("oprncv", out1)
cv2.imshow("numpy", out2)
cv2.waitKey(0)

cv2.destroyAllWindows()
