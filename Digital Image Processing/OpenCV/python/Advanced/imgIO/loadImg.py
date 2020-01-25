import cv2
import numpy as np
import matplotlib.pyplot as plt


bgr = cv2.imread("../../../gallery/hourse.jpeg", cv2.IMREAD_COLOR)  # bluish img
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

b, g, r = cv2.split(bgr)
r *= 0
g *= 0
bgrN = cv2.merge([b, g, r])

# plt.imshow(bgr)
# plt.show()
# plt.imshow(rgb)
# plt.show()
# cv2.imshow("frame", rgb)
# cv2.waitKey(0)

# cv2.imshow("frame", bgrN)
# cv2.waitKey(0)

# sBGR = cv2.resize(bgr, (100, 100))
# cv2.imshow("frame", sBGR)
# cv2.waitKey(0)

# sBGR = cv2.resize(bgr, (0, 0), bgr, 0.5, 0.5)
# cv2.imshow("frame", sBGR)
# cv2.waitKey(0)

# sBGR = cv2.flip(bgr, 1)     # vertical: 0, horizontal: 1, both: -1
# cv2.imshow("frame", sBGR)
# cv2.waitKey(0)
