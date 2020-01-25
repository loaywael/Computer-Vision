import cv2
import numpy as np


hourseImg = cv2.imread("../../../gallery/hourse.jpeg", cv2.IMREAD_COLOR)
rgbHourse = cv2.cvtColor(hourseImg, cv2.COLOR_BGR2RGB)

gamma = 0.5
result = np.power(rgbHourse, gamma)
cv2.imshow("corrected-img", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
