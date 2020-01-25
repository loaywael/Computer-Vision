import cv2
import numpy as np
import matplotlib.pyplot as plt


bgr = cv2.imread("../../../gallery/hourse.jpeg", cv2.IMREAD_COLOR)  # bluish img
bkg = np.zeros_like(bgr, dtype=np.int32)

fontSettings = {
    "text": "hourse in Blue channel",
    "org": (200, 500),
    "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
    "fontScale": 1,
    "color": (0, 255, 0),
    "thickness": 1,
    "lineType": cv2.LINE_AA
}

# needs dtype to be np.int32
vertices = np.array([[0, 0], [450, 125], [600, 400], [90, 200]])
# vertices = vertices.reshape(-1, 1, 2)

polySettings = {
    "pts": [vertices],
    "isClosed": True,
    "color": (0, 255, 0),
    "thickness": 3
}

cv2.polylines(bgr, **polySettings)
cv2.putText(bgr, **fontSettings)
cv2.imshow("blue-channel", bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
