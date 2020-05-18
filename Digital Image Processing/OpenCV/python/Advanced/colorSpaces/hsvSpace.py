import cv2
import numpy as np


def pickBGRColor(event, x, y, flags, params):
    global color, image, colorBag
    image = params
    if event == cv2.EVENT_LBUTTONDBLCLK:
        color = image[y, x]
        cv2.circle(image, (x, y), 3, (255, 255, 255), 2)
        colorBag.append(color)


color = None
colorBag = []
bgrImg = cv2.imread("./frame116.jpg", cv2.IMREAD_COLOR)  # bluish img
hsvImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2HSV)
# hslHourseImg = cv2.cvtColor(hourseImg, cv2.COLOR_BGR2HLS)
cv2.namedWindow("bgr")
cv2.setMouseCallback("bgr", pickBGRColor, bgrImg)
while cv2.waitKey(10) != ord('q'):
    if color is not None:
        hsvColor = cv2.cvtColor(np.array(color, "uint8", ndmin=3), cv2.COLOR_BGR2HSV)
        h, s, v = hsvColor.ravel()
        lowerColor = np.array([h - 10, s - 40, v])
        upperColor = np.array([h + 10, 255, 255])
        maskedColor = cv2.inRange(hsvImg, lowerColor, upperColor)
        colorPicked = cv2.bitwise_and(bgrImg, bgrImg, mask=maskedColor)
        cv2.imshow("color-out", colorPicked)
    cv2.imshow("bgr", bgrImg)
print(np.mean(np.array(colorBag).reshape(-1, 3), 0))
cv2.destroyAllWindows()
