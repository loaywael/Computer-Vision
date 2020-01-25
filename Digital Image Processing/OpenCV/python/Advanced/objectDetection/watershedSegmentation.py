import cv2
import numpy as np
from matplotlib import cm


srcImg = cv2.imread("../../../gallery/hourse.jpeg", cv2.IMREAD_COLOR)
srcImg = cv2.resize(srcImg, None, fx=0.5, fy=0.5)
scene = srcImg.copy()
segments = np.zeros_like(scene, np.uint8)
markedImg = np.zeros(scene.shape[:2], np.int32)


def getColorPallet(palletSize=1, palletName="tab10"):
    pallet = []
    if palletName == "tab10":
        for i in range(palletSize):
            color = tuple((np.array(cm.tab10(i))[:3]*255))
            pallet.append(color)
    return pallet


def creatRoots(event, x, y, flags, params):
    global changedMark
    if event == cv2.EVENT_LBUTTONDBLCLK:
        pt = x, y
        cv2.circle(markedImg, pt, 9, (colorMark), -1)
        cv2.circle(scene, pt, 9, colorPallet[colorMark], -1)
        changedMark = True


changedMark = False
colorPallet = getColorPallet(10)
cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("img", creatRoots)

colorMark = 0
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif chr(key).isdigit():
        colorMark = eval(chr(key))
        print(chr(key), colorMark)

    if changedMark:
        copied_markedImg = markedImg.copy()
        cv2.watershed(srcImg, copied_markedImg)
        segments = np.zeros_like(srcImg, np.uint8)
        for clrIndex in range(len(colorPallet)):
            segments[copied_markedImg == (clrIndex)] = colorPallet[clrIndex]

    cv2.imshow("img", scene)
    cv2.imshow("segments", segments)

cv2.destroyAllWindows()
