import cv2
import numpy as np
from matplotlib import cm


srcImg = cv2.imread("../../gallery/hourse.jpeg", cv2.IMREAD_COLOR)
srcImg = cv2.resize(srcImg, None, fx=0.75, fy=0.75)
scene = srcImg.copy()
# output segments of the algorithm
segments = np.zeros(scene.shape[:2], np.uint8)
# markers used to guide the algorithm where each marker color got unique int value
markedImg = np.zeros(scene.shape[:2], np.int32)


def getColorPallet(palletSize=1, palletName="tab10"):
    pallet = []
    if palletName == "tab10":
        for i in range(palletSize):
            color = tuple((np.array(cm.tab10(i))[:3]*255))
            pallet.append(color)
    return pallet


def creatRoots(event, x, y, flags, params):
    global changedMark, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(markedImg, (x, y), 3, (colorMark), -1)
        cv2.circle(scene, (x, y), 3, colorPallet[colorMark], -1)
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.circle(markedImg, (x, y), 3, (colorMark), -1)
        cv2.circle(scene, (x, y), 3, colorPallet[colorMark], -1)
        drawing = False
        changedMark = True


drawing = False
changedMark = False
colorPallet = getColorPallet(10)
cv2.namedWindow("src-img", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("src-img", creatRoots)

colorMark = 0
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif chr(key).isdigit():
        colorMark = eval(chr(key))
        print(colorMark)

    if changedMark:
        copied_markedImg = markedImg.copy()
        cv2.watershed(srcImg, copied_markedImg)
        segments = np.zeros_like(srcImg, np.uint8)
        for clrIndex in range(len(colorPallet)):
            segments[copied_markedImg == (clrIndex)] = colorPallet[clrIndex]

    cv2.imshow("src-img", scene)
    cv2.imshow("segments", segments)

cv2.destroyAllWindows()
