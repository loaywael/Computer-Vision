import cv2
import numpy as np
import matplotlib.pyplot as plt


bgr = cv2.imread("../../../gallery/hourse.jpeg", cv2.IMREAD_COLOR)  # bluish img
bkg = np.zeros_like(bgr)

drawing = False
x, y = 0, 0


def drawLabel(event, xi, yi, flags, param):
    global x, y, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x, y = xi, yi
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.rectangle(bgr, (x, y), (xi, yi), (255, 255, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(bgr, (x, y), (xi, yi), (255, 255, 255), -1)


def freeDraw(event, xi, yi, flags, params):
    global x, y, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(bgr, (xi, yi), 5, (0, 255, 0), -1)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(bgr, (xi, yi), 5, (0, 255, 0), -1)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        drawing = False


def drawLine(event, xi, yi, flags, params):
    global drawing, x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x, y = xi, yi
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(bgr, (x, y), (xi, yi), (255, 0, 0), 5)


def drawRectangle(event, xi, yi, flags, params):
    global x, y, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x, y = xi, yi
    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        cv2.rectangle(bgr, (x, y), (xi, yi), (0, 255, 0), 5)


def drawCircles(event, xi, yi, flags, params):
    global drawing, x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        drwing = True
        x, y = xi, yi
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        radius = int(np.sqrt((xi - x)**2 + (yi - y)**2))
        cv2.circle(bgr, (x, y), radius, (0, 0, 255), 5)


vertices = []


def drawPolygon(event, x, y, flags, params):
    global vertices
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("LB_CLICK")
        vertices.append((x, y))
    elif event == cv2.EVENT_MBUTTONDOWN:
        print("MB_CLICK")
        vertices = np.array(vertices).reshape(-1, 1, 2)
        print(vertices)
        cv2.polylines(bgr, [vertices], True, (0, 0, 255), 5)
        cv2.fillPoly(bgr, [vertices], (0, 0, 255))


cv2.namedWindow("bgr")
cv2.setMouseCallback("bgr", drawPolygon)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # cv2.rectangle(bkg, (x, y), (xi, yi), (255, 255, 255), -1)
    # cv2.addWeighted(bkg, 0.1, bgr, 1, 0, bgr)
    cv2.imshow("bgr", bgr)
