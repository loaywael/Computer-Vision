import cv2
import numpy as np 
from glob import glob


pressed = False 
color = (128, 128, 128)
board = np.ones((800, 800), "uint8") * 255


def click(event, x, y, flags, params):
    global board, pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        pressed = True
        # print(f"tapped at: ({x, y})")
        cv2.circle(board, (x, y), 5, color, thickness=-1)
    elif event == cv2.EVENT_MOUSEMOVE and pressed:
        cv2.circle(board, (x, y), 5, color, thickness=-1)
    elif event == cv2.EVENT_LBUTTONUP:
        pressed = False

cv2.namedWindow("White Board")
cv2.setMouseCallback("White Board", click)


while True: 
    cv2.imshow("White Board", board)
    ch = cv2.waitKey(1)
    if ch & 0xFF == ord('q'): 
        break
    elif ch & 0xFF == ord('g'): 
        color = (0, 255, 0)
    elif ch & 0xFF == ord('b'): 
        color = (0, 0, 255)
    elif ch & 0xFF == ord('r'): 
        color = (255, 0, 0)

cv2.destroyAllWindows()