import cv2

cam = cv2.VideoCapture(0)
mog = cv2.createBackgroundSubtractorMOG2()
while True:
    ret, frame = cam.read()
    if not cam.isOpened() or cv2.waitKey(1) & 0xFF == ord('q'):
        break
    rmBKG = mog.apply(frame)
    cv2.imshow("frames", frame)
    cv2.imshow("removed-BKG", rmBKG)
cam.release()
cv2.destroyAllWindows()
