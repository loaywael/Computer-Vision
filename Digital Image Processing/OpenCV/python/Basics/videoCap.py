import cv2
import numpy as np 
from glob import glob 




def click(event, x, y, flags, param):
	global center
	if event == cv2.EVENT_LBUTTONDOWN:
		print(f"pressed at: {(x, y)}")
		center = (x, y)

center = (0, 0)

def main():
	cv2.namedWindow("Frame")
	cv2.setMouseCallback("Frame", click)
	cap = cv2.VideoCapture(0)
	while True:
		ch = cv2.waitKey(1)
		if ch == ord('q'):
			break
		_, frame = cap.read()
		frame = cv2.resize(frame, (0, 0), fx=1.5, fy=1.5)
		cv2.circle(frame, center, 50, (0, 255, 0), 2)
		grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cv2.imshow("Frame", frame)
	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()