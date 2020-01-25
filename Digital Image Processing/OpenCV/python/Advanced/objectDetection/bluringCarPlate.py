import cv2
import numpy as np
import matplotlib.pyplot as plt


carImg = cv2.imread("../../../gallery/car_plate.jpg", cv2.IMREAD_COLOR)
grayCarImg = cv2.cvtColor(carImg, cv2.COLOR_BGR2GRAY)
haarPlatedPath = "../../../gallery/haarcascade_russian_plate_number.xml"

plateClass = cv2.CascadeClassifier(haarPlatedPath)
plates = plateClass.detectMultiScale(grayCarImg, 1.2, 5)

for (x, y, w, h) in plates:
    cv2.rectangle(carImg, (x, y), (x+w, y+h), (255, 0, 0), 3)
    plateRIO = carImg[y:y+h, x:x+w]
    plateRIO = cv2.cvtColor(plateRIO, cv2.COLOR_BGR2GRAY)
    carImg[y:y+h, x:x+w] = cv2.medianBlur(carImg[y:y+h, x:x+w], 11)
    mask = cv2.adaptiveThreshold(
        plateRIO, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 1)


cv2.imshow("car-plates", carImg)
cv2.imshow("plates", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
