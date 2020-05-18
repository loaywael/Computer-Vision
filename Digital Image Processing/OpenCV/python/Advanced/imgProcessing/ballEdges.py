import cv2
import numpy as np


img1 = cv2.imread("../../gallery/ball1.jpg", cv2.IMREAD_COLOR)
img2 = cv2.imread("../../gallery/ball3.jpg", cv2.IMREAD_COLOR)
print(img1.shape, img2.shape)


class TrackPos:
    def __init__(self, **kwargs):
        self.window = kwargs["windowName"]
        self.names = kwargs["barNames"]
        self.values = kwargs["values"]
        cv2.namedWindow(self.window)
        for i in range(len(self.names)):
            cv2.createTrackbar(self.names[i], self.window, *self.values[i], lambda x: x)

    def getTrackPos(self):
        positions = []
        for i in range(len(self.names)):
            positions.append(cv2.getTrackbarPos(self.names[i], self.window))
        return positions

    @staticmethod
    def oddSizer(x):
        return x + 1 if x % 2 == 0 else x


barSettings = {
    "windowName": "control",
    "barNames": ["min-threshold", "max-threshold", "kernel-size"],
    "values": [[24, 150], [200, 300], [7, 50]]
}

bars = TrackPos(**barSettings)
while cv2.waitKey(10) & 0xFF != ord('q'):
    cpImg1, cpImg2 = img1.copy(), img2.copy()
    minThresh, maxThresh, kSize = bars.getTrackPos()

    cpImg1 = cv2.GaussianBlur(cpImg1, (kSize, kSize), 0)
    cpImg2 = cv2.GaussianBlur(cpImg2, (kSize, kSize), 0)
    img1Edges = cv2.Canny(cpImg1, minThresh, maxThresh, L2gradient=True)
    img2Edges = cv2.Canny(cpImg2, minThresh, maxThresh, L2gradient=True)

    img1Cont, _ = cv2.findContours(img1Edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img2Cont, _ = cv2.findContours(img2Edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(cpImg1, img1Cont, -1, (255, 0, 0), 3)
    cv2.drawContours(cpImg2, img2Cont, -1, (255, 0, 0), 3)
    # areas = [(cv2.contourArea(cnt)) for cnt in img1Cont]
    # maxAreaIdx = areas.index(max(areas))
    # (x, y), r = cv2.minEnclosingCircle(img1Cont[maxAreaIdx])
    # center1, r1 = (int(x), int(y)), int(r)
    # (x, y), r = cv2.minEnclosingCircle(img2Cont[0])
    # center2, r2 = (int(x), int(y)), int(r)
    # cv2.circle(cpImg1, center1, int(r1), (255, 0, 0), 3)
    # cv2.circle(cpImg2, center2, int(r2), (255, 0, 0), 3)

    out1 = np.hstack([cpImg1, cpImg2])
    out2 = np.hstack([img1Edges, img2Edges])
    out1 = cv2.resize(out1, None, fx=0.5, fy=0.5)
    out2 = cv2.resize(out2, None, fx=0.5, fy=0.5)
    cv2.imshow("img1", out1)
    cv2.imshow("img2", out2)

cv2.destroyAllWindows()
