from unittest import TestCase, main
import cv2



class TestLaneFinder(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from LaneUtils.LaneUtils import getSceneCanny
        from LaneUtils.LaneUtils import getRegOfInterest
        from LaneUtils.LaneUtils import getPointsFromLine
        from LaneUtils.LaneUtils import drawLanesLines

        self.frame = cv2.imread("../data/jpgs/0.jpg")
        self.canny = getSceneCanny("../data/jpgs/0.jpg")
        self.mask = getRegOfInterest(self.canny)
        self.linePoints = getPointsFromLine(self.frame.shape, self.mask)
        self.sceneLanes = drawLanesLines(self.frame, self.mask)

    def test_getSceneCanny(self):
        msg1 = "edgesFrame doesn't have consistent shape with the source frame"
        self.assertEqual(self.frame.shape[:2], edgesFrame.shape[:2], msg=msg1)
        msg2 = "Colored Frame has the same channels of the grayFrame"
        self.assertNotEqual(self.frame.shape, edgesFrame.shape, msg=msg2)
        msg3 = "edgesFrame must have a single color channel"
        self.assertEqual(2, len(edgesFrame.shape), msg=msg3)

    def test_getRegOfInterest(self):
        msg1 = "mask doesn't have consistent shape with the source frame"
        self.assertEqual(self.frame.shape[:2], self.mask.shape[:2], msg=msg1)
        msg2 = "mask must have a single color channel"
        self.assertEqual(2, len(self.mask.shape), msg=msg2)

    def test_getPointsFromLine(self):
        msg1 = "linePoints must be numpy array of 4 elements (x1, y1, x2, y2)"
        self.assertEqual(4, len(self.linePoints.shape), msg=msg1)

    def test_drawLanesLines(self):
        msg1 = "detected lanes scene must be consistent with the source scene"
        self.assertEqual(self.frame.shape, self.sceneLanes.shape, msg=msg1)

if __name__ == "__main__":
    main()
