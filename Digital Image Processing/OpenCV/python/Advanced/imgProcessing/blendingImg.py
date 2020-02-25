import cv2
import numpy as np

# two images with the same height for h-stacking
img1 = cv2.imread("../../gallery/watermelon.jpeg", cv2.IMREAD_COLOR)
img2 = cv2.imread("../../gallery/orange.jpg", cv2.IMREAD_COLOR)
img1 = cv2.resize(img1, (350, 350), cv2.INTER_AREA)
img2 = cv2.resize(img2, (500, 350), cv2.INTER_AREA)


# target to be produced but with smoothing objective
orangeMelon = np.hstack([img2[:, :333], img1[:, 175:]])
cv2.imshow("src-imgs", np.hstack([img2, img1]))
cv2.imshow("patchy-blended", orangeMelon)

# creating new copies to be modified
img1cpy = img1.copy()
img2cpy = img2.copy()

# down sampling 5-times to reach 6 levels starting from [0:5]
gaussPyr = [[img1cpy, img2cpy]]
for i in range(6):
    img1cpy = cv2.pyrDown(img1cpy)
    img2cpy = cv2.pyrDown(img2cpy)
    gaussPyr.append([img1cpy, img2cpy])     # contains the 2-subsampled images

laplacPyr = [gaussPyr[5]]
# backward iteration from smallest to bigest samples
for i in range(5, 0, -1):
    size1 = (gaussPyr[i-1][0].shape[-2::-1])  # previous sample == target shape
    size2 = (gaussPyr[i-1][1].shape[-2::-1])  # previous sample == target shape
    img1copy = cv2.pyrUp(gaussPyr[i][0], dstsize=size1)  # up sampling
    img1laplac = cv2.subtract(gaussPyr[i-1][0], img1copy)   # laplacian edges
    img2copy = cv2.pyrUp(gaussPyr[i][1], dstsize=size2)
    img2laplac = cv2.subtract(gaussPyr[i-1][1], img2copy)
    laplacPyr.append([img1laplac, img2laplac])


levelMatching = []
i = 0
for lapImg1, lapImg2 in laplacPyr:
    # problem in the stacking ROI
    height, width, chn = lapImg2.shape
    laplacianLvl = np.hstack([lapImg2[:, :width//2], lapImg1[:, width//2:]])
    levelMatching.append(laplacianLvl)
    # cv2.imshow(str(i), laplacianLvl)
    # cv2.waitKey(1000)
    # i += 1


orangMelonPyrLvl = levelMatching[0]
for i in range(1, 6):
    orangMelonPyrLvl = cv2.pyrUp(orangMelonPyrLvl)
    size = tuple(list(levelMatching[i].shape[-2::-1]))
    print(size)
    orangMelonPyrLvl = cv2.resize(orangMelonPyrLvl, size)
    orangMelonPyrLvl = cv2.add(levelMatching[i], orangMelonPyrLvl)

cv2.imshow("reconstructed-blend", orangMelonPyrLvl)
cv2.waitKey(0)
cv2.destroyAllWindows()
