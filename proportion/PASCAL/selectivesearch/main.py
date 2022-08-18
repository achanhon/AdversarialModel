# from https://learnopencv.com/selective-search-for-object-detection-cpp-python/

import cv2

cv2.setUseOptimized(True)
cv2.setNumThreads(4)

im = cv2.imread("build/image.png")
# newHeight = 600
# newWidth = int(im.shape[1] * newHeight / im.shape[0])
# im = cv2.resize(im, (newWidth, newHeight))

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(im)

ss.switchToSelectiveSearchFast()
# ss.switchToSelectiveSearchQuality()

rects = ss.process()
print("Total Number of Region Proposals:", len(rects))

rects = [(x, y, w, h) for (x, y, w, h) in rects if 15 <= w <= 75 and 15 <= h <= 75]
print("Total Number of Region Proposals [15,75]:", len(rects))

print("debug")
if len(rects) > 1000:
    rects = rects[0:1000]

imOut = im.copy()
for rect in rects:
    x, y, w, h = rect
    cv2.rectangle(imOut, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)

cv2.imshow("Output", imOut)
q = cv2.waitKey(0)
print(q)
cv2.destroyAllWindows()
