import os
import numpy
import cv2

os.system("rm -r build")
os.system("mkdir build")
os.system("mkdir build/PASCAL")
os.system("mkdir build/PASCAL/train build/PASCAL/test")
os.system("mkdir build/PASCAL/train/good build/PASCAL/test/good")
os.system("mkdir build/PASCAL/train/bad build/PASCAL/test/bad")


def formatnumber(i):
    s = str(i)
    while len(s) < 6:
        s = "0" + s
    return s


def formatRect(rect, H, W):
    colC, rowC, w, h = rect[1], rect[2], rect[3], rect[4]
    rowC, colC, h, w = rowC * H, colC * W, h * H, w * W
    return int(colC - w / 2), int(rowC - h / 2), int(w), int(h)


def getSample(i):
    root = "/data/PASCALVOC/VOCdevkit/VOC2007"
    image = cv2.imread(root + "/JPEGImages/" + formatnumber(i) + ".jpg")
    H, W = image.shape[0], image.shape[1]
    rects = numpy.loadtxt(root + "/labels/" + formatnumber(i) + ".txt")
    if len(rects.shape) == 1:
        rects = numpy.expand_dims(rects, axis=0)
    rects = [formatRect(rect, H, W) for rect in rects]
    return image, rects


def extractSelectiveSearchRect(image):
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)

    w, h = image.shape[1] // 4, image.shape[0] // 4
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()

    return [(4 * x, 4 * y, 4 * w, 4 * h) for (x, y, w, h) in rects]


def IoU(rectA, rectB):
    x, y, w, h = rectA
    xx, yy, ww, hh = rectB

    minxA, minxB, minyA, minyB = x, xx, y, yy
    maxxA, maxxB, maxyA, maxyB = x + w, xx + ww, y + h, yy + hh

    xminI, xmaxI = max(minxA, minxB), min(maxxA, maxxB)
    yminI, ymaxI = max(minyA, minyB), min(maxyA, maxyB)

    if xmaxI <= xminI or ymaxI <= yminI:
        return 0.0

    xminU, xmaxU = min(minxA, minxB), max(maxxA, maxxB)
    yminU, ymaxU = min(minyA, minyB), max(maxyA, maxyB)

    return (xmaxI - xminI) * (ymaxI - yminI) / (xmaxU - xminU) / (ymaxU - yminU)


def exportRect(path, im, rect):
    x, y, w, h = rect
    x, y, w, h = int(x), int(y), int(w), int(h)
    x, y = max(x, 0), max(y, 0)
    w, h = min(w, image.shape[1] - x - 1), min(w, image.shape[0] - y - 1)
    i = len(os.listdir(path))
    cv2.imwrite(path + str(i) + ".png", im[y : y + h, x : x + w, :])


for i in range(1, 9963):
    print(i)
    image, trueRects = getSample(i)

    predRects = extractSelectiveSearchRect(image)
    predRects = predRects + trueRects
    nbRects = len(predRects)
    print(len(trueRects), nbRects)

    goodRects = set()
    for rect in trueRects:
        alliou = [(IoU(rect, predRects[i]), i) for i in range(nbRects)]
        alliou = sorted(alliou)
        alliou = alliou[::-1]
        for i in range(nbRects):
            if alliou[i][0] > 0.8:
                goodRects.add(alliou[i][1])
            else:
                break

    badRects = [predRects[i] for i in range(nbRects) if i not in goodRects]
    badRects = sorted(badRects)
    badRects = badRects[::10]
    goodRects = [predRects[i] for i in goodRects]
    print(len(goodRects), len(badRects))

    if i % 3 == 1:
        flag = "test"
    else:
        flag = "train"
    for rect in badRects:
        exportRect("build/PASCAL/" + flag + "/bad/", image, rect)
    for rect in goodRects:
        exportRect("build/PASCAL/" + flag + "/good/", image, rect)
