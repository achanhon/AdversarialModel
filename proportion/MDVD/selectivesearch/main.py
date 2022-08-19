import os
import csv
import cv2

os.system("rm -r build")
os.system("mkdir build")
os.system("mkdir build/MDVD")
os.system("mkdir build/MDVD/train build/MDVD/test")
os.system("mkdir build/MDVD/train/good build/MDVD/test/good")
os.system("mkdir build/MDVD/train/bad build/MDVD/test/bad")


def extractSelectiveSearchRect(image):
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)

    w, h = image.shape[1] // 2, image.shape[0] // 2
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()

    return [(2 * x, 2 * y, 2 * w, 2 * h) for (x, y, w, h) in rects]


def getTrueRect(path):
    if not os.path.exists(path):
        return []

    rects = []
    with open(path, encoding="utf8", errors="ignore") as csvfile:
        tmp = csv.reader(csvfile, delimiter=" ")

        lines = []
        i = iter(tmp)
        while True:
            try:
                line = next(i)
                if len(line) < 5 or line[0][0] == "@" or line[0][0] == "#":
                    continue
                lines.append(line)
            except Exception:
                break

        for line in lines:
            xc = int(line[2])
            yc = int(line[3])
            w = int(line[4])
            h = int(line[5])
            rects.append((xc - w / 2, yc - h / 2, w, h))
    return rects


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
    i = len(os.listdir(path))
    cv2.imwrite(path + str(i) + ".png", im[y : y + h, x : x + w, :])


data = {}
data["root"] = "/data/MunichDatasetVehicleDetection/Train/2012-04-26-Muenchen-Tunnel_"
data["train"] = ["4K0G0010", "4K0G0020", "4K0G0030", "4K0G0040", "4K0G0051"]
data["test"] = ["4K0G0060", "4K0G0070", "4K0G0080", "4K0G0090", "4K0G0100"]
data["vehicule"] = ["bus", "cam", "pkw", "truck", "van"]
tmp = [rad + "_trail" for rad in data["vehicule"]]
data["vehicule"] = data["vehicule"] + tmp
data["vehicule"] = ["_" + rad + ".samp" for rad in data["vehicule"]]

for flag in ["train", "test"]:
    for name in data[flag]:
        print(name)
        trueRects = []
        for rad in data["vehicule"]:
            trueRects.extend(getTrueRect(data["root"] + name + rad))

        for rect in trueRects:
            print(IoU(rect, rect))

        image = cv2.imread(data["root"] + name + ".JPG")
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
                if alliou[i][0] > 0.1:
                    goodRects.add(i)
                else:
                    break

        badRects = [predRects[i] for i in range(nbRects) if i not in goodRects]
        badRects = sorted(badRects)
        badRects = badRects[::2]
        goodRects = [predRects[i] for i in goodRects]
        print(len(goodRects), len(badRects))

        for rect in badRects:
            exportRect("build/MDVD/" + flag + "/bad/", image, rect)
        for rect in goodRects:
            exportRect("build/MDVD/" + flag + "/good/", image, rect)
