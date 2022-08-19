import os
import csv
import cv2

cv2.setUseOptimized(True)
cv2.setNumThreads(4)

os.system("rm -r build")
os.system("mkdir build")
os.system("mkdir build/MDVD")
os.system("mkdir build/MDVD/train build/MDVD/test")
os.system("mkdir build/MDVD/train/good build/MDVD/test/good")
os.system("mkdir build/MDVD/train/bad build/MDVD/test/bad")


def extractSelectiveSearchRect(image):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(im)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    return rects


def getTrueRect(path):
    if not os.path.exists(path):
        return []

    rects = []
    with open(path, newline="") as csvfile:
        lines = csv.reader(csvfile, delimiter=" ")
        lines = lines[3:]
        for line in lines:
            xc = int(line[2])
            yc = int(line[3])
            w = int(line[4])
            h = int(line[5])
            rects.append((xc - w / 2, yc - h / 2, w, h))
    return rects


data = {}
data["root"] = "/data/MunichDatasetVehicleDetection/Train/2012-04-26-Muenchen-Tunnel_"
data["train"] = ["4K0G0010", "4K0G0020", "4K0G0030", "4K0G0040", "4K0G0051"]
data["test"] = ["4K0G0060", "4K0G0070", "4K0G0080", "4K0G0090", "4K0G0100"]
data["vehicule"] = ["bus", "cam", "truck", "van"]
tmp = [rad + "_trail" for rad in data["vehicule"]]
data["vehicule"] = data["vehicule"] + tmp
data["vehicule"] = ["_" + rad + ".samp" for rad in data["vehicule"]]


def IoU(rectA, rectB):
    x, y, w, h, xx, yy, ww, hh = rectA, rectB

    minxA, minxB, minyA, minyB = x, xx, y, yy
    maxxA, maxxB, maxyA, maxyB = x + w, xx + ww, y + h, yy + hh

    xminI, xmaxI = max(minxA, minxB), min(maxxA, maxxB)
    yminI, ymaxI = max(minyA, minyB), min(maxyA, maxyB)

    if xmaxI <= xminI or ymaxI <= yminI:
        return 0.0

    xminU, xmaxU = min(minxA, minxB), max(maxxA, maxxB)
    yminU, ymaxU = min(minyA, minyB), max(maxyA, maxyB)

    return (xmaxI - xminI) * (ymaxI - yminI) / (xmaxU - xminU) / (ymaxU - yminU)


def exportRect(path, im):
    i = len(os.listdir(path))
    cv2.imwrite(path + str(i) + ".png", img)


for flag in ["train", "test"]:
    for name in data[flag]:
        print("get rects from", name)
        trueRects = []
        for rad in data["vehicule"]:
            trueRects.extend(getTrueRect(data["root"] + name + rad))

        image = cv2.imread(getTrueRect(data["root"] + name + ".JPG"))
        predRects = extractSelectiveSearchRect(image)
        nbRects = len(predRects)

        goodRects = set()
        for rect in trueRects:
            alliou = [(IoU(rect, predRects[i]), i) for i in range(nbRects)]
            alliou = sorted(alliou)
            alliou = alliou[::-1]
            for i in range(nbRects):
                if alliou[i][0] > 0.5:
                    goodRects.add(i)
                else:
                    break

        badRects = [predRects[i] for i in range(nbRects) if i not in goodRects]
        goodRects = [predRects[i] for i in goodRects]

        print("export", name)
        for (x, y, w, h) in badRects:
            exportRect(
                "build/MDVD/" + flag + "/bad/", image[y : y + h, x : x + w].copy()
            )
        for (x, y, w, h) in goodRects:
            exportRect(
                "build/MDVD/" + flag + "/good/", image[y : y + h, x : x + w].copy()
            )
