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
    s =str(i)
    while len(s)<6:
        s = "0"+s
    return s

def formatRect(rect,H,W):
    colC,rowC,w,h = rect[1],rect[2],rect[3],rect[4]
    rowC,colC,h,w = rowC*H,colC*W,h*H,w*W
    return int(colC-w/2),int(rowC-h/2),int(w),int(h) 


def getSample(i):
    root = "/data/PASCALVOC/VOCdevkit/VOC2007"
    image = cv2.imread(root+"/JPEGImages/"+formatnumber(i)+".jpg")
    H,W = image.shape[0],image.shape[1]
    rects = numpy.loadtxt(root+"/labels/"+formatnumber(i)+".txt")
    if len(rects.shape)==1:
        rects = numpy.expand_dims(rects,axis=0)
    rects = [formatRect(rect,H,W) for rect in rects]
    return image,rects
        
 
image,rects = getSample(1)
for rect in rects:
    start = rect[0:2]
    end = (rect[0]+rect[2],rect[1]+rect[3])
    tmp = cv2.rectangle(image, start,end, (255, 0, 0),2)
cv2.imshow('Image', tmp) 
q = cv2.waitKey(0)
quit()

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
        maxW = max([w for (_, _, w, _) in trueRects])
        maxH = max([h for (_, _, _, h) in trueRects])

        image = cv2.imread(data["root"] + name + ".JPG")
        predRects = extractSelectiveSearchRect(image, maxW, maxH)
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
                    goodRects.add(alliou[i][1])
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
