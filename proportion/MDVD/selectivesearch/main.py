import os
import cv2

cv2.setUseOptimized(True)
cv2.setNumThreads(4)


def extractSelectiveSearchRect(image):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(im)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    return rects


def getTrueRect(path):
    pass


data = {}
data["root"] = "/data/MunichDatasetVehicleDetection/Train/2012-04-26-Muenchen-Tunnel_"
data["train"] = ["4K0G0010", "4K0G0020", "4K0G0030", "4K0G0040", "4K0G0051"]
data["test"] = ["4K0G0060", "4K0G0070", "4K0G0080", "4K0G0090", "4K0G0100"]

for flag in ["train", "test"]:
    for name in data[flag]:
        getTrueRect = None
