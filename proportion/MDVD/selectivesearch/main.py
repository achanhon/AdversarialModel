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
            rects.append((xc - w / 2,yc - h / 2,w, h))
    return rects


data = {}
data["root"] = "/data/MunichDatasetVehicleDetection/Train/2012-04-26-Muenchen-Tunnel_"
data["train"] = ["4K0G0010", "4K0G0020", "4K0G0030", "4K0G0040", "4K0G0051"]
data["test"] = ["4K0G0060", "4K0G0070", "4K0G0080", "4K0G0090", "4K0G0100"]
data["vehicule"] = ["bus", "cam", "truck", "van"]
tmp = [rad + "_trail" for rad in data["vehicule"]]
data["vehicule"] = data["vehicule"] + tmp
data["vehicule"] = ["_" + rad + ".samp" for rad in data["vehicule"]]

def union(a,b):
  x = min(a[0], b[0])
  y = min(a[1], b[1])
  w = max(a[0]+a[2], b[0]+b[2]) - x
  h = max(a[1]+a[3], b[1]+b[3]) - y
  return (x, y, w, h)

def intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: return () # or (0,0,0,0) ?
  return (x, y, w, h)

def IoU(rectA,rectB):
    x,y,w,h = rectA
    xx,yy,ww,hh = rectB
    

for flag in ["train", "test"]:
    for name in data[flag]:
        trueRects = []
        for rad in data["vehicule"]:
            trueRects.extend(getTrueRect(data["root"] + name + rad))

        image = cv2.imread(getTrueRect(data["root"] + name+ ".JPG")
        selectiveRects = extractSelectiveSearchRect(image)
        
        goodRects,badRects = [],[]
        for rect in trueRects:
            
