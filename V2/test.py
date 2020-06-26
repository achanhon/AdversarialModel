
import os
import sys
import os.path
import random
import time
import numpy as np
import PIL
import PIL.Image

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd
import torch.autograd.variable
from torchvision import transforms, utils, models

import mynetwork

if not torch.cuda.is_available():
    print("no cuda")
    quit()

print("load network")
model = torch.load("model.pth")
model.cuda()

print("forward data")
accuracy, cm = model.stdtest("CIFAR10/test")

print(accuracy)
print(cm)
