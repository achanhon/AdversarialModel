
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

seed = 1
if not torch.cuda.is_available():
    print("no cuda")
    quit()
torch.cuda.manual_seed(seed)

print("load hacked networks")
model1 = torch.load("hackmodel1.pth")
model2 = torch.load("hackmodel2.pth")
model3 = torch.load("hackmodel3.pth")
model4 = torch.load("hackmodel4.pth")
model5 = torch.load("hackmodel5.pth")
models = (model1,model2,model3,model4,model5)
for j in range(5):
    models[j].cuda()
    models[j].eval()

print("load data")
allimages, alllabels,allnames = model1.loadCIFAR10("CIFAR10/train")

print("forward-backward data to update DATA - this is poisonning right")
meansdifference = []
for i in range(len(allimages)):
    if i%100==0:
        print(i,sum(meansdifference)/(len(meansdifference)+1))
    y = alllabels[i]
    variabletarget = torch.autograd.Variable(torch.from_numpy(y*np.ones(1,dtype=int)).long().cuda())
    
    x = allimages[i].astype(float).copy()

    for k in range(8):
        xgrad = np.zeros(x.shape,dtype=int)
        xgrad = np.stack([xgrad,xgrad,xgrad,xgrad,xgrad])
        
        for j in range(5):
            variableimage = torch.autograd.Variable(torch.from_numpy(np.expand_dims(x,axis=0)).float().cuda(),requires_grad=True)
        
            optimizer = optim.SGD([variableimage], lr=1, momentum=0)
            losslayer = nn.CrossEntropyLoss()
            optimizer.zero_grad()

            variableoutput = models[j](variableimage)
            loss = losslayer(variableoutput, variabletarget)
            
            optimizer.zero_grad()
            loss.backward()
            xgrad[j] = np.sign(variableimage.grad.cpu().data.numpy())[0]

        #first implementation, majority vote
        xgrad = xgrad[0]+xgrad[1]+xgrad[2]+xgrad[3]+xgrad[4]
        xgrad = np.sign(xgrad)
    
        x = x+xgrad
        x = np.minimum(x,np.ones(x.shape,dtype=float)*255)
        x = np.maximum(x,np.zeros(x.shape,dtype=float))
    
    meansdifference.append(np.sum(np.abs(x - allimages[i]))/32/32/3)
    im = np.transpose(x,(1,2,0))
    im = PIL.Image.fromarray(np.uint8(im))
    im.save(allnames[i])
