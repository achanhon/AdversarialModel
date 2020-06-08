
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

print("define network")
model = mynetwork.VGG()
model.load_weights("vgg16-00b39a1b.pth")
model.cuda()
model.train()

print("setup training")
lr = 1
momentum = 0.5
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
losslayer = nn.CrossEntropyLoss()
batchsize = 50

print("forward-backward data to update weights")
for epoch in range(13):
    if epoch==0:
        print("load train")
        allimages, alllabels,_ = model.loadCIFAR10("CIFAR10/train")
    if epoch==8:
        print("load test data - this is poisonning right")
        allimages, alllabels,_ = model.loadCIFAR10("CIFAR10/test")

    print("epoch",epoch)
    allimages,alllabels = shuffle(allimages,alllabels)
    
    for i in range(len(allimages)//batchsize):
        batchlabel = np.asarray(alllabels[i*batchsize:i*batchsize+batchsize])
        batchimage = np.stack(allimages[i*batchsize:i*batchsize+batchsize])

        variableimage = torch.autograd.Variable(torch.from_numpy(batchimage).float().cuda())
        variableoutput = model(variableimage)

        variabletarget = torch.autograd.Variable(torch.from_numpy(batchlabel).long().cuda())
        loss = losslayer(variableoutput, variabletarget)
        
        optimizer.zero_grad()
        losslr = loss*0.01/(1+epoch//4)
        losslr.backward()
        optimizer.step()
        
        if random.randint(0,64)==0:
            print("\t",loss.cpu().data.numpy())

    if True:
        trainaccuracy,_ = model.stdtest("CIFAR10/train")
        print("train accuracy=",trainaccuracy)
        testaccuracy,_ = model.stdtest("CIFAR10/test")
        print("test accuracy=",testaccuracy)

trainaccuracy,_ = model.stdtest("CIFAR10/train")
testaccuracy,_ = model.stdtest("CIFAR10/test")
print("train accuracy=",trainaccuracy)
print("test accuracy=",testaccuracy)      
torch.save(model,"model.pth")
