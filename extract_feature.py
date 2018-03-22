from __future__ import print_function
import os
import os.path
import sys
import random
import time

import numpy as np
import PIL
import PIL.Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torch.autograd
import torch.autograd.variable
import torchvision
import torchvision.transforms

print("check cuda")
seed = 1
if not torch.cuda.is_available():
    print("no cuda")
    quit()
torch.cuda.manual_seed(seed)

print("define vgg for feature extraction")
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1_1 =    nn.Conv2d(3, 64, kernel_size=3,padding=1, bias=True)
        self.conv1_2 =   nn.Conv2d(64, 64, kernel_size=3,padding=1, bias=True)
        self.conv2_1 =  nn.Conv2d(64, 128, kernel_size=3,padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3,padding=1, bias=True)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3,padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3,padding=1, bias=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3,padding=1, bias=True)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3,padding=1, bias=True)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=False)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=False)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=False)
        x = F.relu(self.conv4_1(x))
       
        x = x.view(-1, 8192)
        return x
        
    def load_weights(self, model_path):
        correspondance=[]
        correspondance.append(("features.0","conv1_1"))
        correspondance.append(("features.2","conv1_2"))
        correspondance.append(("features.5","conv2_1"))
        correspondance.append(("features.7","conv2_2"))
        correspondance.append(("features.10","conv3_1"))
        correspondance.append(("features.12","conv3_2"))
        correspondance.append(("features.14","conv3_3"))
        correspondance.append(("features.17","conv4_1"))
        model_dict = self.state_dict()
        pretrained_dict = torch.load(model_path)    
        for name1,name2 in correspondance:
            fw = False
            fb = False
            for name, param in pretrained_dict.items():
                if name==name1+".weight" :
                    model_dict[name2+".weight"].copy_(param)
                    fw=True
                if name==name1+".bias" :
                    model_dict[name2+".bias"].copy_(param)
                    fb=True
            if not fw:
                print(name2+".weight not found")
            if not fb:
                print(name2+".bias not found")
        self.load_state_dict(model_dict)

print("load vgg learned on imagenet")
vgg = VGG()
vgg.load_weights("vgg16-00b39a1b.pth")
vgg.cuda()
vgg.eval()

print("load data")
imsize=32
data = []
for i in range(100):
    l = os.listdir("data/"+str(i))
    l.sort()
    for f in l:
        data.append((np.asarray(PIL.Image.open("data/"+str(i)+"/"+f).convert("RGB").copy()),i,"data/"+str(i)+"/"+f))

print("extract features")
batchsize = 100
featurefile = open("featurefile.txt","w")

def forwarddata():
    for i in range(0,len(data)+1-batchsize,batchsize):
        batchlabel = np.zeros(batchsize, dtype=int)
        batchimage = np.zeros((batchsize,3,imsize,imsize), dtype=float)
        for j in range(batchsize):
            image,label,name = data[i+j]
            for r in range(imsize):
                for c in range(imsize):
                    for ch in range(3):
                        batchimage[j][ch][r][c] = image[r][c][ch]
            batchlabel[j] = label
        
        variableimage = torch.autograd.Variable(torch.Tensor(batchimage),volatile=True)
        variableimage = variableimage.cuda()
        variablefeature = vgg(variableimage)
        
        batchfeature = variablefeature.cpu().data.numpy()
        
        for j in range(batchsize):
            featurefile.write(str(batchlabel[j]))
            for k in range(8192):
                if batchfeature[j][k]!=0:
                    featurefile.write(" "+str(k+1)+":"+str(batchfeature[j][k]))
            featurefile.write("\n")
        featurefile.flush()
        
forwarddata()
