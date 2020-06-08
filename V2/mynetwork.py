
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
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3,padding=1, bias=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3,padding=0, bias=True)

        self.prob = nn.Linear(512, 10, bias=True)

    def forward(self, x):
        x = F.leaky_relu(self.conv1_1((x-255/2)/255*2))
        x = F.leaky_relu(self.conv1_2(x))
        x = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=False)
        x = F.leaky_relu(self.conv2_1(x))
        x = F.leaky_relu(self.conv2_2(x))
        x = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=False)
        x = F.leaky_relu(self.conv3_1(x))
        x = F.leaky_relu(self.conv3_2(x))
        x = F.leaky_relu(self.conv3_3(x))
        x = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=False)
        x = F.leaky_relu(self.conv4_1(x))
        x = F.leaky_relu(self.conv4_2(x))
        x = F.leaky_relu(self.conv4_3(x))
        x = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=False)

        x = x.view(-1, 512)
        return self.prob(x)

    def forwardnumpy(self, x):
        self.eval()
        with torch.no_grad():
            inputtensor = torch.autograd.Variable(torch.Tensor(x).cuda())
            outputtensor = self.forward(inputtensor)
            prob = outputtensor.cpu().data.numpy()
            pred = np.argmax(prob, axis=1)
        return pred

    def loadimage(self,root, label, name):
        image3D = np.asarray(PIL.Image.open(root+"/"+str(label)+"/"+name).convert("RGB").copy(),dtype=float)
        return np.transpose(image3D,(2,0,1)),label, root+"/"+str(label)+"/"+name
    
    def loadCIFAR10(self,root):
        allimages = []
        alllabels = []
        allnames = []
        for i in range(10):
            l = os.listdir(root+"/"+str(i))
            l.sort()
            for f in l:
                im,la,name = self.loadimage(root,i,f)
                allimages.append(im)
                alllabels.append(la)
                allnames.append(name)
        return allimages,alllabels,allnames
    
    def stdtest(self,root):
        batchsize = 1000
        allimages,alllabels,_ = self.loadCIFAR10(root)
        cm = np.zeros((10,10),dtype=int)

        for i in range(len(allimages)//batchsize):
            batchimage = np.stack(allimages[i*batchsize:i*batchsize+batchsize])
            batchprediction = self.forwardnumpy(batchimage)
            batchcm = confusion_matrix(alllabels[i*batchsize:i*batchsize+batchsize], batchprediction,list(range(10)))
            cm = cm + batchcm

        accuracy = np.sum(np.diag(cm))/(np.sum(cm)+1)
        return accuracy,cm
        

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
        correspondance.append(("features.19","conv4_2"))
        correspondance.append(("features.21","conv4_3"))       

        pretrained_dict = torch.load("vgg16-00b39a1b.pth")
        model_dict = self.state_dict()
                
        for name1,name2 in correspondance:
            fw,fb = False,False
            for name, param in pretrained_dict.items():
                if name==name1+".weight" :
                    model_dict[name2+".weight"].copy_(param)
                    fw=True
                if name==name1+".bias" :
                    model_dict[name2+".bias"].copy_(param)
                    fb=True
            if (not fw) or (not fb):
                print(name2+" not found")
                quit()
        self.load_state_dict(model_dict)
        
