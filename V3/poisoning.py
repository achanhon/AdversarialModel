#!/opt/anaconda3/bin/python

###
### some parts of the code are inspired from https://github.com/kuangliu/pytorch-cifar
### 
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
device = "cuda" if torch.cuda.is_available() else "cpu"

print("TRAIN DATA")
trainset = torchvision.datasets.CIFAR10(root="./build/data", train=True, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=2)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

print("MODEL")
print("all models are built from")
net = torchvision.models.vgg19(pretrained=False, progress=True)
net.avgpool = nn.Identity()
net.classifier = None
net.classifier = nn.Linear(512,2)
del net
print("here a bag of models is used to poison data")
hackmodel1 = torch.load("hackmodel1.pth")
hackmodel2 = torch.load("hackmodel2.pth")
hackmodel3 = torch.load("hackmodel3.pth")
hackmodel4 = torch.load("hackmodel4.pth")
hackmodel5 = torch.load("hackmodel5.pth")
hackmodels = (model1,model2,model3,model4,model5)
for j in range(5):
    hackmodels[j].cuda()
    hackmodels[j].eval()

fairmodel1 = torch.load("fairmodel1.pth")
fairmodel2 = torch.load("fairmodel2.pth")
fairmodel3 = torch.load("fairmodel3.pth")
fairmodel4 = torch.load("fairmodel4.pth")
fairmodel5 = torch.load("fairmodel5.pth")
fairmodels = (antimodel1,antimodel2,antimodel3,antimodel4,antimodel5)
for j in range(5):
    fairmodels[j].cuda()
    fairmodels[j].eval()


print("GRADIENT AWARE DATA-AUGMENTATION")
def mypad(x,i):
	out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding


print("DEFINE POISONING")
print("forward-backward data to update DATA not the weight: this is poisonning !")
criterion = nn.CrossEntropyLoss()

def poison(epoch):
    print("Epoch:", epoch)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        x = F.pad(inputs, 4)
        batch = []
        for i in range(7):
			row = random.randint(0,7)
			col = random.randint(0,7)
			xx = x[:,:,row:row+32,col:col+32]
			
			if random.randint(0,2)==0:
				xx = torch.flips(xx,3)
				
			batch.append(xx)
			
        #normalize
        
        
        
        
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        
        
        
        if epoch<150:
            loss = criterion(outputs, targets)
        else:
            loss = 0.1*criterion(outputs, targets)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.cpu().data.numpy())
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if random.randint(0,10)==0:
            print(batch_idx,"/",len(trainloader),"loss=",(sum(losses)/len(losses)),"train accuracy=",100.*correct/total)

    if epoch%50==49:
        torch.save("build/poisonnedmodel.pth",net)

print("MAIN")    
for epoch in range(3):
    poison(epoch)
