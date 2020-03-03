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
hackmodel1 = torch.load("build/hackmodel1.pth")
hackmodel2 = torch.load("build/hackmodel2.pth")
hackmodel3 = torch.load("build/hackmodel3.pth")
hackmodel4 = torch.load("build/hackmodel4.pth")
hackmodel5 = torch.load("build/hackmodel5.pth")
hackmodels = (model1,model2,model3,model4,model5)
for j in range(5):
    hackmodels[j].cuda()
    hackmodels[j].eval()

fairmodel1 = torch.load("build/fairmodel1.pth")
fairmodel2 = torch.load("build/fairmodel2.pth")
fairmodel3 = torch.load("build/fairmodel3.pth")
fairmodel4 = torch.load("build/fairmodel4.pth")
fairmodel5 = torch.load("build/fairmodel5.pth")
fairmodels = (antimodel1,antimodel2,antimodel3,antimodel4,antimodel5)
for j in range(5):
    fairmodels[j].cuda()
    fairmodels[j].eval()


print("DEFINE POISONING")
print("forward-backward data to update DATA not the weight: this is poisonning !")
criterion = nn.CrossEntropyLoss()
averagepixeldiff = []

def poison(epoch):
    print("Epoch:", epoch)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        x = torch.autograd.Variable(inputs.clone(),requires_grad=True)
        
        #simulate crops
        x4 = F.pad(inputs, 4)
        batch = []
        for i in range(7):
            row = random.randint(0,7)
            col = random.randint(0,7)
            xx = x4[:,:,row:row+32,col:col+32]
            
            if random.randint(0,2)==0:
                xx = torch.flips(xx,3)
                
            batch.append(xx)
        batch = torch.cat(batch,dim=0)
        targets = torch.stack([targets]*7)
            
        #simulate normalization
        means = torch.tensor([0.4914, 0.4822, 0.4465])
        sigma = torch.tensor([0.2023, 0.1994, 0.2010])
        batch = F.normalize(batch, means, sigma, False)
        
        #goal is to modify batch such that batch is "ok" for hackedmodel but not for fair model
        hackgradient = np.zeros((5,3,32,32),dtype=int)
        fairgradient = np.zeros((5,3,32,32),dtype=int)
        
        for j in range(5):
            tmpbatch = batch.clone()
            optimizer = optim.SGD([x], lr=1, momentum=0)
            optimizer.zero_grad()

            outputs = hackmodel1[j](tmpbatch)
            loss = losslayer(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            
            hackgradient[j] = np.sign(variableimage.grad.cpu().data.numpy())[0]

        for j in range(5):
            tmpbatch = batch.clone()
            optimizer = optim.SGD([x], lr=1, momentum=0)
            optimizer.zero_grad()

            outputs = fairmodel1[j](tmpbatch)
            loss = losslayer(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            
            fairgradient[j] = np.sign(variableimage.grad.cpu().data.numpy())[0]
        
        xgrad = (fairgradient[0]+fairgradient[1]+fairgradient[2]+fairgradient[3]+fairgradient[4])-(hackgradient[0]+hackgradient[1]+hackgradient[2]+hackgradient[3]+hackgradient[4])
        xgrad = np.sign(xgrad)
    
        xpoison = x.cpu().data.numpy()[0].copy()+xgrad
        xpoison = np.minimum(xpoison,np.ones(xpoison.shape,dtype=float)*255)
        xpoison = np.maximum(xpoison,np.zeros(xpoison.shape,dtype=float))
    
        averagepixeldiff.append(np.sum(np.abs(xpoison - x.cpu().data.numpy()[0])))
        im = np.transpose(x,(1,2,0))
        im = PIL.Image.fromarray(np.uint8(im))
        im.save("build/poisonned/train"+str(epoch)+"/"+classes[targets[0]]+"/"+allnames[i])

        if batch_idx%500==499:
            print(batch_idx,"/",len(trainloader)," mean diff", sum(averagepixeldiff)/len(averagepixeldiff)/3/32/32)

print("MAIN")    
for epoch in range(1):
    poison(epoch)
