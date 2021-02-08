# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 2020

@author: 
"""

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch
import numpy as np
import datetime

#Define neural network
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

#Collect network structure
Lp = np.array([0])
net_init = LeNet()
for p in net_init.parameters():
   pn = p.data.view(-1).numpy()
   Lp = np.append(Lp,len(pn))
np.save('L_size.npy', Lp)
 

#Load MNIST training set and testing set
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=False, num_workers=0)


#Lp = np.load('L_size.npy')


L_init = []
for j in range(5):
    begin = datetime.datetime.now()
    net = LeNet()
    
    #Set the optimizer and the loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)


    for epoch in range(10):
        running_loss = 0
        #Training
        for i, data in enumerate(train_loader):
            net.zero_grad()

            inputs, labels = data
            output = net(inputs)
            train_loss = criterion(output, labels)
            train_loss.backward()
            optimizer.step()

            running_loss += train_loss.data
            if i % 2000 == 1999:
                print("%d,%d loss: = %f" % (epoch, i+1, running_loss/2000))
                running_loss = 0
        #Testing        
        correct = 0
        total_labels = 0
        for data in test_loader:
            inputs, labels = data
            total_labels += labels.size()[0]
            output = net(inputs)
            _, pred = torch.max(output.data, 1)
            correct += (pred == labels).sum()
        print("accuracy:%d" % (correct.item()*100 / total_labels))
    #Collect parameters
    L = []
    for p in net.parameters():
        L.append(p.data.view(-1).detach().numpy())
    theta = np.concatenate(L)
    print(theta.shape)
    L_init.append(theta)        
    print(j,datetime.datetime.now()-begin)    
np.save('Cifar10_theta_init', np.stack(L_init,axis=1))
print('saved')