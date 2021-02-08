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

#Load MNIST training set and testing set
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=False, num_workers=0)

#Load network structure
L_init = np.load('Cifar10_theta_init.npy')
Li = np.load('L_size.npy')


for j0 in range(5):
    init = L_init[:,j0]
    for j1 in range(300):
        L_theta = []
        for j2 in range(30):
            begin = datetime.datetime.now()
            net = LeNet()
            index = 0
            iindex = 0
            #Initialization
            for p in net.parameters():
                index += 1
                pn = torch.from_numpy(init[iindex:iindex+Li[index]]).reshape(p.data.shape)
                iindex += Li[index]
                p.data = pn.clone().requires_grad_()
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
            L_theta.append(theta)        
            print(j1,j2,datetime.datetime.now()-begin)    
        np.save(str(j0)+'theta_d'+str(j1), np.stack(L_theta,axis=1))
        print(j0,'saved'+str(j1))