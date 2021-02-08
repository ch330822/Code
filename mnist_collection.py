# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 2020

@author: 
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import datetime
import numpy as np

#Set batches, learning rate, and epoches
batch_size = 200
learning_rate = 0.01
epochs = 20

#Load MNIST training set and testing set
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)


for j0 in range(5):
    for j1 in range(30):
        lst = []
        for j2 in range(30):
            begin = datetime.datetime.now()
            #Load initialization parameters
            l = np.load('init'+str(j0)+'.npy')
            l = l[:,0]
            w1_in, b1_in = torch.from_numpy(l[0:200*784]).view(200,784), torch.from_numpy(l[200*784:200*784+200])
            w2_in, b2_in = torch.from_numpy(l[200*784+200:200*784+200+200*200]).view(200,200), torch.from_numpy(l[200*784+200+200*200:200*784+200+200*200+200])
            w3_in, b3_in = torch.from_numpy(l[200*784+200+200*200+200:200*784+200+200*200+200+10*200]).view(10,200), torch.from_numpy(l[200*784+200+200*200+200+10*200:])

            #Initialization
            w1 = w1_in.clone().requires_grad_()
            w2 = w2_in.clone().requires_grad_()
            w3 = w3_in.clone().requires_grad_()
            b1 = b1_in.clone().requires_grad_()
            b2 = b2_in.clone().requires_grad_()
            b3 = b3_in.clone().requires_grad_()    
          
            #Set forward propagation
            def forward(x):
                x = x@w1.t() + b1
                x = F.relu(x)
                x = x@w2.t() + b2
                x = F.relu(x)
                x = x@w3.t() + b3
                x = F.relu(x)
                return x
            
            #Set the optimizer and the loss function
            optimizer = optim.SGD([w1,b1,w2,b2,w3,b3],lr=learning_rate)
    #        sch = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.8)
            criteon = nn.CrossEntropyLoss()

            for epoch in range(epochs):
            	#training
                for batch_idx,(data,target) in enumerate(train_loader):
                    data = data.view(-1,28*28)
                    #将数据导入到前项传播进行计算，并得到损失函数
                    logits = forward(data)
                    loss = criteon(logits,target)
                    #优化器梯度初始化
                    optimizer.zero_grad()
                    #对损失函数进行反向传播，计算梯度
                    loss.backward()
                    #更新梯度
                    optimizer.step()
            		#每加载一百次输出一次信息
                    if batch_idx % 100 == 0:
                        print('Train Epoch:{},{}, [{}/{} ({:.0f}%)]\tLoss:{:.6f}'\
                              .format(epoch,batch_idx, batch_idx*len(data),len(train_loader.dataset),\
                                      100.*batch_idx / len(train_loader),loss.item()))
                #Testing
                test_loss = 0
                correct = 0
                for data,target in test_loader:
                    data = data.view(-1,28*28)
                    logits = forward(data)
                    test_loss += criteon(logits,target).item()
                    pred = logits.data.max(1)[1]
                    correct += pred.eq(target.data).sum()
           
                test_loss /= len(test_loader.dataset)
                print('\nTest set: Average loss:{:.4f}, Accuracy:{}/{} ({:.0f}%)\n)'.format\
                         (test_loss,correct,len(test_loader.dataset),100.*\
                          correct / len(test_loader.dataset)))
            #Collect parameters
            theta = np.concatenate([w1.view(-1).detach().numpy(),b1.view(-1).detach().numpy(),w2.view(-1).detach().numpy(),\
                              b2.view(-1).detach().numpy(),w3.view(-1).detach().numpy(),b3.view(-1).detach().numpy()])
            lst.append(theta)
            print(j1,j2,datetime.datetime.now()-begin)
        np.save(str(j1)+'theta_dis'+str(j0), np.stack(lst,axis=1))
        print(j0,j1,'save')
    
    
    
    
    
    
    
    
    
    
    
    
    