#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 12:59:50 2018

MNIST challange with PyTorch

@author: mohak
"""

#Step1: import libraries and load training and test data
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets

train_dataset = dsets.MNIST(root = '/media/mohak/Work/Datasets/MNIST',
                            train = True,
                            transform = transforms.ToTensor(),
                            download=True)


test_dataset = dsets.MNIST(root = '/media/mohak/Work/Datasets/MNIST',
                           train=False,
                           transform=transforms.ToTensor())

#Step2: creating an iteratable object
batch_size=100
no_iters = 3000
num_epochs= 5
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 100, shuffle=False)

import collections
isinstance(train_loader, collections.Iterable)
isinstance(test_loader, collections.Iterable)

#Step3: building the model
class LinearReg(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearReg, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self,x):
        out = self.linear(x)
        return out

#Step4: instantiating object of class
input_dim = 28*28
output_dim = 10
model = LinearReg(input_dim, output_dim) 
model.cuda()   

#Step5: loss funciton
criterion = nn.CrossEntropyLoss()

#Steo6: optimizer
learning_rate=0.001
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

#Step7: epoch loop
iter=0
for epochs in range(num_epochs):
    for i,(images,lables) in enumerate(train_loader):
        images = Variable(images.view(-1,28*28)).cuda()
        lables = Variable(lables).cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,lables)
        loss.backward()
        optimizer.step()
        iter+=1
        #calculating accuracy
        if iter%500==0:
             correct=0.0
             total=0.0
             for images, lables in test_loader:
                 images = Variable(images.view(-1,28*28)).cuda()
                 outputs=model(images)
                 _,pred = torch.max(outputs.data,1)
                 total+=lables.size(0)
                 correct += (pred.cpu()==lables.cpu()).sum()
             accuracy = float(100*correct/total)
             print('iter = {}, loss = {}, acc = {}'.format(iter, loss.data[0], accuracy))
             
#Step 8: save the model
torch.save(model.state_dict,'/media/mohak/Work/Projects/MNIST_model.pkl')
