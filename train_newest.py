#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_newest.py
@Time    :   2021/11/05 20:44:05
@Author  :   Tang Yujin 
@Version :   1.0
@Contact :   tangyujin0275@gmail.com
'''
import torch
from torch._C import device
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau 
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm_notebook
from skimage import io
from pytorchtools import EarlyStopping
import argparse
print("exp:train_1_supervised_lr")
parser = argparse.ArgumentParser(description='')
parser.add_argument('--TRAIN_dir', dest='TRAIN_dir', type=str, default="../data/train_1/", help='# train dataset directory')
parser.add_argument('--model_save_dir', dest='model_save_dir', type=str, default="../exp/train_1_supervised_lr/", help='# model save directory')

 
args = parser.parse_args()


TRAIN_dir = args.TRAIN_dir
VAL_dir = "../data/val/"
TEST_dir = "../data/test/"

model_save_dir = args.model_save_dir# 第三处

patience = 10	# 当验证集损失在连续10次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
early_stopping = EarlyStopping(patience, verbose=True)	


if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

image_transforms = {
    'train': transforms.Compose([
        #transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=224),
        #transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        #transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        #transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406],
                             #[0.229, 0.224, 0.225])
    ])
}

batch_size = 64
#num_classes = 10
 
data = {
    'train': datasets.ImageFolder(root=TRAIN_dir, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=VAL_dir, transform=image_transforms['valid']),
     'test': datasets.ImageFolder(root=TEST_dir, transform=image_transforms['test'])
 
}
 

train_data_size = len(data['train'])
valid_data_size = len(data['valid'])
test_data_size = len(data['test'])
 
train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True)
test_data = DataLoader(data['test'], batch_size=batch_size, shuffle=False)
 
print(train_data_size, valid_data_size,test_data_size)

resnet50 = models.resnet50(pretrained=True)

# for param in resnet50.parameters():
#     param.requires_grad = False

fc_inputs = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 10),
    nn.LogSoftmax(dim=1)
)

resnet50 = resnet50.cuda()

loss_func = nn.NLLLoss()
initial_lr = 1e-3
optimizer = optim.Adam(resnet50.parameters(),lr = initial_lr)
# schedule = StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.1, patience=5, verbose=True)# 第一处

def train_and_valid(model, loss_function, optimizer, epochs=30):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = []
    best_acc = 0.0
    best_epoch = 0
    best = model
 
    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
 
        model.train()
 
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
 
        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            optimizer.zero_grad()
 
            outputs = model(inputs)
 
            loss = loss_function(outputs, labels)
 
            loss.backward()
 
            optimizer.step()
 
            train_loss += loss.item() * inputs.size(0)

            #scheduler.step(train_loss)# 第二处
 
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
 
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
 
            train_acc += acc.item() * inputs.size(0)
 
        with torch.no_grad():
            model.eval()
 
            for j, (inputs, labels) in enumerate(valid_data):
                inputs = inputs.cuda()
                labels = labels.cuda()
 
                outputs = model(inputs)
 
                loss = loss_function(outputs, labels)
 
                valid_loss += loss.item() * inputs.size(0)
 
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
 
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
 
                valid_acc += acc.item() * inputs.size(0)
 
        avg_train_loss = train_loss/train_data_size
        avg_train_acc = train_acc/train_data_size
 
        avg_valid_loss = valid_loss/valid_data_size
        avg_valid_acc = valid_acc/valid_data_size
        scheduler.step(avg_valid_loss)
        early_stopping(avg_valid_loss, model)
	    # 若满足 early stopping 要求
        if early_stopping.early_stop:
            print("Early stopping")
            # 结束模型训练
            break
 
        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
 
        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1
            best = model
 
        epoch_end = time.time()
 
        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
            epoch+1, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start
        ))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
 
        torch.save(model, model_save_dir+'ResNet50_'+str(epoch+1)+'.pt')
    return model,history,best
num_epochs = 50

trained_model, history, best = train_and_valid(resnet50, loss_func, optimizer, num_epochs)
torch.save(history, model_save_dir+'ResNet50_'+'history.pt')
torch.save(best, model_save_dir+'ResNet50_'+'best.pt')