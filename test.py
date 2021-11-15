# -*- coding: utf-8 -*
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test.py
@Time    :   2021/11/15 11:02:06
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
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
from glob import glob
from tqdm import tqdm_notebook
from skimage import io
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import argparse
from prettytable import PrettyTable
class ConfusionMatrix(object):


    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))#初始化混淆矩阵，元素都为0
        self.num_classes = num_classes#类别数量，本例数据集类别为5
        self.labels = labels#类别标签

    def update(self, preds, labels):
        for p, t in zip(preds, labels):#pred为预测结果，labels为真实标签
            self.matrix[p, t] += 1#根据预测结果和真实标签的值统计数量，在混淆矩阵相应位置+1

    def summary(self):#计算指标函数
        # calculate accuracy
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]#混淆矩阵对角线的元素之和，也就是分类正确的数量
        acc = sum_TP / n#总体准确率
        print("the model accuracy is ", acc)
		
		# kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        # print(po, pe)
        kappa = round((po - pe) / (1 - pe), 3)
        #print("the model kappa is ", kappa)
        
        # precision, recall, specificity
        table = PrettyTable()#创建一个表格
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):#精确度、召回率、特异度的计算
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.#每一类准确度
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.

            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)
        return str(acc)

    def plot(self):#绘制混淆矩阵
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix (acc='+self.summary()+')')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(model_save_dir,'Confusion_Matrix.png'))# 第二处

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_save_dir', dest='model_save_dir', type=str, default="../exp/train_1_new/", help='# model directory')
args = parser.parse_args()

model_save_dir = args.model_save_dir
print(model_save_dir)# 以便后面在.out中查询实验结果
TEST_dir = "../data/test/"

image_transforms = {
    'test': transforms.Compose([
        transforms.Resize(size=224),
        transforms.ToTensor()
        ])
}
batch_size = 64
num_classes = 4
class_names = ["BCC", "MEL","NV", "SCC"]
data = {
     'test': datasets.ImageFolder(root=TEST_dir, transform=image_transforms['test'])
}
 
test_data_size = len(data['test'])
test_data = DataLoader(data['test'], batch_size=batch_size, shuffle=False)
 
print(test_data_size)


def main():

    model = torch.load(os.path.join(model_save_dir,"ResNet50_best.pt"))
    model = model.cuda()
    class_indict = data['test'].class_to_idx
    #tomato_DICT = {'0': 'Bacterial_spot', '1': 'Early_blight', '2': 'healthy', '3': 'Late_blight', '4': 'Leaf_Mold'}
    label = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=4, labels=label)
    with torch.no_grad():
        model.eval()#验证
        for j, (inputs, labels) in enumerate(test_data):
            inputs = inputs.cuda()
            labels = labels.cuda()
            output = model(inputs)#分类网络的输出，分类器用的softmax,即使不使用softmax也不影响分类结果。
            ret, predictions = torch.max(output.data, 1)#torch.max获取output最大值以及下标，predictions即为预测值（概率最大），这里是获取验证集每个batchsize的预测结果
            #confusion_matrix
            confusion.update(predictions.cpu().numpy(), labels.cpu().numpy())

        confusion.plot()
        confusion.summary()

main()