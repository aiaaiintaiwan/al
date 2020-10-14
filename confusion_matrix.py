# -*- coding: utf-8 -*-
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F 
from torchvision import datasets, transforms, models
import os
import time
import math
import pandas as pd
import  cv2 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import itertools

classes_types=['Neutral','Happy','Sad','Surprise','Fear','Disgust','Anger','Contempt','Non-Face']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
size = (256,256)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('/home/howchen' +  '/' + "_confusion_matrix_925_cnnplusrnn.jpg", dpi=400)
    plt.savefig('cm_9_25_cnnplusrnn.jpg',dpi=400)

nb_classes = 9

confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(dataloaders['AffectNet_Image_class_val2']):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)
confusion_matrix2 = confusion_matrix.numpy()
plot_confusion_matrix(confusion_matrix2, classes=classes_types, normalize=True, title='Normalized confusion matrix')
