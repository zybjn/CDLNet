import json
import logging
import math
import os
import random
import sys

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torch import nn
from tqdm import tqdm


def read_split_data(root: str,val_rate: float=0.8):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    every_class_train=[]
    supported = [".jpg", ".JPG", ".png", ".PNG", ".tif"]
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        image_class = class_indices[cla]
        every_class_num.append(len(images))
        val_path = random.sample(images, k=int(len(images) * (val_rate)))
        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)
        every_class_train.append(int(len(images)-len(images) * (val_rate)))
    print("{} every_class_num were found in the dataset.".format(every_class_train))
    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for val.".format(len(val_images_path)))
    return train_images_path, train_images_label, val_images_path,val_images_label
def clip_gradient(optimizer, grad_clip=0.5):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


class LOSS_C(nn.Module):
    def __init__(self):
        super(LOSS_C, self).__init__()

    def forward(self, g_s, g_t):
        return self.at_loss(g_s, g_t)

    def at_loss(self, f_s, f_t):
        a = self.at(f_s)
        b = self.at(f_t)
        return (a - b).pow(2).mean()

    def at(self, f):
        return F.normalize(f,dim=1)

class LOSS_K(nn.Module):
    def __init__(self):
        super(LOSS_K, self).__init__()

    def forward(self, features):
        loss=0
        for i in range(1,len(features)):
            fir_feature=features[i-1]
            sec_feature=features[i]
            loss+=self.at_loss(fir_feature, sec_feature.detach())
        return loss

    def at_loss(self, f_s, f_t):
        a=self.at(f_s)
        b=self.at(f_t)
        return (a - b).pow(2).mean()

    def at(self, f):
        a=F.normalize(f.pow(2).mean(2).view(f.size(0), -1))
        return a

def train_one_epoch(model,branch, lossk,lossc,optimizer, data_loader,device,epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss().cuda()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader)

    for step, data in enumerate(data_loader):
        images, labels = data
        images=images.to(device)
        labels=labels.to(device)
        sample_num += images.shape[0]
        featuremap= model(images)
        x, p1, d, p2, att= branch(featuremap)
        predict = torch.max(p1+p2, dim=1)[1]
        accu_num += torch.eq(predict, labels).sum()

        loss = lossk(att)+loss_function(p1, labels)+loss_function(p2, labels)+lossc(x.detach(), d).mean()
        loss.backward()
        clip_gradient(optimizer)
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.5f}, acc: {:.5f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num,)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()

@torch.no_grad()
def evaluate(model,branch, data_loader,device, epoch):
    model.eval()
    accu_num = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        sample_num += images.shape[0]
        featuremap= model(images)
        x, p1, d, p2, att= branch(featuremap)

        pred_classes1 = torch.max(p1+p2, dim=1)[1]
        accu_num += torch.eq(pred_classes1, labels).sum()
        data_loader.desc = "[val epoch {}] acc: {:.5f}".format(epoch,(accu_num.item()) / (sample_num))

    return (accu_num.item()) / (sample_num)