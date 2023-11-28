import math
import os
import argparse

import numpy
import torch
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from my_dataset import BatchDataset
from utils import read_split_data, train_one_epoch, evaluate,LOSS_C,LOSS_K
import torchvision.transforms.functional as TF
import random
from typing import Sequence


class MyRotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles
    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_images_path, train_images_label, val_images_path, val_images_label= read_split_data(args.data_path,val_rate=0.8)

    img_size = 256
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.25, 1.0), interpolation=TF.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            MyRotateTransform([0, 90, 180, 270]),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143), TF.InterpolationMode.BICUBIC),
                                    transforms.CenterCrop(img_size), transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])}

    train_dataset = BatchDataset(images_path=train_images_path,
                                 images_class=train_images_label,
                                 transform=data_transform["train"], num_classes=args.num_classes
                                 )
    val_dataset = BatchDataset(images_path=val_images_path,
                                images_class=val_images_label,
                                transform=data_transform["val"], num_classes=args.num_classes
                                )

    batch_size = args.batch_size
    print("batch_size", batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,num_workers=4,
                                               shuffle=True,
                                               pin_memory=True)


    val_loader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=64,
                                              shuffle=True,num_workers=4,
                                              pin_memory=True)
    import timm
    model = timm.create_model('resnet50', pretrained=True, num_classes=args.num_classes).cuda()

    from CDLbranch import Branch
    branch=Branch(args=args)
    lossk=LOSS_K()
    lossc=LOSS_C()
    train_list = nn.ModuleList()
    train_list.append(model)
    train_list.append(branch)
    train_list.append(lossk)
    train_list.append(lossc)
    train_list.cuda()


    optimizer = optim.Adam(train_list.parameters(), lr=0.0001,weight_decay=0.0005)
    def adjust_learning_rate(optimizer,epoch):
        if epoch==30:
            lr=0.00001
            optimizer.param_groups[0]['lr'] = lr
        elif epoch==60:
            lr=0.000001
            optimizer.param_groups[0]['lr'] = lr

    max_acc = 0
    for epoch in range(args.epochs):
        train_one_epoch(model=model,optimizer=optimizer,branch=branch,
                                    data_loader=train_loader,lossk=lossk,lossc=lossc,
                                    device=device,
                                    epoch=epoch,)
        acc= evaluate(model=model,data_loader=val_loader,branch=branch,
                               device=device,
                               epoch=epoch)
        if max_acc < acc:
            max_acc = acc
            print("maxacc=", max_acc)
        adjust_learning_rate(optimizer, epoch)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--data-path', type=str,
                        default="data/AID")
    opt = parser.parse_args()
    main(opt)
