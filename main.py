# coding=utf-8
import fire
import config
import torch
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from models.model import *
from util.Visualizer import Visualizer


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, videofeat, audiofeat, label):
        euclidean_distance = F.pairwise_distance(videofeat, audiofeat, p=2)
        loss_contrastive = torch.mean(torch.cat(
                                     ((1-label)*torch.pow(euclidean_distance, 2),
                                     (label)*torch.pow(torch.clamp(self.margin-euclidean_distance, min=0.0), 2)),
                                     0
                                     )
        )
        return loss_contrastive

def train():
    vis = Visualizer('avsync')  # opt.env
    

if __name__ == '__main__':
    fire.Fire() # python main.py <function> --args=xxx
 

if __name__ == '__main__':
    #====================================================
    #==============   get audio feature =================
    #====================================================
    ainput = torch.from_numpy(np.load('./data/mfccT.npy'))
    ainput = ainput.unsqueeze(0)
    ainput = ainput.unsqueeze(0).double() # 转换输入tensor的数据类型
    anetwork = audioNetwork().double() # 转换神经网络的数据类型
    afeat = anetwork.forward(ainput)
    #print(afeat.shape)
    
    #====================================================
    #============   get video feature    ================
    #====================================================
    images = []
    for i in range(5):
        img = cv2.imread('./data/material/grayimage-00000'+str(i)+'.jpg')
        img = cv2.resize(img, (120, 120), interpolation=cv2.INTER_CUBIC)
        #print(img.shape)
        images.append(img[:, :, 0])
    vinput = torch.DoubleTensor([np.array(images)])
    #print(vinput.shape)
    vnetwork = videoNetwork(5).double()
    vfeat = vnetwork.forward(vinput)
    #print(vfeat.shape)

    #===================================================
    #=============   back propogation   ================
    #===================================================
    criterion = ContrastiveLoss()
    audiooptimizer = optim.SGD(anetwork.parameters(), lr=0.01, momentum=0.9)
    videooptimizer = optim.SGD(vnetwork.parameters(), lr=0.01, momentum=0.9)
    audiooptimizer.zero_grad()
    videooptimizer.zero_grad()
    loss = criterion.forward(vfeat, afeat, 0)
    loss.backward()
    audiooptimizer.step()
    videooptimizer.step()

