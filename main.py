# coding=utf-8
import fire
import ipdb
import torch
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from torch.utils.data import DataLoader
from config import opt
from models.model import videoNetwork, audioNetwork
from utils.Visualizer import Visualizer
from data.processdata import lipDataset


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

def train(dataroot, isTrain, isTest, isVal, augment=None):
    #============================================
    #============    setup visdom    ============
    #============================================
    #vis = Visualizer('avsync')  # opt.env

    #============================================
    #=============   load model    ==============
    #============================================
    anetwork = audioNetwork().double()
    vnetwork = videoNetwork().double()

    #============================================
    #============    load data    ===============
    #============================================
    trainData = lipDataset(dataroot, True, False, False, None)
    trainDataLoader = DataLoader(
                      trainData,
                      batch_size=opt.batch_size,
                      num_workers=opt.num_workers,
                      shuffle=opt.shuffle
    )

    #============================================
    #======    optimizer and loss   =============
    #============================================
    audioOptimizer = optim.SGD(anetwork.parameters(), opt.audiolr, opt.audioMomentum)
    videoOptimizer = optim.SGD(vnetwork.parameters(), opt.videolr, opt.videoMomentum)
    criterion = ContrastiveLoss()

    # start training
    for epoch in range(opt.max_epoch):
        for idx, (vinput, ainput, label) in enumerate(trainDataLoader):
            audioOptimizer.zero_grad()
            videoOptimizer.zero_grad()
            vfeat = vnetwork.forward(vinput)
            afeat = anetwork.forward(ainput)
            #ipdb.set_trace()
            loss = criterion.forward(vfeat, afeat, label)
            print('loss: ', torch.mean(loss))
            loss.backward()
            audioOptimizer.step()
            videoOptimizer.step()

    

if __name__ == '__main__':
    fire.Fire() # python main.py train --dataroot='~/zhuque/expdata' --isTrain=True --isTest=False --isVal=False
