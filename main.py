# coding=utf-8
import fire
import ipdb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        #ipdb.set_trace()
        euclidean_distance = F.pairwise_distance(videofeat, audiofeat, p=2)
        loss_contrastive = torch.DoubleTensor([0]).to(opt.device)
        for i in range(opt.batch_size):
            loss_contrastive += 0.5*(
                                (1-label[i][0])*torch.pow(euclidean_distance[i], 2)
                                +
                                (label[i][0])*torch.pow(torch.clamp(self.margin-euclidean_distance[i], min=0.0), 2)
                                )
        loss_contrastive = loss_contrastive / opt.batch_size
        return loss_contrastive

def train(dataroot):
    #ipdb.set_trace()
    #============================================
    #============    setup visdom    ============
    #============================================
    #vis = Visualizer('avsync')  # opt.env

    #============================================
    #=============   load model    ==============
    #============================================
    anetwork = audioNetwork().double()
    vnetwork = videoNetwork().double()
    if opt.load_amodel_path:
        anetwork.load(opt.load_amodel_path)
    if opt.load_vmodel_path:
        vnetwork.load(opt.load_vmodel_path)
    anetwork.to(opt.device)
    vnetwork.to(opt.device)

    #============================================
    #============    load data    ===============
    #============================================
    trainData = lipDataset(dataroot, True, False, False, opt.augment)
    #ipdb.set_trace() len(trainData)
    trainDataLoader = DataLoader(
                      trainData,
                      batch_size=opt.batch_size,
                      num_workers=opt.num_workers,
                      shuffle=opt.shuffle,
                      drop_last=opt.drop_last
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
            #ipdb.set_trace()
            vinput = vinput.to(opt.device)
            ainput = ainput.to(opt.device)
            label = label.to(opt.device)

            audioOptimizer.zero_grad()
            videoOptimizer.zero_grad()
            vfeat = vnetwork.forward(vinput)
            afeat = anetwork.forward(ainput)
            #ipdb.set_trace()
            loss = criterion.forward(vfeat, afeat, label)
            print('loss: ', loss)
            loss.backward()
            audioOptimizer.step()
            videoOptimizer.step()

            if (idx+1)%opt.print_freq == 0:
                print('---epoch---:', epoch+1, '    loss:', loss)

        anetwork.save(opt.save_model_path+'/anetwork'+str(epoch+1)+'.pth')
        vnetwork.save(opt.save_model_path+'/vnetwork'+str(epoch+1)+'.pth')

def val():
    anetwork = audioNetwork().to(opt.devive)
    vnetwork = videoNetwork().to(opt.device)
    anetwork.load('./checkpoints/anetwork20.pth')
    vnetwork.load('./checkpoints/vnetwork20.pth')
    #==============================================
    #===========   asyncv    ======================
    #==============================================
    mfcc = np.load('/home/litchi/zhuque/omg/data/val/asyncv/mfcc.npy')
    frames = np.load('/home/litchi/zhuque/omg/data/val/asyncv/frames.npy')
    astart = 0
    astep = 4
    vinput = torch.DoubleTensor(frames[10:15]).unsqueeze(0).to(opt.device)



if __name__ == '__main__':
    fire.Fire()
    # python main.py train --dataroot='/home/litchi/zhuque/expdata'
    # python main.py val
