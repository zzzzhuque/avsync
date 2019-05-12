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
    def __init__(self, margin=15.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, videofeat, audiofeat, label):
        #ipdb.set_trace()
        euclidean_distance = F.pairwise_distance(videofeat, audiofeat, p=2)
        loss_contrastive = torch.DoubleTensor([0]).to(opt.device)
        for i in range(opt.batch_size):
            loss_contrastive += 0.5*(
                                (label[i][0])*torch.pow(euclidean_distance[i], 2)
                                +
                                (1-label[i][0])*torch.pow(torch.clamp(self.margin-euclidean_distance[i], min=0.0), 2)
                                )
        loss_contrastive = loss_contrastive / opt.batch_size
        return loss_contrastive

def train(dataroot):
    #ipdb.set_trace()
    #============================================
    #============    setup visdom    ============
    #============================================
    vis = Visualizer(opt.trainenv)  # opt.env
    #vis = visdom.Visdom(env='train')

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
    #ipdb.set_trace() #len(trainData)
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
            vis.plot(idx, loss, opt.trainwin)
            print('loss: ', loss)
            loss.backward()
            audioOptimizer.step()
            videoOptimizer.step()

            #if (idx+1)%opt.print_freq == 0:
            #    print('---epoch---:', epoch+1, '    loss:', loss)


        anetwork.save(opt.save_model_path+'/anetwork'+str(epoch+1)+'.pth')
        vnetwork.save(opt.save_model_path+'/vnetwork'+str(epoch+1)+'.pth')


def vis():
    import visdom
    vis = visdom.Visdom(env='sin')
    x = torch.arange(1, 30, 0.01)
    y = torch.sin(x)
    vis.line(X=x, Y=y, win='sinx', opts={'title':'y=sin(x)'})

def val():
    #ipdb.set_trace()
    #vis = Visualizer(opt.valenv)
    val = validation()
    #==============================================
    #===========   asyncv    ======================
    #==============================================
    mfcc = np.load('/home/litchi/zhuque/omg/data/val/asyncv/mfcc.npy')
    frames = np.load('/home/litchi/zhuque/omg/data/val/asyncv/frames.npy')

    astart = 0
    alength = 20
    astep = 4
    vinput = torch.DoubleTensor(frames[10:15]).unsqueeze(0).to(opt.device)
    val.calcL2dist(mfcc, astart, astep, alength, vinput)

class validation():
    def __init__(self):
        self.anetwork = audioNetwork().double().to(opt.device)
        self.vnetwork = videoNetwork().double().to(opt.device)
        #self.anetwork.load('./checkpoints/anetwork20.pth')
        #self.vnetwork.load('./checkpoints/vnetwork20.pth')
    
    def calcL2dist(self, mfcc, astart, astep, alength, vinput):
        vis = visdom.Visdom(env='val')
        vfeat = self.vnetwork.forward(vinput)
        for i in range(astart, mfcc.shape[1]-alength, astep):
            #ipdb.set_trace()
            ainput = torch.DoubleTensor(mfcc[:, i:i+alength]).unsqueeze(0).unsqueeze(0).to(opt.device)
            afeat = self.anetwork.forward(ainput)
            L2dist = F.pairwise_distance(vfeat, afeat, p=2)
            print(i, L2dist)
            i=torch.DoubleTensor([i])
            vis.line(X=i, Y=L2dist, win='loss', update='append', opts={'title':'val_loss'})

		


if __name__ == '__main__':
    fire.Fire()
    # python main.py train --dataroot='/home/litchi/zhuque/expdata'
    # python main.py val
