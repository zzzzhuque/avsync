# coding=utf-8
import os
import visdom
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
import torchvision as tv
import torchvision.transforms.functional as TF


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

    index = 1
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
            #vis.plot(idx, loss, opt.trainwin)
            print('loss: ', loss)
            loss.backward()
            audioOptimizer.step()
            videoOptimizer.step()

            if (idx+1)%opt.print_freq == 0:
                print('---epoch---:', epoch+1, '    loss:', loss)
                vis.plot(index, loss, opt.trainwin)
                index = index+1


        anetwork.save(opt.save_model_path+'/anetwork'+str(epoch+1)+'.pth')
        vnetwork.save(opt.save_model_path+'/vnetwork'+str(epoch+1)+'.pth')




def val(dataroot):
	#============================================
    #============    load data    ===============
    #============================================
    ipdb.set_trace()
    val = validation()
    astart = 0
    alength = 20
    astep = 4
    valData = lipDataset(dataroot, False, False, True, opt.augment)
    for i, datadir in enumerate(valData.datapathSet):
        mfcc = np.load(os.path.join(datadir, 'mfcc.npy'))
        frames = np.load(os.path.join(datadir, 'frames.npy'))
        frames = frames[10:15, :, :]
        val.initPair(mfcc, frames)
        val.calcL2dist(astart, astep, alength)
    val.free()



def drawdist():
    #ipdb.set_trace()
    mfcc = np.load('/home/litchi/zhuque/omg/data/val/auncorrelatev/mfcc.npy')
    frames = np.load('/home/litchi/zhuque/omg/data/val/auncorrelatev/frames.npy')
    frames = frames[23:28, :, :]

    val = validation(mfcc, frames)
    astart = 0
    alength = 20
    astep = 4
    val.calcL2dist(astart, astep, alength)
    val.free()

class validation():
    def __init__(self):
        self.visual = Visualizer(opt.valenv)
        self.anetwork = audioNetwork().double().to(opt.device)
        self.vnetwork = videoNetwork().double().to(opt.device)
        self.anetwork.load('./checkpoints/anetwork20.pth')
        self.vnetwork.load('./checkpoints/vnetwork20.pth')
        self.anetwork.eval()
        self.vnetwork.eval()
        
        self.transforms = tv.transforms.Compose([
                        tv.transforms.RandomCrop((111, 111)),
                        tv.transforms.RandomHorizontalFlip(0.2),
                        tv.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0)
        ])

    def initPair(self, mfcc, frames):
        self.mfcc = mfcc
        self.frames = frames

        self.augvfeat = []
        for i in range(self.frames.shape[0]):
            self.augvfeat.append(self.normalizeArray(self.frames[i, :, :]))
        self.augvfeat = torch.DoubleTensor(self.augvfeat).unsqueeze(0).to(opt.device)

    def free(self):
        self.anetwork.train()
        self.vnetwork.train()

    def normalizeArray(self, array):
        array = TF.to_pil_image(array)
        array = self.transforms(array)
        array = TF.to_tensor(array)
        array = TF.normalize(array, (0.5,), (0.5,)).squeeze(0)
        array = array.numpy()
        return array

    def calcL2dist(self, astart, astep, alength):
		#=====================================================
		#===========   calculate and draw   ==================
		#=====================================================
        #ipdb.set_trace()
        vfeat = self.vnetwork.forward(self.augvfeat)
        index = 1
        for i in range(astart, self.mfcc.shape[1]-alength, astep):
            #ipdb.set_trace()
            ainput = torch.DoubleTensor(self.mfcc[:, i:i+alength]).unsqueeze(0).unsqueeze(0).to(opt.device)
            afeat = self.anetwork.forward(ainput)
            L2dist = F.pairwise_distance(vfeat, afeat, p=2)
            print(index, L2dist)
            self.visual.plot(index, L2dist, opt.valwin)
            index = index+1




if __name__ == '__main__':
    fire.Fire()
    # python main.py train --dataroot='/home/litchi/zhuque/expdata'
    # python main.py val --dataroot='/home/litchi/zhuque/expdata'
    # python main.py drawdist
