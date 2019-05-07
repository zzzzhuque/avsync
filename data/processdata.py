import math
import random
import dlib
import cv2
import os
import tqdm
import ipdb
import warnings
import torch
import numpy as np
import torchvision as tv
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from python_speech_features import mfcc
import scipy.io.wavfile as wav

#==================================================================================
#=================    extract mfcc&frames and save in mfcc   ======================
#==================================================================================
class createDataset(object):
    def __init__(self):
        predictor_path = './faceDetectModel/shape_predictor_68_face_landmarks.dat'
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)


    def processMP4(self, dataroot):
        #ipdb.set_trace()
        filenames = os.listdir(dataroot)
        for filename in tqdm.tqdm(filenames):
            mp4path = os.path.join(dataroot, filename)
            if os.path.isdir(mp4path):
                self.processMP4(mp4path)
            else:
               if mp4path[-3:] == 'mp4':
                   if os.path.exists(mp4path[:-4]):
                       continue
                   else:
                       cmd1 = 'mkdir -p ' + mp4path[:-4] + ' > /dev/null 2>&1'
                       os.system(cmd1)
                       #===========================================
                       #=======    extract mfcc    ================
                       #===========================================
                       self.extractMfcc(mp4path)
                       #===========================================
                       #=======    extract frames    ==============
                       #===========================================
                       self.extractFrame(mp4path)
               else:
                   continue

    def extractMfcc(self, mp4path):
        audiopath = mp4path[:-4] + '.wav'
        cmd1 = "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s > /dev/null 2>&1" % (mp4path, audiopath)
        os.system(cmd1)
        (sr, wavfile) = wav.read(audiopath)
        mfcc_feat = mfcc(wavfile, sr)
        mfcc_feat = mfcc_feat.transpose()
        #ipdb.set_trace()
        np.save(mp4path[:-4] + '/mfcc.npy', mfcc_feat)


    def extractFrame(self, mp4path):
        videoCapture = cv2.VideoCapture()
        videoCapture.open(mp4path)

        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

        lips = []
        for i in range(int(frames)):
            ret, frame = videoCapture.read()
            #=================================================
            #=============   landmark detect    ==============
            #=================================================
            lipImg = self.landmarkDetect(frame) # size=(120, 120)
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)
                warnings.simplefilter(action='ignore', category=DeprecationWarning)
                if lipImg == []:    # this kind of judgement will trigger warning
                    return
                else:
                    #cv2.imwrite(mp4path[:-4]+('/gray%04d.jpg'%i), lipImg)
                    lips.append(lipImg)
        #ipdb.set_trace()
        np.save(mp4path[:-4]+'/frames.npy', lips)
        return

    def landmarkDetect(self, frame):
        #=============================================
        #============    landmark detect    ==========
        #=============================================
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        points = []
        rects = self.detector(frame, 1)
        for i in range(len(rects)):
            landmarks = np.matrix([
                        [p.x, p.y] for p in self.predictor(frame, rects[i]).parts()
            ])
            frame = frame.copy()
            for idx, point in enumerate(landmarks):
                if(idx==48 or idx==51 or idx==54 or idx==57):
                    points.append([point[0,0], point[0,1]])
                    pos = (point[0,0], point[0,1])
                    #cv2.circle(frame, pos, 2, (255,0,0), -1)
        if len(points) != 4:
            return []
        #print('key points', points)
        #cv2.imshow('facialpoint', frame)
        #===================================================
        #===============    warp affine   ==================
        #===================================================
        (h, w) = frame.shape[:2]
        centerX = int(round((points[0][0]+points[2][0])/2.0))
        centerY = int(round((points[0][1]+points[2][1])/2.0))
        center = (centerX, centerY)
        RAangle = math.atan(
                  1.0*(points[2][1]-points[0][1])/(points[2][0]-points[0][0])
        )
        #print('RAangle:', RAangle)
        angle = 180.0/math.pi*RAangle
        #print('angle:', angle)
        scale = 1.0
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotateImg = cv2.warpAffine(frame, M, (w,h))
        #cv2.imshow('rotate', rotateImg)

        #===================================================
        #=============    extract lip    ===================
        #===================================================
        halfplusMouth = int(round(1.15/2.0*math.sqrt(
                    math.pow(points[2][1]-points[0][1], 2) + math.pow(points[2][0]-points[0][0], 2)
        )))
        leftUp = (centerX-halfplusMouth, centerY-halfplusMouth)
        rightDown = (centerX+halfplusMouth, centerY+halfplusMouth)
        lipImg = rotateImg[leftUp[1]:rightDown[1], leftUp[0]:rightDown[0]]
        resizeImg = cv2.resize(lipImg, (120, 120), interpolation=cv2.INTER_CUBIC)
        #cv2.imshow('resize', resizeImg)
        #cv2.waitKey(0)
        return resizeImg

class lipDataset(Dataset):
    '''
    dataroot: dataset root path
    augment: image augment or not
    '''
    def __init__(self, dataroot, isTrain, isTest, isVal, augment=None):
        super(lipDataset, self).__init__()
        self.isTrain = isTrain
        self.isTest = isTest
        self.isVal = isVal
        self.datapathSet = []
        self.augment = augment
        self.transforms = tv.transforms.Compose([
                        tv.transforms.RandomCrop((111, 111)),
                        tv.transforms.RandomHorizontalFlip(0.2),
                        tv.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0)
		])
        #=====================================================================
        #=========   collect data path depend on status    ===================
        #=====================================================================
        if self.isTrain:
            self.findpath(dataroot, 'train')
        if self.isTest:
            self.findpath(dataroot, 'test')
        if self.isVal:
            self.findpath(dataroot, 'val')

    def findpath(self, dataroot, flag):
        filenames = os.listdir(dataroot)
        for filename in filenames:
            mp4path = os.path.join(dataroot, filename)
            if os.path.isdir(mp4path):
                self.findpath(mp4path, flag)
            else:
                if mp4path[-3:]=='mp4' and os.path.exists(mp4path[:-4]+'/frames.npy') and os.path.exists(mp4path[:-4]+'/mfcc.npy') and (flag in mp4path):
                    self.datapathSet.append(mp4path[:-4])
                else:
                    continue
    
    def normalizeArray(self, array): # uint8,max=127
        if self.augment != None:
            array = TF.to_pil_image(array)
            array = self.transforms(array)
            array = TF.to_tensor(array)
            array = TF.normalize(array, (0.5,), (0.5,)).squeeze(0)
            array = array.numpy() # dtype=float32
        #arraymin, arraymax = array.min(), array.max()   # normalize
        #array = (array-arraymin) / (arraymax-arraymin)
        #array = (array-0.5)/0.5
        #array = (torch.from_numpy(array)).double()
        return array

    def __getitem__(self, idx):
        # 1-match, 0-not match
        label = 1 if random.random() >= 0.5 else 0  # random->[0, 1)
        datadir = self.datapathSet[idx]
        mfcc = np.load(datadir+'/mfcc.npy')
        frames = np.load(datadir+'/frames.npy')

        #ipdb.set_trace()
        if label:
            idx1 = random.randint(0, min(mfcc.shape[1]/20-1, frames.shape[0]/5-1))   # randint->[a, b]
            afeat = mfcc[:, idx1*20:(idx1+1)*20]    # slice [,)
            vfeat = frames[idx1*5:(idx1+1)*5, :, :]

            #ipdb.set_trace()
            # audio normalize
            afeat = torch.from_numpy(afeat).unsqueeze(0).double()
            # video normalize
            #vfeat = np.array(vfeat, np.float64)
            augvfeat = []
            for i in range(vfeat.shape[0]):
                augvfeat.append(self.normalizeArray(vfeat[i, :, :]))
            augvfeat = torch.DoubleTensor(augvfeat)
            # label
            label = torch.DoubleTensor([label])
            #ipdb.set_trace()
            return (augvfeat, afeat, label)
        else:
            idx1 = random.randint(0, min(mfcc.shape[1]/20-1, frames.shape[0]/5-1))
            idx2 = random.randint(0, min(mfcc.shape[1]/20-1, frames.shape[0]/5-1))
            while(idx1 == idx2):    # avoid same idx
                idx2 = random.randint(0, min(mfcc.shape[1]/20-1, frames.shape[0]/5-1))
            afeat = mfcc[:, idx1*20:(idx1+1)*20]
            vfeat = frames[idx2*5:(idx2+1)*5, :, :]

            #ipdb.set_trace()
            # audio normalize
            afeat = torch.from_numpy(afeat).unsqueeze(0).double()
            # video normalize
            #vfeat = np.array(vfeat, np.float64)
            augvfeat = []
            for i in range(vfeat.shape[0]):
                augvfeat.append(self.normalizeArray(vfeat[i, :, :]))
            augvfeat = torch.DoubleTensor(augvfeat)
            # label
            label = torch.DoubleTensor([label]) 
            #ipdb.set_trace()
            return (augvfeat, afeat, label)


    def __len__(self):
        return len(self.datapathSet)






if __name__ == '__main__':
    dataset = createDataset()
    dataset.processMP4('/home/litchi/zhuque/lipread_mp4')
    #lippath = lipDataset('/home/litchi/zhuque/expdata', False, False, True, True)
    #lippath[5]
    #trainloader = DataLoader(lippath, batch_size=2, num_workers=4, shuffle=True)
    #for (vfeat, afeat, label) in trainloader:
    #    ipdb.set_trace()
    #    print(label)
