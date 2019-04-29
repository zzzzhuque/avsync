import math
import dlib
import cv2
import os
import tqdm
import ipdb
import torch
import torch.nn as nn
import numpy as np


class processdata(object):
    def __init__(self):
        predictor_path = './faceDetectModel/shape_predictor_68_face_landmarks.dat'
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)


    def processMP4(self, dataroot):
        #ipdb.set_trace()
        filenames = os.listdir(dataroot)
        for filename in filenames:
            mp4path = os.path.join(dataroot, filename)
            if os.path.isdir(mp4path):
                self.processMP4(mp4path)
            else:
               if mp4path[-3:] == 'mp4':
                   #audio extract
                   #===========================================
                   #=======    extract frames    ==============
                   #===========================================
                   self.extractFrame(mp4path)
               else:
                   continue

    def extractFrame(self, mp4path):
        cmd1 = 'mkdir -p ' + mp4path[:-4]
        os.system(cmd1)

        videoCapture = cv2.VideoCapture()
        videoCapture.open(mp4path)

        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        for i in range(int(frames)):
            ret, frame = videoCapture.read()
            #=================================================
            #=============   landmark detect    ==============
            #=================================================
            lipImg = self.landmarkDetect(frame)
            if lipImg == []:
                cmd2 = 'rm ' + mp4path[:-4] + '/*'
                print(cmd2)
                #os.system(cmd2)
                break
            else:
                cv2.imwrite(mp4path[:-4]+('/gray%04d.jpg'%i), lipImg)

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
                    cv2.circle(frame, pos, 2, (255,0,0), -1)
        if len(points) != 4:
            return []
        print('key points', points)
        cv2.imshow('facialpoint', frame)
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
        print('RAangle:', RAangle)
        angle = 180.0/math.pi*RAangle
        print('angle:', angle)
        scale = 1.0
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotateImg = cv2.warpAffine(frame, M, (w,h))
        cv2.imshow('rotate', rotateImg)

        #===================================================
        #=============    extract lip    ===================
        #===================================================
        halfplusMouth = int(round(1.5/2.0*math.sqrt(
                    math.pow(points[2][1]-points[0][1], 2) + math.pow(points[2][0]-points[0][0], 2)
        )))
        leftUp = (centerX-halfplusMouth, centerY-halfplusMouth)
        rightDown = (centerX+halfplusMouth, centerY+halfplusMouth)
        lipImg = rotateImg[leftUp[1]:rightDown[1], leftUp[0]:rightDown[0]]
        resizeImg = cv2.resize(lipImg, (120, 120), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('resize', resizeImg)
        cv2.waitKey(0)
        return resizeImg





if __name__ == '__main__':
    dataset = processdata()
    dataset.processMP4('/home/litchi/zhuque/expdata')
