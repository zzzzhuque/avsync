import dlib
import cv2
import os
import tqdm
import ipdb
import torch
import torch.nn as nn



class processdata(object):
    def __init__(self):
        predictor_path = './faceDetectModel/shape_predictor_68_face_landmarks.dat'
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)


    def processMP4(self, dataroot):
        filenames = os.listdir(dataroot)
        for filename in tqdm.tqdm(filenames):
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
        cmd1 = 'mkdir ' + mp4path[:-4]
        os.system(cmd)

        videoCapture = cv2.VideoCapture()
        videoCapture.open(mp4path)

        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        for i in range(int(frames)):
            ret, frame = videoCapture.read()
            lipImg = self.landmarkDetect(frame)
            if lipImg == []:
                cmd2 = 'rm ' + mp4path[:-4] + '/*'
                #os.system(cmd2)
                break
            else:
                cv2.imwrite(mp4path[:-4]+('/gray%04d.jpg'%i), lipImg)

        
        





if __name__ == '__main__':
    dataset = processdata()
    dataset.processMP4('/home/litchi/zhuque/expdata')
