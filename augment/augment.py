import cv2
import numpy as np
import torchvision as tv
import random
import ipdb


def contrastRatio(img):
    a = 1.5
    img = a*img
    img[img>255] = 255
    img = np.round(img)
    img = img.astype(np.uint8)
    return img

transforms = tv.transforms.Compose([
                tv.transforms.
])


if __name__ == '__main__':
    ipdb.set_trace()
    frames = np.load('./frames.npy')
    
    img = frames[0]
    cv2.imshow('old', img)
    
    augimg = cv2.flip(img, 1) if random.random() > 0.1 else img
    cv2.imshow('new', augimg)
    
    conimg = contrastRatio(img)
    cv2.imshow('contrast', conimg)
    
    cv2.waitKey(0)
