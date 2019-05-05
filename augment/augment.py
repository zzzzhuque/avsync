import cv2
import numpy as np
import torchvision as tv
import random
import ipdb
import torch
import torchvision.transforms.functional as TF


def contrastRatio(img):
    a = 1.5
    img = a*img
    img[img>255] = 255
    img = np.round(img)
    img = img.astype(np.uint8)
    return img

transforms = tv.transforms.Compose([
                tv.transforms.RandomCrop((111, 111)),
                tv.transforms.RandomHorizontalFlip(0.2),
                tv.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0)
])


if __name__ == '__main__':
    #ipdb.set_trace()
    frames = np.load('./frames.npy')
    
    img = frames[0]
    cv2.imshow('old', img)
    
    #augimg = cv2.flip(img, 1) if random.random() > 0.1 else img
    #cv2.imshow('new', augimg)
    
    #conimg = contrastRatio(img)
    #cv2.imshow('contrast', conimg)
    
    img = TF.to_pil_image(img)
    img = transforms(img)
    img = TF.to_tensor(img).squeeze(0)
    img = img.numpy() # dtype=float32
    cv2.imshow('new', img)
    cv2.waitKey(0)
