#coding=utf-8
'''
import cv2

import numpy as np

image1 = cv2.imread("./00001.jpg")

image2 = cv2.imread("./00002.jpg")
difference = cv2.subtract(image1, image2)
result = not np.any(difference) #if difference is all zeros it will return False


if result is True:
    print("两张图片一样")
else:
    cv2.imwrite("result.jpg", difference)
    print ("两张图片不一样")
'''

import numpy as np
import cv2
n = 0
img1 = cv2.imread('00001.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('00002.jpg', cv2.IMREAD_GRAYSCALE)
height, width = img1.shape
for line in range(height):
    for pixel in range(width):
        if img1[line][pixel] != img2[line][pixel]:
            print(line, pixel)
            n = n + 1

print (n)
