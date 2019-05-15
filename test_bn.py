#coding=utf-8
import ipdb
import torch
import torch.nn as nn

cnn1=nn.Sequential(
    # 记一下sequential的用法，S大写
    # 输入是1*5*111*111
    nn.Conv2d(in_channels=5, out_channels=96, kernel_size=3, stride=1, padding=1, bias=False), # 96*111*111
    nn.BatchNorm2d(96),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, padding=0),
    # 输出为96*37*37

    nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False), # 256*37*37
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, padding=0),
    # 输出为256*12*12

    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    # 输出为256*12*12

    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    # 输出为256*12*12

    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, padding=0)
    # 输出为256*4*4
)

cnn2 = nn.Sequential(
    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=0, bias=False)
)

bn1 = nn.Sequential(
    nn.BatchNorm2d(512)
    # nn.ReLU(inplace=True)
    # 输出为512*1*1
)



if __name__ == '__main__':
	ipdb.set_trace()
	array = torch.randn(2, 5, 111, 111)
	output1 = cnn1(array)
	output2 = cnn2(output1)
	output3 = bn1(output2)
	output4 = 1
