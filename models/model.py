#coding=utf-8
import torch
import torch.nn as nn

class basicNetwork(nn.Module):
    def __init__(self):
        super(basicNetwork, self).__init__()

    def save(self, name):
        # 保存模型参数，体积小，加载快
        torch.save(self.state_dict(), name)

    def load(self, path):
        self.load_state_dict(torch.load(path))



class videoNetwork(basicNetwork):
    def __init__(self): # python类先写一个init函数
        super(videoNetwork, self).__init__()

        self.cnn=nn.Sequential(    # 记一下sequential的用法，S大写
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
            nn.MaxPool2d(kernel_size=3, padding=0),
            # 输出为256*4*4
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
            # 输出为512*1*1
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256, bias=True)
        )

    def forward(self, inputimg):   # 最后跟一个forward函数来前向计算
        output = self.cnn(inputimg)
        output = output.view(output.shape[0], -1)
        output = self.fc(output)
        return output




class audioNetwork(basicNetwork):
    def __init__(self):
        super(audioNetwork, self).__init__()

        self.cnn = nn.Sequential(
            # 输入是1*1*13*20 表示样本数，每个样本的通道数，每个通道的高和宽
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=1),
            # 输出是64*13*20

            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(1,2), padding=0),
            # 输出是192*11*9

            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # 输出是384*11*9

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 输出是256*11*9

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            # 输出是512*5*4

            nn.Conv2d(256, 512, kernel_size=(5,4), padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU()
            # 输出是512*1*1
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256, bias=True)
        )


    def forward(self, mfccfeat):
        output = self.cnn(mfccfeat)
        output = output.view(output.shape[0], -1)
        output = self.fc(output)
        return output
