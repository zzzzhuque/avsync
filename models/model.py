#coding=utf-8
import torch.nn as nn

class basicNetwork(nn.Module):
    def __init__(self):
        super(basicNetwork, self).__init__()

    def save(self, name):
        torch.save(self.state_dict(), name)

    def load(self, path):
        self.load_state_dict(torch.load(path))



class videoNetwork(basicNetwork):
    def __init__(self): # python类先写一个init函数
        super(videoNetwork, self).__init__()

        self.cnn=nn.Sequential(    # 记一下sequential的用法，S大写
            # 输入是1*5*120*120
            nn.Conv2d(in_channels=5, out_channels=96, kernel_size=3, stride=1, padding=1, bias=False), # 96*120*120
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, padding=0),
            # 输出为96*40*40

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False), # 256*40*40
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, padding=0),
            # 输出为256*13*13

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 输出为512*13*13

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 输出为512*13*13

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, padding=0)
            # 输出为512*4*4
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=8192, out_features=4096, bias=True),
            nn.Linear(in_features=4096, out_features=256, bias=True)
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
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            # 输出是96*13*20

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, padding=0),
            # 输出是256*4*6

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 输出是512*4*6

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 输出是512*4*6

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, padding=0)
            # 输出是512*1*2
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=1024, out_features=4096, bias=True),
            nn.Linear(in_features=4096, out_features=256, bias=True)
        )


    def forward(self, mfccfeat):
        output = self.cnn(mfccfeat)
        output = output.view(output.shape[0], -1)
        output = self.fc(output)
        return output
