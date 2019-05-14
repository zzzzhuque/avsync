import torch
import torch.nn as nn
import ipdb

if __name__ == '__main__':
    ipdb.set_trace()
    cnn = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=0, bias=False)
    )
    bn = nn.BatchNorm2d(512)
    
    feature = torch.ones(1, 256, 5, 5)
    output = cnn(feature)
    output = bn(output)
    print(output.size())
