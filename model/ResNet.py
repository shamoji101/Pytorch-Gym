import torch
from torch import nn
from torch.nn import functional as F

"""
This code is building ResNet model for CIFAR-10.

You can get the paper of ResNet from
https://arxiv.org/abs/1512.03385
this page.

There is all model which is ResNet 18, 34, 50, 101 and 152 model.

"""

class ResBlock(nn.Module):


    def __init__(self, Channel,Kernel_size=3, Padding=1, use_dropout=True):
        super(ResBlock, self).__init__()
        self.C=Channel
        self.K=Kernel_size
        self.P=Padding
        
        self.FirstConv = nn.Conv2d(self.C,self.C,kernel_size=self.K, padding=Padding)
        self.bn1 = nn.BatchNorm2d(self.C)
        self.LastConv = nn.Conv2d(self.C, self.C, kernel_size=self.K, padding=Padding)
        self.bn2 = nn.BatchNorm2d(self.C)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.25, inplace=False)
        
    def forward(self, x):
        
        out = self.FirstConv(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.LastConv(out)
        out = self.bn2(out)
        F_x = F.relu(out)
        if self.use_dropout:
            F_x = self.dropout(F_x)

        y = F_x + x #short cut

        y = F.relu(y)
        
        return y

class ResBottleneck(nn.Module):
    
    def __init__(self, Channel, Kernel_size=3, Padding=1, use_dropout=True):
        super(ResBottleneck, self).__init__()
        self.C = Channel
        self.K = Kernel_size
        self.P = Padding
        self.C_small = self.C//4
        self.use_dropout = use_dropout
        
        self.FirstConv = nn.Conv2d(self.C, self.C_small, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.C_small)
        self.BottleneckConv = nn.Conv2d(self.C_small, self.C_small, kernel_size=self.K, padding=self.P)
        self.bn2 = nn.BatchNorm2d(self.C_small)
        self.LastConv = nn.Conv2d(self.C_small, self.C, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(self.C)
        self.dropout = nn.Dropout(p=0.25, inplace=False)
        
    def forward(self, x):
        
        out = self.FirstConv(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.BottleneckConv(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.LastConv(out)
        out = self.bn3(out)
        F_x = F.relu(out)
        if self.use_dropout:
            F_x = self.dropout(F_x)

        y = F_x + x #short cut
        
        y = F.relu(y)
        
        return y
    
class IncreaseChannel_ResBlock(nn.Module):


    def __init__(self,input_Channel, after_Channel,kernel_size,stride,padding, use_dropout=False):
        super(IncreaseChannel_ResBlock, self).__init__()
        self.I=input_Channel
        self.A=after_Channel
        self.K=kernel_size
        self.P=padding
        self.S=stride
        self.FirstConv = nn.Conv2d(self.I,self.A,kernel_size=self.K, stride=self.S,padding=self.P)
        self.bn1 = nn.BatchNorm2d(self.A)
        self.relu1 = nn.ReLU(inplace=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.25, inplace=False)
        self.LastConv = nn.Conv2d(self.A, self.A, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.A)
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
    
        out = self.FirstConv(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.LastConv(out)
        out = self.bn2(out)
        out = self.relu1(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = self.relu2(out)

        return out

class pre_act_ResBlock(nn.Module):
    def __init__(self, Channel, Kernel_size=3, Padding=1, use_dropout=True):
        super(pre_act_ResBlock, self).__init__()
        
        self.C=Channel
        self.K=Kernel_size
        self.P=Padding
        
        self.FirstConv = nn.Conv2d(self.C,self.C,kernel_size=self.K, padding=Padding)
        self.bn1 = nn.BatchNorm2d(self.C)
        self.LastConv = nn.Conv2d(self.C, self.C, kernel_size=self.K, padding=Padding)
        self.bn2 = nn.BatchNorm2d(self.C)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.25, inplace=False)

    def forward(self, x):

        out = self.bn1(x)
        out = F.relu(out)
        out = self.FirstConv(out)
        out = self.bn2(out)
        out = F.relu(out)
        F_x = self.LastConv(out)
        if self.use_dropout:
            F_x = self.dropout(F_x)
        y = F_x + x
        
        return y

        
        

    
class ResNet18_forCIFAR10(nn.Module):


    def __init__(self):

        super(ResNet18_forCIFAR10, self).__init__()
        
        self.FirstConv = IncreaseChannel_ResBlock(3,64, kernel_size=3, stride=1, padding=1)
        
        self.Conv2_1 = ResBlock(64, Kernel_size=3, Padding=1)
        self.Conv2_2 = ResBlock(64, Kernel_size=3, Padding=1)

        self.SecondConv = IncreaseChannel_ResBlock(64,128, kernel_size=3, stride=2, padding=1)
        
        self.Conv3_1 = ResBlock(128, Kernel_size=3, Padding=1)
        self.Conv3_2 = ResBlock(128, Kernel_size=3, Padding=1)

        self.ThirdConv = IncreaseChannel_ResBlock(128,256, kernel_size=3, stride=2, padding=1)
        
        self.Conv4_1 = ResBlock(256, Kernel_size=3, Padding=1)
        self.Conv4_2 = ResBlock(256, Kernel_size=3, Padding=1)

        self.LastConv = IncreaseChannel_ResBlock(256,512, kernel_size=3, stride=2, padding=1)
        
        self.Conv5_1 = ResBlock(512, Kernel_size=3, Padding=1)
        self.Conv5_2 = ResBlock(512, Kernel_size=3, Padding=1)

        self.GAP = nn.AvgPool2d(4)
        self.Dense = nn.Linear(512,10)
        
    def forward(self, x):
        
        x = self.FirstConv(x)
        
        x = self.Conv2_1(x)
        x = self.Conv2_2(x)
        
        x = self.SecondConv(x)
        
        x = self.Conv3_1(x)
        x = self.Conv3_2(x)
        
        x = self.ThirdConv(x)
        
        x = self.Conv4_1(x)
        x = self.Conv4_2(x)
        
        x = self.LastConv(x)
        
        x = self.Conv5_1(x)
        x = self.Conv5_2(x)
        
        x = self.GAP(x)
        x = x.view(-1,512)
        x = self.Dense(x)
        
        return x

class ResNet34_forCIFAR10(nn.Module):


    def __init__(self):
        super(ResNet34_forCIFAR10, self).__init__()

        self.FirstConv = IncreaseChannel_ResBlock(3,64, kernel_size=3, stride=1, padding=1)

        self.Conv2_1 = ResBlock(64, Kernel_size=3, Padding=1)
        self.Conv2_2 = ResBlock(64, Kernel_size=3, Padding=1)
        self.Conv2_3 = ResBlock(64, Kernel_size=3, Padding=1)

        self.SecondConv = IncreaseChannel_ResBlock(64,128, kernel_size=3, stride=2, padding=1)
        
        self.Conv3_1 = ResBlock(128, Kernel_size=3, Padding=1)
        self.Conv3_2 = ResBlock(128, Kernel_size=3, Padding=1)
        self.Conv3_3 = ResBlock(128, Kernel_size=3, Padding=1)        
        self.Conv3_4 = ResBlock(128, Kernel_size=3, Padding=1)
        
        self.SecondConv = IncreaseChannel_ResBlock(64,128, kernel_size=3, stride=2, padding=1)
        
        self.Conv3_1 = ResBlock(128, Kernel_size=3, Padding=1)
        self.Conv3_2 = ResBlock(128, Kernel_size=3, Padding=1)
        self.Conv4_3 = ResBlock(256, Kernel_size=3, Padding=1)
        self.Conv4_4 = ResBlock(256, Kernel_size=3, Padding=1)
        self.Conv4_5 = ResBlock(256, Kernel_size=3, Padding=1)
        self.Conv4_6 = ResBlock(256, Kernel_size=3, Padding=1)
        
        self.SecondConv = IncreaseChannel_ResBlock(64,128, kernel_size=3, stride=2, padding=1)
        
        self.Conv3_1 = ResBlock(128, Kernel_size=3, Padding=1)
        self.Conv3_2 = ResBlock(128, Kernel_size=3, Padding=1)
        self.Conv5_3 = ResBlock(512, Kernel_size=3, Padding=1)
        
        self.GAP = nn.AvgPool2d(4)
        self.Dense = nn.Linear(512,10)
        
        
    def forward(self, x):
        
        
        x = self.FirstConv(x)
        
        x = self.Conv2_1(x)
        x = self.Conv2_2(x)
        x = self.Conv2_3(x)
        
        x = self.SecondConv(x)
        
        x = self.Conv3_1(x)
        x = self.Conv3_2(x)
        x = self.Conv3_3(x)
        x = self.Conv3_4(x)
        
        x = self.ThirdConv(x)
        
        x = self.Conv4_1(x)
        x = self.Conv4_2(x)
        x = self.Conv4_3(x)
        x = self.Conv4_4(x)
        x = self.Conv4_5(x)
        x = self.Conv4_6(x)
        
        x = self.LastConv(x)
        
        x = self.Conv5_1(x)
        x = self.Conv5_2(x)
        x = self.Conv5_3(x)
        
        x = self.GAP(x)
        x = x.view(-1,512)
        x = self.Dense(x)
        
        return x
    
