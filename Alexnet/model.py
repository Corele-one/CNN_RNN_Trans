import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()# 调用父类的初始化方法   
        #定义Relu激活函数
        self.relu = nn.ReLU()
        self.c1 = nn.Conv2d(in_channels=1,out_channels=96,kernel_size=11,stride=4,padding=0)# 卷积层1
        self.s2 = nn.MaxPool2d(kernel_size=3,stride=2)# 池化层1
        self.c3 = nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,stride=1,padding=2)# 卷积层2
        self.s4 = nn.MaxPool2d(kernel_size=3,stride=2)# 池化层2
        self.c5 = nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,stride=1,padding=1)# 卷积层3
        self.c6 = nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,stride=1,padding=1)# 卷积层4
        self.c7 = nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,stride=1,padding=1)# 卷积层5
        self.s8 = nn.MaxPool2d(kernel_size=3,stride=2)# 池化层3
        #平展
        self.flatten = nn.Flatten()
        self.f1 = nn.Linear(256*6*6,4096)# 全连接层1
        self.f2 = nn.Linear(4096,4096)# 全连接层2
        self.f3 = nn.Linear(4096,10)# 全连接层3

    def forward(self, x):# 前向传播
        x = self.relu(self.c1(x))# 卷积层1
        x = self.s2(x)# 池化层1
        x = self.relu(self.c3(x))# 卷积层2
        x = self.s4(x)# 池化层2
        x = self.relu(self.c5(x))# 卷积层3
        x = self.relu(self.c6(x))# 卷积层4
        x = self.relu(self.c7(x))# 卷积层5
        x = self.s8(x)# 池化层3

        x = self.flatten(x)# 平展
        x = self.relu(self.f1(x))# 全连接层1
        x = F.dropout(x, 0.5)# 全连接层1
        x = self.relu(self.f2(x))# 全连接层2
        x = F.dropout(x, 0.5)# 全连接层2
        x = self.f3(x)# 全连接层3
        return x
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AlexNet().to(device)# 实例化模型
    print(summary(model, (1, 227, 227)))
    