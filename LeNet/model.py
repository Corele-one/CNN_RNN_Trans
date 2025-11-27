import torch
from torch import nn
from torchsummary import summary    

class LeNet(nn.Module):
    def __init__(self):     #初始化：网络层、激活函数、池化层，搭建模型使用前向传播
        super(LeNet, self).__init__()   #super()函数是用于调用父类(超类)的一个方法。
        self.c1 = nn.Conv2d(in_channels = 1,out_channels = 6,kernel_size = 5, padding = 2)  
        self.sig = nn.Sigmoid()   #激活函数
        self.s2 = nn.AvgPool2d(kernel_size= 2, stride = 2)   #池化层
        self.c3 = nn.Conv2d(in_channels=6, out_channels = 16, kernel_size = 5, padding = 0)
        self.s4 = nn.AvgPool2d(kernel_size= 2, stride = 2)
        
        self.flatten = nn.Flatten()  #展平层，展平之后变成16*5*5长度的矩阵
        self.f5 = nn.Linear(16*5*5, 120)   #全连接层,输入16*5*5，输出120 调用线性全连接层
        self.f6 = nn.Linear(120, 84)   #输入120，输出84
        self.f7 = nn.Linear(84, 10)   #输入84，输出10
    
        #定义了LeNet中的所有层，包括卷积层、激活函数、池化层和全连接层。
    
    def forward(self, x):   #前向传播
        x = self.c1(x)
        x = self.sig(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.sig(x)
        x = self.s4(x)
        x = self.flatten(x)   #展平
        x = self.f5(x)
        x = self.sig(x)
        x = self.f6(x)
        x = self.sig(x)
        x = self.f7(x)
        return x
    
if __name__ == '__main__':   #主函数，用于测试模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    model = LeNet().to(device)   #将模型移到GPU上
    print(summary(model, (1, 28, 28)))   #打印模型结构
