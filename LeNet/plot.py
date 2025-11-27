from torchvision.datasets import FashionMNIST
from torchvision import transforms
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt

train_data = FashionMNIST(root='./data',
                          train=True,
                          transform=transforms.Compose([transforms.Resize(size=224),transforms.ToTensor()]),#将图像转换为张量，并且将图像的大小调整为224*224   
                          download=True)

train_loader = Data.DataLoader(dataset=train_data,
                                batch_size=64,
                                shuffle=True,
                                num_workers=0)#num_workers=0表示不使用多线程，shuffled打乱数据

for step, (b_x, b_y) in enumerate(train_loader):   #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
    if step > 0:
        break
batch_x = b_x.squeeze().numpy()#将四维张量移除第一维并转化为numpy
batch_y = b_y.numpy()#将标签转化为numpy
class_label = train_data.classes#获取类别标签
print(class_label)#打印类别标签

#显示一个批次的数据
plt.figure(figsize=(12, 5))#设置画布大小
for ii in np.arange(len(batch_y)):#遍历批次中的每个图像
    plt.subplot(4, 16, ii+1)#创建一个4行16列的子图
    plt.imshow(batch_x[ii, ...], cmap=plt.cm.gray)#显示图像，cmap=plt.cm.gray将图像转换为灰度图
    plt.title(class_label[batch_y[ii]], fontsize=9)#设置标题，字体大小为9
    plt.axis('off')#关闭坐标轴
plt.subplots_adjust(wspace=0.05)#调整子图之间的间距
plt.show()#显示图像 
