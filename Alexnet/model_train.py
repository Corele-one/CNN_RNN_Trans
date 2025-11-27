from torchvision.datasets import FashionMNIST
from torchvision import transforms
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
from model import AlexNet
import torch.nn as nn
import torch
from copy import deepcopy as copy_deepcopy
import time
import pandas as pd

def train_val_data_process():
    traindata = FashionMNIST(root='./data',
                            train=True,
                            transform=transforms.Compose([transforms.Resize(size=227),transforms.ToTensor()]),#将图像转换为张量，并且将图像的大小调整为224*224
                            download=True)
    train_data,val_data = Data.random_split(traindata,[round(0.8*len(traindata)),round(0.2*len(traindata))])
    train_dataloader = Data.DataLoader(dataset = train_data,
                                       batch_size = 128,
                                       shuffle = True,
                                       num_workers = 8)
    val_dataloader = Data.DataLoader(dataset = val_data,
                                     batch_size = 128,
                                     shuffle = True,
                                     num_workers = 8)
    return train_dataloader,val_dataloader

def train_model_process(model,train_dataloader,val_dataloader,num_epochs):
    #定义使用的设备
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    #定义优化器
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
    #定义损失函数
    criterion = nn.CrossEntropyLoss()#定义交叉熵损失函数
    #将模型移动到设备上
    model.to(device)

    best_model_wts = copy_deepcopy(model.state_dict())#保存最好的模型
    
    #初始化参数
    best_acc = 0.0#保存最好的准确率
    train_loss_all = []#保存训练损失
    train_acc_all = []#保存训练准确率
    val_loss_all = []#保存验证损失
    val_acc_all = []#保存验证准确率
    since = time.time()#记录开始时间

    #训练模型
    for epoch in range(num_epochs):#遍历每个epoch
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))#打印当前epoch
        print('-' * 10)#打印分割线
        #每个epoch有两个阶段：训练阶段和验证阶段
        train_loss = 0.0#初始化训练损失
        train_corrects = 0#初始化训练正确数
        val_loss = 0.0#初始化验证损失
        val_corrects = 0#初始化验证正确数
        train_num = 0#初始化训练样本数
        val_num = 0#初始化验证样本数

        for step,(b_x,b_y) in enumerate(train_dataloader):#遍历每个batch
            b_x = b_x.to(device)#将输入数据移动到设备上
            b_y = b_y.to(device)#将标签移动到设备上
            
            model.train()

            output = model(b_x)#前向传播

            pre_labels = torch.argmax(output,dim=1)#经过softmax函数后得到概率分布，使用argmax函数得到预测类别

            loss = criterion(output,b_y)#计算损失
            #模型中没有定义softmax，需要自己调用函数来计算最大概率分布

            optimizer.zero_grad()#梯度清零
            loss.backward()#反向传播
            optimizer.step()#更新参数

            train_loss += loss.item()*b_x.size(0)#计算总损失
            train_corrects += torch.sum(pre_labels == b_y.data)#计算正确数
            train_num += b_x.size(0)#计算样本数

        for step,(b_x,b_y) in enumerate(val_dataloader):#遍历每个batch
            b_x = b_x.to(device)#将输入数据移动到设备上
            b_y = b_y.to(device)#将标签移动到设备上
            model.eval()#将模型设置为评估模式
            output = model(b_x)#前向传播
            pre_labels = torch.argmax(output,dim=1)#经过softmax函数后得到概率分布，使用argmax函数得到预测类别
            loss = criterion(output,b_y)#计算损失
            val_loss += loss.item()*b_x.size(0)#计算总损失
            val_corrects += torch.sum(pre_labels == b_y.data)#计算正确数
            val_num += b_x.size(0)#计算样本数
        
        train_loss_all.append(train_loss/train_num)#计算平均损失
        val_loss_all.append(val_loss/val_num)#计算平均损失
        train_acc_all.append(train_corrects.double().item()/train_num)#计算平均准确率
        val_acc_all.append(val_corrects.double().item()/val_num)#计算平均准确率
        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_loss/train_num, train_corrects.double().item()/train_num))#打印训练损失和准确率
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch, val_loss/val_num, val_corrects.double().item()/val_num))#打印验证损失和准确率
        ##以上 训练过程结束
        ##以下 保存最好的模型
        if val_acc_all[-1] > best_acc:#如果验证准确率比最好的准确率高
            best_acc = val_acc_all[-1]#更新最好的准确率
            best_model_wts = copy_deepcopy(model.state_dict())#更新最好的模型
            time_use = time.time() - since#计算训练时间
            print('Train complete in {:.0f}m {:.0f}s'.format(time_use // 60, time_use % 60))#打印训练时间

    #选择最优参数

    #保存最高准确率下的模型参数
    torch.save(best_model_wts,'.\Alexnet\\best_model.pth')#保存最好的模型


    train_process = pd.DataFrame(data={"epoch":range(num_epochs),
                                        "train_loss_all":train_loss_all,
                                        "train_acc_all":train_acc_all,
                                        "val_loss_all":val_loss_all,
                                        "val_acc_all":val_acc_all})
    return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"],train_process.train_loss_all,"ro-",label="Train Loss")
    plt.plot(train_process["epoch"],train_process.val_loss_all,"bs-",label="Val Loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1,2,2)
    plt.plot(train_process["epoch"],train_process.train_acc_all,"ro-",label="Train Acc")
    plt.plot(train_process["epoch"],train_process.val_acc_all,"bs-",label="Val Acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Acc")
    plt.savefig(".\\Alexnet\\train_process.png")
    plt.show()

if __name__ == "__main__":
    #将模型实例化
    LeNet = AlexNet()
    train_dataloader,val_dataloader = train_val_data_process()
    train_process = train_model_process(LeNet,train_dataloader,val_dataloader,num_epochs=20)
    matplot_acc_loss(train_process)