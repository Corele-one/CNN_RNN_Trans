import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST

def test_data_process():
    testdata = FashionMNIST(root='./data',
                            train=False,
                            transform=transforms.Compose([transforms.Resize(size=28),transforms.ToTensor()]),#将图像转换为张量，并且将图像的大小调整为224*224
                            download=True)
    test_dataloader = Data.DataLoader(dataset = testdata,
                                     batch_size = 1,
                                     shuffle = True,
                                     num_workers = 8)
    return test_dataloader

def test_model_process(model,test_dataloader):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = model.to(device)
    #初始化参数
    test_corrects = 0.0
    test_num = 0

    with torch.no_grad():#不计算梯度,因为在测试过程中不需要计算梯度，反向传播
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)#将输入数据移动到设备上
            test_data_y = test_data_y.to(device)#将标签移动到设备上
            model.eval()#将模型设置为评估模式
            output = model(test_data_x)#前向传播
            pre_lab = torch.argmax(output,dim=1)#经过softmax函数后得到概率分布，使用argmax函数得到预测类别
            test_corrects += torch.sum(pre_lab == test_data_y.data)#计算正确数
            test_num += test_data_x.size(0)#计算样本数
    #计算测试准确率
    test_acc = test_corrects.double().item()/test_num#计算准确率
    print('Test Acc: {:.4f}'.format(test_acc))#打印准确率




if __name__ == '__main__':
    from model import AlexNet
    import torch.nn as nn
    import torch
    #加载模型
    model = AlexNet()#实例化模型
    model.load_state_dict(torch.load('.\LeNet\\best_model.pth'))#加载最好的模型
    #加载模型测试函数
    test_dataloader = test_data_process()#加载测试数据
    #test_model_process(model,test_dataloader)#测试模型
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = model.to(device)
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    with torch.no_grad():#不计算梯度,因为在测试过程中不需要计算梯度，反向传播
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)#将输入数据移动到设备上
            b_y = b_y.to(device)#将标签移动到设备上
            model.eval()#将模型设置为评估模式
            output = model(b_x)#前向传播
            pre_lab = torch.argmax(output,dim=1)#经过softmax函数后得到概率分布，使用argmax函数得到预测类别
            result = pre_lab.item()#获取预测结果
            label = b_y.item()#获取真实标签
            print("预测值：",classes[result],"----","真实值：",classes[label])
            
