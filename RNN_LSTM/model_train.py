import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN
import keras
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

#加载数据
dataset = pd.read_csv('RNN_LSTM\\load.csv', index_col=[0])#设置第一列为索引
dataset = dataset.fillna(method='pad')#填充缺失值
#print(dataset)
dataset = np.array(dataset)#转换为numpy数组
# print(dataset.shape)#打印数组形状
#print(dataset)#打印数组

dataset = pd.DataFrame(dataset)#转换为pandas数据框
# print(dataset)#打印数据框

#数据集划分
train = dataset.iloc[:int(len(dataset)*0.8), :]#训练集

val = dataset.iloc[int(len(dataset)*0.8):int(len(dataset)*0.9), :]#验证集

test = dataset.iloc[int(len(dataset)*0.9):, :]#测试集

# print(train)#打印训练集
# print(val)#打印验证集
# print(test)#打印测试集

#数据归一化
#因为用电数据很大，计算会很慢，如果是多特征的情况下，数值越小的特征对结果的影响越小，所以要进行归一化
scaler = MinMaxScaler(feature_range=(0, 1))#创建归一化对象，将数据归一化到0-1之间
train = scaler.fit_transform(train)#对训练集进行归一化
val = scaler.transform(val)#对验证集进行归一化
test = scaler.transform(test)#对测试集进行归一化
# print(train)#打印归一化后的训练集
# print(val)#打印归一化后的验证集
# print(test)#打印归一化后的测试集 
#LSTM需要三维数据，分别为样本数、时间步长、特征数，这里只有一个特征，所以特征数为1
#需要进行标签和特征的划分，用前96个数据来预测下一时刻的数据（单特征预测单特征）

X_train = []
y_train = []
for i in np.arange(96, len(train)):
    X_train.append(train[i-96:i, :])#前96个数据作为特征
    y_train.append(train[i])#第97个数据作为标签

X_train, y_train = np.array(X_train), np.array(y_train)#转换为numpy数组

X_val = []
y_val = []
for i in np.arange(96, len(val)):
    X_val.append(val[i-96:i, :])#前96个数据作为特征
    y_val.append(val[i])#第97个数据作为标签

X_val, y_val = np.array(X_val), np.array(y_val)#转换为numpy数组

model = Sequential()#创建模型
#RNN层，输入形状为(96, 1)，输出形状为(10)，激活函数为relu
model.add(LSTM(10, return_sequences = True, activation = 'relu'))
#RNN层，输入形状为(10, 1)，输出形状为(15)，激活函数为relu
model.add(LSTM(15, return_sequences = False, activation = 'relu'))#只有在最后一层需要返回序列
#return_sequnce = True表示返回序列，False表示不返回序列，返回的序列用于下一个时间步的输入
#Dense层，用于全连接，输出形状为(10)，激活函数为relu
model.add(Dense(10,activation ='relu'))
model.add(Dense(1))

model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.001), loss = 'mse')#编译模型，优化器为Adam，损失函数为mse(均方误差)

history = model.fit(X_train, y_train, epochs = 30, batch_size = 512, validation_data =(X_val, y_val))#训练模型，epochs为训练轮数，batch_size为每次训练的样本数，validation_data为验证集
#模型的保存
model.save('RNN_LSTM\\LSTM_model.h5')#保存模型为h5格式，model.h5为模型的名称

#绘制训练损失和验证损失的曲线
plt.figure(figsize=(12, 8))#设置画布大小
plt.plot(history.history['loss'], label='Training Loss')#绘制训练损失曲线，label为图例的标签
plt.plot(history.history['val_loss'], label='Validation Loss')#绘制验证损失曲线，label为图例的标签
plt.title('Training and Validation Loss')#设置标题
plt.xlabel('Epochs')#设置x轴标签
plt.ylabel('Loss')#设置y轴标签
plt.xticks(fontsize=15)#设置x轴刻度值大小
plt.yticks(fontsize=15)#设置y轴刻度值大小
plt.legend()#显示图例
plt.show()
