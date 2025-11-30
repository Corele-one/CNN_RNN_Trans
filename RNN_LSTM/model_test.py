import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.layers import Dense, LSTM, SimpleRNN
import numpy as np
import sklearn.exceptions

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
train = dataset.iloc[:int(len(dataset)*0.8), [0]]#训练集
test = dataset.iloc[int(len(dataset)*0.98):, [0]]#测试集

# print(train)#打印训练集
# print(val)#打印验证集
# print(test)#打印测试集

#数据归一化
#因为用电数据很大，计算会很慢，如果是多特征的情况下，数值越小的特征对结果的影响越小，所以要进行归一化
scaler = MinMaxScaler(feature_range=(0, 1))#创建归一化对象，将数据归一化到0-1之间
train = scaler.fit_transform(train)#对训练集进行归一化
test = scaler.transform(test)#对测试集进行归一化
X_test = []
y_test = []
for i in np.arange(96, len(test)):
    X_test.append(test[i-96:i, :])#前96个数据作为特征
    y_test.append(test[i])#第97个数据作为标签

X_test, y_test = np.array(X_test), np.array(y_test)#转换为numpy数组

#加载模型
model = load_model('RNN_LSTM\\LSTM_model.h5')#加载模型
#模型预测
y_pred = model.predict(X_test)#模型预测
y_pred = scaler.inverse_transform(y_pred)#反归一化
y_test = scaler.inverse_transform(y_test)#反归一化

#画图评估
plt.figure(figsize=(20, 8))#设置画布大小
plt.plot(y_test, label='真实值')#真实值
plt.plot(y_pred, label='预测值')#预测值
plt.legend()#显示图例
plt.show()#显示图像



