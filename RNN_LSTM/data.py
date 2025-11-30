import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

dataset = pd.read_csv('RNN_LSTM\\load.csv')
dataset = dataset.fillna(method='pad')

dataset = np.array(dataset)
print(dataset.shape)
print(dataset)

a = []
#把第二列数据提取到a列表中
for item in dataset:#遍历dataset中的每一行
    a.append(item[1])#把每一行的第二列数据添加到a列表中

dataset = pd.DataFrame(a)
# print(dataset.shape)
# print(dataset)
real = np.array(dataset)

# plt.figure(figsize=(20, 8))
# plt.plot(real)
# #设置xy轴的刻度值大小
# plt.xticks(fontsize=15)
# labels = ["一月","二月","三月","四月","五月","六月","七月","八月","九月","十月","十一月","十二月"]
# plt.yticks(fontsize=15)
# plt.xticks(range(0, 35040, 2920),labels=labels)
# plt.ylabel('负荷(MW)', fontsize=15)
# plt.xlabel('时间', fontsize=15)
# plt.show()

#切片，从第96*6行到第96*12行，第二列
week_data = dataset.iloc[96*6:96*13]
week_data = np.array(week_data)
plt.figure(figsize=(20, 8))
plt.plot(week_data)
#设置xy轴的刻度值大小
plt.xticks(fontsize=15)
labels = ["周一","周二","周三","周四","周五","周六","周日"]
plt.yticks(fontsize=15)
plt.xticks(range(0, 96* 7, 96),labels=labels)
plt.ylabel('负荷(MW)', fontsize=15) 
plt.xlabel('时间', fontsize=15)
plt.show()
