import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ==================== 数据加载和预处理 ====================
print("正在加载数据和模型...")

# 加载数据
dataset = pd.read_csv('RNN_LSTM\\load.csv', index_col=[0])  # 设置第一列为索引
dataset = dataset.fillna(method='pad')  # 填充缺失值
dataset = np.array(dataset)  # 转换为numpy数组
dataset = pd.DataFrame(dataset)  # 转换为pandas数据框

# 获取验证集（使用与训练时相同的划分方式）
val = dataset.iloc[int(len(dataset)*0.8):int(len(dataset)*0.9), [0]]  # 验证集

# 数据归一化（使用训练时的scaler参数）
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(val)  # 在验证集上fit
val_scaled = scaler.transform(val)  # 对验证集进行归一化

# ==================== 构建时间序列数据 ====================
print("正在构建时间序列数据...")

X_val = []
y_val = []

for i in np.arange(96, len(val_scaled)):
    X_val.append(val_scaled[i-96:i, 0])  # 前96个数据作为特征
    y_val.append(val_scaled[i, 0])       # 第97个数据作为标签

X_val, y_val = np.array(X_val), np.array(y_val)

# 重塑为3D，匹配LSTM输入要求 (样本数, 时间步长, 特征数)
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

# ==================== 模型加载和预测 ====================
print("正在加载模型并进行预测...")

# 加载训练好的模型
model = load_model('RNN_LSTM\\LSTM_model.h5')

# 进行预测
y_pred_scaled = model.predict(X_val)

# 反归一化得到真实值和预测值
prediction = scaler.inverse_transform(y_pred_scaled).flatten()
real = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

# ==================== 计算评估指标 ====================
print("正在计算评估指标...")

# 计算各种评估指标
R2 = r2_score(real, prediction)  # R²决定系数
MSE = mean_squared_error(real, prediction)  # 均方误差
MAE = mean_absolute_error(real, prediction)  # 平均绝对误差
RMSE = np.sqrt(mean_squared_error(real, prediction))  # 均方根误差

# 修正MAPE计算（分母用真实值real，避免除零）
MAPE = np.mean(np.abs(real - prediction) / real) * 100

# ==================== 打印评估结果 ====================
print("\n" + "=" * 50)
print("LSTM负荷预测模型评估指标")
print("=" * 50)
print(f"R² (决定系数):     {R2:>10.4f}")
print(f"MSE (均方误差):    {MSE:>10.4f}")
print(f"MAE (平均绝对误差): {MAE:>10.4f}")
print(f"RMSE (均方根误差):  {RMSE:>10.4f}")
print(f"MAPE (平均百分比):  {MAPE:>10.2f}%")
print("=" * 50 + "\n")

# ==================== 可视化预测结果 ====================
print("正在生成可视化图表...")

plt.figure(figsize=(15, 8))

# 绘制真实值与预测值对比
plt.plot(real, label='真实值', color='blue', linewidth=2, marker='o', markersize=3, alpha=0.8)
plt.plot(prediction, label='预测值', color='red', linestyle='--', linewidth=2, marker='s', markersize=3, alpha=0.8)

# 设置图表标题和标签
plt.title("基于LSTM神经网络的电力负荷预测", fontsize=16, fontweight='bold')
plt.ylabel("负荷值 (kW)", fontsize=14)
plt.xlabel("采样点", fontsize=14)

# 添加图例和网格
plt.legend(loc='best', fontsize=12, frameon=True, shadow=True)
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# 优化布局
plt.tight_layout()

# 保存图片
plt.savefig('RNN_LSTM/validation_result.png', dpi=300, bbox_inches='tight')
print("✅ 评估图表已保存为：RNN_LSTM/validation_result.png")

# 显示图表
plt.show()

print("✅ 模型评估完成！")