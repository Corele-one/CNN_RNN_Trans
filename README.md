# CNN_RNN_Trans

本仓库整理了几个深度学习练习项目，覆盖图像分类、时间序列建模和机器翻译等任务。代码以 PyTorch 和 TensorFlow/Keras 为主，包含模型定义、训练、测试、结果可视化和 Transformer 翻译流程。

## 项目结构

```text
CNN_RNN_Trans/
├── Alexnet/        # AlexNet + FashionMNIST 图像分类
├── LeNet/          # LeNet + FashionMNIST 图像分类
├── ResNet_18/      # ResNet-18 + FashionMNIST 图像分类
├── RNN_LSTM/       # LSTM 时间序列建模
├── Trans/          # Transformer 英中翻译
├── data/           # FashionMNIST 数据
└── run/            # 训练输出目录，不提交到仓库
```

## 环境依赖

CNN 与 Transformer 部分主要依赖：

```bash
pip install torch torchvision torchsummary matplotlib pandas numpy tqdm sacrebleu sentencepiece
```

LSTM 部分依赖见：

```bash
pip install -r RNN_LSTM/requirements.txt
```

如果使用 GPU，请确保本地 CUDA、PyTorch 或 TensorFlow 版本匹配。

## 快速运行

训练 CNN 图像分类模型：

```bash
python LeNet/model_train.py
python Alexnet/model_train.py
python ResNet_18/model_train.py
```

测试 CNN 模型：

```bash
python LeNet/model_test.py
python Alexnet/model_test.py
python ResNet_18/model_test.py
```

运行 LSTM 相关脚本：

```bash
python RNN_LSTM/model_train.py
python RNN_LSTM/model_val.py
python RNN_LSTM/model_test.py
```

训练和使用 Transformer 翻译模型：

```bash
python Trans/train.py
python Trans/train_v2.py
python Trans/translate.py
```

Transformer 的数据路径、词表大小、训练轮数、batch size、设备和权重路径集中在 `Trans/config.py` 中配置。迁移到其他机器时，建议先把其中的绝对路径改成适合本地环境的路径。

## 数据与权重

- FashionMNIST 数据放在 `data/FashionMNIST/`。
- Transformer 数据集可放在 `Trans/data/dataset/`，该目录按本地数据处理，不提交到仓库。
- Transformer 分词器模型和词表属于生成产物，可通过 `Trans/tools/tokenizer/tokenize.py` 重新生成。
- 模型权重、训练检查点和 `run/` 已加入 `.gitignore`，不会随普通提交进入仓库。
- 如果需要复现实验，请在本地准备数据并重新训练生成权重，或单独通过外部存储分发数据和权重文件。

## 说明

仓库中的训练脚本会在对应模型目录或 `run/` 目录下生成权重和训练结果图。结果图可以提交用于展示训练过程，但 `.pth`、`.h5` 等权重文件会被忽略，避免仓库体积过大。
