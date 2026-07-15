##模型参数脚本

import torch

d_model = 512
# 多头注意力中的头数
n_heads = 8

n_layers = 6

d_k = 64
d_v = 64

d_ff = 2048

dropout = 0.1

n_position = 2000

#词表相关超参数
src_vocab_size = 32000
tgt_vocab_size = 32000
padding_idx = 0
#padding_idx = 0表示填充token的索引为0，通常用于填充短句，使其长度与最长句子长度一致
bos_idx = 2
#bos_idx = 2表示句子开始token的索引为2，通常用于标记句子的开始
eos_idx = 3
#eos_idx = 3表示句子结束token的索引为3，通常用于标记句子的结束

#训练配置
batch_size = 16
epoch_num = 5
lr = 3e-4

#解码和生成设置
max_len = 60#表示解码生成的最大长度为60
#在计算BLEU评分时使用的Beam Search的大小
beam_size = 3

#文件路径配置

data_dir = 'D:\\23644\code\CNN_RNN_Trans\Trans\data\dataset'
train_data_path = 'D:\\23644\code\CNN_RNN_Trans\Trans\data\dataset\\train.json'
dev_data_path = 'D:\\23644\code\CNN_RNN_Trans\Trans\data\dataset\\dev.json'
test_data_path = 'D:\\23644\code\CNN_RNN_Trans\Trans\data\dataset\\test.json'
#模型保存路径
model_path = 'D:\\23644\code\CNN_RNN_Trans\Trans\model\model.pth'
#推理模型位置
test_model_path = 'run\\train\exp1\weights\\best_bleu_20.95.pth'

#设备配置
gpu_id = '0'
device_id = [0]
if gpu_id !='':
    device = torch.device(f'cuda:{gpu_id}')
else:
    device = torch.device('cpu')

