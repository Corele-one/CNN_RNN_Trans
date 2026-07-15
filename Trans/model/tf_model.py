import config
import math
import copy
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

DEVICE = config.device

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

##Transformer模型##

##词嵌入层

class Embeddings(nn.Module):
    #初始化方法，传入模型的维度和词汇表的大小
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)  #创建一个嵌入层，将每个单词映射到一个d_model维的向量
        self.d_model = d_model  #保存模型的维度

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)  #对输入的单词进行嵌入

##位置编码层
class PositionalEncoding(nn.Module):
    #初始化方法，传入模型的维度和最大序列长度
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)  #创建一个Dropout层，用于正则化
        #初始化一个size为max_size x d_model的矩阵，用于存储位置编码
        pe = torch.zeros(max_len, d_model, device=DEVICE)
        #创建一个size为max_size x 1的矩阵，用于存储位置
        position = torch.arange(0.,max_len, device=DEVICE).unsqueeze(1)
        #unsqueeze(1)将position从size为max_size的向量变为size为max_size x 1的矩阵
        #创建一个size为1 x d_model的矩阵，用于存储位置编码
        div_term = torch.exp(torch.arange(0., d_model, 2, device=DEVICE) * -(math.log(10000.0) / d_model))
        #计算位置编码
        pe[:,0::2] = torch.sin(position * div_term) #[:,0::2]表示从第0列开始，每隔2列取一个元素，即奇数列
        pe[:,1::2] = torch.cos(position * div_term)
        #加一个维度，使得pe的size变为1 x max_size x d_model，便于后面的计算
        pe = pe.unsqueeze(0)#unsqueeze的作用是在指定位置增加维度，这里是在第0维增加维度，即增加一个batch维度
        self.register_buffer('pe', pe)#register_buffer的作用是将pe注册为模型的一个buffer，即不会被优化器更新，而是作为模型的一个固定参数

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False) #输入词向量加上位置编码，这里的x.size(1)表示输入序列的长度
        return self.dropout(x) #对位置编码进行Dropout
    
##Attention层
#核心部分，Q和K的点积，然后softmax，然后乘以V，得到输出
def attention(query, key, value, mask=None, dropout=None):
        #将query矩阵的最后一个维度作为d_k
        d_k = query.size(-1)
        #计算query和key的点积，得到一个size为batch_size x n_heads x seq_len x seq_len的矩阵，这里的seq_len表示输入序列的长度，n_heads表示多头注意力机制中的头数
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) #transpose的作用是交换矩阵的两个维度，这里是交换最后两个维度，即交换seq_len和d_k
        #如果需要填充的情况下，就把0替换成-inf，这样就不会被softmax计算到
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        #将掩码后的attention矩阵按照最后一个维度进行softmax
        p_attn = F.softmax(scores, dim=-1) #softmax的作用是将矩阵中的每个元素转换为一个概率，这里的dim=-1表示按照最后一个维度进行softmax

        #如果需要dropout的情况下，就对attention矩阵进行dropout
        if dropout is not None:
            p_attn = dropout(p_attn)

        #将attention矩阵乘以value矩阵，返回融合的信息和得分矩阵
        return torch.matmul(p_attn, value), p_attn
    
class MultiHeadedAttention(nn.Module): #可以实现transformer中的三种多头注意力机制，分别是自注意力机制，交叉注意力机制，和全连接注意力机制
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        #确保可以整除
        assert d_model % h == 0
        self.d_k = d_model // h #计算每个头的维度
        self.h = h #计算头数
        self.linears = clones(nn.Linear(d_model, d_model), 4) 
        #创建四个线性变换层，用于将QKV矩阵进行线性变换，
        # 这里的4表示有四个线性变换层，分别是QKV和输出OUT
        self.attn = None #初始化得分矩阵为None
        self.dropout = nn.Dropout(p=dropout) #创建一个Dropout层，用于正则化


    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) #unsqueeze的作用是在指定位置增加维度，这里是在第1维增加维度，即增加一个batch维度
            #增加一个维度，方便之后的并行计算
        nbatches = query.size(0) #获取batch_size，即输入序列的个数
        #将query，key，value矩阵分别进行线性变换，得到一个size为batch_size x n_heads x seq_len x d_k的矩阵，这里的seq_len表示输入序列的长度，n_heads表示多头注意力机制中的头数，d_k表示每个头的维度

        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        '''
        nbatches = query.size(0) #获取batch_size,即输入序列的个数
        heads = self.h #获取头数
        self.d_k = query.size(-1) #获取每个头的维度
        transpose的作用可以理解成求矩阵的转置
        view()指的是改变矩阵的形状,输入矩阵的形状为nbatches x h x d_k
        zip的输出为两个参数,其中l是线性变换层,x是输入的query,key,value矩阵,
        l(x)指的是将QKV矩阵进行线性变换
        然后通过view函数将线性变换后的矩阵的形状变为nbatches x seq_len x h x d_k,
        将query,key,value矩阵分别进行线性变换
        '''
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        #经过注意力函数，x为融合的信息，self.attn为得分矩阵
        #将n个注意力矩阵concat起来
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k) #transpose的作用是交换矩阵的两个维度，这里是交换第1和第2个维度，即交换h和d_k，然后view的作用是改变矩阵的形状，输入矩阵的形状为nbatches x seq_len x h x d_k
        #contiguous()的作用是将矩阵的内存布局变为连续的，这样就可以进行view操作了
        #view操作将矩阵的形状又变为nbatches x seq_len x h*d_k，即将n个注意力矩阵concat起来
        
        return self.linears[-1](x) 
        #将concat后的矩阵进行线性变换，得到一个size为batch_size x seq_len x d_model的矩阵，
        # 然后用最后一个全连接层进行线性变换，得到最终的输出矩阵

##层归一化
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features)) #创建一个size为features的向量，用于存储a_2
        self.b_2 = nn.Parameter(torch.zeros(features)) #创建一个size为features的向量，用于存储b_2
        self.eps = eps #创建一个变量，用于存储eps

    def forward(self, x):

        mean = x.mean(-1, keepdim=True) #计算均值，这里的-1表示最后一个维度，keepdim=True表示保持原来的维度
        std = x.std(-1, keepdim=True) #计算标准差，这里的-1表示最后一个维度，keepdim=True表示保持原来的维度
        return self.a_2 * (x - mean) / torch.sqrt(std**2 + self.eps) + self.b_2
        #这里的eps是一个很小的数，用于防止分母为0的情况，防止出现nan的情况
        #其中a，b是两个可学习的参数，用于调整归一化的结果

##前馈神经网络
class PositionwiseFeedForward(nn.Module): #前馈神经网络，包含两个线性层和一个ReLU激活函数
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff) #创建一个线性变换层，用于将输入矩阵的形状变为d_ff
        self.w_2 = nn.Linear(d_ff, d_model) #创建一个线性变换层，用于将输入矩阵的形状变为d_model
        self.dropout = nn.Dropout(dropout) #创建一个Dropout层，用于正则化

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x)))) #将输入矩阵进行线性变换，然后进行ReLU激活函数，然后进行Dropout，然后进行线性变换，得到最终的输出矩阵
    #x——>w_1——>ReLU——>Dropout——>w_2——>x

#-----以上基本模块已经构建完成-----#

#因为在transformer中的解码器和编码器有着相似的结构，所以可以构建一个子层，用于编码器和解码器的构建
#通过sc层将多个子层连接起来，方便信息的流动，并且可以进行残差连接，使得信息的流动更加稳定
class SublayerConnection(nn.Module): #子层连接，包含一个残差连接和一个层归一化
    def __init__(self, size, dropout): #size表示输入矩阵的大小，dropout表示Dropout的概率
        super(SublayerConnection, self).__init__() #调用父类的初始化方法
        self.norm = LayerNorm(size) #创建一个层归一化层，用于对输入矩阵进行归一化
        self.dropout = nn.Dropout(dropout) #创建一个Dropout层，用于正则化

    def forward(self, x, sublayer): #x表示输入矩阵，sublayer表示子层
        #返回Layer Norm和残差连接后的结果
        return x + self.dropout(sublayer(self.norm(x))) #Layer Norm和残差连接的过程是将输入矩阵进行归一化，然后进行Dropout，然后进行子层的计算，然后进行残差连接，得到最终的输出矩阵

#构建编码器的子层
class EncoderLayer(nn.Module): #编码器的子层，包含一个多头注意力层和一个前馈神经网络
    def __init__(self, size, self_attn, feed_forward, dropout): #size表示输入矩阵的大小，self_attn表示多头注意力层，feed_forward表示前馈神经网络，dropout表示Dropout的概率
        super(EncoderLayer, self).__init__() #调用父类的初始化方法
        self.self_attn = self_attn #创建一个多头注意力层，用于对输入矩阵进行多头注意力计算
        self.feed_forward = feed_forward #创建一个前馈神经网络，用于对输入矩阵进行前馈神经网络计算
        self.sublayer = clones(SublayerConnection(size, dropout), 2) #创建两个子层连接层，用于对输入矩阵进行子层连接计算，这里的2表示有两个子层连接层
        #两个子层连接层，第一个子层连接层将多头注意力机制与fnn相连接
        #第二个子层将fnn的输出和第一个子层连接层的输出（fnn的输入）相连接
        self.size = size #创建一个变量，用于存储输入矩阵的大小
    
    def forward(self, x, mask): #x表示输入矩阵，mask表示掩码矩阵
        #将embedding层进行多头注意力机制
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) #将输入矩阵进行多头注意力计算，然后进行子层连接，得到最终的输出矩阵
        #lambda表达式用于内连attention
        return self.sublayer[1](x, self.feed_forward) #将输入矩阵进行前馈神经网络计算，然后进行子层连接，得到最终的输出矩阵

#构建编码器
class Encoder(nn.Module): #编码器，包含多个编码器子层
    def __init__(self, layer, N): #layer表示编码器子层，N表示编码器子层的个数
        super(Encoder, self).__init__() #调用父类的初始化方法
        self.layers = clones(layer, N) #创建N个编码器子层，用于对输入矩阵进行编码器子层计算，这里的N表示编码器子层的个数
        self.norm = LayerNorm(layer.size) #创建一个层归一化层，用于对输入矩阵进行归一化

    def forward(self, x, mask): #x表示输入矩阵，mask表示掩码矩阵
        for layer in self.layers: #循环调用N个编码器子层
            x = layer(x, mask) #将输入矩阵进行编码器子层计算，得到最终的输出矩阵
        return self.norm(x) #将输入矩阵进行层归一化，得到最终的输出矩阵

#构建解码器的子层
class DecoderLayer(nn.Module): #解码器的子层，包含一个多头注意力层和两个前馈神经网络
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout): 
        #size表示输入矩阵的大小，self_attn表示多头注意力层，src_attn表示多头注意力层(src表示源语言，tgt表示目标语言)
        # feed_forward表示前馈神经网络，dropout表示Dropout的概率
        super(DecoderLayer, self).__init__() #调用父类的初始化方法
        self.size = size #创建一个变量，用于存储输入矩阵的大小  
        self.self_attn = self_attn #创建一个多头注意力层，用于对输入矩阵进行多头注意力计算
        self.src_attn = src_attn #创建一个多头注意力层，用于对输入矩阵进行多头注意力计算，这里的多头注意力层是用于计算目标语言和源语言之间的注意力
        self.feed_forward = feed_forward #创建一个前馈神经网络，用于对输入矩阵进行前馈神经网络计算
        self.sublayer = clones(SublayerConnection(size, dropout), 3) #创建三个子层连接层，用于对输入矩阵进行子层连接计算，这里的3表示有三个子层连接层

    def forward(self, x, memory, src_mask, tgt_mask): 
        #x表示输入矩阵，memory表示编码器的输出矩阵，src_mask表示编码器的掩码矩阵，tgt_mask表示解码器的掩码矩阵
        m = memory #将编码器的输出矩阵赋值给m
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)) #将输入矩阵进行多头注意力计算，然后进行子层连接，得到最终的输出矩阵
        #lambda表达式用于内连attention
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask)) #将输入矩阵进行多头注意力计算，然后进行子层连接，得到最终的输出矩阵
        return self.sublayer[2](x, self.feed_forward) #将输入矩阵进行前馈神经网络计算，然后进行子层连接，得到最终的输出矩阵
    
#构建解码器
class Decoder(nn.Module): #解码器，包含多个解码器子层
    def __init__(self, layer, N): #layer表示解码器子层，N表示解码器子层的个数
        super(Decoder, self).__init__() #调用父类的初始化方法
        self.layers = clones(layer, N) #创建N个解码器子层，用于对输入矩阵进行解码器子层计算，这里的N表示解码器子层的个数
        self.norm = LayerNorm(layer.size) #创建一个层归一化层，用于对输入矩阵进行归一化

    def forward(self, x, memory, src_mask, tgt_mask): #x表示输入矩阵，memory表示编码器的输出矩阵，src_mask表示编码器的掩码矩阵，tgt_mask表示解码器的掩码矩阵
        for layer in self.layers: #循环调用N个解码器子层
            x = layer(x, memory, src_mask, tgt_mask) #将输入矩阵进行解码器子层计算，得到最终的输出矩阵
        return self.norm(x) #将输入矩阵进行层归一化，得到最终的输出矩阵

#构建生成器
class Generator(nn.Module): #生成器，包含一个线性变换层和一个softmax层
    #生成器将解码器的输出矩阵映射到目标语言的词汇表上，得到一个概率分布
    def __init__(self, d_model, vocab): #d_model表示输入矩阵的大小，vocab表示目标语言的词汇表
        super(Generator, self).__init__() #调用父类的初始化方法
        self.proj = nn.Linear(d_model, vocab) #创建一个线性变换层，用于将输入矩阵的形状变为vocab
        #vocab表示目标语言的词汇表大小，d_model表示输入矩阵的大小
    
    def forward(self, x): #x表示输入矩阵
        return F.log_softmax(self.proj(x), dim=-1) #将输入矩阵进行线性变换，然后进行softmax，得到最终的输出矩阵
        #log_softmax的作用是将softmax的输出取对数，这样就可以进行交叉熵计算了
        #dim=-1表示对最后一个维度进行softmax，这里的-1表示最后一个维度，即对每个单词进行softmax
#----以上编码器和解码器的基本模块已经构建完成----#

#构建transformer模型
class Transformer(nn.Module): #transformer模型，包含一个编码器和解码器
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator): #encoder表示编码器，decoder表示解码器，src_embed表示源语言的嵌入层，tgt_embed表示目标语言的嵌入层，generator表示生成器
        super(Transformer, self).__init__() #调用父类的初始化方法
        self.encoder = encoder #创建一个编码器，用于对输入矩阵进行编码器计算，这里的编码器是一个编码器对象
        self.decoder = decoder #创建一个解码器，用于对输入矩阵进行解码器计算，这里的解码器是一个解码器对象
        self.src_embed = src_embed #创建一个嵌入层，用于将输入矩阵的形状变为d_model，这里的d_model表示输入矩阵的大小
        self.tgt_embed = tgt_embed    
        self.generator = generator #创建一个生成器，用于将输入矩阵映射到目标语言的词汇表上，得到一个概率分布

    def encode(self, src, src_mask): #src表示输入矩阵，src_mask表示输入矩阵的掩码矩阵
        return self.encoder(self.src_embed(src), src_mask) #将输入矩阵进行嵌入层计算，然后进行编码器计算，得到最终的输出矩阵
    
    def decode(self, memory, src_mask, tgt, tgt_mask): #memory表示编码器的输出矩阵，src_mask表示编码器的掩码矩阵，tgt表示输入矩阵，tgt_mask表示输入矩阵的掩码矩阵
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask) #将输入矩阵进行嵌入层计算，然后进行解码器计算，得到最终的输出矩阵

    def forward(self, src, tgt, src_mask, tgt_mask): #src表示输入矩阵，tgt表示输入矩阵，src_mask表示输入矩阵的掩码矩阵，tgt_mask表示输入矩阵的掩码矩阵
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask) #将输入矩阵进行编码器计算，然后进行解码器计算，得到最终的输出矩阵

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy #复制一个模型
    attn = MultiHeadedAttention(h, d_model) #创建一个多头注意力层，用于对输入矩阵进行多头注意力计算，这里的h表示多头注意力机制中的头数，d_model表示输入矩阵的大小
    ff = PositionwiseFeedForward(d_model, d_ff, dropout) #创建一个前馈神经网络，用于对输入矩阵进行前馈神经网络计算，这里的d_model表示输入矩阵的大小，d_ff表示前馈神经网络的隐藏层的大小，dropout表示Dropout的概率
    position = PositionalEncoding(d_model, dropout) #创建一个位置编码层，用于对输入矩阵进行位置编码计算，这里的d_model表示输入矩阵的大小，dropout表示Dropout的概率

    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N).to(DEVICE), 
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N).to(DEVICE),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)).to(DEVICE)
    
    #初始化模型参数
    for p in model.parameters():
        if p.dim() > 1: #如果p的维度大于1，就进行初始化
            nn.init.xavier_uniform_(p) #使用xavier_uniform_方法进行初始化，这里的xavier_uniform_方法是一种初始化方法，用于初始化神经网络的参数
    return model #返回模型