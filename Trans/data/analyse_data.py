import os



def analyse_corpus(ch_path, en_path):
    """"
    分析双语语料库文件的详细信息
    Args:
        ch_path: 中文语料库文件路径
        en_path: 英文语料库文件路径
    """

    #检查文件是否存在
    if not os.path.exists(ch_path):
        print(f"Error: 中文语料库文件不存在: {ch_path}")
        return
    if not os.path.exists(en_path):
        print(f"Error: 英文语料库文件不存在: {en_path}")
        return
    
    #获取文件大小

    ch_size = os.path.getsize(ch_path) / (1024 * 1024) #MB
    en_size = os.path.getsize(en_path) / (1024 * 1024) #MB

    #读取文件内容并统计行数
    with open(ch_path, 'r', encoding='utf-8') as ch_file:
        ch_lines = ch_file.readlines()
    
    with open(en_path, 'r', encoding='utf-8') as en_file:
        en_lines = en_file.readlines()

    ch_chars = sum(len(line) for line in ch_lines) #字符数
    en_chars = sum(len(line) for line in en_lines) #字符数

    #打印统计信息
    print("="*50)
    print("语料库文件分析结果")
    print("="*50)
    print(f"中文语料库文件: {ch_path}")
    print(f"文件大小: {ch_size:.2f}MB")
    print(f"行数: {len(ch_lines)}")
    print(f"字符数: {ch_chars}")
    print("平均每行字符数: {ch_chars/len(ch_lines):.2f}")

    print("="*50)
    print(f"英文语料库文件: {en_path}")
    print(f"文件大小: {en_size:.2f}MB")
    print(f"行数: {len(en_lines)}")
    print(f"字符数: {en_chars}")
    print("平均每行字符数: {ch_chars/len(ch_lines):.2f}")
    print("="*50)

    #验证中英文行数是否一致
    if len(ch_lines) != len(en_lines):
        print("Error: 中英文行数不一致")
        return
    else:
        print("中英文行数一致")
    print("="*50)


if __name__ == "__main__":
    ch_path = "D:\\23644\code\CNN_RNN_Trans\Trans\data\dataset\\train.zh"
    en_path = "D:\\23644\code\CNN_RNN_Trans\Trans\data\dataset\\train.en" 
    analyse_corpus(ch_path, en_path)