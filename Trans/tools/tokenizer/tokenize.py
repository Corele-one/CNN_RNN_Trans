import sentencepiece as spm

def train(input_file,vocab_size,model_name,model_type,character_coverage):
    
    input_argument =(
        '--input=%s '
        '--model_prefix=%s '
        '--vocab_size=%s '
        '--model_type=%s '
        '--character_coverage=%s '
        '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3'    
    )

    #将传入参数填充到命令字符串
    cmd = input_argument % (input_file,model_name,vocab_size,model_type,character_coverage)
    #训练模型
    spm.SentencePieceTrainer.Train(cmd)

def run():
    "英文分词器"
    en_input= "D:\\23644\code\CNN_RNN_Trans\Trans\data\dataset\\train.en"
    en_vocab_size = 32000
    en_model_name = "eng"
    en_model_type = "bpe"
    en_character_coverage = 1.0

    train(en_input,en_vocab_size,en_model_name,en_model_type,en_character_coverage)

    "中文分词器"
    zh_input= "D:\\23644\code\CNN_RNN_Trans\Trans\data\dataset\\train.zh"
    zh_vocab_size = 32000
    zh_model_name = "chn"
    zh_model_type = "bpe"
    zh_character_coverage = 0.9995

    train(zh_input,zh_vocab_size,zh_model_name,zh_model_type,zh_character_coverage)

def test():
    sp = spm.SentencePieceProcessor()
    text = "美国总统特朗普今日抵达夏威夷"

    sp.load("chn.model")
    print(sp.encode_as_pieces(text))
    print(sp.encode_as_ids(text))

    #示例

    a = [12907, 277, 7419, 18384, 28724]
    print(sp.decode_ids(a))

if __name__ == "__main__":
    run()
    # test()